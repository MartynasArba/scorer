import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import math
import logging
import random
from pathlib import Path
import datetime

from scorer.data.loaders import BufferedSleepDataset
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN, SimCLRLoss
from scorer.models.sleep_cnn import SCDSSleepCNN

def setup_logger(save_dir: Path):
    log_file = save_dir / f"pretraining_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def batch_augment(x: torch.Tensor, sample_rate = 250.) -> torch.Tensor:
    """vectorized augmentations per-sample"""
    x = x.clone()
    bsz, n_channels, seq_len = x.shape
    device = x.device
    
    # noise per sample (70% chance)
    apply_noise = (torch.rand(bsz, 1, 1, device=device) < 0.7).float()
    x = x + (apply_noise * torch.randn_like(x) * 0.01)
    
    # log scaling per sample (70% chance)
    apply_scale = (torch.rand(bsz, 1, 1, device=device) < 0.7).float()
    #uniform -1:1
    rand_exponents = (torch.rand(bsz, n_channels, 1, device=device) * 2.0) - 1.0
    scales = torch.exp(rand_exponents * math.log(10.0)) #exp scale
    x = x * ((1 - apply_scale) + apply_scale * scales)
    
    # 50% chance to apply baseline shift (dynamically replicates offset)
    apply_dc = (torch.rand(bsz, 1, 1, device=device) < 0.5).float()
    # uniform random shift between -0.8 and 0.8 per channel
    dc_shifts = (torch.rand(bsz, n_channels, 1, device=device) * 1.4) - 0.7
    x = x + (apply_dc * dc_shifts)
    
    #frequency mask, per-sample (30% chance)
    mask_prob = 0.3
    # choose batch samples that get masked
    apply_mask = torch.rand(bsz, 1, 1, device=device) < mask_prob
    
    if apply_mask.any():
        fft_x = torch.fft.rfft(x, dim=-1)
        freq_bins = fft_x.shape[-1]
        
        # bin array
        freq_idx = torch.arange(freq_bins, device=device).view(1, 1, freq_bins)
        # theta bin bounds
        theta_start = int(6 * (seq_len / sample_rate))
        theta_end = int(9 * (seq_len / sample_rate))
        
        # 50/50 choice: low-pass or band-stop for the batch
        is_lowpass = torch.rand(bsz, 1, 1, device=device) < 0.5
        
        # low-pass
        min_cutoff = min(int(30 * (seq_len / sample_rate)), freq_bins - 1)
        cutoffs = torch.randint(min_cutoff, freq_bins, (bsz, 1, 1), device=device)
        lp_mask = freq_idx < cutoffs
        
        #band stop
        min_bs_start = theta_end + 5
        max_bs_start = max(min_bs_start + 1, freq_bins - 40)
        bs_starts = torch.randint(min_bs_start, max_bs_start, (bsz, 1, 1), device=device)
        # keep everything not in [bs_starts, bs_starts + 40) range
        bs_mask = ~((freq_idx >= bs_starts) & (freq_idx < bs_starts + 40))
        
        # Combine masks based on the 50/50 choice
        action_mask = torch.where(is_lowpass, lp_mask, bs_mask)
        
        # ensure theta is kept
        theta_protected = (freq_idx >= theta_start) & (freq_idx < theta_end)
        action_mask = action_mask | theta_protected
        
        # apply mask for selected samples
        final_mask = torch.where(apply_mask, action_mask, torch.ones_like(action_mask, dtype=torch.bool))
        fft_x = fft_x * final_mask.float()
        x = torch.fft.irfft(fft_x, n=seq_len, dim=-1)
    
    #time shift, 50% chance
    shift_prob = 0.5
    max_shift = int(seq_len * 0.02)
    apply_shift = torch.rand(bsz, device=device) < shift_prob
    shifts = torch.randint(-max_shift, max_shift + 1, (bsz,), device=device)
    shifts = shifts * apply_shift.long()
    # roll
    rows = torch.arange(bsz, device=device).view(bsz, 1, 1)
    chans = torch.arange(n_channels, device=device).view(1, n_channels, 1)
    # create shifted indices
    indices = (torch.arange(seq_len, device=device).view(1, 1, seq_len) - shifts.view(bsz, 1, 1)) % seq_len
    x = x[rows, chans, indices]
        
    # random channel dropout/noise (30% chance) - only if >1 channel
    if x.shape[1] > 1:
        dropout_mask = (torch.rand(bsz, 1, 1, device=device) < 0.3).float()
        noise = torch.randn(bsz, 1, x.shape[2], device=device) * (torch.rand(bsz, 1, 1, device=device) * 1e-4)
        x[:, 1:2, :] = (1 - dropout_mask) * x[:, 1:2, :] + dropout_mask * noise

    return x

def train_supcon(model, dataset, logger, save_dir = '.', epochs=50, batch_size=256):
    """supervised contrastive training loop"""
    torch.set_grad_enabled(True)
    save_dir = Path(save_dir)
    device = next(model.parameters()).device
    base_cnn = model.encoder 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #low lr because should be pretrained
    criterion = SupConLoss(temperature = 0.2)   #increased to make more generalized embeddings 
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=4,
                            pin_memory = True,
                            prefetch_factor = 2)

    best_loss = float('inf')
    early_stop_counter = 0
    patience = 5
    min_delta = 0.001

    for epoch in range(epochs):
        model.train()
        logger.info(f'Epoch {epoch} SupCon training started')
        
        running_loss = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0
        num_batches = 0
        
        for samples, labels in dataloader:
            samples, labels = samples.to(device, non_blocking = True), labels.to(device, non_blocking = True)
            
            # label alignment check
            if samples.shape[0] != labels.shape[0]:
                logger.error(f"Batch mismatch: samples({samples.shape[0]}) != labels({labels.shape[0]})")
                raise ValueError("Input samples and labels must have the same batch size.")

            view1 = batch_augment(samples) 
            view2 = batch_augment(samples)
            
            images = torch.cat([view1, view2], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            # Verify combined label integrity
            assert torch.equal(combined_labels[:samples.shape[0]], combined_labels[samples.shape[0]:]), "Label misalignment after cat"

            optimizer.zero_grad()
            features = model(images)          
            loss = criterion(features, combined_labels)
            
            # Sanity Check: Feature Stability
            if torch.isnan(features).any() or torch.isnan(loss):
                logger.error("NaN detected in features or loss. Aborting.")
                return base_cnn, loss

            feature_var = features.std(dim=0).mean().item()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # debug tracking (compute pos/neg similarity)
            with torch.no_grad():
                sim_matrix = torch.matmul(features, features.T)
                mask = torch.eq(combined_labels.unsqueeze(1), combined_labels.unsqueeze(0)).float()
                
                pos_sim = (sim_matrix * mask).sum() / mask.sum()
                neg_sim = (sim_matrix * (1 - mask)).sum() / (1 - mask).sum()
                
                running_pos_sim += pos_sim.item()
                running_neg_sim += neg_sim.item()
            
        epoch_loss = running_loss / num_batches
        
        if epoch_loss < (best_loss - min_delta):
            # loss improved
            best_loss = epoch_loss
            early_stop_counter = 0
            
            logger.info(f"New best loss: {best_loss:.4f}, saving model.")
            torch.save(base_cnn.state_dict(), save_dir / f'supcon_SCDS_best_model.pt')
            
        else:
            early_stop_counter += 1
            logger.info(f"Early stopping counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                torch.save(base_cnn.state_dict(), save_dir / f'supcon_SCDS_snapshot_{epoch}.pt')
                break
            
        if feature_var < 1e-4:
            logger.warning(f"Low feature variance detected ({feature_var:.6f}). Model might be collapsing.")

        if epoch % 20 == 0:
            torch.save(base_cnn.state_dict(), save_dir / f'supcon_SCDS_snapshot_{epoch}.pt')
            
        logger.info(
            f"Epoch [{epoch}/{epochs}]: loss {epoch_loss:.4f} | "
            f"PosSim: {running_pos_sim/num_batches:.3f} | NegSim: {running_neg_sim/num_batches:.3f} | "
            f"FeatVar: {feature_var:.4f}"
        )
    
    return base_cnn, loss 

def train_unsupervised(model, dataset, logger, save_dir = '.', epochs=50, batch_size=256):
    """unsupervised contrastive training loop (SimCLR)"""
    torch.set_grad_enabled(True)
    save_dir = Path(save_dir)
    device = next(model.parameters()).device
    base_cnn = model.encoder
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = SimCLRLoss(temperature=0.15) 
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=4,
                            pin_memory = True,
                            prefetch_factor = 2)
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 5
    min_delta = 0.001

    for epoch in range(epochs):
        model.train()
        logger.info(f'Epoch {epoch} SimCLR training started')
        
        running_loss = 0.0
        num_batches = 0
        
        for samples, labels in dataloader:
            samples, labels = samples.to(device, non_blocking = True), labels.to(device, non_blocking = True)
            
            view1 = batch_augment(samples) 
            view2 = batch_augment(samples)
            
            optimizer.zero_grad()
            
            # separate views for SimCLR
            z1 = model(view1)          
            z2 = model(view2)
            
            # compute loss
            loss = criterion(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            
        epoch_loss = running_loss / num_batches
        
        if epoch_loss < (best_loss - min_delta):
            # loss improved
            best_loss = epoch_loss
            early_stop_counter = 0
            
            logger.info(f"New best loss: {best_loss:.4f}, saving model.")
            torch.save(base_cnn.state_dict(), save_dir / f'SimCLR_best_model.pt')
            
        else:
            early_stop_counter += 1
            logger.info(f"Early stopping counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                torch.save(base_cnn.state_dict(), save_dir / f'SimCLR_SCDS_snapshot_{epoch}.pt')
                break
        
        if epoch % 20 == 0:
            torch.save(base_cnn.state_dict(), save_dir / f'SimCLR_SCDS_snapshot_{epoch}.pt')
            
        logger.info(f"Epoch [{epoch}/{epochs}]: loss {epoch_loss:.4f}")
    
    return base_cnn, model, loss 


if __name__ == "__main__":
    
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path_for_simclr = r"C:\Users\marty\Desktop\train_sets\unlabeled"
    path_for_supcon= r"C:\Users\marty\Desktop\train_sets\labeled"
    meta = {'ecog_channels' : '0', 'emg_channels' : '1', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 100
    n_epochs = 100
    model_name = '3state_contrastiveSCDS_2026-04-15.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models\weights")
    save_path.mkdir(parents=True, exist_ok=True) # ensure dir exists
    batch_size = 1024
    
    logger = setup_logger(save_path)
    
    logger.info('Starting pretraining pipeline...')
    
    # load full dataset for unsupervised learning
    dataset = BufferedSleepDataset(
        data_path=path_for_simclr,
        n_files_to_pick=None,
        buffer_size=n_files, 
        random_state=0,
        device='cpu',
        transform=None,
        augment=False,
        metadata=meta, 
        normalize= True,
        balance='none',
        exclude_labels=(0,),
        merge_nrem=True
    )    
        
    logger.info('Initializing model architecture')
    base_cnn = SCDSSleepCNN(num_classes=3).to(device)    
    model = SupConSleepCNN(base_cnn).to(device)
    
    # simclr pretraining
    logger.info('Starting unsupervised SimCLR pretraining stage')
    pretrained_cnn, model, unsup_loss = train_unsupervised(model, dataset, logger, save_dir = save_path, epochs=50, batch_size=batch_size)
    
    # save weights just in case
    torch.save(pretrained_cnn.state_dict(), save_path / 'SimCLR_base_weights.pt')
    
    #reload dataset
    del dataset #free ram
    logger.info('Reloading dataset for supervised SupCon stage')
    dataset = BufferedSleepDataset(
        data_path=path_for_supcon,
        n_files_to_pick=None,
        buffer_size=n_files, 
        random_state=0,
        device='cpu',
        transform=None,
        augment=False,
        metadata=meta, 
        normalize= True,
        balance='undersample',  #here make sure categories are equaly represented
        exclude_labels=(0,),
        merge_nrem=True
    )    
    
    # supervised contrastive pretrianing
    logger.info('Starting supervised pretraining stage')
    final_cnn, sup_loss = train_supcon(model, dataset, logger, save_dir = save_path, epochs=50, batch_size=batch_size)
    
    # save final weights for GRU
    torch.save(final_cnn.state_dict(), save_path / model_name)
    
    logger.info('Pretraining pipeline complete.')
    