import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

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

def batch_augment(x: torch.Tensor) -> torch.Tensor:
    """vectorized augmentations per-sample"""
    x = x.clone()
    bsz = x.shape[0]
    device = x.device
    
    # noise per sample (70% chance)
    apply_noise = (torch.rand(bsz, 1, 1, device=device) < 0.7).float()
    x = x + (apply_noise * torch.randn_like(x) * 0.01)
    
    # scaling per sample (50% chance)
    apply_scale = (torch.rand(bsz, 1, 1, device=device) < 0.5).float()
    scales = 0.5 + torch.rand(bsz, x.shape[1], 1, device=device)
    x = x * ((1 - apply_scale) + apply_scale * scales)

    # Vectorized random shift (50% chance)
    shift_mask = torch.rand(bsz, device=device) < 0.5
    max_shift = int(x.shape[2] * 0.02)
    shifts = torch.randint(-max_shift, max_shift + 1, (bsz,), device=device)
    
    for i in range(bsz):
        if shift_mask[i]:
            x[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=1)

    # Random channel dropout/noise (30% chance) - only if model expects >1 channel
    if x.shape[1] > 1:
        dropout_mask = (torch.rand(bsz, 1, 1, device=device) < 0.3).float()
        noise = torch.randn(bsz, 1, x.shape[2], device=device) * (torch.rand(bsz, 1, 1, device=device) * 1e-4)
        x[:, 1:2, :] = (1 - dropout_mask) * x[:, 1:2, :] + dropout_mask * noise

    return x

def train_supcon(model, dataset, logger, save_dir = '.', epochs=50, batch_size=256):
    """supervised contrastive training loop"""
    torch.set_grad_enabled(True)
    device = next(model.parameters()).device
    base_cnn = model.encoder 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #low lr because should be pretrained
    criterion = SupConLoss(temperature = 0.1) 
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
            samples, labels = samples.to(device), labels.to(device)
            
            # Sanity Check: Label Alignment
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
    device = next(model.parameters()).device
    base_cnn = model.encoder
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = SimCLRLoss(temperature=0.15) 
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
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
            samples, labels = samples.to(device), labels.to(device)
            
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