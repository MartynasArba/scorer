import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import random
from pathlib import Path

from scorer.data.loaders import BufferedSleepDataset
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN, SimCLRLoss
from scorer.models.sleep_cnn import SCDSSleepCNN

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

    for i in range(bsz):
        #random shift (50% chance)
        if random.random() < 0.5:
            max_shift = int(x.shape[2] * 0.02)
            shift = random.randint(-max_shift, max_shift)
            x[i] = torch.roll(x[i], shifts=shift, dims=1)
            
        #random emg dropout (30% chance), leftover from two-channel models
        if random.random() < 0.3:
            noise_level = random.random() * 1e-4
            if x.ndim == 3:  # batch
                noise = torch.randn(1, 1, x.shape[2], device=x.device) * noise_level
                x[i, 1:2, :] = noise
    return x

def train_supcon(model, dataset, save_dir = '.', epochs=50, batch_size=256):
    """supervised contrastive training loop"""
    torch.set_grad_enabled(True)
    device = next(model.parameters()).device
    base_cnn = model.encoder # extract encoder for saving
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #low lr because should be pretrained
    criterion = SupConLoss(temperature=0.15) 
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        print(f'epoch {epoch} supcon train started')
        
        running_loss = 0.0
        num_batches = 0
        
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            
            view1 = batch_augment(samples) 
            view2 = batch_augment(samples)
            
            images = torch.cat([view1, view2], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            optimizer.zero_grad()
            features = model(images)          
            loss = criterion(features, combined_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
        epoch_loss = running_loss / num_batches
            
        if epoch % 20 == 0:
            print(f"saving snapshot")
            torch.save(base_cnn.state_dict(), save_dir / f'/supcon_SCDS_snapshot_{epoch}.pt')
            
        print(f"epoch [{epoch}/{epochs}]: average loss {epoch_loss:.4f}")
    
    return base_cnn, loss 

def train_unsupervised(model, dataset, save_dir = '.', epochs=50, batch_size=256):
    """unsupervised contrastive training loop (SimCLR)"""
    torch.set_grad_enabled(True)
    device = next(model.parameters()).device
    base_cnn = model.encoder # extract basecnn for saving 
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = SimCLRLoss(temperature=0.15) 
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        print(f'epoch {epoch} simclr training started')
        
        running_loss = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0
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
            
            # debug tracking (compute pos/neg similarity)
            with torch.no_grad():
                features = torch.cat([z1, z2], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
                
                sim_matrix = torch.matmul(features, features.T)
                mask = torch.eq(combined_labels.unsqueeze(1), combined_labels.unsqueeze(0)).float()
                
                pos_sim = (sim_matrix * mask).sum() / mask.sum()
                neg_sim = (sim_matrix * (1 - mask)).sum() / (1 - mask).sum()
                
                running_pos_sim += pos_sim.item()
                running_neg_sim += neg_sim.item()
            
        epoch_loss = running_loss / num_batches
            
        if epoch % 20 == 0:
            print(f"saving snapshot")
            torch.save(base_cnn.state_dict(), save_dir / f'/SimCLR_SCDS_snapshot_{epoch}.pt')
            
        print(f"epoch [{epoch}/{epochs}]: loss {epoch_loss:.4f} | similarity pos: {running_pos_sim/num_batches:.3f} neg: {running_neg_sim/num_batches:.3f}")
    
    return base_cnn, model, loss 


if __name__ == "__main__":
    
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path = r"G:\oslo_data_train"
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 200
    n_epochs = 100
    model_name = '3state_contrastiveSCDS_2026-03-27.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models\weights")
    save_path.mkdir(parents=True, exist_ok=True) # Good practice to ensure directory exists
    batch_size = 1024
    
    print('starting pretraining...')
    
    # load full dataset for unsupervised learning
    dataset = BufferedSleepDataset(
        data_path=path,
        n_files_to_pick=None,
        buffer_size=n_files, 
        random_state=0,
        device='cpu',
        transform=None,
        augment=False,
        metadata=meta, 
        balance='undersample',
        exclude_labels=(0,),
        merge_nrem=True
    )    
        
        
    print('initializing model')
    base_cnn = SCDSSleepCNN(num_classes=3).to(device)    
    model = SupConSleepCNN(base_cnn).to(device)
    
    # simclr pretraining
    print('starting unsupervised simclr pretraining')
    pretrained_cnn, model, unsup_loss = train_unsupervised(model, dataset, save_dir = save_path, epochs=50, batch_size=batch_size)
    
    # save weights just in case
    torch.save(pretrained_cnn.state_dict(), save_path / 'SimCLR_base_weights.pt')
    
    # supervised contrastive pretrianing
    print('starting supervised pretraining')
    # pass same model
    final_cnn, sup_loss = train_supcon(model, dataset, save_dir = save_path, epochs=50, batch_size=batch_size)
    
    # save final weights for GRU
    torch.save(final_cnn.state_dict(), save_path / model_name)
    
    print('pretraining complete')