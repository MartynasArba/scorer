import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import random

from scorer.data.loaders import BufferedSleepDataset
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN
from scorer.models.sleep_cnn import SCDSSleepCNN

from pathlib import Path

def load_dataset(path, meta: dict, n_files: int):
    """
    loads training dataset class
    """
    #load dataset
    dataset = BufferedSleepDataset(
        data_path = path,
        n_files_to_pick = None,
        buffer_size= n_files, 
        random_state = 0,
        device = 'cpu',
        transform = None,
        augment = False,
        metadata = meta, 
        balance = 'undersample',
        exclude_labels = (0,),#add labels to exclude here
        merge_nrem = True
    )    
    return dataset

def batch_augment(x: torch.Tensor) -> torch.Tensor:
    """Properly vectorizes augmentations so every sample gets unique transforms"""
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
            
        #random emg dropout (30% chance)
        if random.random() < 0.3:
            noise_level = random.random() * 1e-4
            if x.ndim == 3:  # batch
                noise = torch.randn(1, 1, x.shape[2], device=x.device) * noise_level
                x[i, 1:2, :] = noise
    return x

def train_supcon(dataset, epochs=50, batch_size = 256):
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init base model and SupCon wrapper
    base_cnn = SCDSSleepCNN(num_classes = 3).to(device)    #change to 5 if doing 5 states
    model = SupConSleepCNN(base_cnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 5e-5)
    criterion = SupConLoss(temperature = 0.15)  #worked well with 0.15
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        print(f'epoch {epoch} training started')
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            
            # get two augmented views of the same batch
            view1 = batch_augment(samples) 
            view2 = batch_augment(samples)
            
            # combine views for the batch
            images = torch.cat([view1, view2], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            # forward pass
            features = model(images)          
            loss = criterion(features, combined_labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # debug: pos logits should be higher than neg logits
        with torch.no_grad():
            sim_matrix = torch.matmul(features, features.T)
            mask = torch.eq(combined_labels.unsqueeze(1), combined_labels.unsqueeze(0)).float()
            
            pos_sim = (sim_matrix * mask).sum() / mask.sum()
            neg_sim = (sim_matrix * (1 - mask)).sum() / (1 - mask).sum()
            
        if epoch % 10 == 0:
            print(f"saving snapshot")
            torch.save(base_cnn, r'C:\Users\marty\Projects\scorer\scorer\models\weights' + f'/SCDS_snapshot_{epoch}.pt')
            
        print(f"Epoch {epoch}: Loss {loss.item():.4f} | Mean Feature Norm: {torch.norm(features, dim=1).mean().item():.4f} | Similarity Pos: {pos_sim:.3f} Neg: {neg_sim:.3f}")
    
    return base_cnn, loss # Return the encoder for downstream use

if __name__ == "__main__":
    path = r"G:\oslo_data_train"
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 200
    n_epochs = 100
    model_name = 'weights/3state_contrastiveSCDS_2026-03-27.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")
    
    print('starting pretraining...')
    dataset = load_dataset(path = path, meta = meta, n_files = n_files)
    print('data loaded, running train loop')
    pretrained_cnn, loss = train_supcon(dataset, epochs = n_epochs, batch_size=1024)
    torch.save(pretrained_cnn, save_path / model_name)
    print('pretraining done')