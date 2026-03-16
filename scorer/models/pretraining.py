import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

import random

from scorer.data.loaders import SleepTraining, BufferedSleepDataset
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN
from scorer.models.sleep_cnn import EphysSleepCNN, DualStreamSleepCNN, SleepCNN

import matplotlib.pyplot as plt
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
    base_cnn = DualStreamSleepCNN(num_classes = 3).to(device)    #change to 5 if doing 5 states
    model = SupConSleepCNN(base_cnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 5e-5)
    criterion = SupConLoss(temperature = 0.15)
    
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
            torch.save(base_cnn, r'C:\Users\marty\Projects\scorer\scorer\models\weights' + f'/pretrainedCNN_snapshot_{epoch}.pt')
            
        print(f"Epoch {epoch}: Loss {loss.item():.4f} | Mean Feature Norm: {torch.norm(features, dim=1).mean().item():.4f} | Similarity Pos: {pos_sim:.3f} Neg: {neg_sim:.3f}")
    
    return base_cnn, loss # Return the encoder for downstream use

def run_linear_evaluation(
    pretrained_model_path: str, 
    data_path: str, 
    meta: dict, 
    epochs: int = 100, 
    batch_size: int = 256
    ):
    """
    executes linear evaluation on a contrastive pre-trained encoder
    the convolutional hierarchy is frozen, and only the dense classification 
    head is optimized via standard cross-entropy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing linear evaluation on device: {device}")

    dataset = SleepTraining(
        data_path=data_path,
        n_files_to_pick=None,
        random_state=42,
        device=device,
        transform=None,
        augment=False, # augmentations are disabled
        metadata=meta,
        balance='none', 
        exclude_labels=(0,3), # Excluding Unlabeled (0) and IS (3) based on pretraining, plus I don't label them
        merge_nrem=True    #no merging on my data as it shouldn't include it, but can merge on Oslo data
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # load pretrained encoder
    model = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    # #overwrite final layers to correct num classes
    model.fc1 = nn.Linear(128, 64).to(device)
    model.fc2 = nn.Linear(64, 3).to(device)
    
    # freeze feature extraction: iterate through parameters and set all conv layers to no grad
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # new loss for classification
    weights = torch.tensor([1.0, 1.5, 6.0]).to(device) #1.5, 1.0, 1.5, 3.0, 6.0
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # strictly pass only the unfrozen parameters to the optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3,
        weight_decay=5e-5
    )

    best_val_acc = 0.0
    save_dir = Path(pretrained_model_path).parent

    # train loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for samples, labels in val_loader:
                samples, labels = samples.to(device), labels.to(device)
                
                outputs = model(samples)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"epoch [{epoch+1:02d}/{epochs}] | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"val acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, save_dir / "3state_dual_CHANGE.pt")

    print(f"Optimization complete. Peak Validation Accuracy: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    path = 'G:/oslo_data'
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    # meta2 = {'ecog_channels' : '0', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 200
    n_epochs = 100
    model_name = 'weights/3state_contrastiveDualCNN_2026-03_CHANGE.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")

    # print('starting pretraining...')
    # dataset = load_dataset(path = path, meta = meta, n_files = n_files)
    # print('data loaded, running train loop')
    # pretrained_cnn, loss = train_supcon(dataset, epochs = n_epochs, batch_size=1024)
    # torch.save(pretrained_cnn, save_path / model_name)
    
    run_linear_evaluation(
        pretrained_model_path = save_path / model_name,
        data_path = 'G:/oslo_data_val'#r'C:\Users\marty\Desktop\SCORING202602\for_training', #'G:/oslo_data_val'
        meta = meta,
        epochs = 30,
        batch_size = 1024  #batch size can also be smaller as this step uses regular loss 
    )