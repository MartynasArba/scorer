import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from scorer.data.loaders import SleepTraining, SleepTrainingLazy, BufferedSleepDataset
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN
from scorer.models.sleep_cnn import EphysSleepCNN

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
        buffer_size= 100, 
        random_state = 0,
        device = 'cpu',
        transform = None,
        augment = False,
        metadata = meta, 
        balance = 'oversample',
        exclude_labels = (0,3),#add labels to exclude here
        merge_nrem = False
    )    
    return dataset

def train_supcon(dataset, epochs=50):
    #SHOULD PRETRAIN ON ALL FILES INSTEAD!
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init base model and SupCon wrapper
    base_cnn = EphysSleepCNN(num_classes = 3).to(device) # 3 class for now
    model = SupConSleepCNN(base_cnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    criterion = SupConLoss(temperature = 0.1)
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.train()
    for epoch in range(epochs):
        print(f'epoch {epoch} training started')
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            
            # get two augmented views of the same batch
            view1 = dataset._augment(samples) 
            view2 = dataset._augment(samples)
            
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
    epochs: int = 50, 
    batch_size: int = 128
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
        balance='oversample', 
        exclude_labels=(0, 3), # Excluding Unlabeled (0) and IS (3) based on pretraining
        merge_nrem=False
    )

    # could replace random_split with subject-wise splitting for real, biological evaluation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # load pretrained encoder
    model = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    
    # freeze feature extraction: iterate through parameters and set all conv layers to no grad
    for name, param in model.named_parameters():
        if 'conv' in name:
            param.requires_grad = False
        elif 'fc' in name:
            param.requires_grad = True

    # new loss for classification
    criterion = nn.CrossEntropyLoss()
    
    # strictly pass only the unfrozen parameters to the optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )

    best_val_acc = 0.0
    save_dir = Path("weights")
    save_dir.mkdir(exist_ok=True)

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
            torch.save(model, save_dir / "best_linear_eval_model.pt")

    print(f"Optimization complete. Peak Validation Accuracy: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    path = 'G:/oslo_data'
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 100
    n_epochs = 50
    model_name = 'weights/3state_contrastiveCNN_2026-02-25-2.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")

    print('starting pretraining...')
    dataset = load_dataset(path = path, meta = meta, n_files = n_files)
    print('data loaded, running train loop')
    pretrained_cnn, loss = train_supcon(dataset, epochs = n_epochs)
    torch.save(pretrained_cnn, save_path / model_name)
    
    run_linear_evaluation(
        pretrained_model_path = save_path / model_name,
        data_path = 'G:/oslo_data_val',
        meta=meta,
        epochs=30
    )
    
    # plt.plot(loss.detach().cpu().numpy())
    # plt.show()