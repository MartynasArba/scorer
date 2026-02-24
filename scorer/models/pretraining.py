import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from scorer.data.loaders import SleepTraining
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN
from scorer.models.sleep_cnn import EphysSleepCNN

import matplotlib.pyplot as plt
from pathlib import Path

def load_dataset(path, meta: dict, n_files: int):
    """
    loads training dataset class
    """
    #load dataset
    dataset = SleepTraining(
        data_path = path,
        n_files_to_pick = n_files,
        random_state = 0,
        device = 'cuda',
        transform = None,
        augment = False,
        metadata = meta, 
        balance = 'oversample',
        exclude_labels = (0,3),#add labels to exclude here
        merge_nrem = False
    )    
    return dataset

def train_supcon(dataset, epochs=50):
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init base model and SupCon wrapper
    base_cnn = EphysSleepCNN(num_classes = 3).to(device) # 3 class for now
    model = SupConSleepCNN(base_cnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = SupConLoss(temperature=0.05)
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
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
            
            # debug: pos logits should be more than neg logits
            with torch.no_grad():
                sim_matrix = torch.matmul(features, features.T)
                mask = torch.eq(combined_labels.unsqueeze(1), combined_labels.unsqueeze(0)).float()
                
                pos_sim = (sim_matrix * mask).sum() / mask.sum()
                neg_sim = (sim_matrix * (1 - mask)).sum() / (1 - mask).sum()
                
                if epoch % 5 == 0:
                    print(f"Similarity -> Pos: {pos_sim:.3f} | Neg: {neg_sim:.3f}")
            
            
            loss = criterion(features, combined_labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #debug: check if features are normalized, should be around 1
        if epoch % 10 == 0:
            print(f"Mean Feature Norm: {torch.norm(features, dim=1).mean().item():.4f}")
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    
    return base_cnn, loss # Return the encoder for downstream use

if __name__ == "__main__":
    path = 'G:/oslo_data'
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    n_files = 300
    n_epochs = 50
    model_name = 'weights/3state_contrastiveCNN_2026-02-24.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")

    print('starting pretraining...')
    dataset = load_dataset(path = path, meta = meta, n_files = n_files)
    print('data loaded, running train loop')
    pretrained_cnn, loss = train_supcon(dataset, epochs = n_epochs)
    torch.save(pretrained_cnn, save_path / model_name)
    
    plt.plot(loss.detach().cpu().numpy())
    plt.show()
