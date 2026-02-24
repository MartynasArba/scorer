import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from scorer.data.loaders import SleepTraining
from scorer.models.contrastive_embedder import SupConLoss, SupConSleepCNN
from scorer.models.sleep_cnn import EphysSleepCNN
import matplotlib.pyplot as plt

PATH = 'G:/oslo_data'
META = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}

def load_dataset(path, meta: dict):
    """
    loads training dataset class
    """
    #load dataset
    dataset = SleepTraining(
        data_path = path,
        n_files_to_pick = 300,
        random_state = 0,
        device = 'cuda',
        transform = None,
        augment = True,
        metadata = meta, 
        balance = 'oversample',
        exclude_labels = (0,3),#add labels to exclude here
        merge_nrem = False
    )    

def train_supcon(dataset, epochs=50):
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init base model and SupCon wrapper
    base_cnn = EphysSleepCNN(num_classes = 3).to(device) # 3 class for now
    model = SupConSleepCNN(base_cnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = SupConLoss(temperature=0.1)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
            loss = criterion(features, combined_labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    
    return base_cnn, loss # Return the encoder for downstream use

if __name__ == "__main__":
    print('starting pretraining...')
    dataset = load_dataset(path = PATH, meta = META)
    print('data loaded, running train loop')
    pretrained_cnn, loss = train_supcon(dataset, epochs = 3)
    plt.plot(loss)
