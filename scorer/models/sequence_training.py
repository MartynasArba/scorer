#code to train and test sequence models
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


from scorer.data.loaders import SequenceSleepDataset
from scorer.models.sequence_model import ContextAwareSleepScorer, FocalLoss

def train_sequence_model(dataset, encoder_path, epochs = 50, batch_size = 64):
    """
    trains a GRU model to classify sleep states from pretrained embeddings
    """
    device = dataset.device
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    # load pretrained encoder
    encoder = torch.load(encoder_path, map_location = device, weights_only = False)
    #init sequence model
    model = ContextAwareSleepScorer(encoder, embedding_dim = 128, hidden_dim = 64, num_classes = 3, num_layers = 2)
    model = model.to(device)

    # loss for classification
    weights = torch.tensor([1.0, 1.5, 6.0]).to(device) #1.5, 1.0, 1.5, 3.0, 6.0
    criterion = FocalLoss(weight = weights, gamma = 2)
    
    # strictly pass only the unfrozen parameters to the optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3,
        weight_decay=1e-4
    )
    #add scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    save_dir = Path(encoder_path).parent

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
                total += labels.numel()
                correct += (predicted == labels).sum().item()
                
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        #step scheduler by val loss
        scheduler.step(val_loss)

        print(f"epoch [{epoch+1:02d}/{epochs}] | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"val acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, save_dir / "3state_SCDS_GRU.pt")

    print(f"Optimization complete. Peak Validation Accuracy: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    
    data_path = r"C:\Users\marty\Desktop\oslo_data_train"
    device = 'cuda'
    encoder_path = r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_contrastiveSCDS_2026-03-27.pt'
    

    dataset = SequenceSleepDataset(
            data_path = data_path,
            seq_len = 5,       # 5 continuous windows per sequence
            stride = 1,        # slide the window by 1 step (Overlapping data for more training examples!)
            device = device,
            exclude_labels = (0,), 
            merge_nrem = True,
            augment= True
        )
        
    print('data succesfully loaded! starting training')
    train_sequence_model(dataset, encoder_path, epochs = 20, batch_size = 64)