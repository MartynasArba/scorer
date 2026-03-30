import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from scorer.data.loaders import SleepTraining
from pathlib import Path

def run_linear_evaluation(
    pretrained_model_path: str, 
    data_path: str, 
    meta: dict, 
    epochs: int = 100, 
    batch_size: int = 256
    ):
    """
    executes linear evaluation on a contrastive pre-trained encoder embeddings
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
        exclude_labels=(0,), # Excluding Unlabeled (0) and IS (3) based on pretraining, plus I don't label them
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
            torch.save(model, save_dir / "3state_SCDS_2.pt")

    print(f"Optimization complete. Peak Validation Accuracy: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    
    path = r"G:\oslo_data_train"
    meta = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard', 'device':'cuda'}
    model_name = 'weights/3state_contrastiveSCDS_2026-03-27.pt'
    save_path = Path(r"C:\Users\marty\Projects\scorer\scorer\models")

    run_linear_evaluation(
        pretrained_model_path = save_path / model_name,
        data_path = 'G:/val',#r'G:\for_training',#r'C:\Users\marty\Desktop\SCORING202602\for_training', 
        meta = meta,
        epochs = 30,
        batch_size = 1024  #batch size can also be smaller as this step uses regular loss 
    )