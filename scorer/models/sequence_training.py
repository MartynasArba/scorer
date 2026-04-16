#code to train and test sequence models
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, cohen_kappa_score
import logging
import datetime
from pathlib import Path

from scorer.data.loaders import SequenceSleepDataset
from scorer.models.sequence_model import ContextAwareSleepScorer, FocalLoss
from scorer.models.sleep_cnn import SCDSSleepCNN

def setup_logger(save_dir: Path):
    log_file = save_dir / f"sequence_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def validate(model, val_loader, criterion, running_loss, steps_accumulated, epoch, epochs, save_dir, device, logger, best_val_acc, step=0, subset_size=0.1):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    num_val_batches = max(1, int(subset_size * len(val_loader)))
    
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for val_step, (samples, labels) in enumerate(val_loader):
            if val_step >= num_val_batches:
                break            
            
            samples, labels = samples.to(device), labels.to(device)
            
            outputs = model(samples)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()
            all_val_preds.append(predicted)
            all_val_labels.append(labels)
            
    # math for varying chunk sizes
    train_loss = running_loss / steps_accumulated
    val_loss = val_loss / num_val_batches
    val_acc = correct / total
    
    # Calculate advanced metrics
    y_true = torch.cat(all_val_labels).cpu().numpy().flatten()
    y_pred = torch.cat(all_val_preds).cpu().numpy().flatten()
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    all_val_preds = torch.cat(all_val_preds)
    unique_preds, counts = torch.unique(all_val_preds, return_counts=True)
    pred_dist = dict(zip(unique_preds.tolist(), counts.tolist()))

    logger.info(f"Epoch [{epoch+1:02d}/{epochs}], step {step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
    logger.info(f"Validation Predicted Class Distribution: {pred_dist}")

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        logger.info(f"New best validation accuracy ({val_acc:.4f}), saving state_dict.")
        torch.save(model.state_dict(), save_dir / "3state_SCDS_GRU_weights.pt")
        
    # return the best accuracy so the main loop can keep track of it
    return best_val_acc



def train_sequence_model(dataset, val_dataset, encoder_path, logger=None, epochs = 50, batch_size = 64):
    """
    trains a GRU model to classify sleep states from pretrained embeddings
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    device = dataset.device
    
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    # load pretrained encoder
    encoder = SCDSSleepCNN(num_classes = 3)
    # initialize LazyLinear layers with a dummy pass before loading weights
    dummy_input = torch.randn(1, 1, dataset.all_samples.shape[-1])
    encoder(dummy_input)
    
    saved_weights = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder.load_state_dict(saved_weights)
    encoder = encoder.to(device)
    
    #init sequence model
    model = ContextAwareSleepScorer(encoder, embedding_dim = 512, hidden_dim = 64, num_classes = 3, num_layers = 2)
    model = model.to(device)

    # Sanity Check: Ensure encoder is actually frozen
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            logger.error(f"Sanity Check Failed: Encoder parameter {name} is not frozen!")
            raise RuntimeError("Encoder must be frozen during sequence training.")

    # Sanity Check: Label Range
    sample_batch_x, sample_batch_y = next(iter(train_loader))
    if sample_batch_y.max() >= 3 or sample_batch_y.min() < 0:
        logger.error(f"Label Range Error: Found labels up to {sample_batch_y.max()}. Expected range [0, 2].")
        raise ValueError("Dataset labels are not correctly mapped to contiguous indices.")

    logger.info(f"Model initialized. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Sequence length: {dataset.seq_len}, Batch size: {batch_size}")

    # loss for classification
    # weights = torch.tensor([1.0, 1.5, 6.0]).to(device) #1.5, 1.0, 1.5, 3.0, 6.0
    criterion = FocalLoss(gamma = 1.5)  #nn.CrossEntropyLoss(label_smoothing=0.1, weight= weights)#weight = weights, 
    
    # strictly pass only the unfrozen parameters to the optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-5,
        weight_decay=1e-4
    )
    
    total_steps = len(train_loader) * epochs
    
    #add per-batch scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = total_steps, eta_min = 1e-7)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    save_dir = Path(encoder_path).parent   

    # train loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        steps_since_val = 0

        for step, (samples, labels) in enumerate(train_loader):
            samples, labels = samples.to(device), labels.to(device)

            # sanity check: batch alignment
            if samples.shape[0] != labels.shape[0]:
                logger.error(f"Batch mismatch: samples({samples.shape[0]}) != labels({labels.shape[0]})")
                continue

            optimizer.zero_grad()
            outputs = model(samples)
            
            if torch.isnan(outputs).any():
                logger.error("NaN detected in model outputs. Aborting training.")
                return model

            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.1)    #clip gradients to prevent explosion
            
            optimizer.step()
            scheduler.step() 
            
            running_loss += loss.item()
            steps_since_val += 1
            
            #do micro-epoch validation 
            if step > 0 and step % 5000 == 0:

                best_val_acc = validate(model, val_loader, criterion, running_loss, 
                                        steps_since_val, epoch, epochs, save_dir, 
                                        device, logger, best_val_acc, step = step, subset_size= 0.1)
                model.train()
                running_loss = 0.0
        #full validation per-epoch
        if steps_since_val > 0:
            best_val_acc = validate(model, val_loader, criterion, running_loss, 
                                            steps_since_val, epoch, epochs, save_dir, 
                                            device, logger, best_val_acc, step = 'end of epoch', subset_size= 1)
        
    logger.info(f"Optimization complete. Peak Validation Accuracy: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    
    data_path = r'C:\Users\marty\Desktop\train_sets\labeled'
    val_data_path = r'C:\Users\marty\Desktop\train_sets\val'
    
    device = 'cuda'
    encoder_path = r"C:\Users\marty\Projects\scorer\scorer\models\weights\SupCon_final_20260415.pt"#supcon_SCDS_best_model.pt'
    
    save_path = Path(encoder_path).parent
    logger = setup_logger(save_path)

    dataset = SequenceSleepDataset(
            data_path = data_path,
            seq_len = 10,       # 10 continuous windows per sequence
            stride = 1,        # slide windows by 1 step
            device = device,
            exclude_labels = (0,), 
            merge_nrem = True,
            augment= True
        )
    
    logger.info('Training dataset loaded')
    
    val_dataset = SequenceSleepDataset(
            data_path = val_data_path,
            seq_len = 10,       # 10 continuous windows per sequence
            stride = 1,        # slide windows by 1 step
            device = device,
            exclude_labels = (0,), 
            merge_nrem = True,
            augment= False
        )
    logger.info('Validation dataset loaded')
        
    logger.info('Data successfully loaded! Starting training pipeline...')
    train_sequence_model(dataset, val_dataset, encoder_path, logger=logger, epochs = 20, batch_size = 128)