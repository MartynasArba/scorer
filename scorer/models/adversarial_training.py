import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import datetime
import math

from torch.utils.data import DataLoader
import torch.optim as optim

from scorer.data.loaders import BufferedSleepDataset
from scorer.models.sleep_cnn import SCDSSleepCNN

def setup_logger(save_dir: Path):
    log_file = Path(save_dir) / f"adversarial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def train_adversarial_domain(model, good_loader, ood_loader, optimizer, logger, epochs=10):
    model.train()
    device = next(model.parameters()).device
    
    # standard loss for sleep state classification (could also do focal loss, but could get weird) // (labeled data)
    criterion_sleep = nn.CrossEntropyLoss()
    # binary loss for domain classification (all data)
    criterion_domain = nn.BCEWithLogitsLoss()
    
    # both loaders will be reset to the same length
    min_batches = min(len(good_loader), len(ood_loader))
    total_steps = epochs * min_batches
    
    for epoch in range(epochs):
        epoch_loss = 0.
        epoch_domain_loss = 0.
        epoch_state_loss = 0.
        logger.info(f'Epoch {epoch} adversarial training started')
        
        for i, ((x_good, y_sleep), (x_ood, _)) in enumerate(zip(good_loader, ood_loader)):
            
            x_good, y_sleep = x_good.to(device), y_sleep.to(device)
            x_ood = x_ood.to(device)
            # alpha scheduler - increasing from 0 to 1 to keep sleep embeddings
            current_step = i + epoch * min_batches
            p = float(current_step) / total_steps
            warmup_fraction = 0.2
            if p < warmup_fraction:
                alpha = 0.0
            else:
                alpha = (p - warmup_fraction) / (1.0 - warmup_fraction)
            
            optimizer.zero_grad()
            
            # forward pass: labeled data
            features_good, domain_logits_good = model.forward_domain(x_good, alpha=alpha)
            sleep_logits = model.state_classifier(features_good) # pass features through classification head
            # ood data
            _, domain_logits_ood = model.forward_domain(x_ood, alpha=alpha)
            # get state loss on labeled data
            loss_sleep = criterion_sleep(sleep_logits, y_sleep)
            # get domain loss
            domain_labels_good = torch.zeros(x_good.size(0), 1, device = device)
            domain_labels_ood = torch.ones(x_ood.size(0), 1, device = device)
            loss_domain_good = criterion_domain(domain_logits_good, domain_labels_good)
            loss_domain_ood = criterion_domain(domain_logits_ood, domain_labels_ood)
            loss_domain = (loss_domain_good + loss_domain_ood) / 2.0
        
            # get total loss
            total_loss = loss_sleep + loss_domain
            
            epoch_loss += total_loss.item()
            epoch_domain_loss += loss_domain.item()
            epoch_state_loss += loss_sleep.item()
            
            total_loss.backward()
            optimizer.step()
            
        logger.info(f'epoch {epoch} finished, state loss: {(epoch_state_loss/(i+1)):.3f}, domain loss: {(epoch_domain_loss/(i+1)):.3f}, loss: {(epoch_loss/(i+1)):.3f}')
            
    return model

if __name__ == "__main__":
    
    CONFIG = {
        "paths": {
            "unlabeled_data": r"C:\Users\marty\Desktop\train_sets\unlabeled",
            "labeled_data": r"C:\Users\marty\Desktop\train_sets\labeled",
            "weights_dir": Path(r"C:\Users\marty\Projects\scorer\scorer\models\weights"),
            "val_data": r"C:\Users\marty\Desktop\train_sets\val",
            "encoder_path": r"C:\Users\marty\Projects\scorer\scorer\models\weights\SupCon_final_20260424.pt"
        },
        "pretrain": {
            "batch_size": 1024,
            "simclr_epochs": 50,
            "supcon_epochs": 50,
            "n_files_buffer": 100,
        },
        "sequence": {
            "batch_size": 128,
            "epochs": 20,
            "seq_len": 10,
        },
        "metadata": {
            'ecog_channels': '0', 
            'emg_channels': '1', 
            'sample_rate': '250', 
            'ylim': 'standard',
            'win_len': 1000
        }
    }
    save_path = '.'
    logger = setup_logger(save_path)
    
    logger.info('Starting adversarial training')
    
    labeled_dataset = BufferedSleepDataset(
            data_path=CONFIG["paths"]["labeled_data"],
            n_files_to_pick=None,
            buffer_size=CONFIG["pretrain"]["n_files_buffer"],
            metadata=CONFIG["metadata"],
            normalize= True,
            merge_nrem=True,
            device='cpu'
        )
    
    ood_dataset = BufferedSleepDataset(
            data_path=CONFIG["paths"]["unlabeled_data"],
            n_files_to_pick=None,
            buffer_size=CONFIG["pretrain"]["n_files_buffer"],
            metadata=CONFIG["metadata"],
            normalize= True,
            merge_nrem=True,
            device='cpu'
        )
    
    good_loader = DataLoader(labeled_dataset, batch_size= CONFIG["pretrain"]["batch_size"], shuffle = False, drop_last=True)    #shuffle is handled by the dataset itself
    ood_loader = DataLoader(ood_dataset, batch_size= CONFIG["pretrain"]["batch_size"], shuffle = False, drop_last=True)
    
    print('data loaded')

    # load pretrained encoder
    encoder = SCDSSleepCNN(num_classes = 3)
    # initialize LazyLinear layers with a dummy pass before loading weights
    dummy_input = torch.randn(1, 1, CONFIG["metadata"]["win_len"])   #should be win_size
    encoder(dummy_input)
    
    saved_weights = torch.load(CONFIG["paths"]["encoder_path"], map_location='cuda', weights_only=True)
    encoder.load_state_dict(saved_weights, strict = False)
    encoder = encoder.to('cuda')
    
    optimizer = optim.Adam([
        {'params': encoder.time_conv.parameters(), 'lr': 2e-6},
        {'params': encoder.freq_conv.parameters(), 'lr': 2e-6},
        {'params': encoder.freq_flatten.parameters(), 'lr': 1e-5},
        {'params': encoder.domain_classifier.parameters(), 'lr': 1e-3},#, 'weight_decay': 1e-3
        {'params': encoder.state_classifier.parameters(), 'lr': 1e-3}# fast learning for new heads
    ])
    
    model = train_adversarial_domain(encoder, good_loader, ood_loader, optimizer, logger, epochs = 10)
    
    torch.save(encoder.state_dict(), CONFIG["paths"]["weights_dir"] / "Aligned_Encoder_20260427.pt")
    
    print("adversarial training completed, weights saved")
    