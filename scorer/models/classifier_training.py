import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score

from scorer.data.loaders import SequenceSleepDataset, SleepSignals, ContextEmbeddingDataset
from scorer.models.scoring import init_encoder, extract_embeddings
from scorer.models.context_classifiers import ContextResMLP
from scorer.models.diagnostics import plot_embedding_by_class, plot_trace, plot_classification_results

def extract_sequence_feature_tensors(encoder, dataset, batch_size=64):
    """
    helper to handle embedding extraction from SequenceSleepDataset 
    flattens [Batch, Seq, Chan, Time] into [Batch*Seq, Chan, Time]
    for encoder. Returns PyTorch tensors natively.
    """
    device = next(encoder.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader, desc="Extracting Sequence Embeddings"):
            # x: [Batch, Seq, Chan, Time]
            # y: [Batch, Seq]
            b, s, c, t = x.shape
            x_flat = x.view(b * s, c, t).to(device)
            
            # SCDSSleepCNN is single channel
            if x_flat.shape[1] > 1:
                x_flat = x_flat[:, 0:1, :]
            
            emb = encoder(x_flat) # [B*S, 512]
            
            # keep as tensors, but move to RAM to avoid GPU OOM
            all_embeddings.append(emb.cpu())
            # flatten
            all_labels.append(y.flatten().cpu())
            
    # concat tensors along batch
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

def train_context_net(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    context_k: int = 1,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains ContextResMLP on GPU using AdamW + cosine annealing + label smoothing
    """
    train_ds = ContextEmbeddingDataset(train_embeddings, train_labels, context_k=context_k)
    val_ds = ContextEmbeddingDataset(val_embeddings, val_labels, context_k=context_k)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = train_ds.embeddings.shape[1] * (2 * context_k + 1)
    num_classes = len(np.unique(train_labels))

    model = ContextResMLP(input_dim=input_dim, num_classes=num_classes).to(device)

    # weigh classes to account for imbalance
    class_counts = np.bincount(train_labels)
    # class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    # class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(1.0 / np.sqrt(class_counts + 1e-6), dtype=torch.float32).to(device) #less aggressive weigth adjustment

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_model_state = None

    print(f"Starting training on {device} | input dim: {input_dim} | context k: {context_k}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # val step
        val_acc, val_f1, _, _ = validate_context_net(model, val_loader, device=device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | train loss: {train_loss:.4f} | val acc: {val_acc:.4f} | val F1: {val_f1:.4f}")

    model.load_state_dict(best_model_state)
    return model


def validate_context_net(model: nn.Module, val_loader: DataLoader, device: str = 'cuda'):
    """Validates classifier, calculates accuracy, macro F1, and predicted probabilities"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                X_b, y_b = batch
                all_targets.append(y_b.numpy())
            else:
                X_b = batch
            
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets, axis=0)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        return acc, f1, all_preds, all_probs

    return None, None, all_preds, all_probs

def run_training(train_data_path, val_data_path, encoder_weights_path, model_save_path, meta={}, epochs=20, batch_size=512, lr=1e-3):
    """
    extracts embeddings using SequenceSleepDataset and trains ContextResMLP on GPU.
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = meta.get('seq_len', 100)
    win_len = meta.get('win_len', 512)
    sample_rate = meta.get('sample_rate', 128)

    print("initializing CNN encoder")
    encoder = init_encoder(encoder_weights_path, device=device, win_len=win_len, sample_rate=sample_rate)

    print("initializing datasets")
    train_dataset = SequenceSleepDataset(
        data_path=train_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True, # this is an option; should be done per-dataset
        merge_nrem=True,
        exclude_labels=(0,),
        augment=False
    )

    val_dataset = SequenceSleepDataset(
        data_path=val_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True,
        merge_nrem=True,
        exclude_labels=(0,),
        augment=False
    )

    print("extracting base embeddings from CNN")
    X_train_base, y_train = extract_sequence_feature_tensors(encoder, train_dataset)
    X_val_base, y_val = extract_sequence_feature_tensors(encoder, val_dataset)

    print(f"extracted train shape: {X_train_base.shape} | val shape: {X_val_base.shape}")

    print("training ContextResMLP classifier on GPU")
    nn_model = train_context_net(
        train_embeddings=X_train_base,
        train_labels=y_train,
        val_embeddings=X_val_base,
        val_labels=y_val,
        context_k=3, # CHANGED, was k=1 initially
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(nn_model.state_dict(), model_save_path)
    print(f"ContextResMLP weights saved to: {model_save_path}")


def run_validation(val_data_path, encoder_weights_path, nn_model_path, meta={}, extended = True):
    """
    Validates trained ContextResMLP over SequenceSleepDataset sequences.
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = meta.get('seq_len', 100)
    win_len = meta.get('win_len', 512)
    sample_rate = meta.get('sample_rate', 128)

    encoder = init_encoder(encoder_weights_path, device=device, win_len=win_len, sample_rate=sample_rate)

    val_dataset = SequenceSleepDataset(
        data_path=val_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True, #again, a choice
        merge_nrem=True,
        exclude_labels=(0,),
        augment=False
    )

    X_val_base, y_val = extract_sequence_feature_tensors(encoder, val_dataset)
    
    # Wrap embeddings into 1536-D context dataset
    val_ds = ContextEmbeddingDataset(X_val_base, labels=y_val, context_k=3)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    num_classes = len(np.unique(y_val))
    
    input_dim = val_ds.embeddings.shape[1] * (2 * val_ds.context_k + 1)
    nn_model = ContextResMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    nn_model.load_state_dict(torch.load(nn_model_path, map_location=device))

    acc, f1, val_preds, val_probs = validate_context_net(nn_model, val_loader, device=device)
    
    if len(y_val) != len(val_preds):
        y_val = y_val[val_ds.context_k : -val_ds.context_k]

    print(f"\nContext NN Validation Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=['Wake', 'NREM', 'REM']))
    
    if extended:
        plot_embedding_by_class(embeddings=X_val_base, labels=y_val)
        for _ in range(10):
            plot_trace(val_dataset)
        plot_classification_results(val_preds, y_val)

