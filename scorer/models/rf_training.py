import torch
import numpy as np
import pickle
import os
from pathlib import Path
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from scorer.data.loaders import BufferedSleepDataset, SequenceSleepDataset
from scorer.models.scoring import extract_embeddings, init_encoder, apply_heuristics

def train_rf_model(train_data_path, val_data_path, encoder_weights_path, model_save_path, meta={}):
    """
    trains a Random Forest classifier using embeddings extracted from a pre-trained CNN encoder.
    Validates the model and saves the trained RF to a pickle file.
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("init encoder and loading weights")
    encoder = init_encoder(encoder_weights_path, device=device, win_len=meta.get('win_len', 1000))

    # load data
    print("\n loading training data")
    train_dataset = BufferedSleepDataset(
        data_path=train_data_path,
        n_files_to_pick=None,
        buffer_size=meta.get('buffer_size', 100),
        metadata=meta,
        normalize=True,
        merge_nrem=True,
        balance='undersample',
        exclude_labels=(0,),
        device=device
    )

    print("loading validation data")
    val_dataset = BufferedSleepDataset(
        data_path=val_data_path,
        n_files_to_pick=None,
        buffer_size=meta.get('buffer_size', 100),
        metadata=meta,
        normalize=True,
        merge_nrem=True,
        balance='none',
        exclude_labels=(0,),
        device=device
    )

    # get embeddings
    print("\nextracting training embeddings")
    X_train, y_train = extract_embeddings(encoder, train_dataset)
    
    print("extracting validation embeddings")
    X_val, y_val = extract_embeddings(encoder, val_dataset)

    # train RF
    print("\ntraining random forest classifier")
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20, class_weight='balanced_subsample')
    rf.fit(X_train, y_train)

    # validation
    print("\nevaluating model")
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"RF Training Accuracy:   {train_acc:.4f}")
    print(f"RF Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_preds, target_names=['Wake', 'NREM', 'REM']))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_val, val_preds, display_labels=['Wake', 'NREM', 'REM'], 
                                            cmap='Blues', normalize='true', ax=ax)
    plt.title(f"RF Confusion Matrix (Val Acc: {val_acc:.4f})")
    plt.show()

    # save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\nRandom Forest model saved to: {model_save_path}")

def validate_rf_model(val_data_path, encoder_weights_path, rf_model_path, meta={}):
    """
    Loads a trained Random Forest and its associated CNN encoder to run validation on a dataset.
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validating RF model using device: {device}")

    print("init encoder for validation")
    encoder = init_encoder(encoder_weights_path, device=device, win_len=meta.get('win_len', 1000))

    # load model
    print(f"Loading RF model from {rf_model_path}...")
    with open(rf_model_path, 'rb') as f:
        rf = pickle.load(f)

    # load validation data
    print("Loading validation data...")
    val_dataset = BufferedSleepDataset(
        data_path=val_data_path,
        n_files_to_pick=None,
        buffer_size=meta.get('buffer_size', 100),
        metadata=meta,
        normalize=True,
        merge_nrem=True,
        balance='none',
        exclude_labels=(0,),
        device=device
    )

    # extract embeddings and predict
    print("Extracting embeddings and running inference...")
    X_val, y_val = extract_embeddings(encoder, val_dataset)
    val_preds = rf.predict(X_val)
    #DO NOT APPLY HEURISTICS OR SEQUENCE SMOOTHING ON SHUFFLED DATA!!!!!! 

    # report results
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\nLoaded RF Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=['Wake', 'NREM', 'REM']))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_val, val_preds, 
                                            display_labels=['Wake', 'NREM', 'REM'], 
                                            cmap='Blues', 
                                            normalize='true', 
                                            ax=ax)
    plt.title(f"Loaded RF Confusion Matrix (Val Acc: {val_acc:.4f})")
    plt.show()

def extract_sequence_features(encoder, dataset, batch_size=64):
    """
    helper to handle embedding extraction from SequenceSleepDataset
    flattens [Batch, Seq, Chan, Time] into [Batch*Seq, Chan, Time]
    for encoder.
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
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(y.cpu().numpy().flatten())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def validate_rf_sequence(val_data_path, encoder_weights_path, rf_model_path, meta={}, smooth_sigma=1.0):
    """
    eval RF model on SequenceSleepDataset, allows sequence post-processing or further modeling.
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validating RF on sequences (sigma={smooth_sigma}) using device: {device}")

    # load encoder, load rf
    encoder = init_encoder(encoder_weights_path, device=device, win_len=meta.get('win_len', 1000))
    with open(rf_model_path, 'rb') as f:
        rf = pickle.load(f)

    # load data, use stride = seq_len to not duplicate windows
    seq_len = meta.get('seq_len', 100)
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

    # predictions
    X_feat, y_true = extract_sequence_features(encoder, val_dataset)
    
    # probs
    probs = rf.predict_proba(X_feat)

    # post processing
    if smooth_sigma and smooth_sigma > 0:
        print(f"Applying Gaussian smoothing with sigma={smooth_sigma}...")
        probs = gaussian_filter1d(probs, sigma=smooth_sigma, axis=0)
    
    val_preds = np.argmax(probs, axis=1)
    val_preds = apply_heuristics(val_preds, '3_class')

    # results
    val_acc = accuracy_score(y_true, val_preds)
    print(f"\nSequence Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report (Sequence-Smoothed):")
    print(classification_report(y_true, val_preds, target_names=['Wake', 'NREM', 'REM']))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, val_preds, 
                                            display_labels=['Wake', 'NREM', 'REM'], 
                                            cmap='Blues', normalize='true', ax=ax)
    plt.title(f"Sequence RF Confusion Matrix (Acc: {val_acc:.4f}, Sigma: {smooth_sigma})")
    plt.show()
    
# CONTEXT AWARE RF FUNCS

def apply_heuristics_lite(states: np.ndarray, mapping: str = '3_class') -> np.ndarray:
    """
    A gentle heuristic that ONLY cleans up isolated 4-second artifacts.
    It intentionally leaves Wake->REM boundaries alone, trusting the Context RF.
    """
    smoothed = states.copy()
    n = len(smoothed)

    if n < 3: return smoothed
        
    if mapping == '5_class' or mapping == 'gui':
        WAKE = 1
    elif mapping == '3_class':
        WAKE = 0
    else:
        raise ValueError("mapping must be '5_class', '3_class', or 'gui'")

    # ONLY do single-window cleanup. Protect micro-awakenings.
    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    return smoothed

def create_context_features(X_seq, y_seq=None):
    """
    adds t-1, t, and t+1 for chronological sequences
    X_seq shape: [N, 512] returns [N, 1536]
    """
    # pad the sequence
    X_padded = np.vstack([X_seq[0:1], X_seq, X_seq[-1:]])
    
    # stack t-1, t, t+1 
    X_context = np.concatenate([
        X_padded[:-2],  # past window
        X_padded[1:-1], # current window
        X_padded[2:]    # future window
    ], axis=1)
    
    if y_seq is not None:
        return X_context, y_seq
    return X_context

def train_context_rf_model(train_data_path, val_data_path, encoder_weights_path, model_save_path, meta={}):
    """
    trains a context-aware Random Forest (1536-D) using SequenceSleepDataset
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing encoder and loading weights...")
    encoder = init_encoder(encoder_weights_path, device=device, win_len=meta.get('win_len', 1000))
    seq_len = meta.get('seq_len', 100)
    
    # load data
    print("\nLoading chronological training data...")
    train_dataset = SequenceSleepDataset(
        data_path=train_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True,
        merge_nrem=True,
        exclude_labels=(0,),
        augment=False
    )

    print("Loading chronological validation data...")
    val_dataset = SequenceSleepDataset(
        data_path=val_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True,
        merge_nrem=False,
        exclude_labels=(),
        augment=False
    )

    # extract sequential 512-D embeddings
    print("\nExtracting base sequential embeddings...")
    X_train_base, y_train = extract_sequence_features(encoder, train_dataset)
    X_val_base, y_val = extract_sequence_features(encoder, val_dataset)

    # expand to 1536-D context features
    print("Building temporal context (t-1, t, t+1)...")
    X_train_context, y_train = create_context_features(X_train_base, y_train)
    X_val_context, y_val = create_context_features(X_val_base, y_val)

    # train RF
    print("\nTraining Context-Aware Random Forest (1536 features)...")
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20, class_weight='balanced_subsample', max_features=0.15)
    rf.fit(X_train_context, y_train)

    # validation
    print("\nEvaluating Context-Aware model...")
    val_preds = rf.predict(X_val_context)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"Context RF Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report (Unsmoothed):")
    print(classification_report(y_val, val_preds, target_names=['Wake', 'NREM', 'REM']))

    # save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\nContext Random Forest saved to: {model_save_path}")

def validate_context_rf_sequence(val_data_path, encoder_weights_path, rf_model_path, meta={}, smooth_sigma=1.0):
    """
    eval context aware RF on SequenceSleepDataset with optional sequence post-processing
    """
    device = meta.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validating Context RF (sigma={smooth_sigma}) using device: {device}")

    encoder = init_encoder(encoder_weights_path, device=device, win_len=meta.get('win_len', 1000))
    with open(rf_model_path, 'rb') as f:
        rf = pickle.load(f)

    seq_len = meta.get('seq_len', 100)
    val_dataset = SequenceSleepDataset(
        data_path=val_data_path,
        seq_len=seq_len,
        stride=seq_len,
        device=device,
        normalize=True,
        merge_nrem=False,
        exclude_labels=(),
        augment=False
    )

    X_feat, y_true = extract_sequence_features(encoder, val_dataset)
    
    # apply the context window BEFORE predicting
    X_context, y_true = create_context_features(X_feat, y_true)
    
    probs = rf.predict_proba(X_context)
    
    #boost REM probs
    # probs[:, 2] = probs[:, 2] * 1.30
    # # renormalize so rows sum to 1
    # row_sums = probs.sum(axis=1, keepdims=True)
    # probs = probs / row_sums
    # argmax with no smoothing as context is included

    # # post processing
    # if smooth_sigma and smooth_sigma > 0:
    #     print(f"Applying Gaussian smoothing with sigma={smooth_sigma}...")
    #     probs = gaussian_filter1d(probs, sigma=smooth_sigma, axis=0)
    
    val_preds = np.argmax(probs, axis=1)
    val_preds = apply_heuristics_lite(val_preds, '3_class')
    #heuristics basically shouldn;t be needed anymore

    val_acc = accuracy_score(y_true, val_preds)
    print(np.unique(y_true, return_counts= True), np.unique(val_preds, return_counts= True))
    print(f"\nContext-Aware Sequence Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report (Context + Smoothed):")
    print(classification_report(y_true, val_preds, target_names=['Wake', 'NREM', 'REM']))

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, val_preds, 
                                            display_labels=['Wake', 'NREM', 'REM'], 
                                            cmap='Blues', normalize='true', ax=ax)
    plt.title(f"Context RF Confusion Matrix (Acc: {val_acc:.4f}, Sigma: {smooth_sigma})")
    plt.show()

if __name__ == "__main__":
    # config
    BASE_PATH = r"C:\Users\marty\Desktop\DATA_FINAL"
    WEIGHTS_DIR = r"C:\Users\marty\Projects\scorer\scorer\models\weights"
    #context model
    train_context_rf_model(
        train_data_path = os.path.join(BASE_PATH, "labeled-train-oxford"),
        val_data_path = os.path.join(BASE_PATH, "labeled-val-mlsnet"),
        encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"),
        model_save_path = os.path.join(WEIGHTS_DIR, "rf_context_sleep_classifier_fullpretrain_oxfordRF.pkl"),
        meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250', 'seq_len': 100}
    )
    
    validate_context_rf_sequence(
        val_data_path = os.path.join(BASE_PATH, "labeled-val-mlsnet"), 
        encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"), 
        rf_model_path = os.path.join(WEIGHTS_DIR, "rf_context_sleep_classifier_fullpretrain_oxfordRF.pkl"), 
        meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250', 'seq_len': 100}, 
        smooth_sigma = 1.0 
    )
    
    #original model
    # train_rf_model(
    #     train_data_path = os.path.join(BASE_PATH, "labeled"),
    #     val_data_path = os.path.join(BASE_PATH, "val"),
    #     encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"),
    #     model_save_path = os.path.join(WEIGHTS_DIR, "rf_sleep_classifier.pkl"),
    #     meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250'}
    # )
    
    # validate_rf_sequence(os.path.join(BASE_PATH, "val"), 
    #                   encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"), 
    #                   rf_model_path = r"C:\Users\marty\Projects\scorer\scorer\models\weights\rf_sleep_classifier.pkl", 
    #                   meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250', 'seq_len': 100}, 
    #                   smooth_sigma = 0.75)
