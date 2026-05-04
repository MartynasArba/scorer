import torch
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from scorer.models.sleep_cnn import SCDSSleepCNN
from scorer.data.loaders import BufferedSleepDataset
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
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20)
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

if __name__ == "__main__":
    # config
    BASE_PATH = r"C:\Users\marty\Desktop\train_sets"
    WEIGHTS_DIR = r"C:\Users\marty\Projects\scorer\scorer\models\weights"
    
    # train_rf_model(
    #     train_data_path = os.path.join(BASE_PATH, "labeled"),
    #     val_data_path = os.path.join(BASE_PATH, "val"),
    #     encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"),
    #     model_save_path = os.path.join(WEIGHTS_DIR, "rf_sleep_classifier.pkl"),
    #     meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250'}
    # )
    
    validate_rf_model(os.path.join(BASE_PATH, "val"), 
                      encoder_weights_path = os.path.join(WEIGHTS_DIR, "adversarial_adjusted_encoder20260430.pt"), 
                      rf_model_path = r"C:\Users\marty\Projects\scorer\scorer\models\weights\rf_sleep_classifier.pkl", 
                      meta = {'ecog_channels': '0', 'emg_channels': '1', 'sample_rate': '250'})