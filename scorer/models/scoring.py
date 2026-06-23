# code to do pre-trained scoring should go here
# load into SleepSignals
#do scoring: launcher function 
from scorer.data.loaders import SleepSignals
# from scorer.models.heuristic_scorer_v2 import HeuristicScorer2
from scorer.models.sleep_cnn import SleepCNN, EphysSleepCNN, DualStreamSleepCNN, SCDSSleepCNN
from scorer.models.sequence_model import ContextAwareSleepScorer

from scorer.data.storage import save_pickled_states
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import os
import pickle
from scipy.ndimage import gaussian_filter1d

def score_signal(data_path, state_save_folder, meta, selected_ch = 0, scorer_type = 'heuristic', apply_corrections = False, return_confidence = False):
    """
    runs scoring, launched from GUI
    saves states according to selected path
    scorers might have their own preprocessing requirements, check class definitions
    
    data_path should be the folder containing X_ and y_ paths of windowed data that results from preprocessing
    state_save_folder is passed from GUI selection    
    """

    try:
        dataset = SleepSignals(data_path = data_path, 
                        score_path = data_path, 
                        device = meta.get('device', 'cpu'),
                        transform = None,
                        augment = False,
                        normalize = True,
                        scale = 1,
                        spectral_features = None,
                        metadata = meta)
    except Exception as e:      #should be more specific, but it's not filenotfounderror, but runtimeerror in loaders.py
        print(f'whoops in scoring when loading data: {e}')
        return
    
    confidence = None
    
    if scorer_type not in ['3state_GRU', 'random_forest', 'context_rf']:
        print(f'Scorer type {scorer_type} is not supported or has been deprecated.')
        return


    elif scorer_type == '3state_GRU':
        weights_path = Path(meta.get('weights_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\3state_SCDS_GRU_weights.pt"))
        if not weights_path.exists():
            print(f"Weights not found at {weights_path}. Ensure main_pipeline.py has been run successfully.")
            return
            
        model = load_trained_sequence_model(str(weights_path), device=dataset.device)
        # run_sequence_inference handles scoring and remapping to labels
        if return_confidence:
            all_preds, confidence = run_sequence_inference(model, dataset, selected_ch = selected_ch, seq_len=10, batch_size=128, 
                                                           apply_corrections=apply_corrections, 
                                                           return_confidence=True)
            all_preds = all_preds.cpu().numpy()
            confidence = confidence.cpu().numpy()
        else:
            all_preds = run_sequence_inference(model, dataset, selected_ch = selected_ch, seq_len=10, batch_size=128, apply_corrections=apply_corrections)
            all_preds = all_preds.cpu().numpy()

    elif scorer_type == 'random_forest':
        rf_model_path = meta.get('rf_model_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\rf_sleep_classifier.pkl")
        encoder_weights_path = meta.get('weights_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\adversarial_adjusted_encoder20260430.pt")
        
        if not os.path.exists(rf_model_path):
            print(f"RF model not found at {rf_model_path}")
            return
            
        all_preds = score_with_rf(
            dataset=dataset,
            rf_model_path=rf_model_path,
            encoder_weights_path=encoder_weights_path,
            meta=meta,
            selected_ch = selected_ch
        )
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, mapping = 'gui')
            
    elif scorer_type == 'context_rf':
        rf_model_path = meta.get('rf_model_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\rf_context_sleep_classifier_new_adjusted.pkl")
        encoder_weights_path = meta.get('weights_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\adversarial_adjusted_encoder20260615.pt")
        
        if not os.path.exists(rf_model_path):
            print(f"RF context model not found at {rf_model_path}")
            return
            
        all_preds = score_with_context_rf(
            dataset=dataset,
            rf_model_path=rf_model_path,
            encoder_weights_path=encoder_weights_path,
            meta=meta,
            selected_ch = selected_ch
        )
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, mapping = 'gui')

    state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + '_' + meta.get('optional_tag', '') + scorer_type + '_states.pkl')
    save_pickled_states(all_preds, state_save_path)
    
    if confidence is not None:
        conf_save_path = state_save_path.parent / (state_save_path.stem + "_confidence.pkl")
        print(f'mean confidence: {confidence.mean()}')        
        save_pickled_states(confidence, conf_save_path)
        print(f'confidence saved as {conf_save_path}')

    if apply_corrections:
        print('score corrections applied')
    print(f'scoring done, states saved as {state_save_path}')
    print(f'found unique states: {np.unique(all_preds, return_counts = True)}')

def init_encoder(weights_path=None, device='cpu', win_len=1000):
    """helper for encoder init dummy pass and weight cleaning"""
    encoder = SCDSSleepCNN(num_classes=3).to(device)
    dummy_input = torch.randn(1, 1, int(win_len)).to(device)
    encoder(dummy_input)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        cleaned_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        encoder.load_state_dict(cleaned_dict, strict=False)
    encoder.eval()
    return encoder

def score_with_rf(dataset, rf_model_path, encoder_weights_path, meta, selected_ch = 0):
    """
    Uses a pre-trained CNN encoder to extract embeddings and a Random Forest 
    to classify sleep states for a SleepSignals dataset.
    """
    encoder = init_encoder(encoder_weights_path, device=dataset.device, win_len=meta.get('win_len', 1000))

    # get embeddings
    X_embeddings, _ = extract_embeddings(encoder, dataset, selected_ch = selected_ch)

    # load Random Forest model
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    # preds = rf_model.predict(X_embeddings)
    #prob smoothing instead
    probs = rf_model.predict_proba(X_embeddings)
    smoothed_probs = gaussian_filter1d(probs, sigma = 1, axis = 0)
    preds = np.argmax(smoothed_probs, axis = 1)
    
    gui_labels = np.zeros_like(preds)
    gui_labels[preds == 0] = 1 # Wake
    gui_labels[preds == 1] = 2 # NREM
    gui_labels[preds == 2] = 4 # REM

    return gui_labels

def score_with_context_rf(dataset, rf_model_path, encoder_weights_path, meta, selected_ch = 0):
    """uses a pretrained CNN encoder and CONTEXT AWARE random forest model 
    to classify sleep states for a sleepsignals dataset"""
    
    encoder = init_encoder(encoder_weights_path, device=dataset.device, win_len=meta.get('win_len', 1000))

    print("Extracting base embeddings...")
    X_base, _ = extract_embeddings(encoder, dataset, selected_ch = selected_ch)
    
    # build 1536-D context features
    print("Applying context...")
    X_context = create_context_features(X_base)

    # load RF
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    # predict probs
    probs = rf_model.predict_proba(X_context)
    
    # could boost REM by 30% (but this does very little)
    # probs[:, 2] = probs[:, 2] * 1.30
    # row_sums = probs.sum(axis=1, keepdims=True)
    # probs = probs / row_sums
    
    # final state
    preds = np.argmax(probs, axis=1)
    
    # remap to GUI
    gui_labels = np.zeros_like(preds)
    gui_labels[preds == 0] = 1 # Wake
    gui_labels[preds == 1] = 2 # NREM
    gui_labels[preds == 2] = 4 # REM

    return gui_labels    
    
def apply_heuristics(states: np.ndarray, mapping: str = '3_class') -> np.ndarray:
    """
    applies biological heuristic rules to a sequence of predicted sleep states
    states: 1D numpy array of predicted states
    
    mapping options:
      '5_class' : 0(Unlabeled), 1(Wake), 2(NREM), 3(IS), 4(REM)
      '3_class' : 0(Wake), 1(NREM), 2(REM)
      'gui'     : 1(Wake), 2(NREM), 4(REM)
    """
    smoothed = states.copy()
    n = len(smoothed)

    if n < 3:
        return smoothed
        
    # assign target int based on mapping
    if mapping == '5_class' or mapping == 'gui':
        WAKE, REM = 1, 4
    elif mapping == '3_class':
        WAKE, REM = 0, 2
    else:
        raise ValueError("mapping must be '5_class', '3_class', or 'gui'")

    # remove W -> REM transitions
    for i in range(1, n):
        if smoothed[i] == REM and smoothed[i-1] == WAKE:
            smoothed[i] = WAKE
            
    # clean up single window states except W to keep microawakenings
    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    # repeat single window cleanup (catches states created by step 1)
    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    return smoothed

def load_trained_sequence_model(weights_path, device='cuda', window_length = 1000):
    """
    loads trained ContextAwareSleepScorer (CNN encoder + GRU) for inference.
    window_length: sample length of a single window 
    """
    # init base encoder
    encoder = SCDSSleepCNN(num_classes=3)
    
    # dummy pass to init LazyLinear layer shapes
    # shape: [Batch, Channels, Time]
    dummy_input = torch.randn(1, 1, window_length) 
    encoder(dummy_input)
    
    # init GRU wrapper
    model = ContextAwareSleepScorer(
        encoder, 
        embedding_dim=512, 
        hidden_dim=64, 
        num_classes=3, 
        num_layers=2
    )
    
    # load saved weights (this loads BOTH GRU and CNN weights)
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # move to device and lock into eval mode
    model = model.to(device)
    model.eval() 
    model.encoder.eval()
    
    print("Model successfully loaded and ready for inference!")
    return model

import torch

def run_sequence_inference(model, sleep_dataset, selected_ch = 0, seq_len=10, batch_size=128, apply_corrections = False, return_confidence = False):
    """
    passes continuous data from SleepSignals through a sequence model
    """
    device = sleep_dataset.device
    
    # extract continuous tensor of all windows [Total_Windows, Channels, Win_Len]
    X_continuous = sleep_dataset.all_samples
    
    # print(f'X_continous shape: {X_continuous.size()}')
    #select one channel
    if X_continuous.shape[1] > 1:
        X_continuous = X_continuous[:, selected_ch:(selected_ch + 1), :]
        
    # print(f"OOD Data Stats -> Max: {X_continuous.max():.2f}, Min: {X_continuous.min():.2f}, Std: {X_continuous.std():.2f}")
    
    T_total = X_continuous.shape[0]
    
    if T_total < seq_len:
        raise ValueError(f"Recording is too short! Need at least {seq_len} windows.")

    # Accumulators for probability-based voting
    # Shape: [Total_Windows, Num_Classes]
    prob_sums = torch.zeros((T_total, 3), device=device)
    counts = torch.zeros((T_total, 1), device=device)
    
    # lock model for inference
    model.eval()
    model.encoder.eval()    #triple checking
    
    with torch.no_grad():
        # iter through data, create sliding sequences on the fly
        # loop up to (total_windows - seq_len + 1) to ensure full sequences
        for start_idx in range(0, T_total - seq_len + 1, batch_size):
            batch_sequences = []
            
            # find end of batch
            end_idx = min(start_idx + batch_size, T_total - seq_len + 1)
            
            # build sequence batch
            for i in range(start_idx, end_idx):
                sequence = X_continuous[i : i + seq_len] 
                batch_sequences.append(sequence)
            
            X_batch = torch.stack(batch_sequences).to(device)
            
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            
            # add probabilities to total sums for each window in the sequence
            for t in range(seq_len):
                # t-th element of sequence is global window start_idx + t
                batch_indices = torch.arange(start_idx, end_idx, device=device) + t
                prob_sums.index_add_(0, batch_indices, probs[:, :, t])
                counts.index_add_(0, batch_indices, torch.ones((end_idx - start_idx, 1), device=device))

    # compute final labels via majority (highest average probability)
    avg_probs = prob_sums / counts.clamp(min=1)
    confidence, final_predictions = torch.max(avg_probs, dim=1)
    
    if apply_corrections:
        final_predictions = torch.from_numpy(apply_heuristics(final_predictions.cpu().numpy(), mapping='3_class')).to(device)

    # aligned predictions should cover the full recording [0 : T_total]
    #this is here because padding was needed before
    aligned_predictions = final_predictions
    
    # remap to GUI
    gui_labels = torch.zeros_like(aligned_predictions)
    gui_labels[aligned_predictions == 0] = 1 # Wake
    gui_labels[aligned_predictions == 1] = 2 # NREM
    gui_labels[aligned_predictions == 2] = 4 # REM
    
    print(f"Inference complete. Generated {len(gui_labels)} labels.")
    if return_confidence:
        return gui_labels, confidence
    return gui_labels

def extract_embeddings(model, dataset, batch_size=256, selected_ch = 0):
    """helper to run data through the encoder and get features"""
    device = dataset.device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader, desc="Extracting Embeddings"):
            # SleepSignals should return [Window, Channel, Win_Len]
            x = x.to(device)
            if x.shape[1] > 1:
                x = x[:, selected_ch:(selected_ch + 1), :]
            # forward pass through CNN
            embeddings = model(x)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

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

def apply_heuristics_lite(states: np.ndarray, mapping: str = '3_class') -> np.ndarray:
    """
    A gentle heuristic that ONLY cleans up isolated 4-second artifacts.
    It intentionally leaves Wake->REM boundaries alone, trusting the Context RF.
    This might be wrong in some cases. 
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

def score_multiple_signals(meta_df_path):
    import pandas as pd
    meta_df = pd.read_csv(meta_df_path)
    
    print(meta_df)
    
    # score_signal(data_path, 
    #              state_save_folder, 
    #              meta = {}, 
    #              selected_ch = 0, 
    #              scorer_type = 'context_rf', 
    #              apply_corrections = True, 
    #              return_confidence = False)


if __name__ == "__main__":
    meta_df_path = r"C:\Users\marty\Projects\2026analysis\data\meta_paths.csv"
    score_multiple_signals(meta_df_path)
    
    # path = r"C:\Users\marty\Desktop\train_sets\unlabeled\windowed_2026032514575020251207-1_g0_t0.obx0.obx_box3"
    # path = r"C:\Users\marty\Desktop\train_sets\final_test\F\windowed_pilot_ch0"
    # score_signal(path, path, meta = {}, scorer_type = 'context_rf', apply_corrections = True, return_confidence = False)
    

    # labeled_train_path = r"C:\Users\marty\Desktop\train_sets\final_test\F\windowed_pilot_ch0"
    # ood_inference_path = r"C:\Users\marty\Desktop\train_sets\unlabeled\windowed_2026032514575020251207-1_g0_t0.obx0.obx_box3"
    # weights = r"C:\Users\marty\Projects\scorer\scorer\models\weights\adversarial_adjusted_encoder20260430.pt"
    # save_file = r"C:\Users\marty\Desktop\rf_alternative_states.pkl"
