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
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm

def score_signal(data_path, state_save_folder, meta, scorer_type = 'heuristic', apply_corrections = True):
    """
    runs scoring, launched from GUI
    saves states according to selected path
    scorers might have their own preprocessing requirements, check class definitions
    
    data_path should be the folder containing X_ and y_ paths of windowed data that results from preprocessing
    state_save_folder is passed from GUI selection    
    """

    available_scorers = {
                        '3state_pretrained': EphysSleepCNN,
                        '3state_dual': DualStreamSleepCNN,
                        '5state_pretrained': EphysSleepCNN
                         }
    
    selected_scorer = available_scorers.get(scorer_type)
    try:
        dataset = SleepSignals(data_path = data_path, 
                        score_path = data_path, 
                        device = meta.get('device', 'cpu'),
                        transform = None,
                        augment = False,
                        spectral_features = None,
                        metadata = meta)
    except Exception as e:      #should be changed, but it's not filenotfounderror, but runtimeerror in loaders.py
        print(f'whoops: {e}')
        return
    
    if scorer_type == 'heuristic':
        print('heuristic scorer has been depreciated')
        return

        
    elif scorer_type == '3state_pretrained':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        try:
            scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_pretrained_ephys_dropout.pt', weights_only= False)
        except FileNotFoundError:
            print('Check weights folder - selected model not found!')
            return
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
                
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, num_classes = 3)        
            
        #now reset states to include 0 - shift by 1, then reset REM to 4
        all_preds = np.array(all_preds) + 1
        all_preds[all_preds == 3] = 4 
    
    elif scorer_type == '3state_dual':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        try:
            scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_dual_fine_tuned.pt', weights_only= False)
        except FileNotFoundError:
            print('Check weights folder - selected model not found!')
            return
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
                
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, num_classes = 3)        
            
        #now reset states to include 0 - shift by 1, then reset REM to 4
        all_preds = np.array(all_preds) + 1
        all_preds[all_preds == 3] = 4 
    
    elif scorer_type == '3state_SCDS':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        try:
            scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_SCDS_2.pt', weights_only= False)
        except FileNotFoundError:
            print('Check weights folder - selected model not found!')
            return
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
                
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, num_classes = 3)        
            
        #now reset states to include 0 - shift by 1, then reset REM to 4
        all_preds = np.array(all_preds) + 1
        all_preds[all_preds == 3] = 4 
        
    elif scorer_type == '5state_pretrained':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        try:
            scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\5state_pretrained_ephys.pt', weights_only= False)
        except FileNotFoundError:
            print('Check weights folder - selected model not found!')
            return
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
        if apply_corrections:
            all_preds = apply_heuristics(all_preds, num_classes = 5)
        
    else:
        print(f'unavailable scorer selected: {scorer_type}')
        return
        
    state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + '_' + meta.get('optional_tag', '') + scorer_type + '_states.pkl')
    save_pickled_states(all_preds, state_save_path)
    if apply_corrections:
        print('score corrections applied')
    print(f'scoring done, states saved as {state_save_path}')
    print(f'found unique states: {np.unique(all_preds, return_counts = True)}')

def apply_heuristics(states: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    applies biological heuristic rules to a sequence of predicted sleep states.
    states: 1D numpy array of predicted states.
    5 class mapping: 0(Unlabeled), 1(Wake), 2(NREM), 3(IS), 4(REM)
    3 class mapping: 0(Wake), 1(NREM), 2(REM)
    """
    smoothed = states.copy()
    n = len(smoothed)

    if n < 3:
        return smoothed
        
    # Set the integer values based on your model's class mapping
    if num_classes == 5:
        WAKE, REM, IS = 1, 4, 3
    elif num_classes == 3:
        WAKE, REM, IS = 0, 2, -1  # IS doesn't exist in 3-class
    else:
        raise ValueError("num_classes must be 3 or 5")

    # remove W -> REM transitions
    for i in range(1, n):
        if smoothed[i] == REM and smoothed[i-1] == WAKE:
            smoothed[i] = WAKE  # Reclassify the false REM as Wake

    # clean up single window states except wake (to keep microawakenings)
    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    # only in 5 class: brief W -> IS -> W is impossible and is just W
    if num_classes == 5:
        for i in range(1, n - 1):
            if smoothed[i] == IS:
                if smoothed[i-1] == WAKE and smoothed[i+1] == WAKE:
                    smoothed[i] = WAKE

    # repeat of single window cleanup, as more might've been created
    for i in range(1, n - 1):
        if smoothed[i] == WAKE:
            continue
        elif smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    return smoothed

def load_trained_sequence_model(weights_path, device='cuda', window_length = 1000):
    """
    Loads trained ContextAwareSleepScorer (CNN encoder + GRU) for inference.
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
    
    print("Model successfully loaded and ready for inference!")
    return model

import torch

def run_sequence_inference(model, sleep_dataset, seq_len=10, batch_size=128):
    """
    passes continuous data from SleepSignals through a sequence model
    """
    device = sleep_dataset.device
    
    # extract continuous tensor of all windows [Total_Windows, Channels, Win_Len] - might run out of memory, but maybe not
    X_continuous = sleep_dataset.all_samples
    total_windows = X_continuous.shape[0]
    
    if total_windows < seq_len:
        raise ValueError(f"Recording is too short! Need at least {seq_len} windows.")

    all_predictions = []
    
    # lock model for inference
    model.eval()
    
    with torch.no_grad():
        # iter through data, create sliding sequences on the fly
        # loop up to (total_windows - seq_len + 1) to ensure full sequences
        for start_idx in range(0, total_windows - seq_len + 1, batch_size):
            batch_sequences = []
            
            # find end of batch
            end_idx = min(start_idx + batch_size, total_windows - seq_len + 1)
            
            # build sequence batch
            for i in range(start_idx, end_idx):
                # slice 10 continuous windows: [10, Channels, Win_Len]
                sequence = X_continuous[i : i + seq_len] 
                batch_sequences.append(sequence)
            
            # stack into a single batch tensor: [Batch, 10, Channels, Win_Len]
            X_batch = torch.stack(batch_sequences).to(device)
            
            # pass through the model
            outputs = model(X_batch)
            _, predicted_classes = torch.max(outputs, dim=1)
            
            all_predictions.append(predicted_classes)
            
    # combine all batches into one flat tensor
    raw_predictions = torch.cat(all_predictions)
    
    # pad the beginning
    # (missing labels for the first 9 windows on 1st label etc.)
    # copy 1st prediction backward to fill
    padding = raw_predictions[0].repeat(seq_len - 1)
    aligned_predictions = torch.cat([padding, raw_predictions])
    
    # remap to GUI
    gui_labels = torch.zeros_like(aligned_predictions)
    gui_labels[aligned_predictions == 0] = 1 # Wake
    gui_labels[aligned_predictions == 1] = 2 # NREM
    gui_labels[aligned_predictions == 2] = 4 # REM
    
    print(f"Inference complete. Generated {len(gui_labels)} labels.")
    return gui_labels

if __name__ == "__main__":
    print('not implemented as a standalone script')