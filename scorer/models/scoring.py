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

def score_signal(data_path, state_save_folder, meta, scorer_type = 'heuristic', apply_corrections = False, return_confidence = False):
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
                        '3state_GRU': ContextAwareSleepScorer
                         }
    
    selected_scorer = available_scorers.get(scorer_type)
    try:
        dataset = SleepSignals(data_path = data_path, 
                        score_path = data_path, 
                        device = meta.get('device', 'cpu'),
                        transform = None,
                        augment = False,
                        normalize = True,
                        scale = 1.0,
                        spectral_features = None,
                        metadata = meta)
    except Exception as e:      #should be changed, but it's not filenotfounderror, but runtimeerror in loaders.py
        print(f'whoops in scoring when loading data: {e}')
        return
    
    confidence = None
    
    if scorer_type != '3state_GRU':
        print('other model types have been depreciated')
        return
    

    elif scorer_type == '3state_GRU':
        weights_path = Path(meta.get('weights_path', r"C:\Users\marty\Projects\scorer\scorer\models\weights\3state_SCDS_GRU_weights.pt"))
        if not weights_path.exists():
            print(f"Weights not found at {weights_path}. Ensure main_pipeline.py has been run successfully.")
            return
            
        model = load_trained_sequence_model(str(weights_path), device=dataset.device)
        # run_sequence_inference handles scoring and remapping to labels
        if return_confidence:
            all_preds, confidence = run_sequence_inference(model, dataset, seq_len=10, batch_size=128, 
                                                           apply_corrections=apply_corrections, 
                                                           return_confidence=True)
            all_preds = all_preds.cpu().numpy()
            confidence = confidence.cpu().numpy()
        else:
            all_preds = run_sequence_inference(model, dataset, seq_len=10, batch_size=128, apply_corrections=apply_corrections)
            all_preds = all_preds.cpu().numpy()
                
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
        
    # set int values based on model's class mapping
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

def run_sequence_inference(model, sleep_dataset, seq_len=10, batch_size=128, apply_corrections = False, return_confidence = False):
    """
    passes continuous data from SleepSignals through a sequence model
    """
    device = sleep_dataset.device
    
    # extract continuous tensor of all windows [Total_Windows, Channels, Win_Len]
    X_continuous = sleep_dataset.all_samples
    
    # print(f'X_continous shape: {X_continuous.size()}')
    #select one channel
    if X_continuous.shape[1] > 1:
        X_continuous = X_continuous[:, 0:1, :]
        
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
            
            # model output: [Batch, Classes, Seq_Len]
            # print(f"GUI Batch Max: {X_batch.max().item():.4f}, Min: {X_batch.min().item():.4f}, Mean: {X_batch.mean().item():.4f}")
            
            # print(f'batch size before preds: {X_batch.size()}')
            
            # if start_idx == 0:  #  plot first sequence to check
            #     plt.plot(X_batch[0, 0, 0, :].cpu().numpy()) # plot first window, first channel
            #     plt.title("norm input to model")
            #     plt.show()
            
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
        final_predictions = torch.from_numpy(apply_heuristics(final_predictions.cpu().numpy(), num_classes=3)).to(device)

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

if __name__ == "__main__":
    path = r"C:\Users\marty\Desktop\train_sets\unlabeled\windowed_2026032514575020251207-1_g0_t0.obx0.obx_box3"
    score_signal(path, path, meta = {}, scorer_type = '3state_GRU', apply_corrections = False, return_confidence = False)
