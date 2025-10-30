import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Tuple
from scipy.signal import firwin
from torchaudio.functional import filtfilt

def bandpass_filter(signal: torch.Tensor, sr: int = 250, freqs: Tuple[int, int] = (0.5, 30), metadata = {'device': 'cuda'}) -> torch.Tensor:
    """
    designs a FIR bandpass filter based on passed freqs, then uses torch filtfilt to apply it on the signal
    """
    #create filter coefs with scipy
    coefs = firwin(numtaps = 501, 
                   cutoff = freqs, 
                   window = 'hamming',
                   pass_zero = False,
                   fs = sr)
    b = torch.tensor(coefs, dtype=torch.float32).to(device = metadata['device'])
    a = torch.ones_like(b, dtype=torch.float32).to(device = metadata['device']) #filtfilt required denominator param, but in FIR no adaptation so = 1
    signal = signal.to(dtype = torch.float32, device = metadata['device'])
    signal = filtfilt(signal, a, b)
    return signal

def load_from_csv(path: str, metadata: dict = None, states: int = None):
    """
    loads data from csv into a torch tensor
    """
    ecog_channels = metadata.get('ecog_channels', None)
    emg_channels = metadata.get('emg_channels', None)
    device = metadata.get('device', 'cuda')
    #if cuda is not available
    device = "cpu" if not torch.cuda.is_available() else device
    #should be changed later to support chunks // could also move to numpy, but it's probably not necessary now and questionable in general
    data = pd.read_csv(path)
    try:
        ecog_channels = [int(ch) for ch in ecog_channels.split(',')]
        emg_channels = [int(ch) for ch in emg_channels.split(',')]
    except Exception as e:
        print(f'Something went wrong when parsing metadata of channel numbers in load_from_csv: {e}')
        return None, None, None

    ecog = torch.tensor(data.iloc[:, ecog_channels].values, device = device, dtype = torch.float32)
    emg = torch.tensor(data.iloc[:, emg_channels].values, device = device, dtype = torch.float32)

    if states:
        states = torch.tensor(data.iloc[:, states].values, device = device, dtype = torch.float32)
    
    del data
    return ecog, emg, states
            
            
def load_from_csv_in_chunks(path: str, metadata: dict = None, states: int = None, chunk_size: int = None):
    """
    loads data from csv into a torch tensor in chunks, returns a generator
    """
    ecog_channels = metadata.get('ecog_channels', None)
    emg_channels = metadata.get('emg_channels', None)
    device = metadata.get('device', 'cuda')
    #if cuda is not available
    device = "cpu" if not torch.cuda.is_available() else device
    try:
        ecog_channels = [int(ch) for ch in ecog_channels.split(',')]
        emg_channels = [int(ch) for ch in emg_channels.split(',')]
    except Exception as e:
        print(f'Something went wrong when parsing metadata of channel numbers in load_from_csv: {e}')
        return None, None, None
    
    for chunk in pd.read_csv(path, chunksize = chunk_size):
        ecog_chunk = torch.tensor(chunk.iloc[:, ecog_channels].values, device=device)
        emg_chunk = torch.tensor(chunk.iloc[:, emg_channels].values, device=device)
        states_chunk = torch.tensor(chunk.iloc[:, states].values, device=device) if states else None 
        yield ecog_chunk, emg_chunk, states_chunk

def from_Oslo_csv(path: str, sep: str = '\\') -> None:
    """
    Converts Oslo .csv to pickle 
    shouldn't be used at all in the final version
    """
    
    xpath, ypath = _get_data_paths(path)
    
    data = pd.read_csv(path)
    signal = data[['ecog', 'emg']].values
    labels = data['sleep_episode'].values
    X, y = chop_data(states = labels, values = signal, win_len = 1000, labeled = True)
    print(X.shape)

    with open(xpath, 'wb') as f:
        pickle.dump(X, f)
        
    with open(ypath, 'wb') as f:
        pickle.dump(y, f)
        
def from_non_annotated_csv(path: str) -> None:
    """
    Converts non-annotated .csv to chopped pickle 
    """
    xpath, ypath = _get_data_paths(path)
    
    data = pd.read_csv(path)
    signal = data[['ecog', 'emg']].values
    labels = data['sleep_episode'].values
    X, y = chop_data(states = labels, values = signal, win_len = 1000, labeled = False)
    print(X.shape)

    with open(xpath, 'wb') as f:
        pickle.dump(X, f)
        
    with open(ypath, 'wb') as f:
        pickle.dump(y, f)

def chop_data(states: np.array, values: np.array, win_len: int = 1000, labeled: bool = True) -> Tuple[np.array, np.array]:
    """
    runs helper functions to split data into win-length segments 
    helper functions depend on labeled param
    ideally, labeled shouldn't be used, as some data is skipped due to non-win-length labeling
    """
    # in the file: time, ecog, emg, state
    if labeled: #if the data has been labeled, chops by label boundaries
        return _chop_by_state(states, values, win_len)
    else: #if the data is not labeled, chops sequentially
        return _chop(values, win_len)
    
def _get_data_paths(csv_path: str) -> Tuple[str, str]:
    """
    helper: generates paths and folders to save processed data
    """
    
    csv_path = Path(csv_path)
    
    #ensure processed data folder exists
    processed_folder = csv_path.parent.parent / 'processed'
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    base_name = csv_path.stem  # filename without extension
    xpath = processed_folder / f"{base_name}_X.pkl"
    ypath = processed_folder / f"{base_name}_y.pkl"
    
    return str(xpath), str(ypath)
    
def _chop_by_state(states: np.array, values: np.array, win_len: int) -> Tuple[np.array, np.array]:
    X = []
    y = []
    #find all episodes of set len and return their indices
    #state changes found
    state_changes = np.where(np.diff(states).astype(bool))[0] + 1
    #handle boundaries with start/end points
    state_changes = np.concatenate([[0], state_changes, [len(states)]])

    for id in tqdm(range(len(state_changes) - 1)):
        # trying to get better readability
        start_idx = state_changes[id] #state start index 
        end_idx = state_changes[id + 1] #state end index
        current_state = states[start_idx] #keep track of current state
        
        #get segments and append
        if end_idx - start_idx >= win_len: #check if the window is still long enough
            i = start_idx   #mark starting index
            while i + win_len <= end_idx: #iterate through state and save data
                X.append(values[i:i + win_len])
                y.append(current_state)
                i += win_len
    #handle cases where no data was found
    if len(X) == 0:
        return np.array([]), np.array([])
    #return stacked arrays
    return np.stack(X), np.stack(y)

def _chop(values: np.array, win_len: int) -> Tuple[np.array, np.array]:
    """
    chops data into win-len pieces
    state is always 0 (unknown)
    """
    
    X = []
    y = []
    #similar syntax for consistency, generally unnecessary
    start_idx = 0
    end_idx = len(values)
    current_state = 0
    #check whether there is enough data in general
    if end_idx < win_len:
        return np.array([]), np.array([]) 
    else: #main loop
        i = start_idx
        while i + win_len <= end_idx:
            X.append(values[i:i + win_len])
            y.append(current_state)
            i += win_len
    
    return np.stack(X), np.stack(y)