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

def notch_filter(signal: torch.Tensor, sr: int = 250, freq_to_remove: float = 50., metadata = {'device': 'cuda'}) -> torch.Tensor:
    """
    designs a FIR bandstop filter based on passed freq to remove +-1, then uses torch filtfilt to apply it on the signal
    """
    #create filter coefs with scipy
    coefs = firwin(numtaps = 501, 
                   cutoff = (freq_to_remove - 1, freq_to_remove + 1), 
                   window = 'hamming',
                   pass_zero = True,
                   fs = sr)
    b = torch.tensor(coefs, dtype=torch.float32).to(device = metadata['device'])
    a = torch.ones_like(b, dtype=torch.float32).to(device = metadata['device']) #filtfilt required denominator param, but in FIR no adaptation so = 1
    signal = signal.to(dtype = torch.float32, device = metadata['device'])
    signal = filtfilt(signal, a, b)
    return signal


def bandpass_filter(signal: torch.Tensor, sr: int = 250, freqs: Tuple[float, float] = (0.5, 30.), metadata = {'device': 'cuda'}) -> torch.Tensor:
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