import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Tuple
from scipy.signal import firwin
from torchaudio.functional import filtfilt, bandpass_biquad
import torch.nn.functional as F

def notch_filter_old(signal: torch.Tensor, sr: int = 250, freq_to_remove: float = 50., metadata = {'device': 'cuda'}) -> torch.Tensor:
    """
    depreciated, use bandpass filter instead
    very slow due to filtfilt
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


def bandpass_filter_old(signal: torch.Tensor, sr: int = 250, freqs: Tuple[float, float] = (0.5, 30.), metadata = {'device': 'cuda'}) -> torch.Tensor:
    """
    depreciated
    designs a FIR bandpass filter based on passed freqs, then uses torch filtfilt to apply it on the signal
    extremely slow due to filtfilt 
    """
    #create filter coefs with scipy
    coefs = firwin(numtaps = 501, 
                   cutoff = freqs, 
                   window = 'hamming',
                   pass_zero = False,
                   fs = sr)
    b = torch.tensor(coefs, dtype=torch.float32).to(device = metadata['device'])
    a = torch.ones_like(b, dtype=torch.float32).to(device = metadata['device']) #filtfilt required denominator param, but in FIR no adaptation so = 1
    signal = signal.to(dtype = torch.float32, device = metadata['device'])  #size should be channels x time
    signal = filtfilt(signal, a, b)
    return signal

def bandpass_filter(signal: torch.Tensor,
                        sr: int = 250,
                        freqs: tuple = (0.5, 30.0),
                        numtaps: int = 501,
                        device: str = 'cuda',
                        zero_phase: bool = True) -> torch.Tensor:
    """
    FIR bandpass filter via FFT convolution on GPU. signal: [T] or [C, T] - zero_phase: if True, applies forward + reverse convolution (like filtfilt)
    should hopefully be faster
    """
    signal = signal.to(device=device, dtype=torch.float32)
    coefs = firwin(numtaps=numtaps, cutoff=freqs, fs=sr, pass_zero=False, window='hamming')
    coefs = torch.tensor(coefs, dtype=torch.float32, device=device)
    
    # fft length
    n_fft = 2 ** ((signal.shape[-1] + coefs.numel() - 1).bit_length())
    S = torch.fft.rfft(signal, n=n_fft)#signal in freq domain
    H = torch.fft.rfft(coefs, n=n_fft)#filter in freq domain
    y = torch.fft.irfft(S * H, n=n_fft)[..., :signal.shape[-1]]
    # forward convolution in frequency domain by S*H, then irfft converts back to time
    #also trimming of output to match input

    if zero_phase:
        # reverse filter for approximate filtfilt
        Y_rev = torch.fft.rfft(y.flip(-1), n=n_fft)
        y_rev = torch.fft.irfft(Y_rev * H, n=n_fft)[..., :signal.shape[-1]]
        y = y_rev.flip(-1)
    return y

def hilbert_transform(x: torch.Tensor):
    """
    compute the Hilbert transform
    """
    fourier = torch.fft.fft(x)  #get fft
    h = torch.zeros_like(fourier)
    n = fourier.size(0)
    if n % 2 == 0:      #construct which freqs to zero, which to keep: keep 0 and Nyquist freqs, double positives, zero negatives
        h[0] = h[n//2] = 1  #dc and nyquist
        h[1 : n//2] = 2 #positives
    else:   #odd-length
        h[0] = 1
        h[1:(n+1)//2] = 2
    #apply mask
    z = torch.fft.ifft(fourier * h) #this is the analytical signal
    return z

def gaussian_kernel(sigma: float = 0.2, sr: int = 250, device: str = 'cuda'):
    """
    compute gaussian kernel for filtering later
    """
    radius = int(3 * sigma * sr)    #set gaussian radius
    t = torch.arange(-radius, radius + 1, device=device)    #get tensor
    kernel = torch.exp(-0.5 * (t / (sigma * sr))**2)    #calculate gaussian
    kernel /= kernel.sum()  #normalize
    return kernel.view(1, 1, -1)

def band_powers(signal: torch.Tensor, bands: dict = {'delta': (0.5, 4)}, sr: int = 250, device: str = 'cuda', smoothen = 0.2):
    """
    apply a simple bandpass filter, 
    hilbert transform to return analytical signal, 
    take amplitude and add Gaussian smoothing,
    normalize to 0-1,
    return. 
    """
    print(f'in band_powers: {signal.size()}')
    signal = signal.to(device) #make sure just in case
    signal = signal.to(dtype = torch.float32)
    band_envelopes = {}
    if smoothen:
        kernel = gaussian_kernel(sigma = smoothen, sr = sr, device = device)
    else:
        kernel = None
    
    if signal.ndim == 1:
            signal = signal.unsqueeze(0)
    
    for name, (low, high) in bands.items():
        center_freq = (low + high) / 2
        Q = center_freq / (high - low)
        filtered = bandpass_biquad(signal, sr, center_freq, Q)
        filtered = filtered.squeeze(0)
        analytic = hilbert_transform(filtered)
        amplitude = torch.abs(analytic) 
        if amplitude.ndim == 1:
            amplitude = amplitude.unsqueeze(0).unsqueeze(0)
        elif amplitude.ndim == 2:
            amplitude = amplitude.unsqueeze(0)
        
        if kernel is not None:
            amplitude = F.conv1d(amplitude, kernel, padding = 'same').squeeze(0)
        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-12)
        band_envelopes[name] = amplitude
    return band_envelopes   

def sum_power(signal: torch.Tensor, smoothing: float = 0.2, sr: int = 250, device: str = 'cuda', normalize = True):
    """
    calculates sum power by rms with moving average smoothing kernel
    """
    signal = signal.to(dtype = torch.float32)
    
    if signal is not None:
        if device is not None:
            signal = signal.to(device)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        power = signal ** 2    #get power by squaring
        win_len = max(1, int(round(smoothing * sr))) #moving average window size
        kernel = torch.ones(1, 1, win_len, device = power.device) / win_len #construct kernel, pytorch requires shape ch_in, ch_out, size. division to get average instead of sum
        pad = win_len // 2  #add some padding
        avg_power = F.conv1d(power.unsqueeze(1), kernel, padding=pad)
        rms = torch.sqrt(avg_power.squeeze(1) + 1e-12)  #very small value ensures float smoothing is never <0
        if normalize:
            rms = (rms - torch.min(rms)) / (torch.max(rms) - torch.min(rms))    #scale
        return rms

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