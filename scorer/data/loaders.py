import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from pathlib import Path
from typing import Tuple

class SleepSignals(Dataset):
    """Dataset structure for chopped signals and labels, developed using 1000-datapoint windows
    Generally, there are 2 channels (ECoG, EMG)
    Sleep is labeled as: 
          0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'
    """
    def __init__(self, data_path, score_path, device = 'cuda', transform = None, augment = False, spectral_features = None) -> None:
        """        
            Args:
            file_path: path to the folder where chopped data is stored. Can handle one or multiple animals, but the recordings need to be in one file. 
            device: device to use (torch), default cuda
            transform: optional transform to apply. Defaults to None.
            augment: whether to apply data augmentation, relevant when training models
            spectral_features: whether to compute frequency descriptors. options: spectrogram, fourier, None. 
            Applied on the 1st passed channel
        """
        self.all_samples = None
        self.all_labels = None
        
        self._load(data_path, score_path)
                
        #move to device after loading        
        self.all_samples = self.all_samples.to(device)
        self.all_labels = self.all_labels.to(device)
        
        #get quantiles for all channels (alternatively just set limits to largest/smallest value?)
        #channel num = size at dim 1
        #runs out of memory, so if too many samples, use a random subset
        if self.all_labels.size(0) > 1000:
            rand_subset = torch.randint(0, self.all_labels.size(0), size = (1000,))      
            self.channel_ylims = [(torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .01), torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .99)) for dim in range(self.all_samples.size(1))]
        else:
            self.channel_ylims = [(torch.quantile(self.all_samples[:, dim, :].reshape(-1), q = .01), torch.quantile(self.all_samples[:, dim, :].reshape(-1), q = .99)) for dim in range(self.all_samples.size(1))]
        
        self.device = device
        self.transform = transform
        self.augment = augment

        self.spectral = spectral_features
        
        if self.spectral == 'spectrogram':
            self.spect = Spectrogram(n_fft = 100, #changes freq bins
                                    hop_length = 10, #changes time bins
                                    pad = 0, 
                                    power = 2, 
                                    center = False,
                                    normalized = False).to(device)
        
    def __len__(self) -> int:
        return len(self.all_labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.all_samples[idx, :, :]
        label = self.all_labels[idx]
        
        if self.augment:
            sample = self._augment(sample)
        if self.transform:
            sample = self.transform(sample)
        if self.spectral == 'spectrogram':
            sample = (sample, self._spect(sample, channel = 0))
        return sample, label
    
    def _load(self, data_path, score_path):
        """
        loads data that was saved in preprocessing and makes the shape sample x channel x window_len
        """
        with open(data_path, 'rb') as f:
            if Path(data_path).suffix == '.pt':
                self.all_samples = torch.load(data_path)
                self.all_samples = self.all_samples.to(dtype = torch.float32)
                self.all_samples = self.all_samples.permute(1, 0, 2)
                print(f'loaded tensor: {self.all_samples.size()}')
                #loaded tensor of dims channels x samples x win_length
                #samples should probably go to dim 0, so new shape = samples x channels x window size
                
            #legacy option: load a saved numpy array
            else:
                X_list = pickle.load(f)
                if not isinstance(X_list, torch.Tensor):
                    if isinstance(X_list, list):                         
                        self.all_samples = torch.from_numpy(np.concatenate(X_list)).to(dtype = torch.float32)
                    else:
                        self.all_samples = torch.from_numpy(X_list).to(dtype = torch.float32)
                else:
                    self.all_samples = self.all_samples.to(dtype = torch.float32)
                self.all_samples = self.all_samples.permute(0, 2, 1)
                print(f'loaded tensor from numpy: {self.all_samples.size()}')
                    
        with open(score_path, 'rb') as f:
            if Path(score_path).suffix == '.pt':
                self.all_labels = torch.load(score_path)
                self.all_labels = self.all_labels.to(dtype = torch.long)
                self.all_labels = self.all_labels.permute(1, 0, 2)
                if self.all_labels.ndim == 3 and self.all_labels.size(1) == 1:
                    self.all_labels = self.all_labels[..., 0].squeeze(1)    #keep only the 1st state and collapse channel 
                print(f'loaded tensor: {self.all_labels.size()}')
            else:
                y_list = pickle.load(f)
                if not isinstance(X_list, torch.Tensor):
                    if isinstance(y_list, list):
                        self.all_labels = torch.from_numpy(np.concatenate(y_list)).to(dtype=torch.long)
                    else:
                        self.all_labels = torch.from_numpy(y_list).to(dtype=torch.long)
                else:
                    self.all_labels = self.all_labels.to(dtype = torch.long)
                print(f'loaded tensor from numpy: {self.all_labels.size()}')
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """randomly apply time series augmentations - noise, scale, time shift"""
        x = x.clone()
        
        # Random noise
        if random.random() < 0.7:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        # Random scaling
        if random.random() < 0.5:
            scale = 0.8 + random.random() * 0.4  # Scale between 0.8 and 1.2
            x = x * scale
        
        # Random time shift
        if random.random() < 0.5:
            max_shift = int(x.shape[1] * 0.02)  # 2% of sequence length
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=0)
            elif shift < 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=0)
        return x
    
    def _spect(self, x: torch.Tensor, channel: int = 0) -> torch.Tensor:
        """
        compute spectrogram for one channel
        input: 2, time or 2, resample_freq
        output: 2, freq_bins, time_bins
        """
        spect = self.spect(x[channel, :])         
        return spect.to(self.device)