import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from torchaudio.transforms import Spectrogram
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
    def __init__(self, data_path, score_path, device = 'cuda', transform = None, augment = False, resample_freq = 1000, spectral_features = None) -> None:
        """        
            Args:
            file_path: path to the folder where chopped data is stored. Can handle one or multiple animals, but the recordings need to be in one file. 
            device: device to use (torch), default cuda
            transform: optional transform to apply. Defaults to None.
            augment: whether to apply data augmentation, relevant when training models
            resample_freq: whether to resample the signal1
            spectral_features: whether to compute frequency descriptors. options: spectrogram, fourier, band_powers, None
        """
        #implement all options for calculate_freqs!
        # think about scaling and calculation, whether to do it while loading or while retrieving samples
        # write as functions and import from preprocessing

        
        with open(data_path, 'rb') as f:
            X_list = pickle.load(f)
            #check whether it's a list of arrays or already concatenated
            if isinstance(X_list, list):                         
                self.all_samples = torch.from_numpy(np.concatenate(X_list)).to(dtype = torch.float32)
            else:
                self.all_samples = torch.from_numpy(X_list).to(dtype = torch.float32)
                
        with open(score_path, 'rb') as f:
            y_list = pickle.load(f)
    
            if isinstance(y_list, list):
                self.all_labels = torch.from_numpy(np.concatenate(y_list)).to(dtype=torch.long)
            else:
                self.all_labels = torch.from_numpy(y_list).to(dtype=torch.long)
                
        #move to device after loading        
        self.all_samples = self.all_samples.to(device)
        self.all_labels = self.all_labels.to(device)
        
        #summary stats - will be useful for setting limits on plots
        self.q99_0 = torch.quantile(self.all_samples[:, 0], q = .99)
        self.q01_0 = torch.quantile(self.all_samples[:, 0], q = .01)
        self.q99_1 = torch.quantile(self.all_samples[:, 1], q = .99)
        self.q01_1 = torch.quantile(self.all_samples[:, 1], q = .01)
        
        self.device = device
        self.transform = transform
        self.augment = augment
        
        #should move some transforms to preprocessing - if it can happen on raw data, it should. 
        self.resample_freq = resample_freq
        self.resampler = Resample(orig_freq = 1000, new_freq = resample_freq).to(device)
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
            
        sample = self.all_samples[idx, :, :] #should be 1000, 2
        label = self.all_labels[idx]
        
        if self.resample_freq != 1000:
            # Resample each channel separately
            channel_0 = self.resampler(sample[:, 0])  # [1000] -> [resample_freq]
            channel_1 = self.resampler(sample[:, 1])  # [1000] -> [resample_freq]
            sample = torch.stack([channel_0, channel_1], dim=0)  # [2, resample_freq]
        
        if self.augment:
            sample = self._augment(sample)
        if self.transform:
            sample = self.transform(sample)
        if self.spectral == 'spectrogram':
            sample = (sample, self._spect(sample))
            
        return sample, label

    
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
            max_shift = int(x.shape[0] * 0.02)  # 2% of sequence length
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                x = torch.cat([x[shift:], x[:shift]], dim=0)
            elif shift < 0:
                x = torch.cat([x[shift:], x[:shift]], dim=0)
        return x
    
    def _spect(self, x: torch.Tensor) -> torch.Tensor:
        """
        compute spectrogram for each channel
        input: 2, time or 2, resample_freq
        output: 2, freq_bins, time_bins
        """
        spectrograms = []
        for channel in range(x.shape[1]):
            spect = self.spect(x[:, channel]) 
            spectrograms.append(spect)
        x = torch.stack(spectrograms, dim=0)  
        
        return x.to(self.device)