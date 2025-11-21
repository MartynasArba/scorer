import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from torch.fft import rfft, rfftfreq
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
    def __init__(self, data_path, score_path, device = 'cuda', transform = None, augment = False, spectral_features = None, metadata: dict = {}) -> None:
        """        
            Args:
            file_path: path to the folder where chopped data is stored. Can handle one or multiple animals, but the recordings need to be in one file. 
            device: device to use (torch), default cuda
            transform: optional transform to apply. Defaults to None.
            augment: whether to apply data augmentation, relevant when training models
            spectral_features: whether to compute frequency descriptors. options: spectrogram, fourier, None.
            Applied on the 1st passed channel
            metadata is the same as everywhere else, shared between other parts of the program
        """
        self.params = metadata
        
        self.all_samples = None
        self.all_labels = None
        
        self._load(data_path, score_path)
                
        #move to device after loading        
        self.all_samples = self.all_samples.to(device)
        self.all_labels = self.all_labels.to(device)
        
        #get quantiles for all channels (alternatively just set limits to largest/smallest value?)
        #channel num = size at dim 1
        #runs out of memory, so if too many samples, use a random subset
        
        num_ephys_channels = len(self.params.get('ecog_channels', '0').split(',')) + len(self.params.get('emg_channels', '0').split(','))
        
        #approximate num channels
        rand_subset = torch.randint(0, self.all_labels.size(0), size = (1000,))      
        #lims should be center+-spread, so center, spread needed
        #infer for all channels
        if self.params.get('ylims', '') == 'infer':
            self.channel_ylims = [(torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .5).item(), (torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .99).item() - torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .01).item())) for dim in range(self.all_samples.size(1))]
        elif self.params.get('ylims', '') == 'standard':
            self.channel_ylims = [(0, 0.2) for dim in range(num_ephys_channels)]
        elif self.params.get('ylims', '') == 'infer_ephys':
            self.channel_ylims = [(torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .5).item(), (torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .99).item() - torch.quantile(self.all_samples[rand_subset, dim, :].reshape(-1), q = .01).item())) for dim in range(num_ephys_channels)]
        else:
            self.channel_ylims = []

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
        if self.spectral:
            if self.spectral == 'spectrogram':
                sample = (sample, self._spect(sample, channel = 0))
            elif self.spectral == 'fourier':
                sample = (sample, self._fft(sample, channel = 0))
        
        return sample, label
    
    def _load(self, data_path, score_path):
        """
        loads data that was saved in preprocessing and makes the shape sample x channel x window_len
        supported inputs: 
            folder of files (X*.pt, y*.pt)
            single .pt
        """
        data_path = Path(data_path)
        score_path = Path(score_path)
        
        #load folders with chunk data
        if data_path.is_dir():
            x_files = sorted(data_path.glob("X_*.pt"))
            if not x_files:
                raise RuntimeError(f"No X_*.pt found in folder: {data_path}")

            x_chunks = [torch.load(f).float() for f in x_files]
            X = torch.cat(x_chunks, dim=1)        # concat on sample dimension

            if X.ndim == 3:
                X = X.permute(1, 0, 2)            # [samples, channels, win_len]

            self.all_samples = X
            print(f"Loaded {len(x_files)} X chunks: {X.shape}")
            
            #do same for y
            y_files = sorted(data_path.glob("y_*.pt"))
            if not y_files:
                raise RuntimeError(f"No y_*.pt found in folder: {data_path}")

            y_chunks = [torch.load(f).long() for f in y_files]
            #chunk state dimensions are 1channel, samples, win_len
            Y = torch.cat(y_chunks, dim = 1) 
            Y = Y.to(dtype = torch.long)
            Y = Y.permute(1, 0, 2)
            if Y.ndim == 3 and Y.size(1) == 1:
                Y = Y[..., 0].squeeze(1)    #keep only the 1st state and collapse channel 
            else:
                print(f'error when loading states, ndim = {Y.ndim}, shape: {Y.size()}')
                return        
            self.all_labels = Y
            print(f"Loaded y chunks, final y shape: {Y.shape}")
            return
        
        #handle .pt files 
        if data_path.suffix == ".pt":
            X = torch.load(data_path)
            X = X.to(dtype=torch.float32)
            if X.ndim == 3:
                X = X.permute(1, 0, 2)
            self.all_samples = X
            print(f"Loaded tensor: {X.shape}")
        else:
            print('only .pt files are supported')
            return
        if score_path.suffix == ".pt":
            Y = torch.load(score_path).long()
            Y = Y.permute(1, 0, 2)
            if Y.ndim == 3 and Y.size(1) == 1:
                    Y = Y[..., 0].squeeze(1)
            self.all_labels = Y
        else:
            print('only .pt files are supported, states')      
        
        
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
    
    def _fft(self, x: torch.Tensor, channel: int = 0) -> torch.Tensor:
        """
        compute fft for passed data
        """
        fft = rfft(x[channel, :]).to('cuda')
        power = torch.abs(fft) ** 2
        freqs = rfftfreq(x.size(-1), d = 1 / int(self.params.get('sample_rate', 250))).to('cuda')
        output = torch.stack((power, freqs), dim = -1)      #stacked power/freqs tensor
        
        return output.to(self.device)