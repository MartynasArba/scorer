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
        
        self.channel_ylims = []
        
        self._load(data_path, score_path)
        
        #move to device after loading        
        self.all_samples = self.all_samples.to(device)
        self.all_labels = self.all_labels.to(device)
        
        self.mean, self.std = self._compute_mean_std() 
        
        self.device = device
        self.transform = transform
        self.augment = augment

        self.spectral = spectral_features
        
         #for plotting
        
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
            self.channel_ylims = self.compute_ylims()
            
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
        self.channel_ylims = self.compute_ylims()
        
        
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
    
    def compute_ylims(self):
        """
        compute ylims of set mode
        """
        mode = self.params.get("ylim", "standard")
        print(mode)
        X = self.all_samples
        
        n_samples, n_channels, _ = X.size()
        
        ecog_n = len(self.params.get('ecog_channels', '0').split(','))
        emg_n  = len(self.params.get('emg_channels', '0').split(','))
        ephys_n = ecog_n + emg_n
        
        rand_idx = np.random.randint(0, n_samples, size=1000)
        ylims = [] #(center, spread) per channel
        
        for ch in range(n_channels):
                if mode == "standard":
                    if ch < ephys_n:
                        center = 0.0
                        spread = 0.2
                    else:
                        center = 0.5
                        spread = 1
                    ylims.append((center, spread))
                    continue

                infer_channel = ((mode == "infer") or (mode == "infer_ephys" and ch < ephys_n))
                if infer_channel:
                    vals = X[rand_idx, ch, :].reshape(-1).cpu().numpy()
                    median = float(np.quantile(vals, 0.50))
                    q_low  = float(np.quantile(vals, 0.01))
                    q_high = float(np.quantile(vals, 0.99))
                    spread = (q_high - q_low)

                    # avoid degenerate 
                    if spread < 1e-9:
                        spread = 1e-9
                    ylims.append((median, spread))
                    
                else:
                    # non-ephys channels in infer_ephys mode → (0,1)
                    center = 0.5
                    spread = 1.0
                    ylims.append((center, spread))
        print('ylims in loaders.py', ylims)
        return ylims
    
    def _compute_mean_std(self):
            """
            compute mean and std of dataset per-channel (8 vals each)
            """
            X = self.all_samples
            mean = X.mean(dim=(0, 2))
            std  = X.std(dim=(0, 2))
            # avoid divide by zero/low val in later norm
            std[std < 1e-8] = 1.0
            
            return mean, std
    
    
class SleepTraining(Dataset):
    """
    This is a dataset made to load training data with labels, not to score data.
    It follows the main SleepDataset class.
    
    Dataset structure for chopped signals and labels, developed using 1000-datapoint windows
    Generally, there are 2 channels (ECoG, EMG)
    Sleep is labeled as: 
          0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'
    """
    def __init__(self, data_path, random_state = 0,
                 n_files_to_pick = 100, device = 'cuda', 
                 transform = None, augment = False,
                 metadata: dict = {}, balance: str = "none", # "none" | "undersample" | "oversample") -> None:
                 exclude_labels: tuple = ()):  
        
        torch.manual_seed(random_state)

        self.params = metadata
        
        self.all_samples = None
        self.all_labels = None
        
        self._load(data_path, n_files_to_pick)
        
        if balance != "none":
            self._balance_labels(balance=balance, exclude_labels=exclude_labels, seed=random_state)
            self._remap_labels_to_contiguous()
                
        #move to device after loading        
        self.all_samples = self.all_samples.to(device)
        self.all_labels = self.all_labels.to(device)
        
        self.mean, self.std = self._compute_mean_std() 
        
        self.device = device
        self.transform = transform
        self.augment = augment        
        
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
        
        return sample, label
    
    def _load(self, data_path, n_files_to_pick):
        """
        important change from the other dataset:
        1. scans all files from passed data path, selects subset of n_files_to_pick
        2. loads as usual, joins into one output
        """
        import glob
        data_paths = sorted(glob.glob(data_path + './*/X*.pt'))
        score_paths = sorted(glob.glob(data_path + './*/y*.pt'))
        subset_idxs = np.random.choice(len(data_paths), size = n_files_to_pick, replace = False)
        
        x_paths = [data_paths[idx] for idx in subset_idxs]
        y_paths = [score_paths[idx] for idx in subset_idxs]
        
        if not (x_paths or y_paths):
                raise RuntimeError(f"No X_*.pt or no y_*.pt found in folder: {data_path}")
    
        x_chunks = [torch.load(f).float() for f in x_paths]
        X = torch.cat(x_chunks, dim=1)        # concat on sample dimension

        if X.ndim == 3:
            X = X.permute(1, 0, 2)            # [samples, channels, win_len]

        self.all_samples = X
        print(f"Loaded {len(x_paths)} X files: {X.shape}")

        y_chunks = [torch.load(f).long() for f in y_paths]
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
        print(f"Loaded y files, final y shape: {Y.shape}")        
        return
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """randomly apply time series augmentations - noise, scale, time shift"""
        x = x.clone()
        
        # Random noise
        if random.random() < 0.7:
            noise = torch.randn_like(x) * 0.2       #CHANGED FROM 0.01
            x = x + noise
        
        # Random scaling
        if random.random() < 0.5:
            scale = 0.5 + random.random() * 1.0 # Scale between 0.8 and 1.2 #CHANGED FROM 0.8, 0.4
            x = x * scale
        
        # Random time shift
        if random.random() < 0.5:
            max_shift = int(x.shape[1] * 0.02)  # 2% of sequence length
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=1)
            
        return x
    
    def _compute_mean_std(self):
            """
            compute mean and std of dataset per-channel (8 vals each)
            """
            X = self.all_samples
            mean = X.mean(dim=(0, 2))
            std  = X.std(dim=(0, 2))
            # avoid divide by zero/low val in later norm
            std[std < 1e-8] = 1.0
            
            return mean, std
        
    def _balance_labels(self, balance="undersample", exclude_labels=(0,), seed=0):
        """
        balance: "undersample" (downsample to min class) or "oversample" (upsample to max class)
        exclude_labels: labels to remove before balancing  (0 for unlabeled)
        """
        if balance not in ("undersample", "oversample"):
            raise ValueError(f"balance must be 'undersample' or 'oversample', got {balance}")

        # on CPU for indexing
        labels = self.all_labels
        if labels.is_cuda:
            labels = labels.cpu()

        # exclude labels
        keep_mask = torch.ones_like(labels, dtype=torch.bool)
        for lab in exclude_labels or ():
            keep_mask &= (labels != lab)

        keep_idx = torch.where(keep_mask)[0]
        if keep_idx.numel() == 0:
            raise RuntimeError("after excluding labels, no samples remain.")
        labels_kept = labels[keep_idx]

        # get idx per class
        classes = torch.unique(labels_kept).tolist()
        per_class = {}
        for c in classes:
            per_class[c] = keep_idx[torch.where(labels_kept == c)[0]]

        counts = {c: int(per_class[c].numel()) for c in classes}
        print("class counts before balance:", counts)

        if balance == "undersample":
            target = min(counts.values())
            replace = False
        else:  # oversample
            target = max(counts.values())
            replace = True

        g = torch.Generator().manual_seed(seed)
        balanced_idx = []
        for c in classes:
            idx_c = per_class[c]
            if idx_c.numel() == 0:
                continue
            if replace:
                # sample with replacement up to target
                pick = idx_c[torch.randint(0, idx_c.numel(), (target,), generator=g)]
            else:
                # sample without replacement down to target
                perm = idx_c[torch.randperm(idx_c.numel(), generator=g)[:target]]
                pick = perm

            balanced_idx.append(pick)
        balanced_idx = torch.cat(balanced_idx)
        # shuffle final set
        balanced_idx = balanced_idx[torch.randperm(balanced_idx.numel(), generator=g)]

        # Apply subset
        self.all_samples = self.all_samples[balanced_idx]
        self.all_labels = self.all_labels[balanced_idx]

        # Print after
        labels2 = self.all_labels
        if labels2.is_cuda:
            labels2 = labels2.cpu()
        new_counts = {int(c): int((labels2 == c).sum()) for c in torch.unique(labels2)}
        print("class counts after balance :", new_counts)
        
    def _remap_labels_to_contiguous(self):
        # make labels 0..K-1
        labels_cpu = self.all_labels.detach().cpu()
        uniq = torch.unique(labels_cpu)
        mapping = {int(old): i for i, old in enumerate(uniq.tolist())}

        # vectorized remap
        new = torch.empty_like(labels_cpu)
        for old, new_id in mapping.items():
            new[labels_cpu == old] = new_id

        self.label_mapping = mapping          # e.g. {1:0, 2:1, 4:2}
        self.inv_label_mapping = {v:k for k,v in mapping.items()}
        self.all_labels = new.to(self.all_labels.device)

        print("label remap:", self.label_mapping)
    
if __name__ == "__main__":
    dataset = SleepTraining(
        data_path = 'G:/oslo_data',
        n_files_to_pick = 100,
        random_state = 0,
        device = 'cuda',
        transform = None,
        augment = False,
        metadata = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard'}
    )    