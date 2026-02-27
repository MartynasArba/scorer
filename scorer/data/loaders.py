import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, IterableDataset
from torchaudio.transforms import Spectrogram
from torch.fft import rfft, rfftfreq
from pathlib import Path
import bisect
from collections import OrderedDict

class SleepSignals(Dataset):
    """
    Dataset structure for chopped signals and labels, developed using 1000-datapoint windows
    Generally, there are 2 channels (ECoG, EMG), and 6 extracted features: ECoG & EMG power, Delta, Theta, Alpha and Sigma powers
    This dataset is designed and used mostly in the GUI. For training and evaluating models, see SleepTraining class.
    Sleep is labeled as: 
          0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'
    """
    def __init__(self, 
                 data_path: str, 
                 score_path: str, 
                 device:str = 'cuda', 
                 transform:bool = None, 
                 augment:bool = False, 
                 spectral_features:bool = None, 
                 metadata: dict = {}) -> None:
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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        
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
    
    def _load(self, data_path: str, score_path: str) -> None:
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
                Y = Y[..., 0].squeeze(1)    #keep only 1st state
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
    
    def compute_ylims(self) -> list[tuple[float, float]]:
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

                    # avoid division by close to 0
                    if spread < 1e-9:
                        spread = 1e-9
                    ylims.append((median, spread))
                    
                else:
                    # non-ephys channels in infer_ephys mode set to 0.5, 1
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
    It mirrors the main SleepDataset class. 
    
    USE ONLY FOR TRAINING AND ONLY WITH RANDOM SHUFFLE!
    Ensure y contains actual pre-scored labels
    If balance or exclude_labels is passed, the order of frames will not be as recorded. This is also the case as multiple recordings are loaded at random.
    
    Dataset structure for chopped signals and labels, developed using 1000-datapoint windows
    Generally, there are at least 2 channels (ECoG, EMG), and 6 extracted features: ECoG & EMG power, Delta, Theta, Alpha and Sigma power
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
                 metadata: dict = {}, 
                 balance: str = "none", # "none" | "undersample" | "oversample") -> None:
                 exclude_labels: tuple = (), 
                 merge_nrem: bool = False):  
        """
        Initializes the dataset
        """
                
        torch.manual_seed(random_state)

        self.params = metadata
        
        self.all_samples = None
        self.all_labels = None
        
        self._load(data_path, n_files_to_pick)
        
        if merge_nrem:
            self.all_labels[self.all_labels == 3] = 2
        
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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.all_samples[idx, :, :]
        label = self.all_labels[idx]
        
        if self.augment:
            sample = self._augment(sample)
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    
    def _load(self, data_path: str, n_files_to_pick: int = 100) -> None:
        """
        important change from the "normal" dataset:
        1. scans all files from passed data path, selects subset of n_files_to_pick
        2. loads as usual, joins into one output
        """
        import glob
        data_paths = sorted(glob.glob(data_path + './*/X*.pt'))
        score_paths = sorted(glob.glob(data_path + './*/y*.pt'))
        if n_files_to_pick == None:
            n_files_to_pick = len(data_paths)
        
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
            noise = torch.randn_like(x) * 0.01 
            x = x + noise
        
        # Random scaling
        if random.random() < 0.5:
            if x.ndim == 3:
                # [Batch, Channels, 1] allows broadcasting across the Length dimension
                scales = 0.5 + torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
            else:
                # [Channels, 1] for single samples
                scales = 0.5 + torch.rand(x.shape[0], 1, device=x.device)
            x = x * scales
        
        # Random time shift
        if random.random() < 0.5:
            max_shift = int(x.shape[1] * 0.02)  # 2% of sequence length
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=2 if x.ndim == 3 else 1)
            
        return x
    
    def _compute_mean_std(self) -> tuple[float, float]:
            """
            compute mean and std of dataset per-channel (8 vals each)
            """
            X = self.all_samples
            mean = X.mean(dim=(0, 2))
            std  = X.std(dim=(0, 2))
            # avoid divide by zero/low val in later norm
            std[std < 1e-8] = 1.0
            
            return mean, std
        
    def _balance_labels(self, balance:str = "none", exclude_labels:tuple = (0,), seed:int = 0) -> None:
        """
        rebalances labels, important in sleep as there are usually way less REM samples than NREM or W
        balance: "none", "undersample" (downsample to min class) or "oversample" (upsample to max class)
        exclude_labels: labels to remove before balancing  (0 for unlabeled)
        """
        if balance not in ("none", "undersample", "oversample"):
            raise ValueError(f"balance must be 'none', 'undersample' or 'oversample', got {balance}")

        # on CPU for indexing
        labels = self.all_labels
        if labels.is_cuda:
            labels = labels.cpu()

        # 1. ALWAYS apply the exclude_labels mask first
        keep_mask = torch.ones_like(labels, dtype=torch.bool)
        for lab in exclude_labels or ():
            keep_mask &= (labels != lab)

        keep_idx = torch.where(keep_mask)[0]
        if keep_idx.numel() == 0:
            raise RuntimeError("after excluding labels, no samples remain.")
            
        # 2. If no balancing requested, just apply exclusions and exit early!
        if balance == "none":
            self.all_samples = self.all_samples[keep_idx]
            self.all_labels = self.all_labels[keep_idx]
            print("No balancing applied. Removed excluded labels.")
            return

        # 3. Otherwise, proceed with balancing
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
        
    def _remap_labels_to_contiguous(self) -> None:
        """
        if excluding labels, remaps them to start at 0 again
        """
        labels_cpu = self.all_labels.detach().cpu()
        uniq = torch.unique(labels_cpu)
        mapping = {int(old): i for i, old in enumerate(uniq.tolist())}

        # vectorized remap
        new = torch.empty_like(labels_cpu)
        for old, new_id in mapping.items():
            new[labels_cpu == old] = new_id

        self.label_mapping = mapping
        self.inv_label_mapping = {v:k for k,v in mapping.items()}
        self.all_labels = new.to(self.all_labels.device)

        print("label remap:", self.label_mapping)
        

class BufferedSleepDataset(IterableDataset):
    """
    Iterable version of SleepTraining.
    Loads chunks of files into RAM (buffer), shuffles internally, and yields samples.
    Solves I/O bottlenecks while maintaining batch diversity for Contrastive Learning.
    Supports balancing, label remapping, augment, transform, and multiprocessing.
    """

    def __init__(self,
                 data_path,
                 random_state=0,
                 n_files_to_pick=100,
                 device='cuda',
                 transform=None,
                 augment=False,
                 metadata=None,
                 balance="none",
                 exclude_labels=(),
                 merge_nrem=False,
                 buffer_size=100): # Replaced cache_size with buffer_size

        torch.manual_seed(random_state)
        self.device = device
        self.transform = transform
        self.augment = augment
        self.params = metadata or {}
        self.buffer_size = buffer_size

        self.file_map = []              # (start_idx, end_idx, path)
        self.file_start_indices = []
        self.master_labels = None
        self.active_indices = None

        # 1. Load label metadata exactly as before
        self._load(data_path, n_files_to_pick)

        if merge_nrem:
            self.master_labels[self.master_labels == 3] = 2

        # 2. Balance and filter labels to get our global 'active_indices'
        if balance != "none" or exclude_labels:
            self._balance_labels(balance, exclude_labels, random_state)
            self._remap_labels_to_contiguous()
        else:
            self.active_indices = torch.arange(len(self.master_labels))

        self.master_labels = self.master_labels.to(self.device)

    def __len__(self):
        # Even for IterableDatasets, providing length helps DataLoader progress bars
        return len(self.active_indices)

    def __iter__(self):
        """
        The core buffered loading logic. Replaces __getitem__.
        """
        worker_info = torch.utils.data.get_worker_info()
        file_indices = list(range(len(self.file_map)))
        
        # Shuffle the order of files every epoch
        random.shuffle(file_indices)
        
        # If num_workers > 0 (e.g. 4), split the files evenly among the CPU workers
        if worker_info is not None:
            per_worker = int(np.ceil(len(file_indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_indices = file_indices[worker_id * per_worker : (worker_id + 1) * per_worker]

        # Process the assigned files in chunks of `buffer_size`
        for i in range(0, len(file_indices), self.buffer_size):
            chunk_file_indices = file_indices[i : i + self.buffer_size]
            
            chunk_x = []
            chunk_y = []
            
            # Load all X files for this chunk into RAM
            for f_idx in chunk_file_indices:
                start_idx, end_idx, x_path = self.file_map[f_idx]
                
                # Find which active global indices belong to this specific file
                mask = (self.active_indices >= start_idx) & (self.active_indices < end_idx)
                valid_global_indices = self.active_indices[mask]
                
                if len(valid_global_indices) == 0:
                    continue # Skip file entirely if all its samples were excluded/balanced away
                
                # Load the full file
                tensor = torch.load(x_path).float()
                if tensor.ndim == 3:
                    tensor = tensor.permute(1, 0, 2)
                    
                # Extract only the valid samples using local indexing
                local_indices = valid_global_indices - start_idx
                chunk_x.append(tensor[local_indices])
                chunk_y.append(self.master_labels[valid_global_indices])
                
            if not chunk_x:
                continue # Edge case if an entire chunk was filtered out
                
            # Concatenate the active samples from the 300 files into one tensor
            chunk_x_tensor = torch.cat(chunk_x, dim=0)
            chunk_y_tensor = torch.cat(chunk_y, dim=0)
            
            # Shuffle thoroughly WITHIN the RAM chunk to ensure Contrastive batch diversity
            num_samples = chunk_x_tensor.size(0)
            shuffle_idx = torch.randperm(num_samples)
            
            chunk_x_tensor = chunk_x_tensor[shuffle_idx]
            chunk_y_tensor = chunk_y_tensor[shuffle_idx]
            
            # Yield individual samples to the DataLoader
            for j in range(num_samples):
                x = chunk_x_tensor[j]
                y = chunk_y_tensor[j]
                
                if self.augment:
                    x = self._augment(x)
                if self.transform:
                    x = self.transform(x)
                    
                yield x, y

    def _load(self, data_path, n_files_to_pick):
        import glob
        x_paths = sorted(glob.glob(data_path + './*/X*.pt'))
        y_paths = sorted(glob.glob(data_path + './*/y*.pt'))

        if not x_paths or not y_paths:
            raise RuntimeError(f"No X/y files found in {data_path}")

        if n_files_to_pick is None:
            n_files_to_pick = len(x_paths)

        subset = np.random.choice(len(x_paths),
                                  size=min(n_files_to_pick, len(x_paths)),
                                  replace=False)

        x_paths = [x_paths[i] for i in subset]
        y_paths = [y_paths[i] for i in subset]

        labels_list = []
        offset = 0

        for x_path, y_path in zip(x_paths, y_paths):
            y = torch.load(y_path).long()
            if y.ndim == 3:
                y = y.permute(1, 0, 2) 
            if y.ndim == 3 and y.size(1) == 1:
                y = y[..., 0].squeeze(1) 
            else:
                raise RuntimeError(f"Unexpected label shape in {y_path}: {y.shape}")
            n_samples = y.shape[0]
            labels_list.append(y)
            self.file_map.append((offset, offset + n_samples, x_path))
            self.file_start_indices.append(offset)
            offset += n_samples
            
        self.master_labels = torch.cat(labels_list, dim=0)
        print(f"Y example shape: {labels_list[0].shape}")
        print("Total samples:", len(self.master_labels))

    def _balance_labels(self, balance, exclude_labels, seed):
        labels = self.master_labels.cpu()
        keep_mask = torch.ones_like(labels, dtype=torch.bool)
        for lab in exclude_labels:
            keep_mask &= labels != lab

        keep_idx = torch.where(keep_mask)[0]
        labels_kept = labels[keep_idx]

        classes = torch.unique(labels_kept)
        per_class = {int(c): keep_idx[labels_kept == c] for c in classes}

        counts = {c: len(v) for c, v in per_class.items()}
        print("class counts before:", counts)

        if balance == "undersample":
            target = min(counts.values())
            replace = False
        else:
            target = max(counts.values())
            replace = True

        g = torch.Generator().manual_seed(seed)
        balanced = []

        for c, idxs in per_class.items():
            if replace:
                pick = idxs[torch.randint(0, len(idxs), (target,), generator=g)]
            else:
                pick = idxs[torch.randperm(len(idxs), generator=g)[:target]]
            balanced.append(pick)

        self.active_indices = torch.cat(balanced)
        self.active_indices = self.active_indices[
            torch.randperm(len(self.active_indices), generator=g)
        ]

        print("after balance:",
              {int(c): int((self.master_labels[self.active_indices] == c).sum())
               for c in torch.unique(self.master_labels[self.active_indices])})

    def _remap_labels_to_contiguous(self):
        labels = self.master_labels.cpu()
        uniq = torch.unique(labels[self.active_indices])
        mapping = {int(old): i for i, old in enumerate(uniq.tolist())}

        new = torch.empty_like(labels)
        for old, new_id in mapping.items():
            new[labels == old] = new_id

        self.label_mapping = mapping
        self.inv_label_mapping = {v: k for k, v in mapping.items()}
        self.master_labels = new.to(self.device)
        print("label remap:", self.label_mapping)

    def _augment(self, x):
        x = x.clone()
        if random.random() < 0.7:
            x += torch.randn_like(x) * 0.01
        if random.random() < 0.5:
            scale = 0.5 + random.random()
            x *= scale
        if random.random() < 0.5:
            max_shift = int(x.shape[1] * 0.02)
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=1)
        return x

#for testing
if __name__ == "__main__":
    dataset = SleepTrainingLazy(
        data_path = 'G:/oslo_data',
        n_files_to_pick = None,
        random_state = 0,
        device = 'cuda',
        transform = None,
        augment = False,
        metadata = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard'}
    )    
    
    print(dataset[0])