import pickle
import os
import json
from pathlib import Path
from typing import Tuple
from numpy import array
import torch
import numpy as np
import pandas as pd

def construct_paths(data_path: str, format: str = '.pkl', **metadata) ->Tuple[str, str]:    
    """
    Gets data path from QFileDialog, returns save paths for pickle and state annotation arrays   
    """
    
    data_path = Path(data_path)
    
    if data_path.exists():
        
        y_file_path = data_path.with_name(data_path.name.replace('X', 'y'))
        
        scorer = metadata.get('scorer', data_path.stem[:-2] if data_path.suffix else data_path.stem)
        unique_id = metadata.get('scoring_started', 'noID')
        date = metadata.get('date', '')
        animal = metadata.get('animal', '')
        trial = metadata.get('trial', '')
        repetition_id = 0
        
        #create required score dir if it doesn't exist
        score_folder = data_path.parent.parent / 'scores'
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        
        states_array_path = score_folder / f'{unique_id}{metadata.get('optional_tag', '')}_scores_{scorer}_{date}_{animal}_{trial}_{repetition_id}{format}'
        
        #don't overwrite existing files
        while states_array_path.exists():
            repetition_id += 1
            states_array_path = score_folder / f'{unique_id}{metadata.get('optional_tag', '')}_scores_{scorer}_{date}_{animal}_{trial}_{repetition_id}{format}'
    
        return y_file_path, states_array_path

def save_pickled_states(states: list, path: str) -> None:
    """
    dumps states to pickle
    """
    with open(path, 'wb') as f:
        print('TODO: change save_pickled_states to only dump states which are being scored by current scorer or delete this message')
        pickle.dump(states, f)

def load_pickled_states(path: str) -> array:
    """
    loads scores from pickle
    """
    with open(path, 'rb') as f:
        print('TODO: change load_pickled_states usage to have list of states or delete this message')
        states = pickle.load(f)
    return states

def save_json(states: list, path: str) -> None:  
    """
    should save json summary of state results, could be metadata, then state-duration format
    """   
    print('TODO: save_json not implemented')
    pass
    #TODO: implement saving as json with metadata
    
def save_metadata(path: str, metadata: dict) -> None:
    """
    saves metadata to a json file
    """
    with open(path, 'wt') as f:
        json.dump(metadata, f)

def load_metadata(path: str) -> dict:
    """
    loads metadata from a json file
    """
    with open(path, 'rt') as f:
        metadata = json.load(f)
    return metadata

def save_tensor(tensor_seq: tuple = (), 
                metadata: dict = {},
                overwrite = False, 
                chunk = None, 
                raw = False):
    """
    saves tensors that were passed in a sequence
    path is set according to metadata
    """

    proj_path = Path(metadata.get('project_path', '.'))
    
    if raw:
        if isinstance(chunk, int):
            file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_chunk{chunk}_raw.pt' 
        else:
            file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_raw.pt' 
        save_folder = proj_path / "raw"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder / file_name
    else:
        if isinstance(chunk, int):
            file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_chunk{chunk}_processed.pt' 
        else:
            file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_processed.pt' 
        save_folder = proj_path / "processed"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder / file_name
    
    if not overwrite:
        if os.path.exists(save_path):
            print('file already exists, not saving')
            return None
    
    #ensure everything is a tensor
    tensor_seq = [tensor for tensor in tensor_seq if isinstance(tensor, torch.Tensor)]
    #ensure lengths match
    min_len = min(t.size(1) for t in tensor_seq)
    tensor_seq = [tensor[:, : min_len] for tensor in tensor_seq]
    
    if len(set(t.size(1) for t in tensor_seq)) > 1:
        print(f"length mismatch in save_tensor: {[t.size(1) for t in tensor_seq]}, truncating to {min_len}")
    
    if len(tensor_seq) == 0:
        print('no tensors to save')
        return None

    
    to_save = torch.cat(tensor_seq, dim = 0)
    with open(save_path, 'wb') as f:
        torch.save(to_save, f)
            
def save_windowed(tensors: tuple, 
                  states: torch.Tensor = None, 
                  metadata: dict = {}, 
                  win_len: int = 1000,
                  chunked: bool = True, 
                  overwrite: bool = False,
                  testing: bool = False):
    """chops and saves tensor data to be used by the SleepDataset class"""
    
    #get path
    proj_path = Path(metadata.get('project_path', '.'))
    file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_X.pt' 
    save_folder = proj_path / "processed"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder / file_name
    
    if not overwrite:
        if os.path.exists(save_path):
            print('file already exists, not saving')
            return None

    tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    if len(tensors) == 0:
        print('no tensors to save')
        return None
    #do the same as in save_tensor to align times
    min_len = min(t.size(1) for t in tensors)
    tensors = [tensor[:, : min_len] for tensor in tensors]
    
    if len(set(t.size(1) for t in tensors)) > 1:
        print(f"length mismatch in save_windowed: {[t.size(1) for t in tensors]}, truncating to {min_len}")

    #here it should still be channels x time
    if len(tensors) > 1:
        to_save = torch.cat(tensors, dim = 0)
        
    if states is None:
        states = torch.zeros(size = (1, to_save.size()[-1]), dtype=torch.long, device=to_save.device)
    else:
        if states.dim() != 2:       #ensure states shape 1 x time, required to use the same chop function
            states = states.unsqueeze(0)
    
    #chopping to sample x channels x time, stack here
    if not testing:
        to_save = _chop(to_save, win_len = win_len) #
        states = _chop(states, win_len = win_len)
    else:
        to_save, states = _chop_by_state(states, to_save, win_len)
    
    if to_save.size(0) == 0:
        print("Warning: to_save has 0 channels — skipping save.")
        return

    if chunked & os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            prev_data = torch.load(f)
            print(f'shape of prev_data: {prev_data.size()}')
            print(f'shape of to_save: {to_save.size()}')
            if isinstance(prev_data, np.ndarray):
                prev_data = torch.from_numpy(prev_data).to(dtype = torch.float32).to(metadata.get('device', 'cpu'))
            to_save = torch.cat((prev_data.to(device = to_save.device), to_save), dim = 1)
            
    with open(save_path, 'wb') as f:
        torch.save(to_save, f) 

    #handle states       
    states_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}_y.pt'
    states_path = save_folder / states_name

    if chunked & os.path.exists(states_path):
        with open(states_path, 'rb') as f:
            prev_states = torch.load(f)
            print(f'prev_states shape: {prev_states.size()}')
            if isinstance(prev_states, np.ndarray):
                prev_states = torch.from_numpy(prev_states).to(dtype = torch.long).to(metadata.get('device', 'cpu'))
            states = torch.cat((prev_states.to(device = states.device), states.to(dtype = torch.long)), dim  = 1)

    with open(states_path, 'wb') as f:
        torch.save(states, f)
    
    for var in ("prev_data", "prev_states"):
        if var in locals():
            del locals()[var]
    del to_save, states
    
    #importing in SleepDataset
def _chop(values: torch.Tensor, win_len: int):
    """
    chops data into win-len pieces
    """
    length = values.size()[1]
    n_channels = values.size()[0]
    n_samples = length // win_len  #ignoring leftover data
    values = values[:, : win_len * n_samples] #all chs, but trim to fit len
    values = values.reshape(n_channels, n_samples, win_len)
    return values

def _chop_by_state(states: torch.Tensor, 
                  values: torch.Tensor, 
                  win_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    split values by state
    """
    #states must be 1d (not sure if necessary)
    states = states.flatten()
    #detect state changes
    diffs = torch.diff(states)
    state_changes = torch.nonzero(diffs != 0).squeeze(-1) + 1
    # add boundaries
    state_changes = torch.cat([
        torch.tensor([0], device=states.device),
        state_changes,
        torch.tensor([len(states)], device=states.device)
    ])

    X, y = [], []
    for idx in range(len(state_changes) - 1):
        start_idx = int(state_changes[idx])
        end_idx = int(state_changes[idx + 1])
        current_state = states[start_idx]
        # process only if the segment is long enough
        if end_idx - start_idx >= win_len:
            i = start_idx
            while i + win_len <= end_idx:
                segment = values[..., i : i + win_len]
                if segment.size(-1) == win_len:
                    X.append(segment)
                    y.append(current_state)
                i += win_len
    if len(X) == 0:
        return torch.empty(0), torch.empty(0)
    # stack into tensors
    X = torch.stack(X)
    y = torch.stack(y)

    return X, y
  
  
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