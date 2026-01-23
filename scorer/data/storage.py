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
        
        if data_path.is_dir():
            y_file_path = data_path
        else:  
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
        pickle.dump(states, f)

def load_pickled_states(path: str) -> array:
    """
    loads scores from pickle
    """
    with open(path, 'rb') as f:
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
        save_folder = proj_path / "raw"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        if isinstance(chunk, int):
            chunk_folder = save_folder / f'raw_{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}'    #if chunked, create or select a folder to save in
            if not os.path.exists(chunk_folder):
                os.makedirs(chunk_folder)
            file_name = f'chunk{chunk}_raw.pt' 
            save_path = chunk_folder / file_name
        else:
            file_name = f'raw_{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}.pt'
            save_path = save_folder / file_name
    else:
        save_folder = proj_path / "processed"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if isinstance(chunk, int):
            chunk_folder = save_folder / f'preprocessed_{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}'    #if chunked, create or select a folder to save in
            if not os.path.exists(chunk_folder):
                os.makedirs(chunk_folder)
            file_name = f'chunk{chunk}_preprocessed.pt' 
            save_path = chunk_folder / file_name
        else:
            file_name = f'preprocessed_{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}.pt' 
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
                  chunk_id: int = None,
                  overwrite: bool = False,
                  testing: bool = False):
    """chops and saves tensor data to be used by the SleepDataset class"""
    
    #get path
    proj_path = Path(metadata.get('project_path', '.'))
    file_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}_X.pt' 
    save_folder = proj_path / "processed"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not chunked:
        save_path = save_folder / file_name
    else:
        chunk_folder = save_folder / f'windowed_{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}'    #if chunked, create or select a folder to save in
        if not os.path.exists(chunk_folder):
            os.makedirs(chunk_folder)
        save_path = chunk_folder / f'X_chunk{chunk_id}.pt'
    
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
    else:
        to_save = tensors[0]
        
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
            
    with open(save_path, 'wb') as f:
        torch.save(to_save, f) 

    #handle states       
    states_name = f'{metadata.get('scoring_started', 'noID')}{metadata.get('optional_tag', '')}{metadata.get('filename', '')}_y.pt'
    
    if not chunked:
        states_path = save_folder / states_name
    else:
        states_path = chunk_folder / f'y_chunk{chunk_id}.pt'

    with open(states_path, 'wb') as f:
        torch.save(states, f)

    del to_save, states
    
    
def save_windowed_for_testing(tensors: tuple, 
                              save_folder: str,
                              file_name: str,
                                states: torch.Tensor = None, 
                                win_len: int = 1000,
                                chunked: bool = True, 
                                chunk_id: int = None,
                                overwrite: bool = False):
    """chops and saves data for ML testing"""
    #get path
    file_name = f'{file_name}' 
    save_folder = Path(save_folder)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not chunked:
        save_path = save_folder / file_name
    else:
        chunk_folder = save_folder / f'windowed_{file_name}'    #if chunked, create or select a folder to save in
        if not os.path.exists(chunk_folder):
            os.makedirs(chunk_folder)
        save_path = chunk_folder / f'X_{file_name}_chunk{chunk_id}.pt'
    
    if not overwrite:
        if os.path.exists(save_path):
            print('file already exists, not saving')
            return None

    tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    if len(tensors) == 0:
        print('no tensors to save')
        return None
    #do the same as in save_tensor to align times (trim to shortest)
    min_len = min(t.size(1) for t in tensors)
    if len(set(t.size(1) for t in tensors)) > 1:
        print(f"length mismatch in save_windowed: {[t.size(1) for t in tensors]}, truncating to {min_len}")
    tensors = [tensor[:, : min_len] for tensor in tensors]
    
    #here it should still be channels x time
    if len(tensors) > 1:
        to_save = torch.cat(tensors, dim = 0)
    else:
        to_save = tensors[0]
        
    if states is None:
        states = torch.zeros(size = (1, to_save.size()[-1]), dtype=torch.long, device=to_save.device)
    else:
        if states.dim() != 2:       #ensure states shape 1 x time, required to use the same chop function
            states = states.unsqueeze(0)
    to_save, states = _chop_by_state(states, to_save, win_len)
    
    if to_save.size(1) == 0:
        print("Warning: to_save has 0 windows — skipping save.")
        return
            
    with open(save_path, 'wb') as f:
        torch.save(to_save, f) 

    #handle states       
    states_name = f'{file_name}_y.pt'
    
    if not chunked:
        states_path = save_folder / states_name
    else:
        states_path = chunk_folder / f'y_{file_name}_chunk{chunk_id}.pt'

    with open(states_path, 'wb') as f:
        torch.save(states, f)

    del to_save, states
    
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
    Input (same format as _chop):
      values: [channel, time]
      states: [time] or [state, time]

    Output (same format as _chop):
      values_out: [channel, win_num, window(1000)]
      states_out: [state, win_num, window(1000)]
    """
    #validate shapees
    if values.dim() != 2:
        raise ValueError(f"values must be [C, L], got {tuple(values.shape)}")
    C, L = values.shape

    if states.dim() == 1:
        states_ = states.unsqueeze(0)          # [1, L]
    elif states.dim() == 2:
        states_ = states                        # [S, L]
    else:
        raise ValueError(f"states must be [L] or [S, L], got {tuple(states.shape)}")

    S, Ls = states_.shape
    if Ls != L:
        raise ValueError(f"states length ({Ls}) must match values length ({L})")


    # use first state channel to detect changes (typical case S=1, only single state valuation)
    st0 = states_[0].flatten()                  # [time (L)]
    #detect state changes
    diffs = torch.diff(st0)
    change_idx = torch.nonzero(diffs != 0).flatten() + 1

    # add boundaries
    boundaries = torch.cat([
        torch.tensor([0], device=st0.device),
        change_idx.to(st0.device),
        torch.tensor([L], device=st0.device)
    ])

    X_parts = []
    S_parts = []
    
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start = int(start)
        end = int(end)

        seg_len = end - start
        if seg_len < win_len:
            continue

        i = start
        while i + win_len <= end:
            # values window: [C, W] to [C, 1, W]
            v_win = values[:, i:i + win_len].unsqueeze(1)
            X_parts.append(v_win)

            # states window: [S, W] to [S, 1, W]
            s_win = states_[:, i:i + win_len].unsqueeze(1)
            S_parts.append(s_win)

            i += win_len

    if len(X_parts) == 0:
        values_out = values.new_empty((C, 0, win_len))
        states_out = states_.new_empty((S, 0, win_len))
        return values_out, states_out

    values_out = torch.cat(X_parts, dim=1)     # [C, N, W]
    states_out = torch.cat(S_parts, dim=1)     # [S, N, W]
    return values_out, states_out

def __get_start_time(path, time_channel):
    """
    helper to return start of rec
    """
    if time_channel is not None:
        try:
            with open(path, 'rt') as f:
                f.readline() # skip
                firstline = f.readline() #2nd line is 1st of data
                start_time = firstline.split(',')[int(time_channel)]    #take start time
                print(f'selected recording start time: {start_time}')
        except:
            start_time = '2025-12-04 19:01:40.526977527'
            print(f'failed to load file, time set to default: {start_time}')
    else:
        start_time = '2025-12-04 19:01:40.526977527'
        print(f'no time column specified, time set to default: {start_time}')
    return start_time

def __seconds_since_midnight(t) -> float:
    """
    helper which converts time (t) to seconds since midnight
    """
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def __get_num_lines(start_time, times, sr):
    """
    helper to get number of lines to read // cut to time
    """
    seconds_in_day = 24 * 60 * 60

    try:
        recstart = __seconds_since_midnight(pd.to_datetime(start_time).time())
        timestart_dt = __seconds_since_midnight(pd.to_datetime(times[0]).time())
        timeend_dt = __seconds_since_midnight(pd.to_datetime(times[1]).time())
    except:
        print('failed to parse datetimes')
        return None, None    
    
    #handle durations
    line_start = 0
    line_end = None
    
    #midnight crossing is NOT supported here
    if timestart_dt > recstart:     #if requested time start after the start of recording, calculate diff in seconds, multiply, set line start
        diff = timestart_dt - recstart
        line_start = int(diff * sr)    
    else:
        print('warning: select a start time that is after recording start!')
        
    #now calculate end, supporting midnight crossing
    if timeend_dt < timestart_dt:#if end is before start, add a day
        timeend_dt += seconds_in_day
    diff = timeend_dt - timestart_dt
    line_end = int(diff * sr) + line_start  #duration + start    
    return line_start, line_end
  
def load_from_csv(path: str, metadata: dict = None, states: int = None, times = (None, None)):
    """
    loads data from csv into a torch tensor
    """
    time_channel = metadata.get('time_channel', None)
    ecog_channels = metadata.get('ecog_channels', None)
    emg_channels = metadata.get('emg_channels', None)
    sample_rate = int(metadata.get('sample_rate', 1000))
    device = metadata.get('device', 'cuda')
    #if cuda is not available
    device = "cpu" if not torch.cuda.is_available() else device
    
    #read one line to set start time            
    start_time = __get_start_time(path, time_channel)  
    metadata['rec_start'] = start_time
    
    #get line_start, line_end if start and end times are passed
    if (times[0] is not None) and (times[1] is not None):
        line_start, line_end = __get_num_lines(start_time, times, sr = sample_rate)
        #now both need to exist
        if isinstance(line_start, int) & isinstance(line_end, int):    
            #update metadata to reflect cut
            recstart_dt = pd.to_datetime(start_time)
            newstart_dt = pd.to_datetime(times[0])
            if newstart_dt.time() > recstart_dt.time(): #unless start is 0, then keep old
                metadata['rec_start'] = str(recstart_dt.replace(hour = newstart_dt.hour, minute= newstart_dt.minute, second= newstart_dt.second, microsecond= newstart_dt.microsecond).isoformat())
                print(f'updated rec_start: {metadata['rec_start']}')
            data = pd.read_csv(path, skiprows = line_start, nrows= (line_end - line_start))
        else:
            raise ValueError('pass two numbers when clipping time!')
    else:
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
            
            
def load_from_csv_in_chunks(path: str, metadata: dict = None, states: int = None, chunk_size: int = None, times = (None, None)):
    """
    loads data from csv into a torch tensor in chunks, returns a generator
    """
    time_channel = metadata.get('time_channel', None)
    print('time chopping not implemented')
    ecog_channels = metadata.get('ecog_channels', None)
    emg_channels = metadata.get('emg_channels', None)
    device = metadata.get('device', 'cuda')
    sample_rate = int(metadata.get('sample_rate', 1000))
    #if cuda is not available
    device = "cpu" if not torch.cuda.is_available() else device
   
    try:
        ecog_channels = [int(ch) for ch in ecog_channels.split(',')]
        emg_channels = [int(ch) for ch in emg_channels.split(',')]
    except Exception as e:
        print(f'Something went wrong when parsing metadata of channel numbers in load_from_csv: {e}')
        return None, None, None
    
    #read one line to set start time            
    start_time = __get_start_time(path, time_channel)  
    metadata['rec_start'] = start_time
    
    #get line_start, line_end if start and end times are passed
    if (times[0] is not None) and (times[1] is not None):
        line_start, line_end = __get_num_lines(start_time, times, sr = sample_rate)
        
        #now both need to exist
        if isinstance(line_start, int) & isinstance(line_end, int):    
            #update metadata to reflect cut
            recstart_dt = pd.to_datetime(start_time)
            newstart_dt = pd.to_datetime(times[0])
            if newstart_dt.time() > recstart_dt.time(): #unless start is 0, then keep old
                metadata['rec_start'] = str(recstart_dt.replace(hour = newstart_dt.hour, minute= newstart_dt.minute, second= newstart_dt.second, microsecond= newstart_dt.microsecond).isoformat())
                print(f'updated rec_start: {metadata['rec_start']}')
                
            #read data
            for chunk in pd.read_csv(path, chunksize = chunk_size, skiprows = line_start, nrows= (line_end - line_start)):
                ecog_chunk = torch.tensor(chunk.iloc[:, ecog_channels].values, device=device)
                emg_chunk = torch.tensor(chunk.iloc[:, emg_channels].values, device=device)
                states_chunk = torch.tensor(chunk.iloc[:, states].values, device=device) if states else None 
                yield ecog_chunk, emg_chunk, states_chunk
        else:    
            for chunk in pd.read_csv(path, chunksize = chunk_size):
                ecog_chunk = torch.tensor(chunk.iloc[:, ecog_channels].values, device=device)
                emg_chunk = torch.tensor(chunk.iloc[:, emg_channels].values, device=device)
                states_chunk = torch.tensor(chunk.iloc[:, states].values, device=device) if states else None 
                yield ecog_chunk, emg_chunk, states_chunk
    else:    
        for chunk in pd.read_csv(path, chunksize = chunk_size):
            ecog_chunk = torch.tensor(chunk.iloc[:, ecog_channels].values, device=device)
            emg_chunk = torch.tensor(chunk.iloc[:, emg_channels].values, device=device)
            states_chunk = torch.tensor(chunk.iloc[:, states].values, device=device) if states else None 
            yield ecog_chunk, emg_chunk, states_chunk
        
        
if __name__ == "__main__":
    #testing
    for i, _ in enumerate(load_from_csv_in_chunks(path = r"C:\Users\marty\Projects\scorer\proj_data\raw\20251204-1_g0_t0.obx0.obx_box3.csv",
                            metadata = {'time_channel':'3', 'ecog_channels': '0,1','emg_channels':'2', 'device': 'cuda'},
                            chunk_size= 1000000,
                            times = ('19:30:00', '21:00:00'))):
        print(f'chunk {i}')