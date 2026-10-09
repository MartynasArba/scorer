
from scorer.data.preprocessing import bandpass_filter
from torchaudio.functional import resample
from pathlib import Path
import numpy as np
import torch
from pyedflib import highlevel
from tqdm import tqdm
import glob

# script contains code to read, preprocess and convert data from external datasets into 4-s torch tensors for training
#if run, does all conversions

def load_mlsnet_npz(path, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    loads data from mls-net publication, 
    https://doi.org/10.3390/bios14080406
    Params:
    - path, str: path to dataset file, which is .npz for some reason
    - sr is the new resampling rate
    - device, str: device for cuda
    Returns:
    - x_processed, torch.Tensor: ECoG/EEG signals, SHAPE, CODING
    - y_processed, torch.Tensor: label tensor, SHAPE, CODING
    """
    
    device = kwargs.get('device', 'cuda')
    new_sr = kwargs.get('new_sr', 128.)
    
    try:
        obj = np.load(path)
    except:
        print('failed to load object in load_mlsnet_npz')
        return None
    
    #handle x according to dataset "description"
    #ecog is 0-2000 of the rec, kill me 
    x = obj['x'][:, :, 0:2000]     
    #get raw tensor
    ecog_tensor = torch.tensor(x, dtype=torch.float32, device=device).view(7, -1)
    print(f"resamping to {new_sr} and preprocessing")
    ecog_tensor = resample(ecog_tensor, 500, new_sr)    #initial sr is 500
    ecog_tensor = bandpass_filter(ecog_tensor, sr=new_sr, freqs=(0.5, 49.0), device=device)
    #chop back up
    x_processed = ecog_tensor.reshape(-1, 1, int(new_sr*4))  #should dynamically calculate window size, was 1000 points/4 seconds, now 512
    
    print(f'after all this x size: {x_processed.size()}')    
    
    #convert y to tensor
    y_tensor = torch.tensor(obj['y'].reshape(-1), dtype=torch.long, device=device)
    #changes state mapping from initial insanity (0 is NREM, 1 is REM, 2 is W) to my brand of braindead (W - 1, NREM - 2, REM - 4)
    y_tensor[y_tensor == 1] = 4
    y_tensor[y_tensor == 2] = 1
    y_tensor[y_tensor == 0] = 2

    return x_processed, y_tensor

def parse_channel_tsv(path) -> dict:
    """parses tsv file into dict, made for loading mssv recs, returns dict with 'name', 'type', 'units' and lists"""
    with open(path) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        channel_info = {key:[] for key in lines[0]}
        for line in lines[1:]:
            for i in range(len(channel_info.keys())):
                channel_info[lines[0][i]].append(line[i])  
    return channel_info

def parse_state_tsv(path):
    """parses sleep state tsv, returns numpy array, ignores onset and window size (because the dataset is already formatted as 4s, and onset starts from 0s)
    last state is assumed to be the same even if it's shorter"""
    
    states = np.loadtxt(path, skiprows = 1, delimiter = '\t')
    assert (np.sum(states[1] != 4) < 2) and (len(states[np.nonzero(states[1] != 4)]) < 4)   #assert only last duration is different and it's not more than expected
    assert (states[0, 0] == 0)  #assert onset is at 0
    
    #if asserts pass, we can safely return stage only
    return states[:, -1].copy()    

def parse_edf(edf_path, overwrite_channels = []):
    """ used to convert one .edf file to X-tensor, emg is ignored"""
    signals, signal_headers, header = highlevel.read_edf(str(edf_path))
    channels = [(i, sig['label']) for i, sig in enumerate(signal_headers)]
    eeg_chs = [ch for ch in channels if ("EEG" in ch[1] or "eeg" in ch[1])]
    if len(overwrite_channels) != 0:    #if channels are supplied, overwrite
        eeg_chs =  [ch for ch in channels if any(overwrite in ch[1] for overwrite in overwrite_channels)]
    
    sample_rate = signal_headers[0].get('sample_frequency') # sample rate should be the same as stated in the metadata, but will be more accurate from here
    if sample_rate is not None:
        sample_rate = float(sample_rate)
    else:
        raise ValueError('sample rate not found in EDF header!')
    
    return_tensors = []

    for eeg_index, eeg_name in eeg_chs:
        return_tensors.append(torch.Tensor(signals[eeg_index]))

    return torch.stack(return_tensors, axis = 1), sample_rate
        

def parse_mssv_paths(dir_path):
    """parses paths, returns a nice dict, made for mssv structure"""
    filepaths = glob.glob(f'{dir_path}/*')
    paths = {
        'eeg_paths': [],
        'state_paths': [],
        'channel_paths': [],
        'unknown_paths': []
        }
    for path in filepaths:
        if '.edf' in path:
            paths['eeg_paths'].append(path)
        elif ('.tsv' in path) and ('channels' in path):
            paths['channel_paths'].append(path)
        elif ('.tsv' in path) and ('events' in path):
            paths['state_paths'].append(path)
        else:
            paths['unknown_paths'].append(path)
    return paths

def load_mssv_rec(dir_path, **kwargs):
    """
    loads mssv dataset dwonloaded from https://openneuro.org/datasets/ds006366/versions/1.0.1/download#
    
    Args:
        path (Path-like): path to dataset folder
        
    Returns:
        None, but saves X in folder
    """
    device = kwargs.get('device', 'cuda')
    new_sr = kwargs.get('new_sr', 128.)
    
    #can contain multiple runs
    paths = parse_mssv_paths(dir_path)    
    
    assert len(paths['channel_paths']) == 1 #should be one "config"
    if len(paths['unknown_paths']) != 0:
        print(f'found {len(paths['unknown_paths'])} unknown files!')
             
    channel_info = parse_channel_tsv(paths['channel_paths'][0])
    #probably don't even need channel info tbh
    
    #then do for rec in recs
    for run in range(len(paths['state_paths'])):
        states = parse_state_tsv(paths['state_paths'][run])
        #reset states
        states[states == 4] = 0 #reset artifacts
        states[states == 3] = 4 #reset edf
        y = torch.tensor(states, dtype = torch.long, device = device)
        
        #read single edf
        tensor, sample_rate = parse_edf(paths['eeg_paths'][run])
        tensor = resample(tensor.transpose(1, 0).to(device = device), sample_rate, new_sr)
        tensor = bandpass_filter(tensor, sr=new_sr, freqs=(0.5, 49.0), device=device).transpose(-1, 0) 
        #bandpass filter accepts channel, time so transpose, do, transpose
        #now can chop and save one by one
        for ch in range(tensor.size(1)):
            X = tensor[:, ch].reshape(-1, 1, int(new_sr*4)) 
            torch.save(X, dir_path / f'X_ch{ch}_run{run}.pt')
            torch.save(y, dir_path / f'y_ch{ch}_run{run}.pt')
            
def hyp_to_tensor(hypnogram_path, **kwargs):
    """creates state tensor from .hyp file"""
    device = kwargs.get('device', 'cuda')
    duration = 0
    #extract duration
    with open(hypnogram_path) as f:
        line = f.readline()        
        row = line.split('\t')
        if len(row) != 2:
            row = [val for val in line.split(' ') if val != '']
        if 'Duration_sec' in row[0]:
            duration = float(row[1])   
            
    assert duration != 0
    assert isinstance(duration, float)
    
    #init array
    time = np.linspace(0, duration, int(duration/4))
    #read and do
    annotations = np.loadtxt(hypnogram_path, skiprows = 2, usecols= 0, dtype= str, delimiter= '\t')
    timestamps =  np.loadtxt(hypnogram_path, skiprows = 2, usecols= 1, dtype= float, delimiter= '\t')
    #state changes are where time == timestamp, searchsorted gives indices in time where timestamps should be
    indices = np.searchsorted(timestamps, time, side = 'right') - 1
    states_str = annotations[indices]   #fill indices with annotations
    states = np.zeros_like(states_str)
    for state in np.unique(states_str):
        if 'awake' in state.lower():
            states[states_str == state] = 1
        elif (('non' in state.lower()) or ('nrem' in state.lower())) or ('sleep' in state.lower()):
            states[states_str == state] = 2
        elif 'rem' in state.lower():
            states[states_str == state] = 4
        else:
            states[states_str == state] = 0
           
    return torch.tensor(states.astype(int), dtype = torch.long, device = device)

            
def load_oxford_rec(edf_path, hypnogram_path, **kwargs):
    """used to convert Oxford open access .edf files to tensors
    data available at https://doi.org/10.5281/zenodo.10200482"""
    
    device = kwargs.get('device', 'cuda')
    new_sr = kwargs.get('new_sr', 128.)
    overwrite_channels = kwargs.get('channels')
    
    #handle hypnograms
    y = hyp_to_tensor(hypnogram_path, device = device)
    
    #handle data
    tensor, sample_rate = parse_edf(edf_path, overwrite_channels = overwrite_channels )
    print(tensor.size(), sample_rate)
    tensor = resample(tensor.transpose(1, 0).to(device = device), sample_rate, new_sr)
    print(tensor.size())
    tensor = bandpass_filter(tensor, sr=new_sr, freqs=(0.5, 49.0), device=device).transpose(-1, 0) 
    for ch in range(tensor.size(1)):
        X = tensor[:, ch].reshape(-1, 1, int(new_sr*4)) 
        torch.save(X, edf_path.parent / f'X_ch{ch}.pt')
        torch.save(y, edf_path.parent / f'y_ch{ch}.pt')
    print(X.size())
    print(y.size())
    
    
def scan_and_move(initial_dir, dest_dir = '.'):
    import os
    #find all .pt files
    files = [Path(f) for f in glob.glob('*.pt', root_dir = initial_dir, recursive = True)]
    
    # for file in files:
    #     file = Path(file)
        
    print(len(files))
    print(files[:5])
    #get IDs
    #move to destination/ID/file.pt
    # pass
    #create dest_dir if it doesn't exist
    
    
if __name__ == "__main__":
    #runs all conversion funcs
    # new_sr = 128
    # device = 'cuda'

    
    # print('converting mls-net data...')
    # #converts mlsnet data
    # mlsnet_path = Path(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\labeled-val-mlsnet\sleepnet_fea10_xy.npz")
    # x_tensor, y_tensor = load_mlsnet_npz(mlsnet_path, new_sr = new_sr, device = device)
    # assert x_tensor.size()[0] == y_tensor.size()[0]
    # #save to file
    # torch.save(x_tensor, mlsnet_path.parent / 'converted' / 'X.pt')
    # torch.save(y_tensor, mlsnet_path.parent / 'converted' / 'y.pt')


    # print('convertign mssv data...')
    # paths = glob.glob(r'G:\RAW_EXTERNAL_SLEEP_DATASETS\mssv_128hz\sub-*\eeg')
    # for path in tqdm(paths):
    #     mssv_rec_path = Path(path)
    #     load_mssv_rec(mssv_rec_path, device = 'cuda', new_sr = 128.)

    # print('converting oxford data...')
    # converted_recs = 0

    # #in opto experiment: scorer TY
    # edf_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\optogenetic_stimulation\optogenetic_stimulation\recordings\*.edf"))
    # hypnogram_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\optogenetic_stimulation\optogenetic_stimulation\annotations\*_TY.hyp"))
    # for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
    #     load_oxford_rec(Path(edf_path), Path(hyp_path), device = device, new_sr = new_sr, channels = ['Signal 0', 'Signal 1'])
    #     converted_recs += 1


    # #in pilot experiment: scorer consensus
    # edf_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\pilot\pilot\recordings\*.edf"))
    # hypnogram_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\pilot\pilot\annotations\*_consensus.hyp"))
    # for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
    #     load_oxford_rec(Path(edf_path), Path(hyp_path), device = device, new_sr = new_sr, channels = ['Signal 0', 'Signal 1'])
    #     converted_recs += 1

    # #in test experiment: scorer consensus
    # edf_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\test\test\recordings\*.edf"))
    # hypnogram_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\test\test\annotations\*_consensus_state_annotation.hyp"))
    # for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
    #     load_oxford_rec(Path(edf_path), Path(hyp_path), device = device, new_sr = new_sr, channels = ['Signal 2', 'Signal 3'])
    #     converted_recs += 1

    # #in opto experiment: scorer CBD
    # edf_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\sleep_deprivation\sleep_deprivation\recordings\*.edf"))
    # hypnogram_paths = sorted(glob.glob(r"G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford\sleep_deprivation\sleep_deprivation\annotations\*_CBD.hyp"))
    # for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
    #     load_oxford_rec(Path(edf_path), Path(hyp_path), device = device, new_sr = new_sr, channels = ['Signal 16', 'Signal 17'])
    #     converted_recs += 1

    # print(f'done! converted {converted_recs} recs')
    scan_and_move(initial_dir = r'G:\RAW_EXTERNAL_SLEEP_DATASETS\Oxford')

    #should get size datapoints x channels x 512 (4s @ 128Hz)