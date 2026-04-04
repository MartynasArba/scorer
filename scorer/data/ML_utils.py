#implements preprocessing + chop_by_state without GUI to generate a dataset for training/testing models

from scorer.data.preprocessing import bandpass_filter, sum_power, band_powers
from scorer.data.storage import load_from_csv_in_chunks, save_windowed_for_testing, load_from_csv
from torchaudio.functional import resample
import torch.nn.functional as F

from pathlib import Path
import pickle
import numpy as np
import re
import shutil
import torch
from pyedflib import highlevel
from tqdm import tqdm
import glob

def move_into_subfolder(csv_path: str) -> None:
    """moves loose files into folders"""
    file_path = Path(csv_path)
    file_name = file_path.stem.replace('.','')
    Path.mkdir(file_path.parent / file_name)
    new_path = file_path.parent / file_name / file_path.name
    shutil.move(file_path, new_path)
    print(f'moved into subdir: {file_path.parent / file_name}')

def run_default_preprocessing(csv_path: str, save_folder =  r'G:\oslo_data') -> None:
        """
        runs default preprocessing from raw to X, y
        filters ecog 0.5-49 Hz, emg 10-100 Hz
        saves windowed data for use with SleepTraining dataset
        """
        if csv_path is not None:
            
            by_state = True
            file_name = Path(csv_path).stem
            save_folder = save_folder
            chunk_size = 1000000
            states = 4
            times = (None, None)
            win_len = 1000
            metadata = {'time_channel': '1',
                        'ecog_channels':'2',
                        'emg_channels':'3',
                        'device':'cuda',
                        'sample_rate':256
                        }
            if win_len % chunk_size != 0:
                print('warning: window length doesn\'t fit chunk size!')
            
            #run loading, preprocessing, etc
            for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(load_from_csv_in_chunks(csv_path, metadata = metadata, states = states, chunk_size = chunk_size, times = times)):
                ecog_chunk, emg_chunk, states_chunk = _preprocess(ecog_chunk, emg_chunk, states_chunk, metadata) 
                #change this to save chopped
                if by_state:
                    save_windowed_for_testing(tensors = (ecog_chunk, emg_chunk), 
                                              save_folder = save_folder,
                                              file_name = file_name,
                                              states = states_chunk, 
                                              win_len = win_len,
                                              chunked = True, 
                                              chunk_id = i,
                                              overwrite = True)
                else:           
                    print('not implemented')      
        
def _bandpass(signal: torch.Tensor, freqs: tuple, metadata: dict) -> torch.Tensor:
        """
        applies bandpass filtering func
        """
        print(f'in bandpass filter signal size: {signal.size()}')
        signal = bandpass_filter(signal,
                        sr = int(metadata.get('sample_rate', 1000)),
                        freqs = freqs,
                        device = metadata.get('device'))
        return signal        

def _preprocess(ecog: torch.Tensor, emg: torch.Tensor, states: torch.Tensor, metadata: dict) -> tuple:
        """
        leftover from GUI, runs preprocessing helpers
        """
        sample_rate = float(metadata.get('sample_rate', 256))
        new_rate = 250
        
        ecog = ecog.T
        emg = emg.T
        if states is not None:
            states = states.T 
        
        if sample_rate == new_rate:
            print('not resampling: old sr = new sr')

        else:
            ecog = resample(ecog.contiguous(), sample_rate, new_rate)
            emg = resample(emg.contiguous(), sample_rate, new_rate)
            
            if states is not None:
                original_len = states.shape[-1]
                target_len = ecog.shape[-1]
                ratio = original_len / target_len
                indices = (torch.arange(target_len, device=states.device) * ratio).long()
                states = states[:, indices]
        
        freqs = (0.5, 49.0)
        ecog = _bandpass(ecog, freqs, metadata)
        print('ecog data filtered')
                
        freqs = (10.0, 100.0)
        emg = _bandpass(emg, freqs, metadata)
        print('emg data filtered')

        return ecog, emg, states#, ecog_power, emg_power) + tuple(bands.values()
    
def raw_to_edf(csv_path: str) -> None:
    """ 
    goes from raw csv with default preprocessing to edf, first written to test intelliscorer 
    """
    if csv_path is not None:
        out_path = Path(csv_path).with_suffix('.edf')
        
        metadata = {'time_channel': '3',
                    'ecog_channels':'0',
                    'emg_channels':'2',
                    'device':'cuda',
                    'sample_rate':'1000'
                    }
        ecog, emg, _ = load_from_csv(csv_path, metadata = metadata)
        tensor_seq = _preprocess(ecog, emg, metadata)
        ecog, emg = tensor_seq[0].detach().cpu().numpy(), tensor_seq[1].detach().cpu().numpy()
        
        names = ['EEG', 'EMG']
        signal_headers = highlevel.make_signal_headers(names, sample_frequency= int(metadata.get('sample_rate', 1000)))
        header = highlevel.make_header()
        highlevel.write_edf(str(out_path), [ecog, emg], signal_headers, header)
        
def extract_chunk_number(filepath: Path) -> int:
    """extracts the int from filename for numerical sorting"""
    # find all sequences of digits in the filename
    numbers = re.findall(r'\d+', filepath.name)
    if numbers:
        # return last number if more are present
        return int(numbers[-1])
    return 0  # Fallback if no number is found
        
def states_to_yfile(states_pkl_path: str, data_path: str, win_len: int = 1000):
    """converts state files from GUI back to y.pt files for training"""
    print(f"loading states from {states_pkl_path}")
    with open(states_pkl_path, 'rb') as f:
        states = pickle.load(f)
    states = np.array(states)
    print(f"loaded {len(states)} windows")
    print(np.unique(states, return_counts = True))
    
    dataset_dir = Path(data_path)
    
    # find all X files
    raw_x_files = list(dataset_dir.glob("X*.pt"))
    x_files = sorted(raw_x_files, key = extract_chunk_number)
    
    if not x_files:
        print(f"No X files found in {dataset_dir}")
        return
        
    current_idx = 0
    total_samples = 0
    
    # iter through chunks, map states back to their respective files
    for x_path in x_files:
        # print(f'X: {x_path}')
        # Load X to determine the chunk's exact size
        X = torch.load(x_path, weights_only=False)
        print(f'X size: {X.size()}')
        
        # _chop outputs shape [n_channels, n_samples, win_len]
        n_samples = X.shape[1] 
        total_samples += n_samples        
        if current_idx + n_samples > len(states):
            print(f"Error: Ran out of GUI states! Chunk {x_path.name} needs {n_samples} states, "
                  f"but only {len(states) - current_idx} remain.")
            break
            
        # Extract the exact number of states belonging to this specific chunk
        chunk_states = states[current_idx : current_idx + n_samples]
        current_idx += n_samples
        
        # reconstruct the training tensor format: [1, n_samples, win_len]
        y_tensor = torch.tensor(chunk_states, dtype=torch.long)
        
        # Expand a single label across the entire time window (e.g. 1000 points)
        y_tensor = y_tensor.unsqueeze(1).expand(n_samples, win_len)
        
        # Add the channel dimension back
        y_tensor = y_tensor.unsqueeze(0)
        
        # Save the y file (replacing 'X' with 'y' in the filename)
        y_filename = x_path.name.replace('X', 'y')
        y_path = x_path.with_name(y_filename)
        
        torch.save(y_tensor, y_path)
        print(f"Saved {y_filename} | Shape: {list(y_tensor.shape)}")

        print(f'found{total_samples} samples for {len(states)} states')        
    # Validation
    if current_idx < len(states):
        print(f"\nWarning: {len(states) - current_idx} states were leftover. "
              f"The GUI array had more labels than the X_*.pt files combined.")
    elif current_idx == len(states):
        print("\nSuccess: All states perfectly mapped and saved!")
    

def remap_and_fix_state_files(data_dir, states_path, win_len = 1000, new_states_name = "_corrected"):
    
    data_dir = Path(data_dir)
    
    # load original states from GUI
    with open(states_path, 'rb') as f:
        original_states = np.array(pickle.load(f))
        
    print(f"loaded scrambled state labels: {original_states.shape} total epochs")

    # find all X files
    raw_x_files = list(data_dir.glob("X_chunk*.pt"))
    if not raw_x_files:
        print("No X files found!")
        return
    # sort alphabetically to match old incorrect sorting
    x_files_alphabetical = sorted(raw_x_files, key=lambda p: p.name)

    current_idx = 0
    
    # dict to hold corrected slices keyed by chronological number
    chronological_slices = {} 
    
    # slice states array and save individual y_chunk files
    print("extracting and saving individual state chunks")
    for x_path in x_files_alphabetical:
        x_tensor = torch.load(x_path, map_location='cpu')
        if x_tensor.ndim == 3:
            n_samples = x_tensor.shape[1]
        else:
            raise RuntimeError('X tensor must have 3 dims!')
        if current_idx + n_samples > len(original_states):
            print(f"ran out of GUI states: chunk {x_path.name} needs {n_samples} states, but only {len(original_states) - current_idx} left")
            break
        
        # slice labels to match chunks
        end_idx = current_idx + n_samples
        y_chunk = original_states[current_idx:end_idx]
        # reconstruct training tensor format: [1, n_samples, win_len]
        y_tensor = torch.tensor(y_chunk, dtype=torch.long)
        y_tensor = y_tensor.unsqueeze(1).expand(n_samples, win_len)
        y_tensor = y_tensor.unsqueeze(0)
        # save individual y_chunk*.pt
        y_name = x_path.name.replace('X', 'y', 1)
        torch.save(y_tensor, data_dir / y_name)
        # extract the true chunk integer using regex so it can be used for numerical sort
        chunk_num = int(re.search(r'\d+', x_path.name).group())
        # store the slice in the dictionary under its true chronological index
        chronological_slices[chunk_num] = y_chunk
        current_idx = end_idx
        
    if current_idx < len(original_states):
        print(f" {len(original_states) - current_idx} states were leftover,  GUI array had more labels than X_*.pt files")
    elif current_idx == len(original_states):
        print("all individual states mapped and saved!")
        
    print(f"extracted {current_idx} epochs into individual y_chunk files.")
    print("rebuilding states file in correct numerical order")
    # sort dict keys numerically
    sorted_chunk_nums = sorted(chronological_slices.keys())
    # take slices in correct order
    ordered_slices = [chronological_slices[num] for num in sorted_chunk_nums]
    
    # cat all states into new correct file
    corrected_states = np.concatenate(ordered_slices)
    corrected_pkl_path = data_dir / (str(Path(states_path).name)[:-4] + str(new_states_name) + '.pkl')
    
    with open(corrected_pkl_path, 'wb') as f:
        pickle.dump(corrected_states.tolist(), f)
        
    print(f"saved corrected states to {corrected_pkl_path.name}")
    
def edf_to_csv(edf_path, hypnogram_path, channels = [2, 3]):
    """self explanatory, used to convert open source .edf files to compatible .csv"""
    import pandas as pd
    
    signals, signal_headers, header = highlevel.read_edf(edf_path)
    sample_rate = signal_headers[0].get('sample_frequency')
    if sample_rate is not None:
        sample_rate = float(sample_rate)
        print(sample_rate)
    else:
        raise ValueError('sample rate not found in EDF header!')
    
    ch = channels[0]
    num_samples = len(signals[ch])
            
    time = np.arange(0, num_samples) / sample_rate
    ecog = np.array(signals[ch])
    emg = np.array(signals[-1])
    scores = np.zeros_like(ecog)
    #parse state scores
    if hypnogram_path is not None:
        with open(hypnogram_path, 'rt') as f:
            current_start_idx = 0
            for i, line in enumerate(f.readlines()):
                try:
                    state, end_time = line.split('\t')
                    end_idx = int(float(end_time) * sample_rate)
                    if end_idx > num_samples:
                        end_idx = num_samples
                    
                    if 'awake' in state:
                        scores[current_start_idx:end_idx] = 1
                    elif 'non-REM' in state:
                        scores[current_start_idx:end_idx] = 2
                    elif 'REM' in state:
                        scores[current_start_idx:end_idx] = 4
                    else:
                        scores[current_start_idx:end_idx] = 0
                    current_start_idx = end_idx
                    if current_start_idx >= num_samples:
                        break
    
                except:
                    print(f'line {i} failed')
        print('hypnogram parsed! found states:')
        print(np.unique(scores, return_counts= True))
        
    ecog_chs = channels
        
    for ch in ecog_chs:
        fname = Path(edf_path)
        save_name = fname.parent / f'{fname.stem}_ch{ch}.csv'
        print(f'saving csv to: {save_name}')
        pd.DataFrame({
            'time':time,
            'ecog':signals[ch],
            'emg':emg,
            'sleep_episode':scores
                        }).to_csv(save_name)

if __name__ == "__main__":
    # print('nothing uncommented!')
    edf_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\test\test\recordings\*.edf"))       
    hypnogram_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\test\test\annotations\*_consensus_state_annotation.hyp"))
    
    for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
        print(edf_path, hyp_path)
        edf_to_csv(edf_path = edf_path,
                hypnogram_path = hyp_path, 
                channels = [2, 3])
    
    edf_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\sleep_deprivation\sleep_deprivation\recordings\*.edf"))       
    hypnogram_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\sleep_deprivation\sleep_deprivation\annotations\*_CBD.hyp"))
    
    for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
        print(edf_path, hyp_path)
        edf_to_csv(edf_path = edf_path,
                hypnogram_path = hyp_path, 
                channels = [16, 17])
        
    edf_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\optogenetic_stimulation\optogenetic_stimulation\recordings\*.edf"))
    hypnogram_paths = sorted(glob.glob(r"C:\Users\marty\Desktop\train_sets\unsorted\Oxford\optogenetic_stimulation\optogenetic_stimulation\annotations\*_TY.hyp"))
    for edf_path, hyp_path in zip(edf_paths, hypnogram_paths):
        print(edf_path, hyp_path)
        edf_to_csv(edf_path = edf_path,
                hypnogram_path = hyp_path, 
                channels = [0, 1])
    #also convert Oslo data
    #     # converts whole folder to windows for ml
    # paths = glob.glob(r'C:\Users\marty\Desktop\train_sets\unsorted\to_convert\*.csv')
    # for i, path in enumerate(tqdm(paths)):
    #     print(i, path)
    #     run_default_preprocessing(path, save_folder = r'C:\Users\marty\Desktop\train_sets\labeled')
    
    

        

        

    #this is most relevant when converting my data to scorer format
    # states_pkl_path = r"G:\for_training\windowed_2026032514575020251207-1_g0_t0.obx0.obx_box3\noID_scores_windowed_2026032514575020251207-1_g0_t0.ob____0_frame10799.pkl"
    # data_path = r"G:\for_training\windowed_2026032514575020251207-1_g0_t0.obx0.obx_box3"
    # win_len = 1000
    # states_to_yfile(states_pkl_path, data_path, win_len)
    
    #get all subfolders as data dirs
    # subfolders = glob.glob(r'C:\Users\marty\Desktop\SCORING202602\for_training\*')
    # for sub in subfolders:
    #     print(sub)
    #     states_files = glob.glob(sub + '/*.pkl')
    #     for states in states_files:
    #         if 'corrected' not in states:
    #             remap_and_fix_state_files(sub, states)
        # print(states_files)
    # data_folder = r"C:\Users\marty\Desktop\SCORING202602\for_training\windowed_20260224115335 20260107-1_g0_t0.obx0.obx_box1"
    # states_file = r"C:\Users\marty\Desktop\SCORING202602\for_training\windowed_20260224115335 20260107-1_g0_t0.obx0.obx_box1\noID_scores_windowed_20260312140926 20260107-1_g0_t0.ob____0_frame10799_corrected.pkl"
    
    # remap_and_fix_state_files(data_folder, states_file)
    # paths = glob.glob('G:/sleep-ecog-DOWNSAMPLED/*.csv')
    # for path in tqdm(paths):
    #     move_into_subfolder(path)
    
    # data_paths = glob.glob(r'C:\Users\marty\Desktop\SCORING202602\for_training\*')
    # for path in data_paths:
    #     score_paths = glob.glob(path + '/*scores*.pkl')
    #     if len(score_paths) > 1:
    #         print(f'something is wrong, multiple score files in folder: {path}')

    #     win_len = 1000
        
    #     # states_to_yfile(score_paths[0], path, win_len)
    #     print(path, '\n',  score_paths[0])
    
    
    # print('everything is commented out, edit script to do something')
    # import torch
    
    # raw_to_edf(r'g:\sleep-ecog-DOWNSAMPLED\20251124-1_g0_t0.obx0.obx_box1.csv')
    
# converts whole folder to windows for ml
    # paths = glob.glob(r'G:\oslo_data\*.csv')
    # for i, path in enumerate(tqdm.tqdm(paths)):
    #     print(i, path)
    #     run_default_preprocessing(path)
# let's try checking whether states were actually saved
#open one file, check shapes, check unique values in states
    # val_path = r"G:\oslo_data\windowed_trial_1_mouse_b1aqm2\X_trial_1_mouse_b1aqm2_chunk0.pt"
    # score_path = r"G:\oslo_data\windowed_trial_1_mouse_b1aqm2\y_trial_1_mouse_b1aqm2_chunk0.pt"
    # vals = torch.load(val_path)
    # scores = torch.load(score_path)
    # print(vals.size())
    # print(scores.size())
    # print(scores.unique(return_counts = True))
