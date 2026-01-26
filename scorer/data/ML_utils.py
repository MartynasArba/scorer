#implements preprocessing + chop_by_state without GUI to generate a dataset for training/testing models

from scorer.data.preprocessing import bandpass_filter, sum_power, band_powers
from scorer.data.storage import load_from_csv_in_chunks, save_windowed_for_testing, load_from_csv
from pathlib import Path

import torch

from pyedflib import highlevel

def run_default_preprocessing(csv_path: str) -> None:
        """
        runs default preprocessing from raw to X, y
        filters ecog 0.5-49 Hz, emg 10-100 Hz, calculates sum powers and band powers
        saves windowed data for use with SleepTraining dataset
        """
        if csv_path is not None:
            
            by_state = True
            file_name = Path(csv_path).stem
            save_folder = r'G:\oslo_data'
            chunk_size = 1000000
            states = 3
            times = (None, None)
            win_len = 1000
            metadata = {'time_channel': '0',
                        'ecog_channels':'1',
                        'emg_channels':'2',
                        'device':'cuda',
                        'sample_rate':'250'
                        }
            if win_len % chunk_size != 0:
                print('warning: window length doesn\'t fit chunk size!')
            
            #run loading, preprocessing, etc
            for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(load_from_csv_in_chunks(csv_path, metadata = metadata, states = states, chunk_size = chunk_size, times = times)):
                tensor_seq = _preprocess(ecog_chunk, emg_chunk, metadata)     
                #change this to save chopped
                if by_state:
                    save_windowed_for_testing(tensors = tensor_seq, 
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

def _preprocess(ecog: torch.Tensor, emg: torch.Tensor, metadata: dict) -> tuple:
        """
        leftover from GUI, runs preprocessing helpers
        """
        ecog, emg = ecog.T, emg.T       #torch usually requires channels x time
        
        freqs = (0.5, 49.0)
        ecog = _bandpass(ecog, freqs, metadata)
        print('ecog data filtered')
                
        freqs = (10.0, 100.0)
        emg = _bandpass(emg, freqs, metadata)
        print('emg data filtered')
    
        ecog_power = sum_power(ecog, smoothing = 0.2, sr = int(metadata.get('sample_rate', 1000)), device = metadata.get('device'), normalize = True)
        emg_power = sum_power(emg, smoothing = 0.2, sr = int(metadata.get('sample_rate', 1000)), device = metadata.get('device'), normalize = True)
        print('sum pows calculated')
    
        bands = band_powers(signal = ecog, bands = {'delta': (0.5, 4),
                                                    'theta': (5, 9),
                                                    'alpha': (8, 13),
                                                    'sigma': (12, 15)}, 
                            sr = int(metadata.get('sample_rate', 1000)), 
                            device= metadata.get('device'), smoothen = 0.2)

        print('band pows calculated')
        print('preprocessing done')
        return (ecog, emg, ecog_power, emg_power) + tuple(bands.values())
    
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
        
if __name__ == "__main__":
    print('everything is commented out, edit script to do something')
    # import torch
    
    # raw_to_edf(r'g:\sleep-ecog-DOWNSAMPLED\20251124-1_g0_t0.obx0.obx_box1.csv')
    
# converts whole folder to windows for ml
    # import glob
    # import tqdm
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
