#implements preprocessing + chop_by_state without GUI to generate a dataset for training/testing models

from preprocessing import bandpass_filter, sum_power, band_powers
from storage import load_from_csv_in_chunks, save_windowed_for_testing
from pathlib import Path

def run_default_preprocessing(csv_path) -> None:
        """
        runs default preprocessing from raw to X, y
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
                    # save_windowed(tensors = tensor_seq, 
                    #             states = states_chunk, 
                    #             metadata = self.params, 
                    #             win_len = win_len,
                    #             chunked = True, #append_file previously
                    #             chunk_id = i,
                    #             overwrite = False,
                    #             testing = False)
        
def _bandpass(signal, freqs, metadata):
        """
        applies bandpass filtering func
        """
        print(f'in bandpass filter signal size: {signal.size()}')
        signal = bandpass_filter(signal,
                        sr = int(metadata.get('sample_rate', 1000)),
                        freqs = freqs,
                        device = metadata.get('device'))
        return signal        

def _preprocess(ecog, emg, metadata):
        """
        check what's checked, run corresponding funcs
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
    
if __name__ == "__main__":
    import torch
#converts whole folder to windows for ml
    # import glob
    # import tqdm
    # paths = glob.glob(r'G:\oslo_data\*.csv')
    # for path in tqdm.tqdm(paths):
    #     print(path)
    #     run_default_preprocessing(path)
#let's try checking whether states were actually saved
#open one file, check shapes, check unique values in states
    val_path = r"G:\oslo_data\windowed_trial_1_mouse_b1aqm2\X_trial_1_mouse_b1aqm2_chunk0.pt"
    score_path = r"G:\oslo_data\windowed_trial_1_mouse_b1aqm2\y_trial_1_mouse_b1aqm2_chunk0.pt"
    vals = torch.load(val_path)
    scores = torch.load(score_path)
    print(vals.size())
    print(scores.size())
    print(scores.unique(return_counts = True))
