import numpy as np
import pandas as pd

def label_microawakenings(states: np.ndarray, 
                          w_label: int = 1, 
                          nrem_label: int = 2, 
                          max_windows: int = 3, 
                          ma_label: int = 5
                          ) -> np.ndarray:
    """ 
    label MA instead of W if
      - length <= max_windows
      - surounded by NREM
    """        
    states = np.asarray(states)
    n = states.size
    if n == 0:
        print('no states passed to label MA')
        return states.copy()

    out = states.copy() #don't overwrite original

    state_starts = np.empty(n, dtype=bool)
    state_starts[0] = True    
    state_starts[1:] = out[1:] != out[:-1]  #shift by 1 and detect whether they match - basically np.diff

    start_idx = np.flatnonzero(state_starts)          # state start indices
    state_labels = out[start_idx]                 # state for each run
    state_lengths = np.diff(np.append(start_idx, n))   # lengths in windows

    num_states = state_labels.size
    is_ma = np.zeros(num_states, dtype = bool)  #bool array to mark whether some state is ma
    
    #if there are enough states to calculate
    if num_states >= 3:
        state_middle = np.arange(1, num_states - 1) #select safe state indices to not go out of bounds when comparing
        is_ma[state_middle] = (
            (state_labels[state_middle] == w_label)
            & (state_lengths[state_middle] <= max_windows)
            & (state_labels[state_middle - 1] == nrem_label)
            & (state_labels[state_middle + 1] == nrem_label) 
        )
    #now relabel ma as ma
    ma_idx = np.flatnonzero(is_ma)
    for ma in ma_idx:
        start = start_idx[ma]
        end = start + state_lengths[ma]
        out[start:end] = ma_label
    return out

def _subset_report(states: np.ndarray, win_len_s: int, state_mapping: dict = {0:'Unknown', 1:'Wake', 2:'NREM', 3:'IS', 4:'REM', 5: 'MA'}):
    """
    generates a report for passed states
    returns a dict with:
        - number of states
        - total time in states
        - percentage of time in states
        - median, IQR state duration
    """
    result = {}
    
    for state, duration in zip(np.unique(states, return_counts = True)[0], np.unique(states, return_counts = True)[1]):
        result[state_mapping[state] + '_duration'] = duration * win_len_s
        result[state_mapping[state] + '_percentage'] = (duration / len(states))* 100
        
    #get individual state number
    run_starts = np.empty(states.size, dtype=bool)  #create a state change array
    run_starts[0] = True
    run_starts[1:] = states[1:] != states[:-1]      #same as np.diff but will not crash with non-numbers

    run_start_idx = np.flatnonzero(run_starts)                 # indices where runs start
    run_states = states[run_start_idx]                         # state at each run start
    run_lengths = np.diff(np.append(run_start_idx, states.size))# run lengths (windows), states.size returns last state (as it has to go until last element due to diff)
    run_durations_s = run_lengths * win_len_s                  # run durations (seconds)
    
    bout_states, bout_counts = np.unique(run_states, return_counts=True)
    for state, n_bouts in zip(bout_states, bout_counts):
        name = state_mapping.get(int(state), str(state))
        result[f"{name}_bouts"] = int(n_bouts)

        # durations for this state's bouts
        duration = run_durations_s[run_states == state]
        result[f"{name}_bout_mean_s"] = float(np.mean(duration))
        result[f"{name}_bout_std_s"] = float(np.std(duration))
        
    return result
    

def generate_sleep_report(states: str, time_array: np.ndarray, get_by_hour = True, win_len = 1000, save_csv = None, 
                            state_mapping: dict = {0:'Unknown', 1:'Wake', 2:'NREM', 3:'IS', 4:'REM', 5: 'MA'}, 
                            metadata: dict = {}):
    """
    generates a report, complete or by hour
    uses a helper to get state numbers, percentages and durations
    """
    results = {}
    
    if states.size == 0:
        print('empty state array')
        return
    
    win_len_s = win_len / int(metadata.get('sample_rate', '250'))
    print(f'using {win_len_s} seconds per window')
    #ensure states are numpy
    states = np.array(states)
    #get overall results
    results['overall'] = _subset_report(states, win_len_s = win_len_s, state_mapping = state_mapping)
    #now subset by hour
    if get_by_hour:
        for hour in np.unique(time_array.hour, sorted = False)[:-1]:
            if hour < 23:
                next_hour = hour + 1
                in_hour = ((time_array.hour >= hour) & (time_array.hour < next_hour))
            else:   #handle midight crossing
                next_hour = 0
                in_hour = (time_array.hour >= hour)
            subset = states[in_hour]
            print(hour, next_hour, len(states), len(in_hour), in_hour.sum())
            results[str(hour)] = _subset_report(subset, win_len_s = win_len_s, state_mapping= state_mapping)
    if save_csv is not None:
        try:
            pd.DataFrame(results).to_csv(save_csv)
        except Exception as e:
            print(f'report excel saving failed: {e}')
    return results


if __name__ == "__main__":
    import pickle
    from scorer.data.storage import get_timearray_for_states
    path = r"C:\Users\marty\Desktop\SCORING202602\2025-11-27-1\scores\noID_scores_windowed_20260212180921 20251127-1_g0_t0.ob____1_frame10787.pkl"
    meta = {"scoring_started": "20260212180921", 
            "project_path": "C:/Users/marty/Desktop/SCORING202602/20251127-1", 
            "scorer": "Martynas", 
            "animal_id": "4F", 
            "group": "HYDR", 
            "trial": "1", 
            "sample_rate": 250, 
            "time_channel": "3", 
            "ecog_channels": "0", 
            "emg_channels": "2", 
            "ylim": "infer", 
            "spectral_view": None, 
            "device": "cuda", 
            "optional_tag": " ", 
            "date": "2026_02_12", 
            "metadata_path": "C:/Users/marty/Desktop/SCORING202602/20251127-1/20260212180921_meta.json", 
            "filename": "20251127-1_g0_t0.obx0.obx_box1", 
            "rec_start": "2025-11-27 19:16:00.301587302\n", 
            "channels_after_preprocessing": ["ecog", "emg", "t_d_logratio", "b_d_logratio", "s_d_logratio", "delta_logfraction", "emg_logpower"], 
            "preprocessing": ["bandpass", "bandpass", "ratio_signals_calculated", "resampling"], "filter_params": [[10.0, 100.0]], 
            "old_sample_rate": "1000"}
    with open(path, 'rb') as f:
        states = pickle.load(f)
        states = label_microawakenings(states, w_label = 1, nrem_label = 2, max_windows = 3, ma_label=5)
    time_array = get_timearray_for_states(states, win_len = 1000, metadata = meta)
    generate_sleep_report(states, time_array= time_array, save_excel = path[:-4] + '.csv', metadata = meta)