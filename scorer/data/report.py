import numpy as np
import pandas as pd

def label_microawakenings(states: np.ndarray, 
                          w_label: int = 1, 
                          nrem_label: int = 2, 
                          max_windows: int = 10, 
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

    # same as in detect diffs
    state_starts = np.empty(n, dtype=bool)
    state_starts[0] = True    
    state_starts[1:] = out[1:] != out[:-1]

    start_idx = np.flatnonzero(state_starts)          # state start indices
    state_labels = out[start_idx]                      # state for each run
    state_lengths = np.diff(np.append(start_idx, n))   # lengths in windows

    mid = np.arange(state_labels.size)
    valid_mid = (mid > 0) & (mid < state_labels.size - 1)   #have to be surrounded by other states

    is_ma = (
        valid_mid
        & (state_labels == w_label)
        & (state_lengths <= max_windows)
        & (state_labels[mid - 1] == nrem_label)
        & (state_labels[mid + 1] == nrem_label)
    )

    ma_idx = mid[is_ma]
    micro_starts = start_idx[ma_idx]
    micro_ends_excl = micro_starts + state_lengths[ma_idx]

    for s, e in zip(micro_starts, micro_ends_excl):
        out[s:e] = ma_label

    return out

def generate_sleep_report(states: str, time_array: np.ndarray, get_by_hour = True, win_len = 1000, 
                            state_mapping: dict = {0:'Unknown', 1:'Wake', 2:'NREM', 3:'IS', 4:'REM', 5: 'MA'}, 
                            metadata: dict = {}):
    """
    calculates:
        - number of states
        - total time in states
        - percentage of time in states
        - median, IQR state duration
        - split by hour
    """
    results = {}
    
    if states.size == 0:
        print('empty state array')
        return
    
    states = label_microawakenings(states, w_label = 1, nrem_label = 2, max_windows=10, ma_label=5)
    
    win_len_s = win_len / int(metadata.get('sample_rate', '250'))
    print(f'using {win_len_s} seconds per window')
    #ensure states are numpy
    states = np.array(states)
    
    overall = {}
    
    for state, duration in zip(np.unique(states, return_counts = True)[0], np.unique(states, return_counts = True)[1]):
        overall[state_mapping[state] + '_duration'] = duration * win_len_s
        overall[state_mapping[state] + '_percentage'] = (duration / len(states))* 100
        
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
        overall[f"{name}_bouts"] = int(n_bouts)

        # durations for this state's bouts
        duration = run_durations_s[run_states == state]
        overall[f"{name}_bout_mean_s"] = float(np.mean(duration))
        overall[f"{name}_bout_std_s"] = float(np.std(duration))# np.percentile(duration, 75) - np.percentile(duration, 25

    results['overall'] = overall
    print(results)
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
    time_array = get_timearray_for_states(states, win_len = 1000, metadata = meta)
    generate_sleep_report(states, time_array= time_array, metadata = meta)