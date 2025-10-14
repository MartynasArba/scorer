import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

def from_Oslo_csv(path, sep = '\\'):
    """
    Converts Oslo .csv to pickle 
    """
    xpath = path.split(sep)[:]
    xpath[-2] = 'processed'
    xpath = sep.join(xpath)
    xpath = xpath[:-4] + '_X.pkl'
    
    ypath = path.split(sep)[:]
    ypath[-2] = 'processed'
    ypath = sep.join(ypath)
    ypath = ypath[:-4] + '_y.pkl'
    
    data = pd.read_csv(path)
    signal = data[['ecog', 'emg']].values
    labels = data['sleep_episode'].values
    X, y = chop_data(states = labels, values = signal, win_len = 1000, labeled = True)
    print(X.shape)

    with open(xpath, 'wb') as f:
        pickle.dump(X, f)
        
    with open(ypath, 'wb') as f:
        pickle.dump(y, f)

def chop_data(states, values, win_len = 1000, labeled = True):
    # in the file: time, ecog, emg, state
    if labeled: #if the data has been labeled, chops by label boundaries
        return _chop_by_state(states, values, win_len)
    else: #if the data is not labeled, chops sequentially
        return _chop(values, win_len)
    
def _chop_by_state(states, values, win_len):
    X = []
    y = []
    #find all episodes of set len and return their indices
    #state changes found
    state_changes = np.where(np.diff(states).astype(bool))[0] + 1
    #handle boundaries with start/end points
    state_changes = np.concatenate([[0], state_changes, [len(states)]])

    for id in tqdm(range(len(state_changes) - 1)):
        # trying to get better readability
        start_idx = state_changes[id] #state start index 
        end_idx = state_changes[id + 1] #state end index
        current_state = states[start_idx] #keep track of current state
        
        #get segments and append
        if end_idx - start_idx >= win_len: #check if the window is still long enough
            i = start_idx   #mark starting index
            while i + win_len <= end_idx: #iterate through state and save data
                X.append(values[i:i + win_len])
                y.append(current_state)
                i += win_len
    #handle cases where no data was found
    if len(X) == 0:
        return np.array([]), np.array([])
    #return stacked arrays
    return np.stack(X), np.stack(y)

def _chop(values, win_len): #need to also create labels of 0 
    X = []
    y = []
    #similar syntax for consistency, generally unnecessary
    start_idx = 0
    end_idx = len(values)
    current_state = 0
    #check whether there is enough data in general
    if end_idx < win_len:
        return np.array([]), np.array([]) 
    else: #main loop
        i = start_idx
        while i + win_len <= end_idx:
            X.append(values[i:i + win_len])
            y.append(current_state)
            i += win_len
    
    return np.stack(X), np.stack(y)