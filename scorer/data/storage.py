import pickle
import os
from pathlib import Path

def construct_paths(data_path: str, **metadata):    
    """
    Gets data path from QFileDialog, returns save paths for pickle and state annotation arrays   
    """
    
    data_path = Path(data_path)
    
    if data_path.exists():
        
        y_file_path = data_path.with_name(data_path.name.replace('X', 'y'))
        
        scorer = metadata.get('scorer', data_path.stem[:-2] if data_path.suffix else data_path.stem)
        date = metadata.get('date', '')
        animal = metadata.get('animal', '')
        trial = metadata.get('trial', '')
        repetition_id = 0
        
        #create required score dir if it doesn't exist
        score_folder = data_path.parent.parent / 'scores'
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        
        states_array_path = score_folder / f'scores_{scorer}_{date}_{animal}_{trial}_{repetition_id}.pkl'
        
        #don't overwrite existing files
        while states_array_path.exists():
            repetition_id += 1
            states_array_path = score_folder / f'scores_{scorer}_{date}_{animal}_{trial}_{repetition_id}.pkl'
    
        return y_file_path, states_array_path

def save_pickled_states(states: list, path: str):
    """
    dumps states to pickle
    """
    with open(path, 'wb') as f:
        print('TODO: change save_pickled_states to only dump states which are being scored by current scorer')
        pickle.dump(states, f)

def load_pickled_states(path: str):
    """
    loads scores from pickle
    """
    with open(path, 'rb') as f:
        print('TODO: change load_pickled_states usage to have list of states or delete this')
        states = pickle.load(f)
    return states

def save_json(states: list, path: str):  
    print('TODO: save_json not implemented')
    pass
    #todo: implement saving as json with metadata

# testing
if __name__ == "main":
    y, save = construct_paths(r"C:\Users\marty\Projects\scorer\proj_data\processed\trial_1_mouse_b1aqm2_X.pkl")
    print(y, save)