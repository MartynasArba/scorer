# code to do pre-trained scoring should go here
# load into SleepSignals
#do scoring: launcher function 
from scorer.data.loaders import SleepSignals
from scorer.models.heuristic_scorer import HeuristicScorer
# from scorer.models.heuristic_scorer_v2 import HeuristicScorer2
from scorer.models.sleep_cnn import SleepCNN
from scorer.data.storage import save_pickled_states
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm

def score_signal(data_path, state_save_folder, meta, scorer_type = 'heuristic'):
    """
    runs scoring, launched from GUI
    saves states according to selected path
    scorers might have their own preprocessing requirements, check class definitions
    
    data_path should be the folder containing X_ and y_ paths of windowed data that results from preprocessing
    state_save_folder is passed from GUI selection
    
    """
    available_scorers = {'heuristic': HeuristicScorer, 'CNN': SleepCNN, '3state_CNN': SleepCNN}
    
    selected_scorer = available_scorers.get(scorer_type)
    
    dataset = SleepSignals(data_path = data_path, 
                        score_path = data_path, 
                        device = meta.get('device', 'cpu'),
                        transform = None,
                        augment = False,
                        spectral_features = None,
                        metadata = meta)
    
    if selected_scorer == HeuristicScorer:
        scorer = selected_scorer(dataset)
        scorer.score()
        print(scorer)
        state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + scorer_type + '_states.pkl')
        save_pickled_states(np.array(scorer.states), state_save_path)
        print(f'scoring done, states saved as {state_save_path}')
        
    elif scorer_type == 'CNN':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\sleepcnn2026-01-21-1.pt', weights_only= False)
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
        #now reset states to include 0 - shift by 1
        all_preds = np.array(all_preds) + 1
                
        state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + scorer_type + '_states.pkl')
        save_pickled_states(all_preds, state_save_path)
        print(f'scoring done, states saved as {state_save_path}')
        
    elif scorer_type == '3state_CNN':
        loader = DataLoader(dataset, batch_size = 64, shuffle = False)
        #predict 
        scorer = torch.load(r'C:\Users\marty\Projects\scorer\scorer\models\weights\sleepcnn2026-01-20-3.pt', weights_only= False)
        all_preds = []
        scorer.eval()
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                sample, label = data
                outputs = scorer(sample)
                _, pred = torch.max(outputs.data, 1)
                #to get final predictions
                all_preds.extend(pred.to('cpu').numpy().tolist())
        #now reset states to include 0 - shift by 1, then reset REM to 4
        all_preds = np.array(all_preds) + 1
        all_preds[all_preds == 3] = 4    
            
        state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + scorer_type + '_states.pkl')
        save_pickled_states(all_preds, state_save_path)
        print(f'scoring done, states saved as {state_save_path}')

    else:
        print(f'unavailable scorer selected: {scorer_type}')
    

if __name__ == "__main__":
    
    save_path = 'C:/Users/marty/Projects/scorer/proj_data/scores/trial_1_mouse_b1aqm2_heuristic_score.pkl'

    dataset = SleepSignals(
        data_path = 'G:/oslo_data/windowed_trial_1_mouse_b1aqm2',
        score_path = 'G:/oslo_data/windowed_trial_1_mouse_b1aqm2',
        device = 'cuda',
        transform = None,
        augment = False,
        spectral_features = None,
        metadata = {'ecog_channels' : '1', 'emg_channels' : '2', 'sample_rate' : '250', 'ylim' : 'standard'}
    )    
    scorer = HeuristicScorer(dataset)
    scorer.score()
    print(scorer)

    save_pickled_states(np.array(scorer.states), save_path)

    old_states = dataset.all_labels.cpu().numpy()
    print(np.unique(old_states, return_counts = True))

    print(f'accuracy: {accuracy_score(old_states, scorer.states)}')

    cm1 = confusion_matrix(old_states, scorer.states)
    print(cm1)

    disp1 = ConfusionMatrixDisplay(cm1)
    disp1.plot()
    plt.show()