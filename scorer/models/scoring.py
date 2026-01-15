# code to do pre-trained scoring should go here

# load into SleepSignals
#do scoring: launcher function 
from scorer.data.loaders import SleepSignals
from scorer.models.heuristic_scorer import HeuristicScorer
from scorer.models.heuristic_scorer_v2 import HeuristicScorer2
from scorer.data.storage import save_pickled_states
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

def score_signal(data_path, state_save_folder, meta, scorer_type = 'heuristic'):
    """
    runs scoring, launched from GUI
    saves states according to selected path
    scorers might have their own preprocessing requirements, check class definitions
    
    data_path should be the folder containing X_ and y_ paths of windowed data that results from preprocessing
    state_save_folder is passed from GUI selection
    
    """
    available_scorers = {'heuristic': HeuristicScorer}
    
    selected_scorer = available_scorers.get(scorer_type)
    if selected_scorer is not None:
        dataset = SleepSignals(data_path = data_path, 
                               score_path = data_path, 
                               device = meta.get('device', 'cpu'),
                               transform = None,
                               augment = False,
                               spectral_features = None,
                               metadata = meta)
        scorer = selected_scorer(dataset)
        scorer.score()
        print(scorer)
        state_save_path = Path(state_save_folder) / str(meta.get('scoring_started', '') + '_' + meta.get('filename', '') + scorer_type + '_states.pkl')
        save_pickled_states(np.array(scorer.states), state_save_path)
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
        
    scorer = HeuristicScorer2(dataset)
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