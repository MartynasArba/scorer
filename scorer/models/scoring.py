# code to do pre-trained scoring should go here

# load into SleepSignals
#do scoring: launcher function 
from scorer.data.loaders import SleepSignals
from scorer.models.heuristics_scorer import HeuristicScorer
from scorer.data.storage import save_pickled_states
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

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
    
    disp1 = ConfusionMatrixDisplay(cm1)
    disp1.plot()
    plt.show()