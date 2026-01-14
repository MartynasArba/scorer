import torch
import numpy as np
from tqdm import tqdm
from scorer.data.loaders import SleepSignals

#heuristics-based scoring
class HeuristicScorer():
    """
    Heuristic-based scorer class
    expected channels in dataset:
    0 - ecog
    1 - emg
    2 - ecog band power
    3 - emg band power
    4 - 'delta': (0.5, 4)
    5 - 'theta': (5, 9)
    6 - 'alpha': (8, 13)
    7 - 'sigma': (12, 15)
    
    Args:
        data (SleepSignals): dataset of one recording, not shuffled, with no loader

    Returns:
        states, np.ndarray: score for each window, ready for manual review (should check how they're saved in manual scoring)
    """
    
    def __init__(self, dataset):
        self.data = dataset
        self.states = []
        self.q10, self.q50, self.q90 = self._get_qs(self.data, sub_factor = 10)
        if int(self.data.all_samples.size(dim = 1)) != 8:
            raise ValueError('there must be 8 channels in data: ecog, emg, ecog_pow, emg_pow, delta, theta, alpha, sigma')
        
        
    def score(self):
        print('heuristic-based scoring started')
        for window_id in tqdm(range(self.data.all_samples.size(dim = 0))):
            state = 0   #unknown
            window = self.data.all_samples[window_id, :, :]
            max_vals = window.max(dim = 1).values
            #first, checking by EMG power, channel 3
            if (max_vals[3] >= self.q90[3]).item():
                state = 1   #W
            # checking REM - high theta, low emg, could add low delta?
            elif (max_vals[5] >= self.q50[5]).item() and (max_vals[3] <= self.q50[3]).item() and (max_vals[4] <= self.q50[4]).item():
                if window_id != 0:
                    if self.states[window_id-1] != 1:#previous state can't be W
                        state = 4 #REM
            #checking for IS - high sigma, sigma > theta, low emg
            elif (max_vals[7] > self.q50[7]).item() and (max_vals[7] > max_vals[5]).item() and (max_vals[3] <= self.q50[3]).item():
                state = 3
            #NREM: high delta, low emg
            elif (max_vals[4] > self.q50[4]).item() and (max_vals[3] <= self.q50[3]).item():
                state = 2
            self.states.append(state)
        print('scoring done')

    def __str__(self):
        if self.states:
            return f'unique state values: {np.unique(np.array(self.states), return_counts = True)}'
        else:
            return 'not scored yet'
        
    def _get_qs(self, data: SleepSignals, sub_factor: int = 10):
        """
        returns q10, q50, and q90
        uses every sub_factor-th sample for efficiency
        """
        q10, q50, q90 = [], [], []
        n_channels = data.all_samples.size(dim = 1)
        
        #for each channel, select a subset and calculate summary stats
        for ch in range(n_channels):
            q10.append(torch.quantile(data.all_samples[::sub_factor, ch, :].float(), q = 0.10).cpu().numpy())#"low"
            q50.append(torch.quantile(data.all_samples[::sub_factor, ch, :].float(), q = 0.50).cpu().numpy())#"mid"
            q90.append(torch.quantile(data.all_samples[::sub_factor, ch, :].float(), q = 0.90).cpu().numpy())#'high"
        return np.stack(q10), np.stack(q50), np.stack(q90)
