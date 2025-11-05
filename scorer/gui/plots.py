import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def plot_signals(selected_data, labels: list, ylims: list[(float, float)] = [(0, 1)], sample_rate: int = 250, names: list[str] = ['']) -> plt.figure:
    """
    plots selected "raw" signals and labels
    label plotting could be moved to a separate function
    """

    sleep_labels = {0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'}
    
    start_label = labels[0] if isinstance(labels, list) else labels
        
    #get time axis
    n_samples = selected_data.shape[-1]
    time_axis = np.arange(n_samples) / sample_rate
    #get channel num
    n_channels = selected_data.shape[0]

    fig, axs = plt.subplots(n_channels + 1, 1, figsize = (10, 1 * n_channels + 1), sharex = True)
    for i, ax in enumerate(axs[:-1]):
        ax.plot(time_axis, selected_data[i, :])
        ax.grid(True, alpha = 0.3)
        if (len(ylims) >= i) & (isinstance(ylims[i], tuple)):
            ax.set_ylim(ylims[i][0], ylims[i][1])
        if (len(names) >= i) & (isinstance(names, str)):
            ax.set(ylabel = f'{names[i]}')
    
    # Plot state labels if multiple provided
    if isinstance(labels, list) and len(labels) > 1:
        win_size = int(n_samples/len(labels))
        label_arr = np.ravel(np.array([[l] * win_size for l in labels]))
        axs[-1].plot(time_axis, label_arr, drawstyle='steps-post', linewidth=2)
        axs[-1].set_yticks([0, 1, 2, 3, 4])
        
    else:       #if single label, write in text
        axs[-1].text(0.5, 0.5, f'State: {sleep_labels[start_label]}', 
                   ha='center', va='center', transform = axs[-1].transAxes, fontsize=14)
    
    axs[-1].set(xlabel = 'Time (s)', ylabel = 'Sleep State')
    axs[-1].grid(True, alpha = 0.3)
    
    plt.tight_layout()
    return fig

def plot_spectrogram(spect_data, ylim: Tuple[int, int] = (0, 20)) -> plt.figure:
    """
    plots pre-generated spectrogram
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.imshow(spect_data, aspect = 'auto', origin = 'lower', cmap = 'viridis')
    ax.set(title = 'ECoG spectrogram', ylabel = 'frequency bin')  
    if any(ylim):
        ax.set_ylim(ylim)   
    plt.tight_layout()
    return fig

def plot_fourier(fourier) -> plt.figure:
    """
    plots pre-generated fourier transform
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(fourier[1], fourier[0])
    ax.set(title = 'ECoG fft', xlabel = 'power', ylabel = 'frequency')
    plt.tight_layout()
    return fig