import matplotlib.pyplot as plt
import numpy as np

def plot_signals(selected_data, labels, sample_rate = 250, ecog_ylim = (None, None), emg_ylim = (None, None)):
    
    #support for plotting multiple scorer results
    #labels will always be a list?
    # label_list = labels.copy()
    # labels = labels[0]

    sleep_labels = {0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'}
    
    start_label = labels[0] if isinstance(labels, list) else labels
    
    #get time axis
    n_samples = selected_data.shape[0]
    time_axis = np.arange(n_samples) / sample_rate

    fig, ax = plt.subplots(selected_data.shape[1] + 1, 1, figsize = (10, 6), sharex = True)
    ax[0].plot(time_axis, selected_data[:, 0])
    ax[0].set(title = f'starting label is {sleep_labels[start_label]}', ylabel = 'ECoG (μV)')
    ax[0].grid(True, alpha = 0.3)
    if any(ecog_ylim):
          ax[0].set_ylim(ecog_ylim[0], ecog_ylim[1])
    
    ax[1].plot(time_axis, selected_data[:, 1])
    ax[1].set(ylabel = 'EMG (μV)')
    ax[1].grid(True, alpha = 0.3)
    if any(emg_ylim):
          ax[1].set_ylim(emg_ylim[0], emg_ylim[1])
    
    # Plot state labels if multiple provided
    if isinstance(labels, list) and len(labels) > 1:
        win_size = int(n_samples/len(labels))
        label_arr = np.ravel(np.array([[l] * win_size for l in labels]))
        ax[2].plot(time_axis, label_arr, drawstyle='steps-post', linewidth=2)
        ax[2].set_yticks([0, 1, 2, 3, 4])
        
    else:       #if single label, write in text
        ax[2].text(0.5, 0.5, f'State: {sleep_labels[start_label]}', 
                   ha='center', va='center', transform = ax[2].transAxes, fontsize=14)
    
    ax[2].set(xlabel = 'Time (s)', ylabel = 'Sleep State')
    ax[2].grid(True, alpha = 0.3)
    
    plt.tight_layout()
    return fig

def plot_spectrogram(spect_data, ylim = (0, 20)):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    
    im0 = ax[0].imshow(spect_data[0], aspect = 'auto', origin = 'lower', cmap = 'viridis')
    ax[0].set(title = f'ECoG', ylabel = 'Frequency bin')  
    # plt.colorbar(im0, ax = ax[0], label = 'Power')
    
    im1 = ax[1].imshow(spect_data[1], aspect = 'auto', origin = 'lower', cmap = 'viridis')
    ax[1].set(title = 'EMG', ylabel = 'Frequency bin', xlabel = 'Time bin')
    # plt.colorbar(im1, ax = ax[1], label = 'Power')
    
    if any(ylim):
        for a in ax:
            a.set_ylim(ylim)
    
    plt.tight_layout()
    return fig