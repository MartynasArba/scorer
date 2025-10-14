import matplotlib.pyplot as plt

def plot_signals(selected_data, labels):
    
    sleep_labels = {0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'}
    
    start_label = labels[0] if isinstance(labels, list) else labels

    fig, ax = plt.subplots(selected_data.shape[1] + 1, 1, figsize = (10, 6), sharex = True)
    ax[0].plot(selected_data[:, 0])
    ax[0].set(title = f'starting label is {sleep_labels[start_label]}', ylabel = 'voltage, ECoG')
    ax[1].plot(selected_data[:, 1])
    ax[1].set(xlabel = 'time, datapoint', ylabel = 'voltage, EMG')
    ax[2].plot(labels)
    ax[2].set(ylabel = 'sleep state')
    plt.tight_layout()
    # plt.show()
    return fig

def plot_spectrogram(spect_data, labels):
    
    sleep_labels = {0:'Unlabeled',
          1:'Wake',
          2:'NREM',
          3:'IS',
          4:'REM'}
    
    start_label = labels[0] if isinstance(labels, list) else labels
    
    fig, ax  = plt.subplots(1, 2, figsize = (10,5), sharex = True, sharey = True)
    
    ax[0].imshow(spect_data[0])
    ax[0].set(title = f'starting label is {sleep_labels[start_label]}', ylabel = 'freq bin', xlabel = 'time bin, ECoG')
    
    ax[1].imshow(spect_data[1])
    ax[1].set(ylabel = 'freq bin', xlabel = 'time bin, EMG')
    
    plt.tight_layout()
    # plt.show()
    return fig