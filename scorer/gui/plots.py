import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def plot_signals(selected_data, labels: list, ylims: list[(float, float)] = [(0, 0.5, 1)], sample_rate: int = 250, names: list[str] = ['']) -> plt.figure:
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

def plot_signals_init(selected_data, ylims, sample_rate=250, names=None):
    """
    initializes plots to be modified later by update func
    returns (fig, axs, lines)
    """
    n_samples = selected_data.shape[-1]
    time_axis = np.arange(n_samples) / sample_rate
    n_channels = selected_data.shape[0]

    fig, axs = plt.subplots(n_channels + 1, 1, figsize=(10, 1 * n_channels + 1), sharex=True)
    lines = []

    for i, ax in enumerate(axs[:-1]):
        # create lines with initial data
        line, = ax.plot(time_axis, selected_data[i], lw=0.8)
        lines.append(line)
        lo, hi = ylims[i] if i < len(ylims) else (np.min(selected_data[i]), np.max(selected_data[i]))
        ax.set_ylim(lo, hi)
        if names and i < len(names):
            ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)

    # label axis: create both a line (invisible) and a text object
    label_ax = axs[-1]
    label_ax.grid(True, alpha=0.3)
    label_ax.set(xlabel="Time (s)", ylabel="Sleep State")
    # persistent line: start invisible with empty data
    label_line, = label_ax.plot([], [], drawstyle='steps-post', lw=2, visible=False)
    # persistent text (centered); start empty
    label_text = label_ax.text(0.5, 0.5, "", ha='center', va='center', transform=label_ax.transAxes, fontsize=14)
    # set y-ticks for discrete states (but keep minor ticks)
    label_ax.set_yticks([0, 1, 2, 3, 4])

    fig.tight_layout()
    return fig, axs, lines, label_line, label_text

def plot_signals_update(selected_data, labels, lines, label_ax,
                        label_line, label_text, ylims, sample_rate=250):
    """
    update existing main ecog plots
    """
    n_samples = selected_data.shape[-1]
    time_axis = np.arange(n_samples) / sample_rate
    n_channels = selected_data.shape[0]

    # --- Update signal lines and y-limits ---
    for i in range(min(n_channels, len(lines))):
        lines[i].set_xdata(time_axis)
        lines[i].set_ydata(selected_data[i])
        lo, hi = (ylims[i] if i < len(ylims) else (np.min(selected_data[i]), np.max(selected_data[i])))
        lines[i].axes.set_ylim(lo, hi)

    # ensure x-limits span full time
    if len(lines) > 0:
        lines[-1].axes.set_xlim(time_axis[0], time_axis[-1])

    # --- Update label panel ---
    # If labels is a list of segment labels -> draw step trace
    if isinstance(labels, (list, tuple)) and len(labels) > 1:
        # compute label array of length n_samples
        n_segments = len(labels)
        win_size = max(1, n_samples // n_segments)

        # Expand labels to length win_size * n_segments, then pad or trim to n_samples
        label_arr = np.repeat(np.asarray(labels, dtype=int), win_size)
        if label_arr.size < n_samples:
            # pad with last label
            pad_len = n_samples - label_arr.size
            label_arr = np.concatenate([label_arr, np.full(pad_len, label_arr[-1], dtype=int)])
        elif label_arr.size > n_samples:
            label_arr = label_arr[:n_samples]

        # show/hide artists correctly
        label_text.set_text("")            # clear text
        if not label_line.get_visible():
            label_line.set_visible(True)
        label_line.set_xdata(time_axis)
        label_line.set_ydata(label_arr)
        # adjust y-limits and ticks for discrete labels
        label_ax.set_ylim(-0.5, 4.5)
        label_ax.set_yticks([0, 1, 2, 3, 4])

    else:
        # single label: show centered text, hide the step line
        single = labels[0] if isinstance(labels, (list, tuple)) else labels
        # map numeric code -> label text if you want descriptive names here
        sleep_labels = {0: 'Unlabeled', 1: 'Wake', 2: 'NREM', 3: 'IS', 4: 'REM'}
        label_text.set_text(f"State: {sleep_labels.get(int(single), str(single))}")
        # hide label line to prevent stale trace
        if label_line.get_visible():
            label_line.set_visible(False)
        # optional: keep ticks but don't autoscale them
        label_ax.set_ylim(-0.5, 1.5)
        label_ax.set_yticks([])  # hide ticks when showing single text (makes it cleaner)

    # Force draw of axis autoscale if necessary (we don't recreate axis)
    label_ax.relim()
    label_ax.autoscale_view(scalex=False, scaley=True)

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

def plot_spectrogram_init(spect_data):
    """
    init pre-generated spectrogram plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    img = ax.imshow(spect_data, aspect='auto', origin='lower', cmap='viridis')
    ax.set_ylabel("frequency bin")
    fig.tight_layout()
    return fig, ax, img

def plot_spectrogram_update(img, ax, spect_data, time_axis):
    """
    update spectrogram plot
    """
    img.set_data(spect_data)
    ax.set_xlim(time_axis[0], time_axis[-1])

def plot_fourier(fourier) -> plt.figure:
    """
    plots pre-generated fourier transform
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(fourier[:, 1], fourier[:, 0])
    ax.set(title = 'ECoG fft', xlabel = 'power', ylabel = 'frequency')
    plt.tight_layout()
    return fig

def plot_fourier_init(data):
    """
    initializes plot for pre-generated fourier transform
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    line, = ax.plot(data[:, 1], data[:, 0])
    ax.set(xlabel='power', ylabel='frequency')
    fig.tight_layout()
    return fig, ax, line

def plot_fourier_update(line, fourier):
    """
    updates plot for pre-generated fourier transform
    """
    line.set_xdata(fourier[:, 1])
    line.set_ydata(fourier[:, 0])