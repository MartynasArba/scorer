import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import matplotlib.ticker as mticker
from datetime import datetime
DAY = 24 * 60 * 60

def plot_signals_init(
        selected_data,
        ylims,
        sample_rate=250,
        names=None,
        scorer_names=None,
        time_formatter = None):
    """
    initialize signal and label plots

    Returns:
        fig, axs, signal_lines, label_lines, label_text
    """

    # shape
    n_samples = selected_data.shape[-1]
    time_axis = np.arange(n_samples) / sample_rate
    n_channels = selected_data.shape[0]

    # figure
    fig, axs = plt.subplots(
        n_channels + 1, 1,
        figsize=(10, n_channels + 1),
        sharex=True
    )
    # signal axes - ecog, emg, powers etc
    signal_lines = []

    for i, ax in enumerate(axs[:-1]):
        line, = ax.plot(time_axis, selected_data[i], lw=0.8)
        signal_lines.append(line)

        # y-limits
        if i < len(ylims):
            center, spread = ylims[i]
            low = center - spread / 2
            high = center + spread / 2
        else:
            low, high = np.min(selected_data[i]), np.max(selected_data[i])
        ax.set_ylim(low, high)

        # optional channel names
        if names and i < len(names):
            ax.set_ylabel(names[i], rotation = 'horizontal', labelpad=4)

        ax.grid(True, alpha=0.3)

    # label axis (last axis)
    label_ax = axs[-1]
    label_ax.grid(True, alpha=0.3)
    label_ax.set_xlabel("time")
    label_ax.set_ylabel("state")
    
    if time_formatter is not None:
        label_ax.xaxis.set_major_formatter(time_formatter)
        # label_ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = 8))    #, integer = True
    
    #one line per scorer
    if scorer_names is None:
        scorer_names = ["default_scorer"]

    colors = plt.cm.tab10.colors
    label_lines = {}

    for i, name in enumerate(scorer_names):
        color = colors[i % len(colors)]
        linestyle = "-" if i == 0 else "--" 
        ln, = label_ax.plot([], [], lw=2, alpha=0.9, color=color, linestyle=linestyle)
        ln.set_visible(False) # hidden until update
        label_lines[name] = ln

    #text for single label
    label_text = label_ax.text(
        0.5, 0.5, "",
        ha='center', va='center',
        transform=label_ax.transAxes,
        fontsize=14
    )

    #default ticks for integer states
    label_ax.set_yticks([0, 1, 2, 3, 4])
    label_ax.set_ylim(-0.5, 4.5)

    fig.tight_layout()
    fig.subplots_adjust(left=0.06)
    return fig, axs, signal_lines, label_lines, label_text

def plot_signals_update(
        selected_data,
        signal_lines,
        label_ax,
        label_lines,
        label_text,
        ylims,
        sample_rate=250,
        labels=None):
    """
    update existing plots
    labels: dict of scorer -> list of ints/None
    show_all_scorers: if False, only the first scorer (main) is displayed
    """
    if labels is None:
        labels = {}
    #backward compatibility
    if isinstance(labels, (list, tuple)):
        labels = {"main": list(labels)}    

    # update signal plots
    n_samples = selected_data.shape[-1]
    time_axis = np.arange(n_samples) / sample_rate
    n_channels = selected_data.shape[0]

    for i in range(min(n_channels, len(signal_lines))):
        signal_lines[i].set_xdata(time_axis)
        signal_lines[i].set_ydata(selected_data[i])

        if i < len(ylims):
            center, spread = ylims[i]
            low = center - spread / 2
            high = center + spread / 2
        else:   #if no ylims are available
            low, high = np.min(selected_data[i]), np.max(selected_data[i])
            
        signal_lines[i].axes.set_ylim(low, high)

    if signal_lines:
        signal_lines[-1].axes.set_xlim(time_axis[0], time_axis[-1])

    # labels
    scorer_names = list(labels.keys())
    #if no labels at all, clear text and hide lines
    if not scorer_names:
        label_text.set_text("")
        for ln in label_lines.values():
            ln.set_visible(False)
        return

    #if single scorer, plot text on center
    if len(scorer_names) == 1:
        scorer = scorer_names[0]
        arr = labels[scorer]

        # ensure it's a list
        arr = list(arr)

        if len(arr) == 1:
            # single label -> show centered text, hide all lines
            state_val = arr[0]
            sleep_labels = {0: 'Unlabeled', 1: 'Wake', 2: 'NREM', 3: 'IS', 4: 'REM'}

            # hide all scorer lines
            for ln in label_lines.values():
                ln.set_visible(False)

            # show single text
            label_text.set_text(f"State: {sleep_labels.get(int(state_val), str(state_val))}")
            label_ax.set_ylim(-0.5, 1.5)
            label_ax.set_yticks([])
            return
    #if multi scorer or multi label
    # no more text
    label_text.set_text("")

    # ylims reflect sleep states
    label_ax.set_ylim(-0.5, 4.5)
    label_ax.set_yticks([0, 1, 2, 3, 4])
    
    # hide all lines, re-enable ones we use later
    for ln in label_lines.values():
        ln.set_visible(False)

    # for each scorer we actually have labels
    for scorer, arr in labels.items():
        if scorer not in label_lines:
            continue  # no line for this scorer

        ln = label_lines[scorer]
        vals = list(arr)

        if len(vals) == 0:
            ln.set_visible(False)
            continue

        # Convert values to float, allow None -> NaN (gap)
        y = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)

        # scale to x axis
        n_labels = len(y)
        win_size = max(1, n_samples // n_labels)

        # repeat each label value win_size times
        stretched = np.ravel([[v] * win_size for v in y])

        # pad or trim to exactly n_samples
        if stretched.size < n_samples:
            pad_len = n_samples - stretched.size
            stretched = np.concatenate([stretched, np.full(pad_len, stretched[-1])])
        elif stretched.size > n_samples:
            stretched = stretched[:n_samples]

        # update this scorer’s line
        ln.set_xdata(time_axis)
        ln.set_ydata(stretched)
        ln.set_visible(True)
    


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
    ax.set(title = 'ECoG fft', xlabel = 'frequency', ylabel = 'power')
    plt.tight_layout()
    return fig

def plot_fourier_init(data):
    """
    initializes plot for pre-generated fourier transform
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    line, = ax.plot(data[:, 1], data[:, 0])
    ax.set(xlabel='frequency', ylabel='power')
    fig.tight_layout()
    return fig, ax, line

def plot_fourier_update(line, fourier):
    """
    updates plot for pre-generated fourier transform
    """
    line.set_xdata(fourier[:, 1])
    line.set_ydata(fourier[:, 0])
    
class TimeOfDayFormatter(mticker.Formatter):
    """
    format x axis(seconds from window start) instead as time of day: recording start timestamp + window offset seconds
    """
    def __init__(self, rec_start_epoch_s: float, window_offset_s: float = 0.0, fmt: str = "%H:%M:%S"):
        """
        rec_start_epoch_s - recording start, seconds from midnight
        window_offset_s - current_idx * win_len / sr    
        fmt - format
        """
        self.rec_start_epoch_s = float(rec_start_epoch_s)
        self.window_offset_s = float(window_offset_s)
        self.fmt = fmt

    def set_window_offset(self, window_offset_s: float):
        self.window_offset_s = float(window_offset_s)   #updates window position in recording

    def __call__(self, x, pos=None):
        #x is seconds from window start in plot func
        #converts to time of day in set format
        t = (self.rec_start_epoch_s + self.window_offset_s + float(x)) % DAY
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if self.fmt == "%H:%M:%S":
            return f"{h:02d}:{m:02d}:{s:02d}"
        # fallback: build a datetime for weird fmt
        dt = datetime(2000, 1, 1, h, m, s)
        return dt.strftime(self.fmt)