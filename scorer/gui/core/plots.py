"""
Matplotlib drawing for the GUI (Qt-free; widgets wrap these Figures in a FigureCanvas).

Changes vs gui/plots.py:
  * matplotlib.figure.Figure instead of pyplot: pyplot keeps every figure alive in its global
    registry, so each hypnogram / reload leaked a figure (and can open stray windows);
  * state tracks are drawn as steps with one point per WINDOW (old: every label repeated for every
    sample - 150 windows x 1000 samples = 150,000 points per scorer in macro view, built in Python);
  * hypnograms are step plots (old line plot drew diagonals between states);
  * optional model-confidence trace and spectrogram panel in the scoring view;
  * the unused spectral-feature plots (SleepSignals no longer produced them) are gone.
"""
from __future__ import annotations

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure

from scorer.gui.core.recording import STATE_NAMES

DAY = 24 * 60 * 60
STATE_TICKS = [0, 1, 2, 3, 4]


def get_visual_envelope(time_array: np.ndarray, data_array: np.ndarray, max_points: int = 10000):
    """Downsample a trace to about ``max_points`` by keeping each block's min and max.

    Unlike plain decimation this keeps short spikes visible. Returns ``(t, values)``.
    """
    n = len(time_array)
    if n <= max_points:
        return time_array, data_array
    step = int(np.ceil(n / (max_points / 2)))
    pad = (-n) % step
    t = np.pad(time_array, (0, pad), mode="edge") if pad else time_array
    d = np.pad(data_array, (0, pad), mode="edge") if pad else data_array
    d = d.reshape(-1, step)
    env = np.empty(d.shape[0] * 2, dtype=data_array.dtype)
    env[0::2], env[1::2] = d.min(axis=1), d.max(axis=1)
    return np.repeat(t.reshape(-1, step)[:, 0], 2), env


class TimeOfDayFormatter(mticker.Formatter):
    """Axis formatter that shows seconds-from-view-start as time of day.

    Args:
        rec_start_sod: Recording start in seconds since midnight.
        window_offset_s: Seconds from the recording start to the left edge of the view.
        fmt: ``strftime`` format of the labels.
    """

    def __init__(self, rec_start_sod: float, window_offset_s: float = 0.0, fmt: str = "%H:%M:%S"):
        """Store the recording start, view offset and format."""
        self.rec_start_sod = float(rec_start_sod)
        self.window_offset_s = float(window_offset_s)
        self.fmt = fmt

    def set_window_offset(self, window_offset_s: float) -> None:
        """Update the offset when the view moves."""
        self.window_offset_s = float(window_offset_s)

    def __call__(self, x, pos=None):
        """Return the label for axis value ``x`` (wraps past midnight)."""
        t = (self.rec_start_sod + self.window_offset_s + float(x)) % DAY
        h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
        if self.fmt == "%H:%M:%S":
            return f"{h:02d}:{m:02d}:{s:02d}"
        return datetime(2000, 1, 1, h, m, s).strftime(self.fmt)


def parse_rec_start(ts) -> tuple[float, bool]:
    """Parse a recording start such as ``'2025-11-27 19:16:00.301587302\n'``.

    Returns:
        ``(seconds since midnight, parsed_ok)``; ``(0.0, False)`` if it cannot be parsed.
    """
    if not ts:
        return 0.0, False
    ts = str(ts).strip()
    if "." in ts:
        head, frac = ts.split(".", 1)
        ts = f"{head}.{frac[:6]}"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0, False
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6, True


def _state_steps(values: np.ndarray, win_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return x edges and y values (n + 1 points) to draw per-window values with ``drawstyle='steps-post'``."""
    v = np.asarray(values, dtype=float)
    x = np.arange(len(v) + 1) * win_s
    return x, np.append(v, v[-1]) if len(v) else v


class SignalView:
    """The manual-scoring figure: one axis per channel, a state axis and an optional spectrogram.

    The state axis shows one step line per scorer and, on a twin axis, the model
    confidence when it is available.

    Args:
        n_channels: Number of signal axes.
        names: Channel names for the y labels.
        scorer_names: One state line per name (the first solid, others dashed).
        formatter: Optional x-axis formatter (e.g. ``TimeOfDayFormatter``).
        spectrogram: Add a spectrogram axis at the bottom.
    """

    def __init__(self, n_channels: int, names: list[str], scorer_names: list[str], formatter=None,
                 spectrogram: bool = False):
        """Create the axes and empty lines; see the class docstring."""
        self.n_channels = n_channels
        self.spectrogram = spectrogram
        n_axes = n_channels + 1 + (1 if spectrogram else 0)
        ratios = [1] * n_channels + [1.2] + ([1.6] if spectrogram else [])
        self.fig = Figure(figsize=(10, n_axes))
        self.axs = self.fig.subplots(n_axes, 1, sharex=True, gridspec_kw={"height_ratios": ratios})
        self.axs = np.atleast_1d(self.axs)
        self.signal_axs = list(self.axs[:n_channels])
        self.label_ax = self.axs[n_channels]
        self.spec_ax = self.axs[n_channels + 1] if spectrogram else None
        self.lines = []
        for i, ax in enumerate(self.signal_axs):
            (ln,) = ax.plot([], [], lw=0.8)
            self.lines.append(ln)
            if i < len(names):
                ax.set_ylabel(names[i], rotation="horizontal", labelpad=4, ha="right")
            ax.grid(True, alpha=0.3)
        colors = [f"C{i}" for i in range(10)]
        self.state_lines = {}
        for i, name in enumerate(scorer_names):
            (ln,) = self.label_ax.plot([], [], lw=2, alpha=0.9, color=colors[i % 10], ls="-" if i == 0 else "--",
                                       drawstyle="steps-post", label=str(name))
            ln.set_visible(False)
            self.state_lines[name] = ln
        self.conf_ax = self.label_ax.twinx()
        (self.conf_line,) = self.conf_ax.plot([], [], lw=1, color="0.45", ls=":", drawstyle="steps-post")
        self.conf_ax.set_ylim(0, 1.05)
        self.conf_ax.set_yticks([0.5, 1.0])
        self.conf_ax.tick_params(labelsize=7, colors="0.45")
        self.conf_ax.set_visible(False)
        self.label_text = self.label_ax.text(0.5, 0.5, "", ha="center", va="center", transform=self.label_ax.transAxes,
                                             fontsize=14)
        self.label_ax.set_ylabel("state")
        self.label_ax.grid(True, alpha=0.3)
        if len(scorer_names) > 1:
            self.label_ax.legend(loc="upper right", fontsize=7, ncol=len(scorer_names))
        self.spec_img = None
        bottom = self.spec_ax if spectrogram else self.label_ax
        bottom.set_xlabel("time")
        if formatter is not None:
            bottom.xaxis.set_major_formatter(formatter)
        self._grid_key = None
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.08, hspace=0.08)

    def update(self, signals: np.ndarray, ylims, sample_rate: float, labels: dict[str, np.ndarray], win_s: float,
               confidence: np.ndarray | None = None, spec_channel: int = 0) -> None:
        """Redraw the view for the visible windows.

        Args:
            signals: Visible data [C, samples].
            ylims: ``(center, spread)`` per channel; channels without one use their min-max.
            sample_rate: Sample rate in Hz.
            labels: Scorer name -> states of the visible windows.
            win_s: Window length in seconds.
            confidence: Model confidence of the visible windows, or None.
            spec_channel: Channel shown in the spectrogram.
        """
        n_samples = signals.shape[-1]
        t = np.arange(n_samples) / sample_rate
        for i, ln in enumerate(self.lines):
            if i >= signals.shape[0]:
                break
            te, de = get_visual_envelope(t, signals[i])
            ln.set_data(te, de)
            if i < len(ylims):
                c, s = ylims[i]
                ln.axes.set_ylim(c - s / 2, c + s / 2)
            else:
                lo, hi = float(np.min(signals[i])), float(np.max(signals[i]))
                ln.axes.set_ylim(lo, hi if hi > lo else lo + 1)
        self.signal_axs[0].set_xlim(0, n_samples / sample_rate)
        n_win = len(next(iter(labels.values()))) if labels else 0
        self._set_grid(n_win, win_s)

        # states
        for ln in self.state_lines.values():
            ln.set_visible(False)
        if n_win == 1 and len(labels) == 1:
            v = int(next(iter(labels.values()))[0])
            self.label_text.set_text(f"State: {STATE_NAMES.get(v, str(v))}")
            self.label_ax.set_ylim(-0.5, 1.5)
            self.label_ax.set_yticks([])
        else:
            self.label_text.set_text("")
            self.label_ax.set_ylim(-0.5, 4.5)
            self.label_ax.set_yticks(STATE_TICKS)
            self.label_ax.set_yticklabels([STATE_NAMES[k][:5] for k in STATE_TICKS], fontsize=8)
            for name, vals in labels.items():
                ln = self.state_lines.get(name)
                if ln is None or len(vals) == 0:
                    continue
                ln.set_data(*_state_steps(vals, win_s))
                ln.set_visible(True)
        if confidence is not None and len(confidence) == n_win and n_win:
            self.conf_line.set_data(*_state_steps(confidence, win_s))
            self.conf_ax.set_visible(True)
        else:
            self.conf_ax.set_visible(False)

        if self.spec_ax is not None and spec_channel < signals.shape[0]:
            self._update_spectrogram(signals[spec_channel], sample_rate)

    def _set_grid(self, n_win: int, win_s: float) -> None:
        """Draw a minor grid line per window and major lines every ~1/8 of the view.

        Only redone when the number of windows shown changes.
        """
        key = (n_win, round(win_s, 6))
        if key == self._grid_key:
            return
        self._grid_key = key
        xax = self.signal_axs[0].xaxis
        if n_win > 1:
            xax.set_minor_locator(mticker.MultipleLocator(win_s))
            xax.set_major_locator(mticker.MultipleLocator(max(1, round(n_win / 8)) * win_s))
            for ax in self.axs:
                ax.grid(True, which="minor", axis="x", alpha=0.12)
                ax.grid(True, which="major", axis="x", alpha=0.30)
        else:
            xax.set_minor_locator(mticker.NullLocator())
            xax.set_major_locator(mticker.AutoLocator())
            for ax in self.axs:
                ax.grid(False, which="minor", axis="x")

    def _update_spectrogram(self, x: np.ndarray, fs: float, fmax: float = 30.0) -> None:
        """Draw a 1-s-segment spectrogram of ``x`` up to ``fmax`` Hz (dB, 5-99% colour range)."""
        from scipy.signal import spectrogram

        nper = int(min(len(x), fs))  # 1-s segments, 50 % overlap
        if nper < 8:
            return
        f, tt, sxx = spectrogram(x.astype(np.float64), fs=fs, nperseg=nper, noverlap=nper // 2)
        keep = f <= fmax
        img = 10 * np.log10(sxx[keep] + 1e-12)
        extent = (0, len(x) / fs, 0, f[keep][-1] if keep.any() else fmax)
        vmin, vmax = np.percentile(img, [5, 99])
        if self.spec_img is None:
            self.spec_img = self.spec_ax.imshow(img, aspect="auto", origin="lower", extent=extent, cmap="viridis",
                                                vmin=vmin, vmax=vmax, interpolation="nearest")
            self.spec_ax.set_ylabel("Hz")
        else:
            self.spec_img.set_data(img)
            self.spec_img.set_extent(extent)
            self.spec_img.set_clim(vmin, vmax)


def hypnogram(states: np.ndarray, time_array, state_mapping: dict | None = None, metadata: dict | None = None,
              confidence: np.ndarray | None = None) -> Figure:
    """Return a step hypnogram of a whole recording, optionally with a confidence panel.

    Args:
        states: State code per window.
        time_array: One timestamp per window (``storage.get_timearray_for_states``) or
            plain numbers.
        state_mapping: Code -> name for the y axis.
        metadata: Used for the title (file name, scorer, animal).
        confidence: Model confidence per window, shown below when given.
    """
    state_mapping = state_mapping or {0: "Unknown", 1: "Wake", 2: "NREM", 3: "IS", 4: "REM"}
    metadata = metadata or {}
    states = np.asarray(states)
    fig = Figure(figsize=(15, 5 if confidence is None else 6))
    axs = fig.subplots(2 if confidence is not None else 1, 1, sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]} if confidence is not None else None)
    ax = axs[0] if confidence is not None else axs
    t = np.asarray(time_array)
    ax.step(t, states, where="post", lw=1)
    is_time = t.dtype.kind == "M" or (len(t) and isinstance(t[0], datetime))
    if is_time:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    if len(t):
        ax.set_xlim(t.min(), t.max())
    ax.set_yticks(list(state_mapping))
    ax.set_yticklabels(list(state_mapping.values()))
    ax.grid(True, axis="x", alpha=0.3)
    if confidence is not None:
        axs[1].step(t[: len(confidence)], confidence[: len(t)], where="post", lw=0.8, color="0.4")
        axs[1].set_ylabel("confidence")
        axs[1].set_ylim(0, 1.02)
    (axs[-1] if confidence is not None else ax).tick_params(axis="x", rotation=90)
    parts = [metadata.get(k) or "" for k in ("filename", "scorer", "animal_id")]
    fig.suptitle("hypnogram " + " ".join(p for p in map(str, parts) if p))
    fig.tight_layout()
    return fig


def confusion_figure(res: dict, name_a: str = "A", name_b: str = "B") -> Figure:
    """Return a row-normalised confusion matrix figure for a ``compare_states`` result."""
    cm = np.asarray(res["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = Figure(figsize=(5.5, 5))
    ax = fig.subplots()
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = res["names"]
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel(f"B: {name_b}"[:60])
    ax.set_ylabel(f"A: {name_a}"[:60])
    ax.set_title(f"agreement: acc {res['accuracy']:.3f}, kappa {res['kappa']:.3f} (rows normalised)", fontsize=9)
    fig.tight_layout()
    return fig
