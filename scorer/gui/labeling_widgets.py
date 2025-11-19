from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel,
    QFileDialog, QSlider, QRadioButton, 
    QButtonGroup, QCheckBox, QSizePolicy
)
from PyQt5 import QtCore

import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from gui.plots import (
    plot_signals_init, plot_signals_update,
    plot_spectrogram_init, plot_spectrogram_update,
    plot_fourier_init, plot_fourier_update
)
from data.storage import construct_paths, save_pickled_states, load_pickled_states
from data.loaders import SleepSignals


class SleepGUI(QWidget):
    """
    Sleep labeling GUI with efficient persistent plotting (no canvas recreation).
    """

    def __init__(self, dataset: SleepSignals = None, metadata: dict = None) -> None:
        super().__init__()

        self.params = metadata or {}

        # paths / metadata
        self.score_save_path = ''
        self.data_path = ''
        self.metadata = {}

        # plotting / artist attributes
        self.signal_fig = None
        self.signal_axs = None
        self.signal_lines = None
        self.label_ax = None
        self.label_line = None
        self.label_text = None

        self.signal_canvas = None

        self.spect_fig = None
        self.spect_ax = None
        self.spect_img = None
        self.spect_canvas = None

        # data tracking
        self.dataset = dataset
        self._len_dataset = len(dataset) if dataset is not None else 0
        self.current_idx = 0
        self.scale = 1

        # ylims: stored as (center, spread) for each channel
        self.ylim_defaults = [(0, 1)]
        self.ylims = list(self.ylim_defaults)
        self.yscale = 1.0

        # annotation array and settings
        self.states = np.array([], dtype=int)
        self.label_whole_screen = False

        # small cache for last requested window (idx, scale)
        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}

        # update guard
        self._updating = False

        # IMPORTANT: UI
        layout = QVBoxLayout(self)

        # canvas area for matplotlib plots
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        # labeling layout
        self.labeling_layout = QHBoxLayout()
        layout.addLayout(self.labeling_layout)

        # controls
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        # sliders
        slider_layout = QVBoxLayout()
        layout.addLayout(slider_layout)

        # navigation and control buttons
        btn_load = QPushButton("load file")
        btn_load.clicked.connect(self.select_dataset)
        control_layout.addWidget(btn_load)

        btn_save_states = QPushButton("save annotated sleep states")
        btn_save_states.clicked.connect(self.save_states)
        btn_save_states.setShortcut("Ctrl+s")
        control_layout.addWidget(btn_save_states)

        btn_load_states = QPushButton("load saved states")
        btn_load_states.clicked.connect(self.load_states)
        control_layout.addWidget(btn_load_states)

        btn_prev = QPushButton("prev frame (<)")
        btn_prev.clicked.connect(self.prev)
        btn_prev.setShortcut("<")
        control_layout.addWidget(btn_prev)

        btn_next = QPushButton("next frame (>)")
        btn_next.clicked.connect(self.next)
        btn_next.setShortcut(">")
        control_layout.addWidget(btn_next)

        btn_jump = QPushButton("jump to frame")
        btn_jump.clicked.connect(self.jump)
        control_layout.addWidget(btn_jump)

        btn_plus = QPushButton("increase time scale (+)")
        btn_plus.clicked.connect(self.increase_scale)
        btn_plus.setShortcut("+")
        control_layout.addWidget(btn_plus)

        btn_minus = QPushButton("decrease time scale (-)")
        btn_minus.clicked.connect(self.decrease_scale)
        btn_minus.setShortcut("-")
        control_layout.addWidget(btn_minus)

        btn_y_plus = QPushButton("+ y scale")
        btn_y_plus.clicked.connect(self.increase_yscale)
        control_layout.addWidget(btn_y_plus)

        btn_y_minus = QPushButton("- y scale")
        btn_y_minus.clicked.connect(self.decrease_yscale)
        control_layout.addWidget(btn_y_minus)

        btn_reset = QPushButton("reset")
        btn_reset.clicked.connect(self.reset_settings)
        control_layout.addWidget(btn_reset)

        # labeling radio buttons
        self.labeling_group = QButtonGroup()
        labeling_buttons = [
            QRadioButton('Unknown: 0'),
            QRadioButton('Awake: 1'),
            QRadioButton('NREM: 2'),
            QRadioButton('IS: 3'),
            QRadioButton('REM: 4')
        ]
        for i, b in enumerate(labeling_buttons):
            self.labeling_group.addButton(b, id=i)
            b.setShortcut(str(i))
            self.labeling_layout.addWidget(b)
        self.labeling_group.setExclusive(True)
        self.labeling_group.buttonClicked[int].connect(self.label_data)

        whole_screen_check = QCheckBox("Label the whole screen?")
        whole_screen_check.stateChanged.connect(self.label_screen_toggle)
        self.labeling_layout.addWidget(whole_screen_check)

        # sliders
        self.slider_yscale = QSlider(value=100, minimum=5, maximum=195, singleStep=5)
        self.slider_yscale.setOrientation(QtCore.Qt.Horizontal)
        self.slider_yscale.setTracking(False)
        self.slider_yscale.valueChanged.connect(self.change_yscale)
        self.yscale_label = QLabel("Y scale: 1")
        slider_layout.addWidget(self.yscale_label)
        slider_layout.addWidget(self.slider_yscale)

        self.slider_frame = QSlider(value=0, minimum=0, maximum=max(0, self._len_dataset - 1), singleStep=1)
        self.slider_frame.setOrientation(QtCore.Qt.Horizontal)
        self.slider_frame.setTracking(False)
        self.slider_frame.valueChanged.connect(self.frame_slider_func)
        self.slider_frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.slider_frame_label)
        slider_layout.addWidget(self.slider_frame)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    # DATA LOADING
    def select_dataset(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption = "select data file to load",
            directory = self.params.get('project_path', '.'),
            filter = "Data files (*.pt *.pkl)"
        )
        if not file_name:
            return

        # load dataset and pre-compute a few things
        self.data_path = file_name
        score_path, self.score_save_path = construct_paths(self.data_path)

        # create dataset object by passing path
        self.dataset = SleepSignals(
            data_path=self.data_path,
            score_path=score_path,
            augment=False,
            spectral_features=self.params.get('spectral_view', None),
            metadata=self.params
        )

        # initialize states and indexing
        self.states = self.dataset.all_labels.cpu().numpy()
        self.current_idx = 0

        # store ylims and defaults
        self.ylims = list(self.dataset.channel_ylims)
        self.ylim_defaults = list(self.dataset.channel_ylims)

        # store dataset length to avoid repeated calls
        self._len_dataset = len(self.dataset)

        # set slider ranges
        if self._len_dataset > 0:
            self.slider_frame.setMaximum(self._len_dataset - 1)
            self.slider_frame.setValue(0)
        else:
            self.slider_frame.setMaximum(0)
            self.slider_frame.setValue(0)

        # clear cached plotting objects so they'll be re-initialized (basically if reloading)
        for c in (self.signal_canvas, self.spect_canvas):
            if c is not None:
                try:
                    c.setParent(None)
                except Exception:
                    pass

        self.signal_fig = None
        self.signal_axs = None
        self.signal_lines = None
        self.label_ax = None
        self.label_line = None
        self.label_text = None
        self.signal_canvas = None

        self.spect_fig = None
        self.spect_ax = None
        self.spect_img = None
        self.spect_canvas = None

        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}

        # initial plot
        self.update_screen()
        self.status_label.setText(f"Loaded: {file_name}")

    # DATA RETRIEVAL
    def _to_numpy_sample(self, sample):
        """Convert a sample (tensor or tuple) to numpy arrays (cpu)."""
        if isinstance(sample, tuple):
            s0 = sample[0].cpu().numpy()
            s1 = sample[1].cpu().numpy() if sample[1] is not None else None
            return (s0, s1)
        else:
            return sample.cpu().numpy()

    def get_data(self):
        """
        Return (sample_numpy, labels) for the current window (idx..idx+scale).
        Caches results for identical (idx, scale).
        sample_numpy is either array (channels, time) or tuple (signals_np, spect_np).
        """
        idx, scale = self.current_idx, self.scale
        if self._len_dataset == 0:
            return np.zeros((1, 1)), [0]

        end = min(idx + scale, self._len_dataset)

        # return from cache if available
        if self._plot_cache["idx"] == idx and self._plot_cache["scale"] == scale \
        and self._plot_cache["sample"] is not None \
        and self._plot_cache["label"] is not None:
            return self._plot_cache["sample"], self._plot_cache["label"]
        
        samples = []
        spects = []
        labels = []
        last_is_tuple = False

        for i in range(idx, end):
            sample = self.dataset[i][0]
            labels.append(int(self.states[i]))
            if isinstance(sample, tuple):
                last_is_tuple = True
                samples.append(sample[0])
                spects.append(sample[1])
            else:
                samples.append(sample)

        # single concat
        if last_is_tuple:
            signals = torch.cat(samples, dim=-1) if len(samples) > 1 else samples[0]
            spect_concat = torch.cat(spects, dim=-1) if len(spects) > 1 else (spects[0] if spects else None)
            sample_cpu = (signals.cpu().numpy(), spect_concat.cpu().numpy() if spect_concat is not None else None)
            self._plot_cache.update({"idx": idx, "scale": scale, "sample": sample_cpu, "label": labels})
            return sample_cpu, labels
        else:
            signals = torch.cat(samples, dim=-1) if len(samples) > 1 else samples[0]
            sample_cpu = signals.cpu().numpy()
            self._plot_cache.update({"idx": idx, "scale": scale, "sample": sample_cpu, "label": labels})
            return sample_cpu, labels

    # LABELING
    def label_data(self, value) -> None:
        if self.label_whole_screen:
            end = min(self.current_idx + self.scale, self._len_dataset)
            self.states[self.current_idx:end] = value
        else:
            self.states[self.current_idx] = value
            
        self._plot_cache["label"] = None
        self.update_screen()

    def label_screen_toggle(self) -> None:
        self.label_whole_screen = not self.label_whole_screen

    # UPDATE UI
    def update_screen(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.update_plot()
            self.update_labels()
        finally:
            self._updating = False

    def update_labels(self) -> None:
        self.status_label.setText(
            f"frame index: {self.current_idx}, time scale: {self.scale} frames; y scale: {self.yscale}"
        )

        if self.slider_frame.value() != self.current_idx:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(self.current_idx)
            self.slider_frame.blockSignals(False)
        self.slider_frame_label.setText(f"frame: {self.current_idx}")

        yval = int(self.yscale * 100)
        if self.slider_yscale.value() != yval:
            self.slider_yscale.blockSignals(True)
            self.slider_yscale.setValue(yval)
            self.slider_yscale.blockSignals(False)
        self.yscale_label.setText(f"Y scale: {self.yscale:.3g}")

        # labeling group
        state_idx = int(self.states[self.current_idx])
        btn = self.labeling_group.button(state_idx)
        if btn and not btn.isChecked():
            self.labeling_group.blockSignals(True)
            btn.setChecked(True)
            self.labeling_group.blockSignals(False)

    # PLOT UPDATES
    def update_plot(self) -> None:
        """
        Efficiently update existing artists (lines, image, text).
        If first call, create figures & canvases via the *_init helpers.
        """
        sample, labels = self.get_data()

        # number of samples and time axis (based on visible window)
        n_samples = sample[0].shape[-1] if isinstance(sample, tuple) else sample.shape[-1]
        sample_rate = int(self.params.get('sample_rate', 250))
        time_axis = np.arange(n_samples) / sample_rate

        # --- SIGNALS update/init ---
        signals = sample[0] if isinstance(sample, tuple) else sample

        # initialize signal figure and artists on first call
        if self.signal_fig is None:
            (
                self.signal_fig,
                self.signal_axs,
                self.signal_lines,
                self.label_line,
                self.label_text
            ) = plot_signals_init(
                signals,
                [(c - s / 2, c + s / 2) for c, s in self.ylims],
                sample_rate,
                self.params.get('channels_after_preprocessing', [])
            )

            # create and configure canvas
            self.signal_canvas = FigureCanvas(self.signal_fig)
            self.signal_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.signal_canvas.updateGeometry()
            self.canvas_layout.addWidget(self.signal_canvas)
            self.signal_fig.tight_layout()

        # update artists (lines, label area)
        plot_signals_update(
            signals,
            labels,
            self.signal_lines,
            self.signal_axs[-1],      # label axis is last one
            self.label_line,
            self.label_text,
            [(c - s / 2, c + s / 2) for c, s in self.ylims],
            sample_rate
        )
        self.signal_canvas.draw_idle()

        # --- SPECTRAL (spectrogram or fourier) ---
        if isinstance(sample, tuple) and self.params.get('spectral_view'):
            spect = sample[1]

            # initialize spectral figure if needed
            if self.spect_fig is None:
                if self.params['spectral_view'] == 'spectrogram':
                    self.spect_fig, self.spect_ax, self.spect_img = plot_spectrogram_init(
                        spect if isinstance(spect, np.ndarray) else spect.cpu().numpy()
                    )
                else:
                    self.spect_fig, self.spect_ax, self.spect_img = plot_fourier_init(
                        spect if isinstance(spect, np.ndarray) else spect.cpu().numpy()
                    )

                self.spect_canvas = FigureCanvas(self.spect_fig)
                self.spect_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.spect_canvas.updateGeometry()
                self.canvas_layout.addWidget(self.spect_canvas)
                self.spect_fig.tight_layout()

            # update spectra (pass numpy arrays to update helpers)
            spect_np = spect if isinstance(spect, np.ndarray) else spect.cpu().numpy()

            if self.params['spectral_view'] == 'spectrogram':
                plot_spectrogram_update(self.spect_img, self.spect_ax, spect_np, time_axis)
            else:
                plot_fourier_update(self.spect_img, spect_np)

            self.spect_canvas.draw_idle()

    # SAVE/LOAD SCORES
    def save_states(self) -> None:
        _, self.score_save_path = construct_paths(self.data_path)
        save_pickled_states(self.states, self.score_save_path)
        self.status_label.setText(f"Saved states to {self.score_save_path}")

    def load_states(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select states file to load",
            directory=".",
            filter="Pickle files (*.pkl)"
        )
        if file_name:
            self.states = load_pickled_states(file_name)
            self.update_screen()

    # NAVIGATE
    def _set_idx_and_update(self, idx: int) -> None:
        if self._len_dataset == 0:
            return
        idx = max(0, min(idx, self._len_dataset - 1))
        if idx == self.current_idx:
            return
        self.current_idx = idx
        self.update_screen()

    def next(self) -> None:
        self._set_idx_and_update(self.current_idx + 1)

    def prev(self) -> None:
        self._set_idx_and_update(self.current_idx - 1)

    def jump(self) -> None:
        if self._len_dataset == 0:
            return
        idx, ok = QInputDialog.getInt(self, "Jump", "Enter frame index:", value=self.current_idx)
        if ok:
            self._set_idx_and_update(idx)

    def frame_slider_func(self, value: int) -> None:
        if 0 <= value < max(1, self._len_dataset):
            self._set_idx_and_update(value)

    # CHANGE Y SCALE
    def set_yscale(self, s: float) -> None:
        new_yscale = max(float(s), 0.001)
        if abs(new_yscale - self.yscale) < 1e-9:
            return
        self.yscale = new_yscale
        self.ylims = [(center, spread * self.yscale) for center, spread in self.ylim_defaults]
        self.update_screen()

    def change_yscale(self, value: int) -> None:
        self.set_yscale(value * 0.01)

    def increase_yscale(self) -> None:
        self.set_yscale(self.yscale + 0.05)

    def decrease_yscale(self) -> None:
        self.set_yscale(self.yscale - 0.05)

    # CHANGE TIME SCALE
    def increase_scale(self) -> None:
        self.scale += 1
        self._plot_cache["idx"] = None  # invalidate cache
        self.update_screen()

    def decrease_scale(self) -> None:
        if self.scale > 1:
            self.scale -= 1
            self._plot_cache["idx"] = None
            self.update_screen()

    # RESET ALL
    def reset_settings(self) -> None:
        self.scale = 1
        self.ylims = list(self.ylim_defaults)
        self.yscale = 1.0
        self.label_whole_screen = False
        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}
        self.update_screen()
