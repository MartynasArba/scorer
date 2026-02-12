from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel,
    QFileDialog, QSlider, QRadioButton, 
    QButtonGroup, QCheckBox, QSizePolicy
)
from PyQt5 import QtCore
from PyQt5.QtCore import QFileInfo
import matplotlib.ticker as mticker

import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math

from scorer.gui.plots import (
    plot_signals_init, plot_signals_update,
    plot_spectrogram_init, plot_spectrogram_update,
    plot_fourier_init, plot_fourier_update, TimeOfDayFormatter
)
from scorer.data.storage import construct_paths, save_pickled_states, load_pickled_states
from scorer.data.loaders import SleepSignals


class SleepGUI(QWidget):
    """
    Sleep labeling GUI with plotting (efficient, no canvas recreation).
    """

    def __init__(self, dataset: SleepSignals = None, metadata: dict = None) -> None:
        """
        initializes manual labeling tab in GUI
        """
        super().__init__()

        # paths / metadata
        self.score_save_path = ''
        self.data_path = ''
        self.params = metadata or {}
        self.active_scorer = self.params.get('scorer', 'unknown_scorer')
        self.load_folder = False

        # plotting / artist attributes
        self.signal_fig = None
        self.signal_axs = None
        self.signal_lines = None
        self.label_lines = None
        self.label_text = None

        self.signal_canvas = None

        self.spect_fig = None
        self.spect_ax = None
        self.spect_img = None
        self.spect_canvas = None
        
        self._mouse_canvas = None
        self._mouse_cids = []
        
        self.show_all_scorers = False
        self.active_label_value = 0 #for mouse labeling, tracks current label ID, defaults to 0

        # data tracking
        self.dataset = dataset
        self._len_dataset = len(dataset) if dataset is not None else 0
        self.current_idx = 0
        self.scale = 1

        # ylims: stored as (center, spread) for each channel
        self.ylim_defaults = [(0, 1)]
        self.ylims = list(self.ylim_defaults)
        self.yscale = 1.0
        self.yscale_log_min = math.log(0.1)   # for log slider
        self.yscale_log_max = math.log(10.0)  
        

        # annotation array and settings
        self.states = np.array([], dtype=int)
        # self.label_whole_screen = False
        self.passive_scorers = {}   #other scorers that are displayed but can't be interacted with. name:state_arr structure

        # small cache for last requested window (idx, scale)
        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}
        self._sep_cache_key = None  #cache of grid lines

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
        load_folder_check = QCheckBox("Load multiple files from folder?")
        load_folder_check.stateChanged.connect(self.load_folder_toggle)
        control_layout.addWidget(load_folder_check)
        
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
        
        btn_load_other_scorers = QPushButton("load non-interactable annotations")
        btn_load_other_scorers.clicked.connect(self.load_passive_scorer)
        control_layout.addWidget(btn_load_other_scorers)
        
        btn_prev_state = QPushButton("<< prev state")
        btn_prev_state.clicked.connect(self.prev_change)
        btn_prev_state.setShortcut("Shift+Left")
        control_layout.addWidget(btn_prev_state)

        btn_prev = QPushButton("< prev frame")
        btn_prev.clicked.connect(self.prev)
        btn_prev.setShortcut("Left")
        control_layout.addWidget(btn_prev)


        btn_jump = QPushButton("jump to frame")
        btn_jump.clicked.connect(self.jump)
        control_layout.addWidget(btn_jump)
        
        btn_next = QPushButton("next frame >")
        btn_next.clicked.connect(self.next)
        btn_next.setShortcut("Right")
        control_layout.addWidget(btn_next)

        btn_next_state = QPushButton("next state >>")
        btn_next_state.clicked.connect(self.next_change)
        btn_next_state.setShortcut("Shift+Right")
        control_layout.addWidget(btn_next_state)
        
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

        # whole_screen_check = QCheckBox("Label the whole screen?")
        # whole_screen_check.stateChanged.connect(self.label_screen_toggle)
        # self.labeling_layout.addWidget(whole_screen_check)
        
        all_scorers_check = QCheckBox("Show all loaded scorers?")
        all_scorers_check.stateChanged.connect(self.show_all_scorers_toggle)
        self.labeling_layout.addWidget(all_scorers_check)
        
        # sliders
        self.slider_yscale = QSlider(QtCore.Qt.Horizontal)
        self.slider_yscale.setMinimum(0)
        self.slider_yscale.setMaximum(1000)
        self.slider_yscale.setSingleStep(10)
        self.slider_yscale.valueChanged.connect(self.change_yscale)
        self.yscale_label = QLabel("Y scale: 1")
        slider_layout.addWidget(self.yscale_label)
        slider_layout.addWidget(self.slider_yscale)

        self.slider_frame = QSlider(value=0, minimum=0, maximum=max(0, self._len_dataset - 1), singleStep=1, pageStep = 8)
        self.slider_frame.setOrientation(QtCore.Qt.Horizontal)
        self.slider_frame.setTracking(False)
        self.slider_frame.valueChanged.connect(self.frame_slider_func)
        self.slider_frame_label = QLabel("Min from start: 0")
        slider_layout.addWidget(self.slider_frame_label)
        slider_layout.addWidget(self.slider_frame)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
    def load_folder_toggle(self):
        self.load_folder = not self.load_folder

    # DATA LOADING
    def select_dataset(self) -> None:
        """
        selects and loads dataset in GUI, precomputes a few things, creates matplotlib canvas
        """
        if self.load_folder:
            file_name = QFileDialog.getExistingDirectory(self,
                                                      caption="Or select a folder with partial chunk files",
                                                      directory=self.params.get('project_path', '.'))
        elif not self.load_folder:
            file_name, _ = QFileDialog.getOpenFileName(self,
                                                       caption="Select file to preprocess",
                                                       directory=self.params.get('project_path', '.'),
                                                       filter="Data files (*.pt *.npy)")
        else:
            print('not file, not folder - weird')
            
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
        
        sample_rate = int(self.params.get("sample_rate", 250))
        sample0 = self.dataset[0][0]    #get first sample
        signals0 = sample0[0] if isinstance(sample0, tuple) else sample0
        self.frame_n_samples = int(signals0.shape[-1])  #get frame sample numers - win len
        self.frame_duration_s = self.frame_n_samples / sample_rate  # copnvert window len to seconds
        #parse rec start
        start_ts = self.params.get("rec_start", '2025-11-26 19:10:39.127974128')  # required
        dt = _parse_iso(start_ts)
        #store as seconds since midnight
        self.rec_start_sod = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        # initialize states and indexing
        self.states = self.dataset.all_labels.cpu().numpy()
        self.current_idx = 0

        # store ylims and defaults
        self.ylims = list(self.dataset.channel_ylims)
        self.ylim_defaults = list(self.dataset.channel_ylims).copy()
        self.yscale = 1.0
        self.ylims = [(c, s * self.yscale) for (c, s) in self.ylim_defaults]
        
        print(f'loaded ylims: {self.ylims}')
        # store dataset length to avoid repeated calls
        self._len_dataset = len(self.dataset)

        # set slider ranges
        if self._len_dataset > 0:
            self.slider_frame.setMaximum(self._len_dataset - 1)
            self.slider_frame.setValue(0)
        else:
            self.slider_frame.setMaximum(0)
            self.slider_frame.setValue(0)

        # clear cached plotting objects so they'll be re-initialized (if reloading)
        for c in (self.signal_canvas, self.spect_canvas):
            if c is not None:
                try:
                    c.setParent(None)
                except Exception:
                    pass

        self.signal_fig = None
        self.signal_axs = None
        self.signal_lines = None
        self.label_lines = None
        self.label_text = None
        self.signal_canvas = None

        self.spect_fig = None
        self.spect_ax = None
        self.spect_img = None
        self.spect_canvas = None
        
        self._mouse_canvas = None
        self._mouse_cids = []
        self._dragging = False
        self._drag_x0 = None
        self._drag_preview = None
        
        self._sep_cache_key = None

        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}

        # initial plot
        self.update_screen()
        self.status_label.setText(f"Loaded: {file_name}")

    # DATA RETRIEVAL
    def _to_numpy_sample(self, sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """converts a sample (tensor or tuple) to numpy array (cpu)"""
        if isinstance(sample, tuple):
            s0 = sample[0].cpu().numpy()
            s1 = sample[1].cpu().numpy() if sample[1] is not None else None
            return (s0, s1)
        else:
            return sample.cpu().numpy()

    def get_data(self):
        """
        returns (sample_numpy, labels) for current window (idx : idx + scale).
        caches results for identical (idx, scale).
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
    def label_data(self, value: float) -> None:
        
        self.active_label_value = value # update mouse val
        
        #this is commented out as keyboard controlls clash with mouse. Mouse is more convenient, so keeping it
        # if self.label_whole_screen:
        #     end = min(self.current_idx + self.scale, self._len_dataset)
        #     self.states[self.current_idx:end] = value
        # else:
        #     self.states[self.current_idx] = value
        # #clear cache
        # self._plot_cache["label"] = None
        # self._plot_cache["idx"] = None
        # self.update_screen()

    # def label_screen_toggle(self) -> None:
    #     self.label_whole_screen = not self.label_whole_screen
    
    def show_all_scorers_toggle(self) -> None:
        self.show_all_scorers = not self.show_all_scorers
        self.update_screen()
        
    #MOUSE CONTROLLS
    def _install_mouse_labeling(self):
        """connects mouse controls to current signal_canvas"""
        if self.signal_canvas is None:
            return

        # disconnect from old canvas if it changed
        if self._mouse_canvas is not None and self._mouse_canvas is not self.signal_canvas:
            try:
                for cid in self._mouse_cids:
                    self._mouse_canvas.mpl_disconnect(cid)
            except Exception:
                pass
            self._mouse_cids = []

        # already connected to this canvas
        if self._mouse_canvas is self.signal_canvas and self._mouse_cids:
            return
        
        #connect
        self._mouse_canvas = self.signal_canvas
        self._mouse_cids = [
            self.signal_canvas.mpl_connect("button_press_event", self._on_mouse_press),
            self.signal_canvas.mpl_connect("motion_notify_event", self._on_mouse_move),
            self.signal_canvas.mpl_connect("button_release_event", self._on_mouse_release),
            self.signal_canvas.mpl_connect("scroll_event", self._on_scroll), 
        ]

        # reset drag state
        self._dragging = False
        self._drag_x0 = None
        self._drag_preview = None    
        
    def _x_to_global_frame(self, x_seconds: float) -> int:
        """converts mouse x to frame ID"""
        if x_seconds is None or self._len_dataset == 0:
            return self.current_idx
        #window goes from 0 to scale * frame_duration s
        win_offset = int(math.floor(x_seconds / max(self.frame_duration_s, 1e-9)))
        win_offset = max(0, min(win_offset, self.scale - 1))
        
        sel_window = self.current_idx + win_offset
        sel_window = max(0, min(sel_window, self._len_dataset - 1))
        return sel_window
    
    def _frames_to_x_span(self, i0: int, i1: int) -> tuple[float, float]:
        """        
        converts frame idx to x in seconds in current window
        returns (x_start, x_end) in seconds for preview shading
        """
        # clamp to current window
        w0 = self.current_idx
        w1 = min(self.current_idx + self.scale - 1, self._len_dataset - 1)

        i0 = max(w0, min(i0, w1))
        i1 = max(w0, min(i1, w1))
        if i1 < i0:
            i0, i1 = i1, i0

        # local frame offsets
        l0 = i0 - self.current_idx
        l1 = i1 - self.current_idx

        x0 = l0 * self.frame_duration_s
        x1 = (l1 + 1) * self.frame_duration_s  # end at end of last frame
        return x0, x1
       
    def _on_mouse_press(self, event):
        if event.button != 1:
            return
        if event.inaxes is None:
            return
        if self.signal_axs is None or len(self.signal_axs) == 0:
            return

        label_ax = self.signal_axs[-1]
        if event.inaxes != label_ax:
            return

        self._dragging = True
        self._drag_x0 = event.xdata

        # remove old preview
        if getattr(self, "_drag_preview", None) is not None:
            try:
                self._drag_preview.remove()
            except Exception:
                pass
            self._drag_preview = None

        x0 = float(event.xdata) if event.xdata is not None else 0.0
        #gives a tiny non-zero width so it’s a rectangle
        self._drag_preview = label_ax.axvspan(x0, x0 + 1e-9, alpha=0.2)
        self.signal_canvas.draw_idle()


    def _on_mouse_move(self, event):
        if not getattr(self, "_dragging", False):
            return
        if event.inaxes is None:
            return
        if self.signal_axs is None or len(self.signal_axs) == 0:
            return

        label_ax = self.signal_axs[-1]
        if event.inaxes != label_ax:
            return

        if self._drag_x0 is None or event.xdata is None:
            return
        if getattr(self, "_drag_preview", None) is None:
            return

        x0 = float(self._drag_x0)
        x1 = float(event.xdata)
        left = min(x0, x1)
        right = max(x0, x1)

        try:
            self._drag_preview.set_x(left)
            self._drag_preview.set_width(max(right - left, 1e-12))
        except Exception:
            # fallback: recreate preview if patch type isn't compatible
            try:
                self._drag_preview.remove()
            except Exception:
                pass
            self._drag_preview = label_ax.axvspan(left, right, alpha=0.2)

        self.signal_canvas.draw_idle()


    def _on_mouse_release(self, event):
        if not getattr(self, "_dragging", False):
            return
        self._dragging = False

        if self.signal_axs is None or len(self.signal_axs) == 0:
            return

        # remove preview
        if getattr(self, "_drag_preview", None) is not None:
            try:
                self._drag_preview.remove()
            except Exception:
                pass
            self._drag_preview = None

        x0 = self._drag_x0
        self._drag_x0 = None

        if x0 is None:
            self.signal_canvas.draw_idle()
            return

        x1 = event.xdata if (event is not None and event.xdata is not None) else x0

        # map to global frame indices
        i0 = self._x_to_global_frame(x0)
        i1 = self._x_to_global_frame(x1)
        if i1 < i0:
            i0, i1 = i1, i0

        val = int(getattr(self, "active_label_value", 0))
        self.states[i0:i1 + 1] = val

        self._plot_cache["label"] = None
        self._plot_cache["idx"] = None
        self.update_screen()
        
    def _on_scroll(self, event):
        """
        mouse wheel scroll:
        - up is previous frame
        - down is next frame
        """
        if self._len_dataset == 0:
            return

        if event.step > 0:
            new_idx = self.current_idx - 1
        else:
            new_idx = self.current_idx + 1

        new_idx = max(0, min(new_idx, self._len_dataset - 1))

        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.update_screen()

        
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
        self.slider_frame_label.setText(f"Min from start: {(self.current_idx * self.frame_duration_s / 60):.2f}")

        slider_val = self._yscale_to_slider(self.yscale)
        if self.slider_yscale.value() != slider_val:
            self.slider_yscale.blockSignals(True)
            self.slider_yscale.setValue(slider_val)
            self.slider_yscale.blockSignals(False)
        self.yscale_label.setText(f"Y scale: ×{self.yscale:.2f}")

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
        # build scorer dict
        all_scorers = {self.active_scorer: self.states}
        if self.show_all_scorers:
            all_scorers.update(self.passive_scorers)

        # get sample + labels for visible window
        sample, _ = self.get_data()

        # extract signals from either form
        if isinstance(sample, tuple):
            signals = sample[0]
            spect = sample[1]
        else:
            signals = sample
            spect = None  # no spectral data

        n_samples = signals.shape[-1]
        sample_rate = int(self.params.get('sample_rate', 250))
        time_axis = np.arange(n_samples) / sample_rate

        # initialize signal figure and artists if not initialized
        if self.signal_fig is None:
            self.time_formatter = TimeOfDayFormatter(self.rec_start_sod, window_offset_s = 0.0)
            (   self.signal_fig,
                self.signal_axs,
                self.signal_lines,
                self.label_lines,
                self.label_text
                ) = plot_signals_init(
                    signals,
                    self.ylims,
                    sample_rate,
                    self.params.get('channels_after_preprocessing', []),
                    scorer_names = list(all_scorers.keys()),
                    time_formatter=self.time_formatter
            )

            # create and configure canvas
            self.signal_canvas = FigureCanvas(self.signal_fig)
            self.signal_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.signal_canvas.updateGeometry()
            self.canvas_layout.addWidget(self.signal_canvas)
            # self.signal_fig.tight_layout()
            #also add mouse labeling from signalcanvas
            self._install_mouse_labeling()
            
        # convert scorer label arrays for the visible window
        scorer_window_labels = {}
        idx0 = self.current_idx
        idx1 = min(self.current_idx + self.scale, self._len_dataset)
        for scorer_name, arr in all_scorers.items():
            window_vals = arr[idx0:idx1]
            # map list to scorer name, keep None values untouched
            scorer_window_labels[scorer_name] = [
                (int(x) if x is not None else None)
                for x in window_vals
            ]
        
        #GRID LINES
        # infer how many windows are displayed from any scorer's visible labels
        if scorer_window_labels:
            n_windows = len(next(iter(scorer_window_labels.values())))
        else:
            n_windows = 0

        sep_key = (n_windows, n_samples, sample_rate)
        if self._sep_cache_key != sep_key:
            self._sep_cache_key = sep_key

            # shared x axis: set locators once on any shared axis (use first signal axis)
            xax = self.signal_axs[0].xaxis

            if n_windows > 1:
                win_s = (n_samples / sample_rate) / n_windows  # seconds per frame
                # minor ticks exactly at frame boundaries
                minor = mticker.MultipleLocator(win_s)
                # major ticks every k frames (k*win_s)
                target_major_lines = 8
                k = max(1, int(round(n_windows / target_major_lines)))
                major = mticker.MultipleLocator(k * win_s)
                xax.set_minor_locator(minor)
                xax.set_major_locator(major)
                # apply grids to ALL axes (signals + label)
                for ax in self.signal_axs:
                    ax.grid(True, which="minor", axis="x", alpha=0.12)
                    ax.grid(True, which="major", axis="x", alpha=0.30)

            else:
                # disable minor grid when scale==1
                xax.set_minor_locator(mticker.NullLocator())
                for ax in self.signal_axs:
                    ax.grid(False, which="minor", axis="x")

        #update window offset
        window_offset_s = self.current_idx * self.frame_duration_s
        self.time_formatter.set_window_offset(window_offset_s)

        # update artists (lines, label area)
        plot_signals_update(signals, 
                            self.signal_lines,
                            self.signal_axs[-1],      # label axis is last one
                            self.label_lines,
                            self.label_text,
                            self.ylims,
                            sample_rate,
                            labels = scorer_window_labels)
        self.signal_canvas.draw_idle()

        # spectrograms/fourier
        if isinstance(sample, tuple) and (self.params.get('spectral_view') in ('spectrogram', 'fourier')):
            spect = sample[1]
            if spect is None:   #safety check if spectral view is set but spect is not available
                return

            # initialize spectral figure if not available
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

            # update plots
            spect_np = spect if isinstance(spect, np.ndarray) else spect.cpu().numpy()

            if self.params['spectral_view'] == 'spectrogram':
                plot_spectrogram_update(self.spect_img, self.spect_ax, spect_np, time_axis)
            else:
                plot_fourier_update(self.spect_img, spect_np)

            self.spect_canvas.draw_idle()

    # SAVE/LOAD SCORES
    def save_states(self) -> None:
        _, self.score_save_path = construct_paths(self.data_path, add = str(self.current_idx))
        save_pickled_states(self.states, self.score_save_path)
        self.status_label.setText(f"Saved states to {self.score_save_path}")

    def load_states(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption = "Select states file to load",
            directory = self.params.get('project_path', '.'),
            filter = "Pickle files (*.pkl)"
        )
        if file_name:
            loaded = load_pickled_states(file_name)
            self.states = np.array(loaded, dtype=int)
            
            if self._len_dataset and len(self.states) != self._len_dataset:
                self.status_label.setText("Loaded states length does not match dataset!")
                print("Loaded states length does not match dataset!")
            self.update_screen()
            
    def load_passive_scorer(self) -> None:
        
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            caption = "load other scorer labels",
            directory = self.params.get('project_path', '.'),
            filter = "Pickle files (*.pkl)"
        )
        if not file_name:
            return
        
        labels = load_pickled_states(file_name)
        if len(labels) != self._len_dataset:
            self.status_label.setText("Scorer file length does not match dataset!")
            return

        scorer_name = QFileInfo(file_name).baseName()   #set scorer name same as file
        self.passive_scorers[scorer_name] = np.array(labels)
        #reset signal_fig so it is created with new scorer
        self.reset_scorers()
        
        # update plots
        self.update_screen()
        self.status_label.setText(f"Loaded passive scorer: {scorer_name}")

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
            
    def _segment_starts(self) -> np.ndarray:
        """
        get idx of state segment starts
        """
        if self._len_dataset == 0:
            return np.array([0], dtype=int)

        s = self.states
        #start of a new segment is where state changes from previous, but 1st always starts from 0
        starts = np.flatnonzero(s[1:] != s[:-1]) + 1
        return np.concatenate(([0], starts)).astype(int)
            
    def next_change(self) -> None:
        """jump to next state change of active scorer"""
        if self._len_dataset == 0:
            return

        starts = self._segment_starts()
        i = int(self.current_idx)

        next_starts = starts[starts > i]    #filter out segment start idxs that are after current idx

        if next_starts.size == 0:
            self.status_label.setText("no next state available")
            return

        self._set_idx_and_update(int(next_starts[0]))

    def prev_change(self) -> None:
        """jump to previous state change"""
        if self._len_dataset == 0:
            return

        starts = self._segment_starts() #same logic as before
        i = int(self.current_idx)
                
        prev_starts = starts[starts < i]#filter out segment starts that are before current idx

        if prev_starts.size == 0:
            self.status_label.setText("no previous state available")
            return

        self._set_idx_and_update(int(prev_starts[-1]))

    # CHANGE Y SCALE
    def _slider_to_yscale(self, slider_val: int) -> float:
        """map slider 0 to 1000 > yscale in log"""
        frac = slider_val / 1000.0
        log_y = self.yscale_log_min + frac * (self.yscale_log_max - self.yscale_log_min)
        return math.exp(log_y)

    def _yscale_to_slider(self, yscale: float) -> int:
        """map yscale to slider position."""
        log_y = math.log(max(yscale, 1e-9))
        frac = (log_y - self.yscale_log_min) / (self.yscale_log_max - self.yscale_log_min)
        return int(np.clip(frac * 1000, 0, 1000))
    
    def set_yscale(self, s: float) -> None:
        new_yscale = max(float(s), 0.001)
        if abs(new_yscale - self.yscale) < 1e-9:
            return
        self.yscale = new_yscale
        self.ylims = [(center, spread * self.yscale) for center, spread in self.ylim_defaults]
        self.update_screen()

    def change_yscale(self, value: int) -> None:
        y = self._slider_to_yscale(value)
        self.set_yscale(y)

    def increase_yscale(self) -> None:
        self.set_yscale(self.yscale * 1.1 )

    def decrease_yscale(self) -> None:
        self.set_yscale(self.yscale / 1.1)

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
        self.yscale = 1.0
        self.ylims = [(center, spread * self.yscale) for center, spread in self.ylim_defaults]    #explicit copy
        
        # self.label_whole_screen = False
        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}
        self.passive_scorers.clear()
        self.reset_scorers()
        self.update_screen()

    def reset_scorers(self):
        """
        resets plots so new ones are initialized, esp. relevant when new plots are created
        """
        #clear cache
        self._plot_cache = {"idx": None, "scale": None, "sample": None, "label": None}
        
        # reset signal_fig 
        if self.signal_canvas is not None:
            self.signal_canvas.setParent(None)   # remove from layout
            self.signal_canvas.deleteLater()
            plt.close(self.signal_fig)
            
        if self.spect_canvas is not None:
            self.spect_canvas.setParent(None)
            self.spect_canvas.deleteLater()
            plt.close(self.spect_fig)


        # clear all figure-related states
        self.signal_fig = None
        self.signal_axs = None
        self.signal_lines = None
        self.label_lines = None
        self.label_text = None
        self.spect_fig = None
        self.spect_canvas = None
        self.spect_ax = None
        self.spect_img = None
        
        #clear mouse canvas and params
        self._mouse_canvas = None
        self._mouse_cids = []
        self._dragging = False
        self._drag_x0 = None
        self._drag_preview = None
        
        self._sep_cache_key = None
        
def _parse_iso(ts: str) -> datetime:
    """
    helper to generate timestamp from metadata rec start
    """
    ts = ts.strip()  # remove \n, spaces

    # truncate fractional seconds to 6 digits if present
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = frac[:6]          # keep microseconds
        ts = f"{head}.{frac}"
    try:
        dt = datetime.fromisoformat(ts)
    except:
        print('datetime conversion failed, setting to default time')
        dt = datetime.now()
    return dt