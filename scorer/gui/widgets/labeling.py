"""
Manual scoring tab (replaces labeling_widgets.SleepGUI).

Kept: the layout, buttons, keyboard shortcuts, mouse drag-labeling on the state axis, wheel
navigation, macro (10-min) view, y-scale slider, extra read-only scorers.
New: RecordingData instead of SleepSignals (CPU numpy, no torch import); StateTrack with undo
(Ctrl+Z), unsaved-change tracking and autosave; model confidence trace + "next uncertain window";
"next unlabeled window"; optional spectrogram and display normalisation; jump by time of day;
state files of the wrong length are fitted only after you confirm.
"""
from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore
from PyQt5.QtCore import QFileInfo, QTimer
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QPushButton,
                             QRadioButton, QShortcut, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget)
from PyQt5.QtGui import QKeySequence

from scorer.gui.core import meta as M
from scorer.gui.core.annotations import (StateTrack, fit_array, fit_states_to, load_confidence_for,
                                        load_states_file)
from scorer.gui.core.plots import SignalView, TimeOfDayFormatter, parse_rec_start
from scorer.gui.core.recording import STATE_NAMES, RecordingData
from scorer.gui.widgets import common as C

log = logging.getLogger(__name__)
AUTOSAVE_MS = 3 * 60 * 1000
MACRO_SECONDS = 600


class LabelingWidget(QWidget):
    """Manual-scoring tab: browse a windowed recording and label it with the mouse or keys.

    Drag across the state axis to label windows with the selected state (keys 0-4 pick
    the state). Arrow keys move, + and - change how many windows are shown, T toggles a
    10-minute view, Ctrl+Z undoes, Ctrl+S saves. Labels are autosaved every 3 minutes to
    one file per recording. Other scorers' files can be shown read-only, and a model's
    confidence can guide review ("next uncertain").

    Args:
        metadata: The shared metadata dict (sample rate, channels, scorer name).
    """
    def __init__(self, metadata: dict | None = None) -> None:
        """Build the controls and shortcuts and start the autosave timer."""
        super().__init__()
        self.params = metadata if metadata is not None else {}
        self.data: RecordingData | None = None
        self.track = StateTrack.empty(0)
        self.passive: dict[str, np.ndarray] = {}
        self.data_path = ""
        self.score_save_path = ""
        self._autosave_path = None
        self.current_idx = 0
        self.scale = 1
        self._prev_scale = 1
        self.frame_duration_s = 4.0
        self.sample_rate = 250.0
        self.rec_start_sod = 0.0
        self.ylim_defaults: list[tuple[float, float]] = []
        self.ylims: list[tuple[float, float]] = []
        self.yscale = 1.0
        self.yscale_log_min, self.yscale_log_max = math.log(0.1), math.log(10.0)
        self.active_label_value = 0
        self.show_all_scorers = False
        self.view: SignalView | None = None
        self.canvas: FigureCanvas | None = None
        self._cids: list[int] = []
        self._drag_x0 = None
        self._drag_preview = None
        self._autosaved_version = 0
        self._updating = False

        layout = QVBoxLayout(self)
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout, 1)

        # state buttons
        lab = QHBoxLayout()
        self.labeling_group = QButtonGroup(self)
        for code in range(5):
            b = QRadioButton(f"{STATE_NAMES[code]}: {code}")
            b.setShortcut(str(code))
            self.labeling_group.addButton(b, code)
            lab.addWidget(b)
        self.labeling_group.setExclusive(True)
        self.labeling_group.buttonClicked[int].connect(self._set_active_label)
        self.all_scorers_check = QCheckBox("show other scorers")
        self.all_scorers_check.toggled.connect(self._toggle_all_scorers)
        self.spec_check = QCheckBox("spectrogram of channel")
        self.spec_check.toggled.connect(lambda _: self._rebuild_view())
        self.spec_channel = QSpinBox()
        self.spec_channel.setRange(0, 15)
        self.spec_channel.valueChanged.connect(lambda _: self.update_screen())
        self.norm_check = QCheckBox("normalise display")
        self.norm_check.setToolTip("median/IQR scaling per channel (reloads the recording)")
        for w in (self.all_scorers_check, self.spec_check, self.spec_channel, self.norm_check):
            lab.addWidget(w)
        lab.addStretch(1)
        layout.addLayout(lab)

        # file + navigation controls
        ctl = QHBoxLayout()
        self.folder_check = QCheckBox("folder")
        self.folder_check.setToolTip("load a folder of X_*.pt chunks instead of one file")
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["auto", "cnt", "nct"])
        self.layout_combo.setToolTip("cnt = channels first (GUI preprocessing), nct = windows first (conversions)")
        buttons = [
            ("load data", self.select_dataset, None),
            ("save states", self.save_states, "Ctrl+S"),
            ("load states", self.load_states, None),
            ("load other scorer", self.load_passive_scorer, None),
            ("<< prev state", self.prev_change, "Shift+Left"),
            ("< prev", self.prev, "Left"),
            ("jump...", self.jump, "Ctrl+J"),
            ("next >", self.next, "Right"),
            ("next state >>", self.next_change, "Shift+Right"),
            ("next uncertain", self.next_uncertain, "Ctrl+Right"),
            ("next unlabeled", self.next_unlabeled, "Ctrl+U"),
            ("+ time", self.increase_scale, "+"),
            ("- time", self.decrease_scale, "-"),
            ("macro view", self.toggle_macro_view, "T"),
            ("+ y", lambda: self.set_yscale(self.yscale * 1.1), None),
            ("- y", lambda: self.set_yscale(self.yscale / 1.1), None),
            ("reset view", self.reset_settings, None),
        ]
        ctl.addWidget(self.folder_check)
        ctl.addWidget(self.layout_combo)
        for text, slot, key in buttons:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if key:
                b.setShortcut(key)
                b.setToolTip(key)
            ctl.addWidget(b)
        self.ephys_only = QCheckBox("y only ECoG/EMG")
        self.ephys_only.setToolTip("y-scale changes apply only to the ECoG/EMG channels from the settings tab")
        self.ephys_only.toggled.connect(lambda _: (self._apply_yscale(), self.update_screen()))
        ctl.addWidget(self.ephys_only)
        layout.addLayout(ctl)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("uncertain = confidence below"))
        self.conf_thr = C.float_field("0.6", 0, 1)
        self.conf_thr.setMaximumWidth(60)
        conf_row.addWidget(self.conf_thr)
        conf_row.addStretch(1)
        layout.addLayout(conf_row)

        self.slider_yscale = QSlider(QtCore.Qt.Horizontal)
        self.slider_yscale.setRange(0, 1000)
        self.slider_yscale.valueChanged.connect(lambda v: self.set_yscale(self._slider_to_yscale(v)))
        self.yscale_label = QLabel("Y scale: x1.00")
        self.slider_frame = QSlider(QtCore.Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.setPageStep(8)
        self.slider_frame.setTracking(False)
        self.slider_frame.valueChanged.connect(self._set_idx)
        self.slider_label = QLabel("")
        for w in (self.yscale_label, self.slider_yscale, self.slider_label, self.slider_frame):
            layout.addWidget(w)
        self.status_label = QLabel("load windowed data to start")
        layout.addWidget(self.status_label)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(AUTOSAVE_MS)

    # ---- properties ---------------------------------------------------------------------------
    @property
    def states(self) -> np.ndarray:
        """The editable labels (raw codes), one per window."""
        return self.track.states

    @property
    def n(self) -> int:
        """Number of windows in the loaded recording (0 if none)."""
        return len(self.data) if self.data is not None else 0

    @property
    def active_scorer(self) -> str:
        """Name of the person labeling (settings-tab ``scorer``), used for the state line."""
        return M.get_str(self.params, "scorer", "me")

    def has_unsaved(self) -> bool:
        """True if labels were changed since the last save or load."""
        return self.data is not None and self.track.dirty

    # ---- loading ------------------------------------------------------------------------------
    def select_dataset(self) -> None:
        """Ask for a data file or folder (after confirming if labels are unsaved) and load it."""
        if self.has_unsaved() and not C.confirm(self, "unsaved labels", "discard unsaved labels and load new data?"):
            return
        if self.folder_check.isChecked():
            path = C.pick_dir(self, "select a recording folder with X_*.pt chunks", self.params, "label_data")
        else:
            path = C.pick_file(self, "select windowed data file", self.params, "Data files (*.pt *.npy)", "label_data")
        if path:
            self.load_data(path)

    def load_data(self, path: str) -> bool:
        """Load a recording (file or folder) and reset the view.

        The sample rate comes from the settings tab; if it is unset, 4-s windows are
        assumed. The time axis starts at the metadata's ``rec_start``.

        Returns:
            True on success; on failure an error dialog is shown and False returned.
        """
        try:
            if Path(path).is_dir():
                data = RecordingData.from_folder(path, layout=self.layout_combo.currentText())
            else:
                data = RecordingData.from_file(path, "auto", layout=self.layout_combo.currentText())
        except Exception as e:
            C.error(self, "load data", f"could not load {path}:\n{e}")
            return False
        if self.norm_check.isChecked():
            data.normalize_for_display()
        n_ecog, n_emg = M.channel_counts(self.params)
        data.set_ylims(M.get_str(self.params, "ylim", "infer"), n_ecog + n_emg if n_ecog + n_emg else 2)
        self.data = data
        self.data_path = path
        self._autosave_path = None
        self.sample_rate = M.sample_rate(self.params, 0) or 0
        if not self.sample_rate:
            self.sample_rate = data.win_len / 4.0
            self.status_label.setText(f"sample_rate not set: assuming {self.sample_rate:g} Hz (4-s windows)")
        self.frame_duration_s = data.win_len / self.sample_rate
        self.rec_start_sod, ok = parse_rec_start(self.params.get("rec_start"))
        self.track = StateTrack(data.states.copy())
        self._autosaved_version = self.track.version
        self.passive.clear()
        self.current_idx, self.scale = 0, 1
        self.ylim_defaults = list(data.ylims)
        self.yscale = 1.0
        self._apply_yscale()
        self.slider_frame.blockSignals(True)
        self.slider_frame.setRange(0, max(0, self.n - 1))
        self.slider_frame.setValue(0)
        self.slider_frame.blockSignals(False)
        self.spec_channel.setRange(0, max(0, data.n_channels - 1))
        self._rebuild_view()
        msg = f"loaded {Path(path).name}: {self.n} windows x {data.n_channels} channels"
        if not data.labeled:
            msg += " (no labels found: all unlabeled)"
        if not ok:
            msg += " | rec_start missing: time axis starts at 00:00:00"
        self.status_label.setText(msg)
        return True

    def open_result(self, data_folder: str, states_path: str) -> None:
        """Load a data folder and its autoscored state file (called from the automatic-scoring tab)."""
        self.folder_check.setChecked(True)
        if self.load_data(data_folder):
            self._load_states_from(states_path)

    # ---- view ---------------------------------------------------------------------------------
    def _rebuild_view(self) -> None:
        """Create a new figure and reconnect the mouse events.

        Needed after loading data, toggling the spectrogram or adding a scorer.
        """
        if self.data is None:
            return
        if self.canvas is not None:
            for cid in self._cids:
                self.canvas.mpl_disconnect(cid)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        names = self.params.get("channels_after_preprocessing") or []
        scorers = [self.active_scorer] + list(self.passive)
        self.formatter = TimeOfDayFormatter(self.rec_start_sod)
        self.view = SignalView(self.data.n_channels, list(names), scorers, self.formatter, self.spec_check.isChecked())
        self.canvas = FigureCanvas(self.view.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_layout.addWidget(self.canvas)
        self._cids = [self.canvas.mpl_connect(ev, fn) for ev, fn in (
            ("button_press_event", self._on_press), ("motion_notify_event", self._on_move),
            ("button_release_event", self._on_release), ("scroll_event", self._on_scroll))]
        self._drag_preview = None
        self.update_screen()

    def update_screen(self) -> None:
        """Redraw the visible windows, labels and confidence (re-entrant calls are ignored)."""
        if self.data is None or self.view is None or self._updating:
            return
        self._updating = True
        try:
            i0, i1 = self.current_idx, min(self.current_idx + self.scale, self.n)
            labels = {self.active_scorer: self.states[i0:i1]}
            if self.show_all_scorers:
                labels.update({k: v[i0:i1] for k, v in self.passive.items()})
            conf = self.track.confidence[i0:i1] if self.track.confidence is not None else None
            self.formatter.set_window_offset(i0 * self.frame_duration_s)
            self.view.update(self.data.signals(i0, self.scale), self.ylims, self.sample_rate, labels,
                             self.frame_duration_s, conf, self.spec_channel.value())
            self.canvas.draw_idle()
            self._update_controls()
        finally:
            self._updating = False

    def _update_controls(self) -> None:
        """Sync the sliders and status texts with the current position and scale."""
        if self.slider_frame.value() != self.current_idx:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(self.current_idx)
            self.slider_frame.blockSignals(False)
        t = (self.rec_start_sod + self.current_idx * self.frame_duration_s) % 86400
        self.slider_label.setText(f"window {self.current_idx}/{self.n - 1} | {self.current_idx * self.frame_duration_s / 60:.1f} "
                                  f"min from start | {int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}")
        v = self._yscale_to_slider(self.yscale)
        if self.slider_yscale.value() != v:
            self.slider_yscale.blockSignals(True)
            self.slider_yscale.setValue(v)
            self.slider_yscale.blockSignals(False)
        self.yscale_label.setText(f"Y scale: x{self.yscale:.2f}")
        state = int(self.states[self.current_idx]) if self.n else 0
        conf = ""
        if self.track.confidence is not None and self.current_idx < len(self.track.confidence):
            conf = f" | model confidence {self.track.confidence[self.current_idx]:.2f}"
        dirty = " | UNSAVED" if self.track.dirty else ""
        self.status_label.setText(f"state {STATE_NAMES.get(state, state)} | {self.scale} window(s) shown{conf}{dirty} | "
                                  f"drag on the state axis to label with: {STATE_NAMES[self.active_label_value]}")

    # ---- labeling -----------------------------------------------------------------------------
    def _set_active_label(self, value: int) -> None:
        """Select the state that dragging will write."""
        self.active_label_value = int(value)
        self._update_controls()

    def _is_state_axis(self, ax) -> bool:
        # the confidence trace lives on a twin axis that sits on top of the state axis
        """True if ``ax`` is the state axis or the confidence axis drawn over it."""
        return self.view is not None and ax in (self.view.label_ax, self.view.conf_ax)

    def _x_to_frame(self, x) -> int:
        """Return the window under axis position ``x`` (seconds from the view start), clipped to the view."""
        if x is None:
            return self.current_idx
        off = int(math.floor(float(x) / max(self.frame_duration_s, 1e-9)))
        off = max(0, min(off, self.scale - 1))
        return max(0, min(self.current_idx + off, self.n - 1))

    def _on_press(self, event) -> None:
        """Start a drag on the state axis with a shaded preview."""
        if event.button != 1 or not self._is_state_axis(event.inaxes) or event.xdata is None:
            return
        self._drag_x0 = float(event.xdata)
        self._drag_preview = self.view.label_ax.axvspan(self._drag_x0, self._drag_x0 + 1e-9, alpha=0.2)
        self.canvas.draw_idle()

    def _on_move(self, event) -> None:
        """Stretch the drag preview to the mouse position."""
        if self._drag_x0 is None or self._drag_preview is None or event.xdata is None:
            return
        lo, hi = sorted((self._drag_x0, float(event.xdata)))
        # axvspan returns a Polygon (matplotlib < 3.9) or a Rectangle (>= 3.9): recreate instead of resizing
        self._drag_preview.remove()
        self._drag_preview = self.view.label_ax.axvspan(lo, max(hi, lo + 1e-9), alpha=0.2)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        """Finish a drag: label every window between press and release."""
        if self._drag_x0 is None:
            return
        x0, self._drag_x0 = self._drag_x0, None
        if self._drag_preview is not None:
            try:
                self._drag_preview.remove()
            except (ValueError, NotImplementedError):
                pass
            self._drag_preview = None
        x1 = event.xdata if event.xdata is not None else x0
        if self.track.set_range(self._x_to_frame(x0), self._x_to_frame(x1), self.active_label_value):
            self.update_screen()
        else:
            self.canvas.draw_idle()

    def undo(self) -> None:
        """Undo the last labeling and move the view there if needed."""
        rng = self.track.undo()
        if rng is None:
            self.status_label.setText("nothing to undo")
            return
        if not (self.current_idx <= rng[0] < self.current_idx + self.scale):
            self.current_idx = max(0, rng[0])
        self.update_screen()

    def _on_scroll(self, event) -> None:
        """Move by a tenth of the view per wheel step."""
        step = max(1, self.scale // 10)
        self._set_idx(self.current_idx + (-step if event.step > 0 else step))

    # ---- navigation ---------------------------------------------------------------------------
    def _set_idx(self, idx: int) -> None:
        """Move the view so it starts at window ``idx`` (clipped)."""
        if self.n == 0:
            return
        idx = max(0, min(int(idx), self.n - 1))
        if idx != self.current_idx:
            self.current_idx = idx
            self.update_screen()

    def _macro_frames(self) -> int:
        """Return the number of windows in the 10-minute macro view."""
        return max(1, int(MACRO_SECONDS / self.frame_duration_s))

    def next(self) -> None:
        """Move one window on (two thirds of the view in macro view)."""
        self._set_idx(self.current_idx + (max(1, self.scale * 2 // 3) if self.scale == self._macro_frames() else 1))

    def prev(self) -> None:
        """Move one window back (two thirds of the view in macro view)."""
        self._set_idx(self.current_idx - (max(1, self.scale * 2 // 3) if self.scale == self._macro_frames() else 1))

    def _goto(self, idx: int | None, none_msg: str) -> None:
        """Move to ``idx``, or show ``none_msg`` if it is None."""
        if idx is None:
            self.status_label.setText(none_msg)
        else:
            self._set_idx(idx)

    def next_change(self) -> None:
        """Move to the next state change."""
        self._goto(self.track.next_change(self.current_idx), "no later state change")

    def prev_change(self) -> None:
        """Move to the previous state change."""
        self._goto(self.track.prev_change(self.current_idx), "no earlier state change")

    def next_uncertain(self) -> None:
        """Move to the next window, past the current view, whose model confidence is below the threshold."""
        if self.track.confidence is None:
            self.status_label.setText("no confidence loaded (load an autoscored state file with its _confidence.pkl)")
            return
        try:
            thr = C.read_float(self.conf_thr, "threshold")
        except ValueError as e:
            self.status_label.setText(str(e))
            return
        self._goto(self.track.next_uncertain(self.current_idx, thr, skip=max(1, self.scale)),
                   f"no window with confidence < {thr} after this point")

    def next_unlabeled(self) -> None:
        """Move to the next unlabeled window."""
        self._goto(self.track.next_unlabeled(self.current_idx), "no unlabeled window after this point")

    def jump(self) -> None:
        """Ask for a window index or a time of day (HH:MM[:SS]) and move there."""
        if self.n == 0:
            return
        text, ok = QInputDialog.getText(self, "jump", "window index, or time of day HH:MM[:SS]:")
        if not ok or not text.strip():
            return
        text = text.strip()
        if ":" in text:
            try:
                parts = [int(p) for p in text.split(":")]
                sod = parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) > 2 else 0)
            except (ValueError, IndexError):
                self.status_label.setText(f"cannot read time {text!r}")
                return
            offset = (sod - self.rec_start_sod) % 86400  # recordings may pass midnight
            self._set_idx(int(offset // self.frame_duration_s))
        elif text.isdigit():
            self._set_idx(int(text))
        else:
            self.status_label.setText(f"cannot read {text!r}")

    def increase_scale(self) -> None:
        """Show one more window."""
        self.scale = min(self.scale + 1, max(1, self.n))
        self.update_screen()

    def decrease_scale(self) -> None:
        """Show one window fewer (at least one)."""
        if self.scale > 1:
            self.scale -= 1
            self.update_screen()

    def toggle_macro_view(self) -> None:
        """Switch between the 10-minute view and the previous number of windows."""
        if self.n == 0:
            return
        macro = self._macro_frames()
        if self.scale != macro:
            self._prev_scale, self.scale = self.scale, macro
        else:
            self.scale = self._prev_scale
        self.update_screen()

    # ---- y scale ------------------------------------------------------------------------------
    def _slider_to_yscale(self, v: int) -> float:
        """Map the y slider (0-1000) to a gain between 0.1 and 10 (log scale)."""
        return math.exp(self.yscale_log_min + v / 1000 * (self.yscale_log_max - self.yscale_log_min))

    def _yscale_to_slider(self, y: float) -> int:
        """Map a y gain back to the slider position."""
        frac = (math.log(max(y, 1e-9)) - self.yscale_log_min) / (self.yscale_log_max - self.yscale_log_min)
        return int(np.clip(frac * 1000, 0, 1000))

    def _apply_yscale(self) -> None:
        """Scale the default y ranges by the gain (only ECoG/EMG channels if 'y only ECoG/EMG' is ticked)."""
        n_ecog, n_emg = M.channel_counts(self.params)
        n_ephys = n_ecog + n_emg if n_ecog + n_emg else 2
        only = self.ephys_only.isChecked()
        self.ylims = [(c, s * self.yscale if (not only or ch < n_ephys) else s)
                      for ch, (c, s) in enumerate(self.ylim_defaults)]

    def set_yscale(self, s: float) -> None:
        """Set the y gain and redraw."""
        s = max(float(s), 1e-3)
        if abs(s - self.yscale) < 1e-9:
            return
        self.yscale = s
        self._apply_yscale()
        self.update_screen()

    def reset_settings(self) -> None:
        """Back to one window and gain 1."""
        self.scale, self.yscale = 1, 1.0
        self._apply_yscale()
        self.update_screen()

    def _toggle_all_scorers(self, on: bool) -> None:
        """Show or hide the other scorers' state lines."""
        self.show_all_scorers = on
        self.update_screen()

    # ---- state files --------------------------------------------------------------------------
    def save_states(self) -> None:
        """Save the labels to a new file in the project's ``scores`` folder (never overwrites)."""
        if self.data is None:
            self.status_label.setText("nothing to save: no data loaded")
            return
        from scorer.data.storage import construct_paths, save_pickled_states

        _, self.score_save_path = construct_paths(self.data_path, add=str(self.current_idx))
        save_pickled_states(self.states, self.score_save_path)
        self.track.mark_saved()
        self._autosaved_version = self.track.version
        self.status_label.setText(f"saved {self.score_save_path}")

    def autosave(self) -> None:
        """Save the labels to this recording's autosave file if they changed since the last save.

        One autosave file is used per loaded recording; errors are logged and never interrupt
        labeling.
        """
        if self.data is None or not self.track.dirty or self.track.version == self._autosaved_version:
            return
        from scorer.data.storage import construct_paths

        try:
            # one autosave file per loaded recording: construct_paths never reuses a name, so calling it
            # every time would leave a new file every 3 minutes
            if self._autosave_path is None:
                _, self._autosave_path = construct_paths(self.data_path, add="autosave")
            path = self._autosave_path
            with open(path, "wb") as f:
                pickle.dump(self.states.copy(), f)
            self._autosaved_version = self.track.version
            log.info("autosaved labels to %s", path)
        except Exception as e:  # never interrupt labeling because of autosave
            log.warning("autosave failed: %s", e)

    def load_states(self) -> None:
        """Ask for a state file and load it as the editable labels."""
        path = C.pick_file(self, "select states file", self.params, "Pickle files (*.pkl)", "states")
        if path:
            self._load_states_from(path)

    def _fitted(self, path: str) -> np.ndarray | None:
        """Load a state file and fit it to the recording's length.

        Returns:
            The states, or None if loading failed or the user declined a length fix.
        """
        try:
            states = load_states_file(path)
        except Exception as e:
            C.error(self, "load states", f"{path}: {e}")
            return None
        fitted, msg = fit_states_to(states, self.n)
        if msg and not C.confirm(self, "length mismatch", f"{Path(path).name}: {msg}.\nLoad anyway?"):
            return None
        return fitted

    def _load_states_from(self, path: str) -> None:
        """Replace the editable labels with a state file (and its confidence, if present).

        Asks first if the current labels are unsaved.
        """
        if self.data is None:
            C.error(self, "load states", "load the data first")
            return
        if self.track.dirty and not C.confirm(self, "unsaved labels", "replace the current, unsaved labels?"):
            return
        states = self._fitted(path)
        if states is None:
            return
        conf = load_confidence_for(path)
        if conf is not None:
            conf = fit_array(conf, self.n, np.nan)
        self.track.replace(states, conf)
        self._autosaved_version = self.track.version
        self.status_label.setText(f"loaded {Path(path).name}" + (" with model confidence" if conf is not None else ""))
        self.update_screen()

    def load_passive_scorer(self) -> None:
        """Load another scorer's file to show read-only next to your labels."""
        if self.data is None:
            C.error(self, "load other scorer", "load the data first")
            return
        path = C.pick_file(self, "other scorer's states (shown, not editable)", self.params, "Pickle files (*.pkl)",
                           "states")
        if not path:
            return
        states = self._fitted(path)
        if states is None:
            return
        self.passive[QFileInfo(path).baseName()] = states
        self.all_scorers_check.setChecked(True)
        self._rebuild_view()
