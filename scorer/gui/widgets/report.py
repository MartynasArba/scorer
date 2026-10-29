"""
Report tab: hypnogram, state report, scorer agreement.

Changes: epoch length is entered in SECONDS (the window length in samples is derived from the
settings-tab sample rate, so autoscored files and manual files use the same time base); the
hypnogram is a step plot and shows model confidence when a _confidence.pkl sits next to the
states; scorer comparison runs over the real state codes (the old num_classes count dropped REM),
checks lengths, and shows a confusion matrix; figures are released when replaced.
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
                             QWidget)

from scorer.gui.core import meta as M
from scorer.gui.core.agreement import compare_states, format_agreement
from scorer.gui.core.annotations import load_confidence_for, load_states_file
from scorer.gui.core.plots import confusion_figure, hypnogram
from scorer.gui.widgets import common as C

STATE_MAPPING = {0: "Unknown", 1: "Wake", 2: "NREM", 3: "IS", 4: "REM", 5: "MA"}


class ReportWidget(QWidget):
    """Report tab: hypnogram, sleep-state report and agreement between two scorers.

    The epoch length is entered in seconds; the window length in samples follows from
    the settings-tab sample rate, so manual and automatic state files share one time
    base.

    Args:
        metadata: The shared metadata dict.
    """
    def __init__(self, metadata: dict) -> None:
        """Build the buttons and options."""
        super().__init__()
        self.params = metadata
        self.states_path = ""
        self.canvas = None
        main = QVBoxLayout(self)  # not `self.layout`: that shadowed QWidget.layout()
        self.plot_area = QVBoxLayout()
        main.addLayout(self.plot_area, 1)
        self.label = QLabel("select a states .pkl file")
        main.addWidget(self.label)

        row = QHBoxLayout()
        b = QPushButton("select states file")
        b.clicked.connect(self.set_states_path)
        row.addWidget(b)
        row.addWidget(QLabel("epoch length (s):"))
        self.epoch_s = C.float_field("4", 0.1, 3600)
        self.epoch_s.setMaximumWidth(60)
        row.addWidget(self.epoch_s)
        self.win_info = QLabel()
        row.addWidget(self.win_info)
        row.addStretch(1)
        main.addLayout(row)
        self.epoch_s.textChanged.connect(self._update_win_info)

        row = QHBoxLayout()
        self.ma_check = QCheckBox("mark micro-arousals (MA = 5): Wake runs of at most")
        self.ma_check.setToolTip("report.label_microawakenings: short Wake runs between two NREM runs become MA")
        self.ma_windows = QSpinBox()
        self.ma_windows.setRange(1, 30)
        self.ma_windows.setValue(3)
        row.addWidget(self.ma_check)
        row.addWidget(self.ma_windows)
        row.addWidget(QLabel("windows between NREM (hypnogram and state report)"))
        row.addStretch(1)
        main.addLayout(row)

        row = QHBoxLayout()
        b = QPushButton("hypnogram")
        b.clicked.connect(self.get_hypnogram)
        self.save_png = QCheckBox("save as .png")
        row.addWidget(b)
        row.addWidget(self.save_png)
        b = QPushButton("state report")
        b.clicked.connect(self.get_state_report)
        self.save_csv = QCheckBox("save .csv")
        self.by_hour = QCheckBox("by hour")
        for w in (b, self.save_csv, self.by_hour):
            row.addWidget(w)
        row.addStretch(1)
        main.addLayout(row)

        row = QHBoxLayout()
        b = QPushButton("compare two scorers...")
        b.clicked.connect(self.compare_scorers)
        self.ignore_unlabeled = QCheckBox("ignore unlabeled windows")
        self.ignore_unlabeled.setChecked(True)
        self.merge_is = QCheckBox("count IS as NREM")
        for w in (b, self.ignore_unlabeled, self.merge_is):
            row.addWidget(w)
        row.addStretch(1)
        main.addLayout(row)
        self._update_win_info()

    # ---------------------------------------------------------------------------------------
    def _win_len(self) -> int:
        """Return the window length in samples (epoch seconds x sample rate).

        Raises:
            ValueError: If the sample rate is unset or the epoch length is not a number.
        """
        sr = M.get_float(self.params, "sample_rate")
        if not sr:
            raise ValueError("set sample_rate in the settings tab (the rate of the scored data)")
        return int(round(C.read_float(self.epoch_s, "epoch length") * sr))

    def _update_win_info(self) -> None:
        """Show the window length in samples, or why it cannot be computed."""
        try:
            self.win_info.setText(f"= {self._win_len()} samples at {M.sample_rate(self.params):g} Hz")
        except ValueError as e:
            self.win_info.setText(str(e))

    def set_states_path(self) -> None:
        """Ask for the state file to report on."""
        path = C.pick_file(self, "states file", self.params, "Pickle files (*.pkl)", "states")
        if path:
            self.states_path = path
            self.label.setText(f"selected {path}")
            self._update_win_info()

    def _load(self):
        """Load the selected state file, optionally mark micro-arousals, and build its time axis.

        Returns:
            ``(states, timestamps, window length in samples)``.
        """
        if not self.states_path:
            raise ValueError("select a states file first")
        from scorer.data.storage import get_timearray_for_states

        states = load_states_file(self.states_path)
        if self.ma_check.isChecked():
            from scorer.data.report import label_microawakenings

            states = label_microawakenings(states, w_label=1, nrem_label=2, max_windows=self.ma_windows.value(),
                                           ma_label=5)
        win_len = self._win_len()
        return states, get_timearray_for_states(states, win_len=win_len, metadata=self.params), win_len

    def get_hypnogram(self) -> None:
        """Show the hypnogram (with confidence if available) and optionally save it as PNG."""
        try:
            states, times, _ = self._load()
        except Exception as e:
            C.error(self, "hypnogram", str(e))
            return
        fig = hypnogram(states, times, STATE_MAPPING, self.params, load_confidence_for(self.states_path))
        if self.save_png.isChecked():
            out = str(Path(self.states_path).with_suffix(".png"))
            fig.savefig(out, dpi=300, transparent=True)
            self.label.setText(f"saved {out}")
        if self.canvas is not None:
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        self.canvas = FigureCanvas(fig)
        self.plot_area.addWidget(self.canvas)
        self.canvas.draw()

    def get_state_report(self) -> None:
        """Compute the state report (optionally by hour), show it and optionally save it as CSV."""
        from scorer.data.report import generate_sleep_report

        try:
            states, times, win_len = self._load()
            save_csv = str(Path(self.states_path).with_suffix(".csv")) if self.save_csv.isChecked() else None
            results = generate_sleep_report(states, times, get_by_hour=self.by_hour.isChecked(), win_len=win_len,
                                            save_csv=save_csv, state_mapping=STATE_MAPPING, metadata=self.params)
        except Exception as e:
            C.error(self, "state report", f"{type(e).__name__}: {e}")
            return
        self.label.setText("report generated" + (f", saved {save_csv}" if save_csv else ""))
        self._popup("state report", json.dumps(results, indent=2, default=str))

    def compare_scorers(self) -> None:
        """Ask for two state files and show their agreement with a confusion matrix."""
        a = C.pick_file(self, "states file A (reference)", self.params, "Pickle files (*.pkl)", "states")
        if not a:
            return
        b = C.pick_file(self, "states file B", self.params, "Pickle files (*.pkl)", "states")
        if not b:
            return
        try:
            res = compare_states(load_states_file(a), load_states_file(b), self.ignore_unlabeled.isChecked(),
                                 self.merge_is.isChecked())
        except Exception as e:
            C.error(self, "compare scorers", f"{type(e).__name__}: {e}")
            return
        text = format_agreement(res, Path(a).name, Path(b).name)
        fig = confusion_figure(res, Path(a).stem, Path(b).stem) if "error" not in res else None
        self._popup("scorer agreement", text, fig, save_stem=str(Path(b).with_suffix("")) + "_vs_" + Path(a).stem,
                    payload=res)

    def _popup(self, title: str, text: str, fig=None, save_stem: str | None = None, payload=None) -> None:
        """Show text (and optionally a figure) in a dialog.

        Args:
            title: Dialog title.
            text: Monospaced text.
            fig: Figure shown above the text, or None.
            save_stem: If given, add a button that saves ``payload`` as
                ``<save_stem>_agreement.json`` and the figure as PNG.
            payload: Data saved with the JSON.
        """
        box = QDialog(self)
        box.setWindowTitle(title)
        box.resize(900, 800 if fig is not None else 700)
        lay = QVBoxLayout(box)
        if fig is not None:
            lay.addWidget(FigureCanvas(fig), 1)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setLineWrapMode(QTextEdit.NoWrap)
        t.setFontFamily("monospace")
        t.setPlainText(text)
        lay.addWidget(t, 1)
        row = QHBoxLayout()
        if save_stem:
            s = QPushButton("save .json + .png")

            def save():
                """Save the agreement JSON and PNG next to file B."""
                Path(save_stem + "_agreement.json").write_text(json.dumps(payload, indent=1, default=str))
                if fig is not None:
                    fig.savefig(save_stem + "_agreement.png", dpi=200)
                s.setText("saved next to file B")
            s.clicked.connect(save)
            row.addWidget(s)
        close = QPushButton("close")
        close.clicked.connect(box.accept)
        row.addWidget(close)
        lay.addLayout(row)
        box.exec_()
