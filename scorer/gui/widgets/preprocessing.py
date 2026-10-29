"""
Preprocessing tab: collects options into PreprocessOptions, asks every question BEFORE starting,
then runs gui.core.preprocess_job.run_preprocessing in a background thread.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QWidget)

from scorer.gui.core import meta as M
from scorer.gui.core.preprocess_job import PreprocessOptions, check_options, model_preset, run_preprocessing
from scorer.gui.widgets import common as C
from scorer.gui.workers import run_task, with_prints_to


class PreprocessWidget(QWidget):
    """Preprocessing tab: collects the options, asks every question before starting, then
    runs ``preprocess_job.run_preprocessing`` in a background thread.

    Signals:
        preprocessing_done(dict): The job's summary, after a successful run.

    Args:
        metadata: The shared metadata dict.
    """
    preprocessing_done = pyqtSignal(dict)

    def __init__(self, metadata: dict) -> None:
        """Build the option groups, run buttons and log."""
        super().__init__()
        self.params = metadata
        self.selected_file = ""
        self._task = None
        layout = QVBoxLayout(self)

        self.label = QLabel("select a .csv file to preprocess")
        layout.addWidget(self.label)
        row = QHBoxLayout()
        b = QPushButton("select file")
        b.clicked.connect(self.select_file)
        row.addWidget(b)
        preset = QPushButton("preset: windows for the 128 Hz models")
        preset.setToolTip("resample to 128 Hz, band-pass ECoG 0.5-49 Hz, save 512-sample (4 s) windows")
        preset.clicked.connect(self.apply_model_preset)
        row.addWidget(preset)
        row.addStretch(1)
        layout.addLayout(row)

        g = QGridLayout()
        r = 0

        def add(check: QCheckBox, *fields):
            """Put a checkbox and its fields on the next grid row."""
            nonlocal r
            g.addWidget(check, r, 0)
            for i, f in enumerate(fields):
                g.addWidget(f, r, 1 + i)
            r += 1

        self.chunk_check = QCheckBox("load in chunks of (samples)")
        self.chunk_size = C.int_field("1000000", 1)
        add(self.chunk_check, self.chunk_size)
        self.timechop_check = QCheckBox("only load between (HH:MM:SS)")
        self.t_start, self.t_end = QLineEdit("19:00:00"), QLineEdit("07:00:00")
        add(self.timechop_check, self.t_start, self.t_end)
        self.resample_check = QCheckBox("resample to (Hz)")
        self.new_sr = C.int_field("250", 1)
        add(self.resample_check, self.new_sr)
        self.ecog_filter = QCheckBox("band-pass ECoG (Hz)")
        self.ecog_lo, self.ecog_hi = C.float_field("0.5"), C.float_field("49")
        add(self.ecog_filter, self.ecog_lo, self.ecog_hi)
        self.emg_filter = QCheckBox("band-pass EMG (Hz)")
        self.emg_lo, self.emg_hi = C.float_field("10"), C.float_field("100")
        add(self.emg_filter, self.emg_lo, self.emg_hi)
        self.notch_check = QCheckBox("notch filter at (Hz)")
        self.notch_f = C.float_field("50")
        add(self.notch_check, self.notch_f)
        steps = QGroupBox("steps")
        steps.setLayout(g)
        layout.addWidget(steps)

        feats = QGroupBox("features (pick at most one band-based option)")
        fl = QHBoxLayout(feats)
        self.sum_power = QCheckBox("sum power")
        self.band_power = QCheckBox("band power")
        self.band_signals = QCheckBox("band signals")
        self.ratio = QCheckBox("ratio features")
        self.smooth_check = QCheckBox("Gaussian smoothing, sigma (s)")
        self.smooth_sigma = C.float_field("0.25")
        for w in (self.sum_power, self.band_power, self.band_signals, self.ratio, self.smooth_check, self.smooth_sigma):
            fl.addWidget(w)
        layout.addWidget(feats)

        out = QGroupBox("output")
        ol = QGridLayout(out)
        self.prescore = QCheckBox("prescore (Watson 2016)")
        self.save_raw = QCheckBox("save raw tensors")
        self.save_pre = QCheckBox("save preprocessed tensors")
        self.save_win = QCheckBox("save windowed data, window (samples)")
        self.win_len = C.int_field("1000", 1)
        self.win_info = QLabel()
        self.overwrite = QCheckBox("overwrite files")
        self.add_name = QCheckBox("add original file name")
        for i, w in enumerate((self.prescore, self.save_raw, self.save_pre, self.overwrite, self.add_name)):
            ol.addWidget(w, 0, i)
        ol.addWidget(self.save_win, 1, 0)
        ol.addWidget(self.win_len, 1, 1)
        ol.addWidget(self.win_info, 1, 2, 1, 3)
        layout.addWidget(out)
        for w in (self.win_len, self.new_sr):
            w.textChanged.connect(self._update_win_info)
        self.resample_check.toggled.connect(self._update_win_info)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("run preprocessing")
        self.run_btn.clicked.connect(self.run_preprocessing)
        self.cancel_btn = QPushButton("stop after current chunk")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self._task and self._task.cancel())
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        layout.addLayout(run_row)
        self.log = C.LogPanel()
        layout.addWidget(self.log, 1)
        self._update_win_info()

    # ---------------------------------------------------------------------------------------
    def select_file(self) -> None:
        """Ask for the CSV file to preprocess."""
        path = C.pick_file(self, "select file to preprocess", self.params, "CSV files (*.csv)", "preprocess_in")
        if path:
            self.selected_file = path
            self.label.setText(f"{Path(path).name} selected - choose steps and press run")

    def apply_model_preset(self) -> None:
        """Set the options for windows the 128 Hz models can score (chunk size rounded to whole windows)."""
        p = model_preset(self.params)
        self.resample_check.setChecked(p["resample"])
        self.new_sr.setText(str(p["new_sr"]))
        self.ecog_filter.setChecked(p["ecog_filter"])
        self.ecog_lo.setText(str(p["ecog_band"][0]))
        self.ecog_hi.setText(str(p["ecog_band"][1]))
        self.save_win.setChecked(True)
        self.win_len.setText(str(p["win_len"]))
        if self.chunk_check.isChecked():
            self.chunk_size.setText(str(max(1, int(self.chunk_size.text() or 1) // p["win_len"]) * p["win_len"]))
        self._update_win_info()

    def _update_win_info(self) -> None:
        """Show the window duration in seconds next to the window length, with a hint if it is not 4 s."""
        try:
            sr = float(self.new_sr.text()) if self.resample_check.isChecked() else M.sample_rate(self.params, 0)
            dur = int(self.win_len.text()) / sr if sr else 0
            ok = abs(dur - 4.0) < 1e-6
            self.win_info.setText(f"= {dur:g} s at {sr:g} Hz" + ("" if ok else "  (models expect 4 s)"))
        except (ValueError, ZeroDivisionError):
            self.win_info.setText("")

    def options(self) -> PreprocessOptions:
        """Return the current choices as ``PreprocessOptions``.

        Raises:
            ValueError: If a number field holds an invalid value (the message names it).
        """
        return PreprocessOptions(
            file=self.selected_file, chunked=self.chunk_check.isChecked(),
            chunk_size=C.read_int(self.chunk_size, "chunk size"), timechop=self.timechop_check.isChecked(),
            time_start=self.t_start.text().strip(), time_end=self.t_end.text().strip(),
            resample=self.resample_check.isChecked(), new_sr=C.read_int(self.new_sr, "new sample rate"),
            ecog_filter=self.ecog_filter.isChecked(),
            ecog_band=(C.read_float(self.ecog_lo, "ECoG low"), C.read_float(self.ecog_hi, "ECoG high")),
            emg_filter=self.emg_filter.isChecked(),
            emg_band=(C.read_float(self.emg_lo, "EMG low"), C.read_float(self.emg_hi, "EMG high")),
            notch=self.notch_check.isChecked(), notch_freq=C.read_float(self.notch_f, "notch"),
            sum_power=self.sum_power.isChecked(), band_power=self.band_power.isChecked(),
            band_signals=self.band_signals.isChecked(), ratio_features=self.ratio.isChecked(),
            smooth=self.smooth_check.isChecked(), smooth_sigma=C.read_float(self.smooth_sigma, "sigma"),
            prescore=self.prescore.isChecked(), save_raw=self.save_raw.isChecked(),
            save_preprocessed=self.save_pre.isChecked(), save_windowed=self.save_win.isChecked(),
            win_len=C.read_int(self.win_len, "window length"), overwrite=self.overwrite.isChecked(),
            add_filename=self.add_name.isChecked())

    def run_preprocessing(self) -> None:
        """Check the options, ask the needed confirmations and start the job."""
        if self._task is not None and self._task.running:
            return
        try:
            opts = self.options()
        except ValueError as e:
            C.error(self, "preprocessing", str(e))
            return
        checks = check_options(opts, self.params)
        errors = [m for lvl, m in checks if lvl == "error"]
        if errors:
            C.error(self, "preprocessing", "\n".join(errors))
            return
        for lvl, m in checks:
            if lvl == "confirm" and not C.confirm(self, "preprocessing", m):
                return
            if lvl == "info":
                self.log.add(m)
        if not any((opts.save_raw, opts.save_preprocessed, opts.save_windowed)):
            if not C.confirm(self, "preprocessing", "no save option is ticked - nothing will be written. Run anyway?"):
                return
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(opts.chunked)
        self.log.add(f"started: {Path(opts.file).name}")
        params = self.params
        job = with_prints_to(lambda progress, cancelled: run_preprocessing(opts, params, progress, cancelled))
        self._task = run_task(self, job, on_done=self._done, on_error=self._failed, on_progress=self.log.add,
                              on_finished=self._finished)

    def _done(self, summary: dict) -> None:
        """Show the result and tell the other tabs (metadata changed)."""
        self.label.setText(f"done: {summary.get('chunks')} chunk(s); metadata in {summary.get('metadata_path')}")
        self.preprocessing_done.emit(summary)

    def _failed(self, msg: str) -> None:
        """Log and show the error."""
        self.log.add("FAILED: " + msg.split("\n")[0])
        C.error(self, "preprocessing failed", msg)

    def _finished(self) -> None:
        """Re-enable the run button after success or failure."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
