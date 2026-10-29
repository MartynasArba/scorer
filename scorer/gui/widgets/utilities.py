"""
Utilities tab (OneBox conversion, quality report, motion-sensor split), run in the background.

Fixed: 'convert all' read `save_folder` before asking for it (UnboundLocalError every time);
sample_rate None crashed int(); a long conversion froze the window. The data modules' print()
progress now appears in the tab's log; the quality report uses the csv's real sample rate and
also finds csv files directly in the chosen folder (where conversion writes them).
"""
from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from scorer.gui.core import meta as M
from scorer.gui.widgets import common as C
from scorer.gui.workers import run_task, with_prints_to


def _target_rate(meta: dict) -> int:
    """Return the settings-tab sample rate as the conversion's target rate.

    Raises:
        ValueError: If it is not set.
    """
    sr = M.get_float(meta, "sample_rate")
    if not sr:
        raise ValueError("set sample_rate (target rate for the conversion) in the settings tab")
    return int(sr)


class UtilWidget(QWidget):
    """Utilities tab: OneBox conversion, folder quality report and motion-sensor split.

    Every job runs in the background; the data modules' printed progress appears in the
    tab's log. Only one job runs at a time.

    Args:
        metadata: The shared metadata dict.
    """
    def __init__(self, metadata: dict):
        """Build the buttons and the log."""
        super().__init__()
        self.params = metadata
        self._task = None
        layout = QVBoxLayout(self)
        self.label = QLabel("ready")
        layout.addWidget(self.label)
        self.overwrite = QCheckBox("overwrite existing files (folder conversion)")
        layout.addWidget(self.overwrite)
        self.buttons = []
        for text, slot in (("convert one OneBox recording (.bin) to .csv", self.obx_to_csv),
                           ("convert all OneBox recordings in a folder to .csv", self.folder_to_csv),
                           ("quality report for all files in a folder", self.quality_report),
                           ("split motion-sensor files by cage", self.parse_sensor_data)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            layout.addWidget(b)
            self.buttons.append(b)
        self.quality_plots = QCheckBox("save quality-report plots")
        layout.addWidget(self.quality_plots)
        self.log = C.LogPanel()
        layout.addWidget(self.log, 1)

    def _run(self, title: str, fn, *args) -> None:
        """Run ``fn(*args)`` in the background with prints sent to the log; buttons are disabled meanwhile."""
        if self._task is not None and self._task.running:
            C.info(self, title, "another utility is still running")
            return
        for b in self.buttons:
            b.setEnabled(False)
        self.log.add(f"{title}: started")
        self._task = run_task(self, with_prints_to(lambda: fn(*args)),
                              on_progress=self.log.add,
                              on_done=lambda r: self.log.add(f"{title}: done" + (f" - {r}" if r else "")),
                              on_error=lambda m: (self.log.add(f"{title}: FAILED"), C.error(self, title, m)),
                              on_finished=lambda: [b.setEnabled(True) for b in self.buttons])

    def obx_to_csv(self) -> None:
        """Convert one OneBox ``.bin`` recording to CSV at the settings-tab sample rate."""
        from scorer.data.onebox_utils import run_conversion

        f = C.pick_file(self, "OneBox .bin file", self.params, "BIN files (*.bin)", "obx")
        if not f:
            return
        out = C.pick_dir(self, "folder to save .csv files in", self.params, "obx_out")
        if not out:
            self.label.setText("cancelled")
            return
        try:
            sr = _target_rate(self.params)
        except ValueError as e:
            C.error(self, "conversion", str(e))
            return
        self._run(f"convert {f}", lambda: run_conversion(f, self.params, save_folder=out, sr_new=sr))

    def folder_to_csv(self) -> None:
        """Convert every OneBox recording in a folder (existing outputs are skipped unless overwrite is ticked)."""
        from scorer.data.onebox_utils import convert_multiple_recs

        src = C.pick_dir(self, "folder with OneBox recordings", self.params, "obx")
        if not src:
            return
        out = C.pick_dir(self, "folder to save .csv files in", self.params, "obx_out")
        if not out:
            self.label.setText("cancelled")
            return
        try:
            sr = _target_rate(self.params)
        except ValueError as e:
            C.error(self, "conversion", str(e))
            return
        ow = self.overwrite.isChecked()
        self._run(f"convert folder {src}",
                  lambda: convert_multiple_recs(src, self.params, save_folder=out, sr_new=sr, overwrite=ow))

    def quality_report(self) -> None:
        """Write a quality report (and optional plots) for every CSV in a folder."""
        from scorer.data.onebox_utils import get_folder_quality_report

        src = C.pick_dir(self, "folder", self.params, "obx")
        if not src:
            return
        try:
            sr = _target_rate(self.params)  # the csv files were written at the conversion rate
        except ValueError as e:
            C.error(self, "quality report", str(e))
            return
        plots, out = self.quality_plots.isChecked(), M.project_dir(self.params)
        # figures are matplotlib.figure.Figure objects now (no pyplot), so this can run in the background
        self._run(f"quality report {src}",
                  lambda: get_folder_quality_report(src, savepath=out, save_fig=plots, sr=sr))

    def parse_sensor_data(self) -> None:
        """Split motion-sensor files into one file per cage."""
        from scorer.data.motionsensor_utils import parse_sensors

        src = C.pick_dir(self, "folder with motion-sensor files", self.params, "sensors")
        if src:
            self._run(f"split sensors {src}", lambda: f"{len(parse_sensors(src))} files written")
