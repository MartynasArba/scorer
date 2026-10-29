"""
Model training tab: edit/check a pipeline config, choose stages, run `scorer.pipeline.run` in a
separate process, follow its progress, and browse the run folder's results.
See gui/core/training.py for why training runs in a child process.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from scorer.gui.core import training as T
from scorer.gui.widgets import common as C

STAGE_HELP = {
    "inspect": "label counts per dataset (fast, no training)",
    "prepare": "normalise every recording once into the cache",
    "simclr": "unsupervised contrastive pretraining",
    "supcon": "supervised contrastive pretraining",
    "adversarial": "align to the target domain (mode tuned only)",
    "embed": "encoder -> embeddings for the datasets the heads need",
    "context_rf": "random forest head",
    "context_mlp": "context MLP head",
    "gru": "GRU sequence head",
    "linear_probe": "linear probe head",
    "evaluate": "metrics, confusion matrices, hypnograms",
}


class TrainingWidget(QWidget):
    """Model-training tab: edit and check a pipeline config, pick stages, run the pipeline
    as a separate process, follow its progress and browse the run folder.

    The run is ``python -m scorer.pipeline.run`` in a child process, so a crash or GPU
    out-of-memory error cannot take the GUI down, and "copy command" gives the same run
    for the command line. Overrides (run name, mode, device, extra keys) are passed as
    ``--set`` and never change the config file.

    Signals:
        use_run_for_scoring(str): Run folder to load in the automatic-scoring tab.

    Args:
        metadata: The shared metadata dict (used for dialog start folders).
    """
    use_run_for_scoring = pyqtSignal(str)  # run folder

    def __init__(self, metadata: dict):
        """Build the editor, override fields, stage list, run controls and result tables."""
        super().__init__()
        self.params = metadata
        self.config_path: Path | None = None
        self.check: T.ConfigCheck | None = None
        self.proc: QProcess | None = None
        self.progress = T.Progress()
        self._buf = ""
        self._run_dir: Path | None = None
        self._stopping = False
        layout = QVBoxLayout(self)

        # --- config file + editor ---
        row = QHBoxLayout()
        self.path_label = QLabel("no config loaded")
        for text, slot in (("open config...", self.open_config), ("new from template...", self.new_from_template),
                           ("save", self.save_config), ("save as...", self.save_config_as)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addWidget(self.path_label, 1)
        layout.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        self.editor = QPlainTextEdit()
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.modificationChanged.connect(self._update_title)
        split.addWidget(self.editor)

        right = QWidget()
        rl = QVBoxLayout(right)
        ob = QGroupBox("this run (command-line overrides, the file is not changed)")
        ol = QFormLayout(ob)
        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("from config")
        self.mode = QComboBox()
        self.mode.addItems(["from config", "tuned", "zeroshot"])
        self.mode.setToolTip("tuned: align to the target's unlabeled inputs | zeroshot: skip alignment")
        self.device = QComboBox()
        self.device.addItems(["from config", "auto", "cuda", "cpu"])
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("e.g. context_mlp.epochs=5; simclr.max_batches_per_epoch=20")
        self.force_prepare = QCheckBox("re-normalise cached recordings (--force-prepare)")
        ol.addRow("run name", self.run_name)
        ol.addRow("mode", self.mode)
        ol.addRow("device", self.device)
        ol.addRow("more overrides", self.extra)
        ol.addRow("", self.force_prepare)
        rl.addWidget(ob)
        b = QPushButton("check config")
        b.clicked.connect(self.check_config)
        rl.addWidget(b)
        rl.addWidget(QLabel("stages to run (ticked = enabled in the config):"))
        self.stage_list = QListWidget()
        for s in STAGE_HELP:
            it = QListWidgetItem(f"{s}  -  {STAGE_HELP[s]}")
            it.setData(Qt.UserRole, s)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.stage_list.addItem(it)
        rl.addWidget(self.stage_list, 1)
        self.check_box = QPlainTextEdit()
        self.check_box.setReadOnly(True)
        self.check_box.setMaximumHeight(120)
        rl.addWidget(self.check_box)
        split.addWidget(right)
        split.setSizes([700, 500])
        layout.addWidget(split, 3)

        # --- run controls ---
        row = QHBoxLayout()
        self.run_btn = QPushButton("run selected stages")
        self.run_btn.clicked.connect(self.run_selected)
        self.inspect_btn = QPushButton("inspect datasets only")
        self.inspect_btn.clicked.connect(lambda: self.run_stages(["inspect"]))
        self.stop_btn = QPushButton("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.copy_btn = QPushButton("copy command")
        self.copy_btn.clicked.connect(self.copy_command)
        for w in (self.run_btn, self.inspect_btn, self.stop_btn, self.copy_btn):
            row.addWidget(w)
        self.status = QLabel("idle")
        row.addWidget(self.status, 1)
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(260)
        row.addWidget(self.bar)
        layout.addLayout(row)
        self.log = C.LogPanel(max_lines=20000)
        layout.addWidget(self.log, 2)

        # --- results ---
        res = QGroupBox("run folder")
        rlay = QVBoxLayout(res)
        row = QHBoxLayout()
        self.run_label = QLabel("-")
        for text, slot in (("refresh", self.refresh_results), ("open folder", self.open_run_folder),
                           ("use this run for automatic scoring", self._use_for_scoring)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addWidget(self.run_label, 1)
        rlay.addLayout(row)
        self.manifest_table = QTableWidget(0, 3)
        self.manifest_table.setHorizontalHeaderLabels(["stage", "finished", "main output"])
        self.manifest_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.summary_table = QTableWidget(0, 0)
        tables = QSplitter(Qt.Horizontal)
        tables.addWidget(self.manifest_table)
        tables.addWidget(self.summary_table)
        rlay.addWidget(tables)
        layout.addWidget(res, 2)
        self._restore()

    # ---- config file ------------------------------------------------------------------------
    def _restore(self) -> None:
        """Reopen the config used last time (from QSettings)."""
        last = C.settings().value("training/config", "", type=str)
        if last and Path(last).exists():
            self.load_config(last)

    def _update_title(self, modified: bool = False) -> None:
        """Show the config path, with '*' while the editor has unsaved changes."""
        name = str(self.config_path) if self.config_path else "unsaved config"
        self.path_label.setText(("* " if self.editor.document().isModified() else "") + name)

    def load_config(self, path: str | Path) -> None:
        """Load a config file into the editor, remember it and check it."""
        self.config_path = Path(path)
        self.editor.setPlainText(self.config_path.read_text(encoding="utf-8"))
        self.editor.document().setModified(False)
        self._update_title()
        C.settings().setValue("training/config", str(self.config_path))
        self.check_config()

    def open_config(self) -> None:
        """Ask for a config file (after confirming if the editor has unsaved changes)."""
        if not self._discard_ok():
            return
        path = C.pick_file(self, "pipeline config", self.params, "TOML (*.toml)", "train_config")
        if path:
            self.load_config(path)

    def new_from_template(self) -> None:
        """Start a new, unsaved config from a bundled template."""
        if not self._discard_ok():
            return
        names = list(T.templates())
        if not names:
            C.error(self, "templates", f"no .toml templates in {T.config_dir()}")
            return
        name, ok = QInputDialog.getItem(self, "template", "start from:", names, 0, False)
        if not ok:
            return
        self.editor.setPlainText(T.templates()[name].read_text(encoding="utf-8"))
        self.config_path = None
        self.editor.document().setModified(True)
        self._update_title()
        self.check_box.setPlainText("edit [run] name, runs_dir, [cache] root and [datasets], then 'save as'")

    def save_config(self) -> bool:
        """Save the editor to the current file (asks for a name if there is none); True if saved."""
        if self.config_path is None:
            return self.save_config_as()
        self.config_path.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.editor.document().setModified(False)
        self._update_title()
        return True

    def save_config_as(self) -> bool:
        """Ask for a file name and save; warns before overwriting a bundled template. True if saved."""
        path = C.pick_save(self, "save config as", self.params, "pipeline.toml", "TOML (*.toml)", "train_config")
        if not path:
            return False
        if Path(path).resolve().parent == T.config_dir().resolve() and Path(path).stem in T.templates():
            if not C.confirm(self, "overwrite template", "overwrite a bundled template? (better: save a copy)"):
                return False
        self.config_path = Path(path)
        C.settings().setValue("training/config", path)
        return self.save_config()

    def _discard_ok(self) -> bool:
        """True if the editor has no unsaved changes or the user agrees to discard them."""
        return not self.editor.document().isModified() or C.confirm(
            self, "unsaved config", "discard the unsaved changes in the config editor?")

    # ---- checks -------------------------------------------------------------------------------
    def overrides(self) -> list[str]:
        """Return the ``--set`` overrides from the run-name, mode, device and extra fields.

        Raises:
            ValueError: If an extra override is malformed.
        """
        mode = self.mode.currentText()
        dev = self.device.currentText()
        return T.build_overrides(self.run_name.text(), mode if mode != "from config" else "",
                                 dev if dev != "from config" else "", self.extra.text())

    def check_config(self) -> bool:
        """Check the saved config with the current overrides and show the result.

        Unsaved edits must be saved first (the user is asked). On success the stage list is
        ticked with the config's default stages and the run folder's results are shown.

        Returns:
            True if the config has no errors.
        """
        if self.config_path is None or self.editor.document().isModified():
            if not self.editor.toPlainText().strip():
                return False
            if not C.confirm(self, "check config", "the config must be saved before it can be checked. Save now?"):
                return False
            if not self.save_config():
                return False
        try:
            ov = self.overrides()
        except ValueError as e:
            self.check_box.setPlainText(str(e))
            return False
        self.check = T.check_config(self.config_path, ov)
        lines = [f"ERROR: {e}" for e in self.check.errors] + [f"warning: {w}" for w in self.check.warnings]
        if self.check.cfg is not None:
            lines.insert(0, f"run folder: {self.check.run_dir} | mode: {self.check.mode} | "
                            f"default stages: {', '.join(self.check.default_stages)}")
            for i in range(self.stage_list.count()):
                it = self.stage_list.item(i)
                it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in self.check.default_stages else Qt.Unchecked)
            self._run_dir = self.check.run_dir
            self.refresh_results()
        self.check_box.setPlainText("\n".join(lines) or "config OK")
        return self.check.ok

    def selected_stages(self) -> list[str]:
        """Return the ticked stages in list order."""
        return [self.stage_list.item(i).data(Qt.UserRole) for i in range(self.stage_list.count())
                if self.stage_list.item(i).checkState() == Qt.Checked]

    # ---- running ------------------------------------------------------------------------------
    def is_running(self) -> bool:
        """True while a training process is running."""
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def run_selected(self) -> None:
        """Run the ticked stages."""
        self.run_stages(self.selected_stages())

    def _command(self, stages: list[str]) -> list[str]:
        """Return the command for ``stages`` with the current config and overrides."""
        return T.build_command(self.config_path, stages, self.overrides(), self.force_prepare.isChecked())

    def copy_command(self) -> None:
        """Copy the command for the ticked stages to the clipboard (and the log)."""
        from PyQt5.QtWidgets import QApplication

        if self.config_path is None:
            return
        try:
            text = T.command_text(self._command(self.selected_stages()))
        except ValueError as e:
            C.error(self, "command", str(e))
            return
        QApplication.clipboard().setText(text)
        self.log.add("command copied: " + text)

    def run_stages(self, stages: list[str]) -> None:
        """Check the config and the stage selection, then start the pipeline process.

        Errors (e.g. a head without embeddings) block the run; warnings ask to confirm. The
        process runs in the package root with stdout and stderr merged into the log.
        """
        if self.is_running():
            return
        if not stages:
            C.error(self, "training", "tick at least one stage")
            return
        if not self.check_config() or self.check is None or self.check.cfg is None:
            C.error(self, "training", "fix the config problems listed on the right first")
            return
        problems = T.stage_problems(self.check.cfg, stages, T.read_manifest(self.check.run_dir))
        errors = [m for lvl, m in problems if lvl == "error"]
        if errors:
            C.error(self, "training", "\n".join(errors))
            return
        for lvl, m in problems:
            if lvl == "warning" and not C.confirm(self, "training", m + "\nRun anyway?"):
                return
        cmd = self._command(stages)
        self.progress = T.Progress(stages=list(stages))
        self._run_dir = self.check.run_dir
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(T.package_root()))
        env = QProcessEnvironment.systemEnvironment()
        for k, v in T.child_env().items():
            if k in ("PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONIOENCODING"):
                env.insert(k, v)
        self.proc.setProcessEnvironment(env)
        self._stopping = False
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(lambda err: self.log.add(f"process error: {err}"))
        self.log.add("starting: " + T.command_text(cmd))
        self.proc.start(cmd[0], cmd[1:])
        self.run_btn.setEnabled(False)
        self.inspect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._show_progress()

    def _read_output(self) -> None:
        """Append new output lines to the log and update the progress from them."""
        self._buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        changed = False
        for line in lines:
            line = line.rstrip("\r")
            if line:
                self.log.appendPlainText(line)
                changed |= self.progress.feed(line)
        if changed:
            self._show_progress()

    def _show_progress(self) -> None:
        """Show the stage summary and an epoch/recording or stage progress bar."""
        self.status.setText(self.progress.summary())
        i, n = self.progress.step
        if n:
            self.bar.setRange(0, n)
            self.bar.setValue(i)
            self.bar.setFormat(f"{self.progress.step_label} %v/%m")
        else:
            self.bar.setRange(0, max(1, len(self.progress.stages)))
            self.bar.setValue(len(self.progress.done))
            self.bar.setFormat("stage %v/%m")

    def _finished(self, code: int, status) -> None:
        """Handle the end of the process: flush output, refresh results and report the outcome.

        A failure (not caused by "stop") shows the last 40 lines of ``pipeline.log``.
        """
        if self._buf.strip():
            self.log.appendPlainText(self._buf)
            self.progress.feed(self._buf)
            self._buf = ""
        self.run_btn.setEnabled(True)
        self.inspect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._show_progress()
        self.refresh_results()
        if code == 0 and not self.progress.failed:
            self.log.add(f"finished: {', '.join(self.progress.done)}")
        elif self._stopping:
            self.log.add(f"stopped by you; finished stages: {', '.join(self.progress.done) or 'none'}")
        else:
            where = f" in stage {self.progress.failed}" if self.progress.failed else ""
            self.log.add(f"run ended with exit code {code}{where}")
            tail = T.tail_lines(Path(self._run_dir) / "pipeline.log", 40) if self._run_dir else []
            C.error(self, "training failed", f"the pipeline stopped{where} (exit code {code}).\n\n" + "\n".join(tail))

    def stop(self) -> None:
        """After confirmation, terminate the run (killed after 8 s if it does not stop); finished stages stay."""
        if not self.is_running():
            return
        if not C.confirm(self, "stop training", "stop the running pipeline? The current stage is lost; finished "
                                                "stages stay in the run folder and can be resumed."):
            return
        self.log.add("stopping...")
        self._stopping = True
        self.proc.terminate()  # console processes on Windows ignore this; killed after 8 s
        QTimer.singleShot(8000, lambda: self.is_running() and self.proc.kill())

    def shutdown(self) -> None:
        """Kill a running process without asking (called by the main window when it closes)."""
        if self.is_running():
            self.proc.kill()
            self.proc.waitForFinished(5000)

    # ---- results ------------------------------------------------------------------------------
    def refresh_results(self) -> None:
        """Fill the tables from the run folder: finished stages (manifest) and evaluation summaries."""
        if self._run_dir is None:
            return
        self.run_label.setText(str(self._run_dir))
        rows = T.manifest_rows(T.read_manifest(self._run_dir))
        self.manifest_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("stage", "finished", "output")):
                item = QTableWidgetItem(row[key])
                item.setToolTip(row["details"])
                self.manifest_table.setItem(r, c, item)
        summ = T.read_summaries(self._run_dir)
        cols = ["mode", "head", "dataset", "accuracy", "macro_f1", "kappa", "accuracy_argmax_only", "scored_windows"]
        cols = [c for c in cols if summ and c in summ[0]]
        self.summary_table.setColumnCount(len(cols))
        self.summary_table.setHorizontalHeaderLabels(cols)
        self.summary_table.setRowCount(len(summ))
        for r, row in enumerate(summ):
            for c, key in enumerate(cols):
                v = row.get(key, "")
                try:
                    v = f"{float(v):.4f}" if key in ("accuracy", "macro_f1", "kappa", "accuracy_argmax_only") else v
                except ValueError:
                    pass
                self.summary_table.setItem(r, c, QTableWidgetItem(str(v)))

    def open_run_folder(self) -> None:
        """Open the run folder in the system file browser."""
        if self._run_dir and Path(self._run_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run_dir)))

    def _use_for_scoring(self) -> None:
        """Send the run folder to the automatic-scoring tab (it needs a manifest)."""
        if self._run_dir and (Path(self._run_dir) / "manifest.json").exists():
            self.use_run_for_scoring.emit(str(self._run_dir))
        else:
            C.error(self, "automatic scoring", "this run has no manifest.json yet")
