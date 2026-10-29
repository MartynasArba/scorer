"""
Automatic scoring tab, built on inference.scoring.SleepScorer.

Changes vs AutoScoringWidget: models are chosen from a pipeline run folder (encoder + head always
come from the same run and mode) or as two explicit files - no hard-coded weight paths; one data
folder or a folder of many recordings can be scored in one go; data at another sample rate is
resampled to the model's rate; scoring runs in the background with a log and per-recording errors;
results can be opened straight in the manual-scoring tab.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from scorer.gui.core import meta as M
from scorer.gui.core.autoscore import ModelChoice, ScoreRequest, discover_models, expand_targets, score_targets
from scorer.gui.widgets import common as C
from scorer.gui.workers import run_task


def score_requests(choice: ModelChoice, requests: list[ScoreRequest], progress=None, cancelled=None):
    """Load the model once, then score every request (runs in the worker thread).

    Loading happens here because importing torch takes seconds and must not freeze the
    window.

    Returns:
        ``(outcomes, model description)``.
    """
    from scorer.inference.scoring import SleepScorer

    progress(f"loading model: {choice.label}")
    scorer = SleepScorer(choice.encoder, choice.head, device=requests[0].device)
    info = scorer.describe()
    progress(f"model: {info['head']} on {info['encoder_stage'] or 'encoder'} | {info['sample_rate']:g} Hz, "
             f"{info['win_len']} samples | classes {info['classes']} | trained on {info['trained_on']}")
    outcomes = []
    for r in requests:
        outcomes += score_targets(scorer, r, progress, cancelled)
        if cancelled():
            break
    return outcomes, info


class AutoScoringWidget(QWidget):
    """Automatic-scoring tab.

    Models come from a pipeline run folder (encoder and head always from the same run
    and mode) or from two explicit files. Data can be one recording, a folder of
    recordings or a single X file. Scoring runs in the background; results can be
    opened in the manual-scoring tab.

    Signals:
        open_in_labeling(str, str): Data folder and state file to review.

    Attributes:
        gpu_busy: Callable set by the main window; True while a training run uses the GPU.

    Args:
        meta: The shared metadata dict.
    """
    open_in_labeling = pyqtSignal(str, str)  # data folder, states file

    def __init__(self, meta: dict):
        """Build the model, data and option groups and restore the last run folder."""
        super().__init__()
        self.params = meta
        self.choices: list[ModelChoice] = []
        self.out_dir = ""
        self.results = []
        self._task = None
        self.gpu_busy = lambda: False  # set by the main window (training tab running)
        layout = QVBoxLayout(self)

        # --- model ---
        mbox = QGroupBox("model")
        ml = QVBoxLayout(mbox)
        row = QHBoxLayout()
        b_run = QPushButton("choose pipeline run folder...")
        b_run.clicked.connect(self.choose_run)
        b_files = QPushButton("choose encoder + head files...")
        b_files.clicked.connect(self.choose_files)
        row.addWidget(b_run)
        row.addWidget(b_files)
        row.addStretch(1)
        ml.addLayout(row)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        ml.addWidget(self.model_combo)
        self.model_label = QLabel("no model selected")
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ml.addWidget(self.model_label)
        layout.addWidget(mbox)

        # --- data ---
        dbox = QGroupBox("data")
        dl = QVBoxLayout(dbox)
        row = QHBoxLayout()
        b_data = QPushButton("add data folder...")
        b_data.setToolTip("a recording folder (X_*.pt chunks) or a folder that contains several recording folders")
        b_data.clicked.connect(self.add_data)
        b_file = QPushButton("add single X file...")
        b_file.setToolTip("a non-chunked windowed file saved by preprocessing (<id><name>_X.pt)")
        b_file.clicked.connect(self.add_file)
        b_clear = QPushButton("clear list")
        b_clear.clicked.connect(lambda: self.targets.clear())
        b_out = QPushButton("output folder...")
        b_out.clicked.connect(self.choose_out)
        self.out_label = QLabel("output: <project>/scores (two levels above each recording, like manual state files)")
        for w in (b_data, b_file, b_clear, b_out):
            row.addWidget(w)
        row.addWidget(self.out_label, 1)
        dl.addLayout(row)
        self.targets = QListWidget()
        self.targets.setMaximumHeight(140)
        dl.addWidget(self.targets)
        layout.addWidget(dbox)

        # --- options ---
        obox = QGroupBox("options")
        ol = QFormLayout(obox)
        self.channel = QSpinBox()
        self.channel.setRange(0, 15)
        self.channel.setToolTip("index of the ECoG/EEG channel in the windowed data")
        self.heuristics = QComboBox()
        self.heuristics.addItems(["lite", "none", "full"])
        self.heuristics.setToolTip("lite: fix isolated single-window states | full: also forbid Wake->REM | none")
        self.sigma = C.float_field("0", 0, 10)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["auto", "cnt", "nct"])
        self.layout_combo.setToolTip("cnt = channels first (GUI preprocessing), nct = windows first (conversions)")
        self.use_meta_rate = QCheckBox("data sample rate = settings-tab sample_rate (else inferred from 4-s windows)")
        self.use_meta_rate.setChecked(True)
        self.tag = QLineEdit()
        self.overwrite = QCheckBox("overwrite existing state files")
        ol.addRow("scoring channel", self.channel)
        ol.addRow("post-processing", self.heuristics)
        ol.addRow("probability smoothing sigma (windows)", self.sigma)
        ol.addRow("data layout", self.layout_combo)
        ol.addRow("", self.use_meta_rate)
        ol.addRow("file tag", self.tag)
        ol.addRow("", self.overwrite)
        layout.addWidget(obox)

        row = QHBoxLayout()
        self.run_btn = QPushButton("score")
        self.run_btn.clicked.connect(self.run_scoring)
        self.cancel_btn = QPushButton("stop after current recording")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self._task and self._task.cancel())
        self.open_btn = QPushButton("open selected result in manual scoring")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_result)
        for w in (self.run_btn, self.cancel_btn, self.open_btn):
            row.addWidget(w)
        layout.addLayout(row)
        self.log = C.LogPanel()
        layout.addWidget(self.log, 1)
        self._restore()

    # ---- model selection ----------------------------------------------------------------------
    def _restore(self) -> None:
        """Reload the last used run folder and model choice (from QSettings)."""
        s = C.settings()
        run = s.value("autoscore/run_dir", "", type=str)
        if run and Path(run, "manifest.json").exists():
            self._load_run(run, quiet=True)
            self.model_combo.setCurrentIndex(max(0, self.model_combo.findText(s.value("autoscore/choice", "", type=str))))

    def choose_run(self) -> None:
        """Ask for a pipeline run folder and list its models."""
        path = C.pick_dir(self, "select a pipeline run folder (contains manifest.json)", self.params, "runs")
        if path:
            self._load_run(path)

    def _load_run(self, path: str, quiet: bool = False) -> None:
        """List the model pairs of a run folder whose files exist.

        Args:
            path: Run folder (with ``manifest.json``).
            quiet: Do not show error dialogs (used when restoring at start-up).
        """
        try:
            choices = [c for c in discover_models(path) if c.exists()]
        except Exception as e:
            if not quiet:
                C.error(self, "pipeline run", str(e))
            return
        if not choices:
            if not quiet:
                C.error(self, "pipeline run", f"{path}: no trained heads with existing files in manifest.json")
            return
        self.choices = choices
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems([c.label for c in choices])
        self.model_combo.blockSignals(False)
        self._model_changed(0)
        C.settings().setValue("autoscore/run_dir", path)

    def choose_files(self) -> None:
        """Ask for an encoder and a head file and add them as a model choice."""
        enc = C.pick_file(self, "encoder checkpoint (.pt)", self.params, "PyTorch (*.pt *.pth)", "models")
        if not enc:
            return
        head = C.pick_file(self, "classifier head (.pt / .pkl)", self.params, "heads (*.pt *.pkl)", "models")
        if not head:
            return
        c = ModelChoice(f"files: {Path(head).name} on {Path(enc).name}", enc, head)
        self.choices.append(c)
        self.model_combo.addItem(c.label)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _model_changed(self, i: int) -> None:
        """Show the chosen pair's file paths and remember the choice."""
        if 0 <= i < len(self.choices):
            c = self.choices[i]
            self.model_label.setText(f"encoder: {c.encoder}\nhead: {c.head}")
            C.settings().setValue("autoscore/choice", c.label)

    @property
    def choice(self) -> ModelChoice | None:
        """The selected ``ModelChoice``, or None."""
        i = self.model_combo.currentIndex()
        return self.choices[i] if 0 <= i < len(self.choices) else None

    # ---- data ---------------------------------------------------------------------------------
    def add_data(self) -> None:
        """Ask for a folder and add every recording found in it."""
        path = C.pick_dir(self, "recording folder, or a folder of recording folders", self.params, "score_data")
        if not path:
            return
        try:
            folders = expand_targets(path)
        except Exception as e:
            C.error(self, "data folder", str(e))
            return
        self._add_targets(folders)
        self.log.add(f"{len(folders)} recording(s) found in {path}")

    def add_file(self) -> None:
        """Ask for a single windowed X file and add it."""
        path = C.pick_file(self, "windowed data file", self.params, "Data files (*.pt *.npy)", "score_data")
        if path:
            self._add_targets(expand_targets(path))

    def _add_targets(self, folders) -> None:
        """Add targets to the list (checked), skipping ones already listed."""
        known = {self.targets.item(i).data(Qt.UserRole) for i in range(self.targets.count())}
        for f in folders:
            if str(f) in known:
                continue
            item = QListWidgetItem(str(f))
            item.setData(Qt.UserRole, str(f))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.targets.addItem(item)

    def choose_out(self) -> None:
        """Ask for one output folder for all state files (default: each project's ``scores``)."""
        path = C.pick_dir(self, "folder for state files", self.params, "score_out")
        if path:
            self.out_dir = path
            self.out_label.setText(f"output: {path}")

    def _checked_targets(self) -> list[str]:
        """Return the targets that are ticked."""
        return [self.targets.item(i).data(Qt.UserRole) for i in range(self.targets.count())
                if self.targets.item(i).checkState() == Qt.Checked]

    # ---- run ----------------------------------------------------------------------------------
    def run_scoring(self) -> None:
        """Check the inputs and start scoring in the background.

        Targets are grouped by output folder (one ``ScoreRequest`` each). If a training run
        is using the GPU, the user is asked first.
        """
        if self._task is not None and self._task.running:
            return
        choice = self.choice
        if choice is None:
            C.error(self, "scoring", "choose a model first")
            return
        targets = self._checked_targets()
        if not targets:
            C.error(self, "scoring", "add at least one data folder")
            return
        try:
            sigma = C.read_float(self.sigma, "smoothing sigma")
        except ValueError as e:
            C.error(self, "scoring", str(e))
            return
        groups = {}
        for t in targets:  # grouped so each output folder gets one request
            # default: the project's scores/ folder, like manual state files (storage.construct_paths)
            out = self.out_dir or str(Path(t).parent.parent / "scores")
            groups.setdefault(out, []).append(t)
        if self.gpu_busy() and M.device(self.params) == "cuda" and not C.confirm(
                self, "GPU in use", "a training run is using the GPU; scoring at the same time can run out of GPU "
                                    "memory. Score anyway?"):
            return
        meta_rate = M.get_float(self.params, "sample_rate") if self.use_meta_rate.isChecked() else None
        requests = []
        for out, ts in groups.items():
            Path(out).mkdir(parents=True, exist_ok=True)
            r = ScoreRequest(targets=ts, out_dir=out, channel=self.channel.value(),
                             heuristics=self.heuristics.currentText(), smooth_sigma=sigma,
                             layout=self.layout_combo.currentText(), metadata_sample_rate=meta_rate,
                             tag=self.tag.text().strip(), overwrite=self.overwrite.isChecked(),
                             device=M.get_str(self.params, "device", "auto"))
            requests.append(r)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.results = []
        self._task = run_task(self, score_requests, choice, requests, on_done=self._done, on_error=self._failed,
                              on_progress=self.log.add, on_finished=self._finished)

    def _done(self, result) -> None:
        """Log the outcomes, record the model in the metadata and enable 'open in manual scoring'."""
        outcomes, info = result
        self.results = [o for o in outcomes if o.ok]
        failed = [o for o in outcomes if not o.ok]
        self.log.add(f"finished: {len(self.results)} scored, {len(failed)} not scored")
        for o in failed:
            self.log.add(f"  {Path(o.target).name}: {o.message}")
        self.params["autoscoring"] = {"model": info, "scored": [o.states_path for o in self.results]}
        self.open_btn.setEnabled(bool(self.results))

    def _failed(self, msg: str) -> None:
        """Log and show an error that stopped the whole job (e.g. the model failed to load)."""
        self.log.add("FAILED: " + msg.split("\n")[0])
        C.error(self, "scoring failed", msg)

    def _finished(self) -> None:
        """Re-enable the score button."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _open_result(self) -> None:
        """Send the selected (or first) scored recording to the manual-scoring tab."""
        if not self.results:
            return
        sel = self.targets.currentItem()
        chosen = next((o for o in self.results if sel and o.target == sel.data(Qt.UserRole)), self.results[0])
        self.open_in_labeling.emit(chosen.target, chosen.states_path)
