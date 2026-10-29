"""Main window: one shared metadata dict, seven tabs, a status bar, and guards against losing labels or a run."""
from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from scorer.gui.core import meta as M
from scorer.gui.widgets.autoscoring import AutoScoringWidget
from scorer.gui.widgets.labeling import LabelingWidget
from scorer.gui.widgets.preprocessing import PreprocessWidget
from scorer.gui.widgets.report import ReportWidget
from scorer.gui.widgets.settings import SettingsWidget
from scorer.gui.widgets.training import TrainingWidget
from scorer.gui.widgets.utilities import UtilWidget


class SleepWindow(QMainWindow):
    """Main window: seven tabs sharing one metadata dict, plus a status bar.

    Tabs: settings and metadata, utilities, preprocess data, automatic scoring, manual
    scoring, report, model training. The window wires them together (scoring result ->
    manual review, finished training run -> automatic scoring) and guards against
    closing with unsaved labels or a running training.

    Args:
        metadata: Shared metadata dict; None starts a new session.
    """
    def __init__(self, metadata: dict | None = None) -> None:
        """Create the tabs and connect their signals."""
        super().__init__()
        self.setWindowTitle("Scoring GUI")
        self.resize(1600, 900)
        self.metadata = metadata if metadata is not None else M.new_metadata()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.settings_tab = SettingsWidget(self.metadata)
        self.util_tab = UtilWidget(self.metadata)
        self.preprocess_tab = PreprocessWidget(self.metadata)
        self.auto_scoring_tab = AutoScoringWidget(self.metadata)
        self.scoring_tab = LabelingWidget(self.metadata)
        self.report_tab = ReportWidget(self.metadata)
        self.training_tab = TrainingWidget(self.metadata)
        for w, name in ((self.settings_tab, "settings and metadata"), (self.util_tab, "utilities"),
                        (self.preprocess_tab, "preprocess data"), (self.auto_scoring_tab, "automatic scoring"),
                        (self.scoring_tab, "manual scoring"), (self.report_tab, "report"),
                        (self.training_tab, "model training")):
            self.tabs.addTab(w, name)

        # tabs that change the shared metadata tell the others
        self.preprocess_tab.preprocessing_done.connect(lambda _: self.settings_tab.refresh())
        self.settings_tab.metadata_changed.connect(self.report_tab._update_win_info)
        self.auto_scoring_tab.open_in_labeling.connect(self._review_scoring)
        self.training_tab.use_run_for_scoring.connect(self._score_with_run)
        self.auto_scoring_tab.gpu_busy = self.training_tab.is_running
        self.statusBar().showMessage(f"session {self.metadata.get('scoring_started')}")

    def _review_scoring(self, data_folder: str, states_path: str) -> None:
        """Switch to manual scoring and open a scoring result for review."""
        self.tabs.setCurrentWidget(self.scoring_tab)
        self.scoring_tab.open_result(data_folder, states_path)

    def _score_with_run(self, run_dir: str) -> None:
        """Switch to automatic scoring with the models of a finished training run."""
        self.tabs.setCurrentWidget(self.auto_scoring_tab)
        self.auto_scoring_tab._load_run(run_dir)

    def closeEvent(self, event) -> None:
        """Ask before closing while training runs or labels are unsaved.

        A running training is stopped (finished stages are kept); unsaved labels can be
        saved, discarded or the close cancelled.
        """
        if self.training_tab.is_running():
            if QMessageBox.question(self, "training running", "A training run is still going. Stop it and close? "
                                    "(finished stages are kept and can be resumed)",
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                event.ignore()
                return
            self.training_tab.shutdown()
        if self.scoring_tab.has_unsaved():
            r = QMessageBox.question(self, "unsaved labels", "Manual labels are not saved. Save before closing?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                event.ignore()
                return
            if r == QMessageBox.Save:
                self.scoring_tab.save_states()
        event.accept()
