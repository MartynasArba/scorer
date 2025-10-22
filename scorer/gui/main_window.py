from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.labeling_widgets import SleepGUI
from gui.preprocessing_widgets import PreprocessWidget
from gui.settings_widgets import SettingsWidget
from gui.autoscoring_widgets import AutoScoringWidget
from gui.report_widgets import ReportWidget

class SleepWindow(QMainWindow):
    """
    (WIP) Class that creates the main application window
    Current tabs are:
    - settings and metadata
    - preprocessing
    - automatic scoring
    - manual scoring
    - report generation
    """
    
    def __init__(self, dataset = None):
        super().__init__()
        
        #window options
        self.setWindowTitle("Scoring GUI")
        self.setGeometry(100, 100, 1800, 1000)
        
        #create multiple tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # initialize widgets
        self.settings_tab = SettingsWidget()
        self.preprocess_tab = PreprocessWidget()
        self.auto_scoring_tab = AutoScoringWidget()
        self.scoring_tab = SleepGUI(dataset = None)
        self.report_tab = ReportWidget()

        #add as tabs
        tabs.addTab(self.settings_tab, "settings and metadata")
        tabs.addTab(self.preprocess_tab, "preprocess data")
        tabs.addTab(self.auto_scoring_tab, "automatically score data")
        tabs.addTab(self.scoring_tab, "manual scoring")
        tabs.addTab(self.report_tab, "report")