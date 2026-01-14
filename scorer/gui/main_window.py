from PyQt5.QtWidgets import QMainWindow, QTabWidget
from scorer.gui.labeling_widgets import SleepGUI
from scorer.gui.preprocessing_widgets import PreprocessWidget
from scorer.gui.settings_widgets import SettingsWidget
from scorer.gui.autoscoring_widgets import AutoScoringWidget
from scorer.gui.report_widgets import ReportWidget
from scorer.gui.util_widgets import UtilWidget

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
    
    def __init__(self, dataset: SleepGUI = None) -> None:
        """
        initializes the main window
        """
        super().__init__()
        
        #window options
        self.setWindowTitle("Scoring GUI")
        self.setGeometry(100, 100, 1600, 800)
        
        #create multiple tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # initialize metadata
        # params to track in metadata
        params =    [
            'scoring_started', 
            'project_path', 
            'scorer', 
            'animal_id', 
            'group', 
            'trial', 
            'sample_rate', 
            'time_channel',
            'ecog_channels', 
            'emg_channels',
            'ylim',
            'spectral_view',
            'device',
            'optional_tag'
            ]
        #create metadata dict (global, to be shared across widgets)
        self.metadata = {param : None for param in params}
        
        #initialize widgets
        self.settings_tab = SettingsWidget(self.metadata)
        self.util_tab = UtilWidget(self.metadata)
        self.preprocess_tab = PreprocessWidget(self.metadata)
        self.auto_scoring_tab = AutoScoringWidget(self.metadata)
        self.scoring_tab = SleepGUI(dataset = None, metadata = self.metadata)
        self.report_tab = ReportWidget()

        #add as tabs
        tabs.addTab(self.settings_tab, "settings and metadata")
        tabs.addTab(self.util_tab, "utilities")
        tabs.addTab(self.preprocess_tab, "preprocess data")
        tabs.addTab(self.auto_scoring_tab, "automatically score data")
        tabs.addTab(self.scoring_tab, "manual scoring")
        tabs.addTab(self.report_tab, "report")