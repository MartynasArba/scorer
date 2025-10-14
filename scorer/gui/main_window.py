from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.widgets import SleepGUI, ChopWidget

class SleepWindow(QMainWindow):
    def __init__(self, dataset = None):
        super().__init__()
        
        #window options
        self.setWindowTitle("Scoring GUI")
        self.setGeometry(100, 100, 1800, 1000)
        
        #create multiple tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # initialize widgets
        self.scoring_tab = SleepGUI(dataset = None)
        self.chopping_tab = ChopWidget()

        #add as tabs
        tabs.addTab(self.scoring_tab, "Score data")
        tabs.addTab(self.chopping_tab, "Chop data")