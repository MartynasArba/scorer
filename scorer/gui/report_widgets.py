from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QCheckBox
)
from scorer.gui.plots import hypnogram
from scorer.data.storage import load_pickled_states, get_timearray_for_states
from scorer.data.report import generate_sleep_report

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ReportWidget(QWidget):
    """
    class for generating reports, wip
    """
    
    def __init__(self, metadata: dict = {}) -> None:
        super().__init__()
        
        self.params = metadata
        self.states_path = None
        
        #main layout
        self.layout = QVBoxLayout(self)
        
        #hypnogram screen
        self.plot_layout = QVBoxLayout()
        self.layout.addLayout(self.plot_layout)
        
        #label
        self.mainlabel = QLabel("Select a states .pkl file to load")
        self.layout.addWidget(self.mainlabel)
    
         # project path selection button
        states_path_btn = QPushButton('select states path')
        states_path_btn.clicked.connect(self.set_states_path)
        self.layout.addWidget(states_path_btn)
        
        #win len textbox
        tbox_layout = QHBoxLayout()
        winlen_label = QLabel("Enter window length used:")
        self.winlen_textbox = QLineEdit(self)
        self.winlen_textbox.setText('1000')
        tbox_layout.addWidget(winlen_label)
        tbox_layout.addWidget(self.winlen_textbox)
        self.layout.addLayout(tbox_layout)
        
        #generate hypnogram button
        hypno_layout = QHBoxLayout()        
        hypnogram_btn = QPushButton('generate a hypnogram')
        hypnogram_btn.clicked.connect(self.get_hypnogram)
        hypno_layout.addWidget(hypnogram_btn)
        self.plot_save_check = QCheckBox('save plot to file?')  #save to file?
        hypno_layout.addWidget(self.plot_save_check)
        self.layout.addLayout(hypno_layout)
        
        #get state report button
        statereport_layout = QHBoxLayout()
        get_report_btn = QPushButton('generate a state report')
        get_report_btn.clicked.connect(self.get_state_report)
        self.save_json_check = QCheckBox('save to .json file?')
        self.by_hour_check = QCheckBox('generate results by hour?')
        statereport_layout.addWidget(get_report_btn)
        statereport_layout.addWidget(self.save_json_check)
        statereport_layout.addWidget(self.by_hour_check)
        self.layout.addLayout(statereport_layout)
                
    def set_states_path(self) -> str:
        """dialog box to select states path"""
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                    caption="select states file to open",
                                                    directory=self.params.get('project_path', '.'),
                                                    filter="Pickle files (*.pkl)")
        self.states_path = file_name
        self.mainlabel.setText(f'file selected: {file_name}')
        return file_name

    def get_state_report(self) -> None:
        """
        generates the following info:
        - number of states
        - total time in states
        - percentage of time in states
        - median, IQR state duration
        - saves result to json (should also add meta)
        - split by hour
        """
        path = self.states_path
        if path is None:
            self.mainlabel.setText('select a valid file')
            return
        states = load_pickled_states(self.states_path)
        times = get_timearray_for_states(states, win_len = int(self.winlen_textbox.text()), metadata = self.params)
        #ensure numpy states:
        states = np.array(states)

    def _save_plot(self, fig: Figure, path: str) -> None:
        """helper to save figure to file"""
        figpath = path[:-4] + '.png'
        fig.savefig(figpath, dpi = 300, transparent= True)

    def get_hypnogram(self) -> Figure:
        """generates a hypnogram and saves it to file"""
        #load pickle
        if self.states_path is not None:
            states = load_pickled_states(self.states_path)
        else:
            self.mainlabel.setText('no states file selected')
            return
        
        times = get_timearray_for_states(states, win_len = int(self.winlen_textbox.text()), metadata = self.params)
        fig = hypnogram(states = states, time_array = times, metadata = self.params)       
        
        if self.plot_save_check.isChecked():
            self._save_plot(fig, path = self.states_path)
            
        #clear old plots
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()