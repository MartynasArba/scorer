from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel,
    QFileDialog, QSlider, QRadioButton, 
    QButtonGroup, QCheckBox
)

from PyQt5 import QtCore

import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from gui.plots import plot_signals, plot_spectrogram
from data.storage import construct_paths, save_pickled_states, load_pickled_states
from data.loaders import SleepSignals

class SleepGUI(QWidget):
    """
    class of the labeling GUI tab
    """
    def __init__(self, dataset: SleepSignals = None) -> None:
        """
        creates main interface
        """
        super().__init__()
        self.path = ''
        self.score_save_path = ''
        self.data_path = ''
        
        self.metadata = {}
        
        self.dataset = dataset
        self.current_idx = 0
        self.scale = 1
        
        self.ecog_ylim = [0, 0]
        self.emg_ylim = [0, 0]
        self.ecog_ylim_defaults = [0, 0]
        self.emg_ylim_defaults = [0, 0]
        self.yscale = 1
        
        #empty copy array to store and load data
        self.states = np.array([], dtype = int)
        self.label_whole_screen = False
        
        layout = QVBoxLayout(self)

        # canvas area for matplotlib plots
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        #labeling layout
        self.labeling_layout = QHBoxLayout()
        layout.addLayout(self.labeling_layout)

        # controls       
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)
        
        #sliders
        slider_layout = QVBoxLayout()
        layout.addLayout(slider_layout)
    
        # navigation
        btn_load = QPushButton("load file")
        btn_load.clicked.connect(self.select_dataset)
        control_layout.addWidget(btn_load)
        
        btn_save_states = QPushButton("save annotated sleep states")
        btn_save_states.clicked.connect(self.save_states)
        btn_save_states.setShortcut("Ctrl+s")
        control_layout.addWidget(btn_save_states)
        
        btn_load_states = QPushButton("load saved states")
        btn_load_states.clicked.connect(self.load_states)
        control_layout.addWidget(btn_load_states)
        
        btn_prev = QPushButton("prev frame (<)")
        btn_prev.clicked.connect(self.prev)
        btn_prev.setShortcut("<")
        control_layout.addWidget(btn_prev)

        btn_next = QPushButton("next frame (>)")
        btn_next.clicked.connect(self.next)
        btn_next.setShortcut(">")
        control_layout.addWidget(btn_next)

        btn_jump = QPushButton("jump to frame")
        btn_jump.clicked.connect(self.jump)
        control_layout.addWidget(btn_jump)

        btn_plus = QPushButton("increase time scale (+)")
        btn_plus.clicked.connect(self.increase_scale)
        btn_plus.setShortcut("+")
        control_layout.addWidget(btn_plus)

        btn_minus = QPushButton("decrease time scale (-)")
        btn_minus.clicked.connect(self.decrease_scale)
        btn_minus.setShortcut("-")
        control_layout.addWidget(btn_minus)
        
        btn_y_plus = QPushButton("+ y scale")
        btn_y_plus.clicked.connect(self.increase_yscale)
        control_layout.addWidget(btn_y_plus)

        btn_y_minus = QPushButton("- y scale")
        btn_y_minus.clicked.connect(self.decrease_yscale)
        control_layout.addWidget(btn_y_minus)
        
        btn_reset = QPushButton("reset")
        btn_reset.clicked.connect(self.reset_settings)
        control_layout.addWidget(btn_reset)
        
        #labeling button group
        
        self.labeling_group = QButtonGroup()
        
        labeling_buttons = [QRadioButton('Unknown: 0'),
                   QRadioButton('Awake: 1'),
                   QRadioButton('NREM: 2'),
                   QRadioButton('IS: 3'),
                   QRadioButton('REM: 4')]
        
        for i, b in enumerate(labeling_buttons):
            self.labeling_group.addButton(b, id = i)
            b.setShortcut(str(i))
            self.labeling_layout.addWidget(b)
            
        self.labeling_group.setExclusive(True)
        self.labeling_group.buttonClicked[int].connect(self.label_data)
        # checkbox whether to label the whole screen or just sample
        whole_screen_check = QCheckBox("Label the whole screen?")
        whole_screen_check.stateChanged.connect(self.label_screen_toggle)
        self.labeling_layout.addWidget(whole_screen_check)
        
        
        #slider widgets        
        self.slider_yscale = QSlider(value = 100, minimum = 5, maximum = 195, singleStep = 5, tracking = True)
        self.slider_yscale.setOrientation(QtCore.Qt.Horizontal)
        self.slider_yscale.valueChanged.connect(self.change_yscale)
        self.yscale_label = QLabel("Y scale: 1")
        slider_layout.addWidget(self.yscale_label)
        slider_layout.addWidget(self.slider_yscale)
       
       
        self.slider_frame = QSlider(value = 0, minimum = 0, maximum = 100, singleStep = 1, tracking = True)
        self.slider_frame.setOrientation(QtCore.Qt.Horizontal)
        self.slider_frame.valueChanged.connect(self.frame_slider_func)
        self.slider_frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.slider_frame_label)
        slider_layout.addWidget(self.slider_frame)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def select_dataset(self) -> None:
        """
        pop up window to select pre-processed data, then loads it to the scorer
        """
        file_name, _ = QFileDialog.getOpenFileName(self, caption = "Select file to load", directory = ".", filter = "Pickle files (*.pkl)")
        if file_name:
            try:
                #should automatically generate paths and metadata here
                self.data_path = file_name
                
                score_path, self.score_save_path = construct_paths(self.data_path)

                self.dataset = SleepSignals(data_path = self.data_path, score_path = score_path, augment = False, spectral_features = 'spectrogram')
                self.states = self.dataset.all_labels.to('cpu').numpy()
                self.current_idx = 0
                
                self.update_screen()
                self.status_label.setText(f"Loaded: {file_name}")
                
                self.ecog_ylim = [self.dataset.q01_0.item(), self.dataset.q99_0.item()]
                self.emg_ylim = [self.dataset.q01_1.item(), self.dataset.q99_1.item()]
                self.ecog_ylim_defaults = self.ecog_ylim.copy()
                self.emg_ylim_defaults = self.emg_ylim.copy()
                
                self.slider_frame.setMaximum(len(self.dataset))
                self.slider_frame.setValue(0)
                
                #start plotting
                self.update_screen()
                
            except Exception as e:
                self.status_label.setText(f"Error loading file: {e}")
        
        
    def get_data(self) -> torch.Tensor:
        """
        returns selected samples from the dataset, most relevant when scale != 0
        """
        if self.scale == 1:
            sample = self.dataset[self.current_idx][0]
            label = self.states[self.current_idx]       
            return sample, label
        
        # Concatenate multiple consecutive samples
        samples, spects, labels = [], [], []
        for i in range(self.current_idx, min(self.current_idx + self.scale, len(self.dataset))):
            sample = self.dataset[i][0]
            label = self.states[i]
            if isinstance(sample, tuple):
                samples.append(sample[0])
                spects.append(sample[1])
            else:
                samples.append(sample)
            labels.append(label)

        # Concatenate along time dimension - time dimension is 0 here
        if isinstance(sample, tuple):
            signals = torch.cat(samples, dim = 0)
            spectrograms = torch.cat(spects, dim = 2)
            return (signals, spectrograms), labels
        else:
            return torch.cat(samples, dim = 0), labels
        
    def label_data(self, value) -> None:
        """
        marks selection as some sleep state value
        """
        if self.label_whole_screen:
            self.states[self.current_idx:self.current_idx + self.scale] = value 
        else:
            self.states[self.current_idx] = value
        
        self.update_screen()
        
    def update_screen(self) -> None:
        """
        runs helpers to update plots and sync labels
        """
        self.update_plot()
        self.update_labels()

    def update_labels(self) -> None:
        """
        syncs labels
        """
        #update sliders and labels
        self.slider_frame.blockSignals(True)
        self.slider_yscale.blockSignals(True)

        self.status_label.setText(f"frame index: {self.current_idx}, time scale: {self.scale} frames; y scale: {self.yscale}")
        self.slider_frame.setValue(self.current_idx)
        self.slider_frame_label.setText(f"frame: {self.current_idx}")
        self.slider_yscale.setValue(int(self.yscale * 100))
        self.yscale_label.setText(f"Y scale: {self.yscale}")

        self.slider_frame.blockSignals(False)
        self.slider_yscale.blockSignals(False)
        
        #updating state buttons
        self.labeling_group.blockSignals(True)
        self.labeling_group.button(self.states[self.current_idx]).setChecked(True)
        self.labeling_group.blockSignals(False)
        
    
    def update_plot(self) -> None:
        """
        generates plots and updates the widgets
        """
        sample, label = self.get_data()
        # Handle tuple (signals, spectrogram)
        if isinstance(sample, tuple):
            signals, spect = sample
            fig = plot_signals(signals.to("cpu"), label, ecog_ylim = self.ecog_ylim, emg_ylim = self.emg_ylim)
            fig2 = plot_spectrogram(spect.to("cpu"))
            figs = [fig, fig2]
        else:
            fig = plot_signals(sample.to("cpu"), label)
            figs = [fig]

        # Clear old canvases
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Embed figures in Qt
        for fig in figs:
            canvas = FigureCanvas(fig)
            self.canvas_layout.addWidget(canvas)
            canvas.draw()
            plt.close(fig) #close fig after embedding
            
    def save_states(self) -> None:
        """
        saves labeled states
        """
        _, self.score_save_path = construct_paths(self.data_path)
        print(f'saved in {self.score_save_path}')
        save_pickled_states(self.states, self.score_save_path)
    
    def load_states(self) -> None:
        """
        loads saved states
        """
        file_name, _ = QFileDialog.getOpenFileName(self, caption = "Select states file to load", directory = ".", filter = "Pickle files (*.pkl)")
        if file_name:
            self.states = load_pickled_states(file_name)
        self.update_screen()
            
    def label_screen_toggle(self) -> None:
        """
        toggles whether to label the whole screen at once
        """
        self.label_whole_screen = not self.label_whole_screen

    def next(self) -> None:
        """
        updates the index and screen = goes to the next frame
        """
        self.current_idx = min(self.current_idx + 1, len(self.dataset) - 1)
        self.update_screen()

    def prev(self) -> None:
        """
        goes to the previous frame
        """
        self.current_idx = max(self.current_idx - 1, 0)
        self.update_screen()

    def jump(self) -> None:
        """
        jumps to a selected frame via a popup
        """
        idx, ok = QInputDialog.getInt(self, "Jump", "Enter frame index:", value=self.current_idx)
        if ok and 0 <= idx < len(self.dataset):
            self.current_idx = idx
            self.update_screen()
            
    def frame_slider_func(self, value: int) -> None:
        """
        updates the frame based on slider input
        """
        if 0 <= value < len(self.dataset):
            self.current_idx = value
            self.update_screen()
            
    def change_yscale(self, value: int) -> None:
        """
        updates y limits based on slider input
        """
        
        self.yscale = value * 0.01
        self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
        self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
        self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
        self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
        self.update_screen()
    
    def increase_yscale(self) -> None:
        """
        increases y scale
        """     
        self.yscale += 0.05
        self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
        self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
        self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
        self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
        
        self.update_screen()
        
    def decrease_yscale(self) -> None:
        """
        decreases y scale
        """
        self.yscale -= 0.05
        
        if self.yscale > 0:
            self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
            self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
            self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
            self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
            
            self.update_screen()
        
        else:
            pass
    
    def increase_scale(self) -> None:
        """
        increases time scale, updates modifier on how many frames to take
        """
        self.scale += 1
        self.update_screen()

    def decrease_scale(self) -> None:
        """
        decreases time scale by decreasing the modifier
        """
        if self.scale > 1:
            self.scale -= 1
        self.update_screen()
        
    def reset_settings(self) -> None:
        """
        resets settings to default values
        """
        # self.current_idx = 0 #debatable whether it should change, probably not
        self.scale = 1
        self.ecog_ylim = self.ecog_ylim_defaults.copy()
        self.emg_ylim = self.emg_ylim_defaults.copy()
        self.yscale = 1
        self.label_whole_screen = False
        
        self.update_screen()