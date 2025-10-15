from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel,
    QFileDialog, QSlider
)

from PyQt5 import QtCore

import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from gui.plots import plot_signals, plot_spectrogram
from data.preprocessing import from_Oslo_csv
from data.loaders import SleepSignals

class SleepGUI(QWidget):
    def __init__(self, dataset = None):
        super().__init__()
        self.dataset = dataset
        self.current_idx = 0
        self.scale = 1
        
        self.ecog_ylim = [0, 0]
        self.emg_ylim = [0, 0]
        self.ecog_ylim_defaults = [0, 0]
        self.emg_ylim_defaults = [0, 0]
        self.yscale = 1
        
        layout = QVBoxLayout(self)

        # Canvas area (for matplotlib plots)
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        # Controls       
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)
        
        #Sliders
        slider_layout = QVBoxLayout()
        layout.addLayout(slider_layout)
        
        btn_load = QPushButton("load file")
        btn_load.clicked.connect(self.select_dataset)
        control_layout.addWidget(btn_load)

        btn_prev = QPushButton("prev frame")
        btn_prev.clicked.connect(self.prev)
        control_layout.addWidget(btn_prev)

        btn_next = QPushButton("next frame")
        btn_next.clicked.connect(self.next)
        control_layout.addWidget(btn_next)

        btn_jump = QPushButton("jump to frame")
        btn_jump.clicked.connect(self.jump)
        control_layout.addWidget(btn_jump)

        btn_plus = QPushButton("+ time scale")
        btn_plus.clicked.connect(self.increase_scale)
        control_layout.addWidget(btn_plus)

        btn_minus = QPushButton("- time scale")
        btn_minus.clicked.connect(self.decrease_scale)
        control_layout.addWidget(btn_minus)
        
        btn_y_plus = QPushButton("+ y scale")
        btn_y_plus.clicked.connect(self.increase_yscale)
        control_layout.addWidget(btn_y_plus)

        btn_y_minus = QPushButton("- y scale")
        btn_y_minus.clicked.connect(self.decrease_yscale)
        control_layout.addWidget(btn_y_minus)
        
        self.slider_yscale = QSlider(value = 100, minimum = 5, maximum = 195, singleStep = 5, tracking = True)
        self.slider_yscale.setOrientation(QtCore.Qt.Horizontal)
        self.slider_yscale.valueChanged.connect(self.change_yscale)
        self.yscale_label = QLabel("Y scale: 1")
        # slider_yscale.valueChanged.connect(lambda v: self.yscale_label.setText(f"Y scale: {v/100}"))
        slider_layout.addWidget(self.yscale_label)
        slider_layout.addWidget(self.slider_yscale)
       
       
        self.slider_frame = QSlider(value = 0, minimum = 0, maximum = 100, singleStep = 1, tracking = True)
        self.slider_frame.setOrientation(QtCore.Qt.Horizontal)
        self.slider_frame.valueChanged.connect(self.frame_slider_func)
        self.slider_frame_label = QLabel("Frame: 0")
        # self.slider_frame.valueChanged.connect(lambda v: self.slider_frame_label.setText(f"Frame: {v}"))
        slider_layout.addWidget(self.slider_frame_label)
        slider_layout.addWidget(self.slider_frame)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def select_dataset(self):
        file_name, _ = QFileDialog.getOpenFileName(self, caption = "Select file to chop", directory = ".", filter = "Pickle files (*.pkl)")
        if file_name:
            try:
                data_path = file_name
                score_path = data_path[:-5] + 'y.pkl'
                self.dataset = SleepSignals(data_path = data_path, score_path = score_path, augment=False, compute_spectrogram=True)
                self.current_idx = 0
                self.scale = 1
                self.update_plot()
                self.status_label.setText(f"Loaded: {file_name}")
                
                self.ecog_ylim = [self.dataset.q01_0.item(), self.dataset.q99_0.item()]
                self.emg_ylim = [self.dataset.q01_1.item(), self.dataset.q99_1.item()]
                self.ecog_ylim_defaults = self.ecog_ylim.copy()
                self.emg_ylim_defaults = self.emg_ylim.copy()
                
                self.slider_frame.setMaximum(len(self.dataset))
                self.slider_frame.setValue(0)
                
                #start plotting
                self.update_plot()
                
            except Exception as e:
                self.status_label.setText(f"Error loading file: {e}")
        
        
    def get_data(self):
        if self.scale == 1:
            sample, label = self.dataset[self.current_idx]
            return sample, [label.item()]   
        
        # Concatenate multiple consecutive samples
        samples, spects, labels = [], [], []
        for i in range(self.current_idx, min(self.current_idx + self.scale, len(self.dataset))):
            sample, label = self.dataset[i]
            if isinstance(sample, tuple):
                samples.append(sample[0])
                spects.append(sample[1])
            else:
                samples.append(sample)
            labels.append(label.item())

        # Concatenate along time dimension - time dimension is 0 here
        if isinstance(sample, tuple):
            signals = torch.cat(samples, dim = 0)
            spectrograms = torch.cat(spects, dim = 2)
            return (signals, spectrograms), labels
        else:
            return torch.cat(samples, dim = 0), labels

    def update_plot(self):

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

    def next(self):
        self.current_idx = min(self.current_idx + 1, len(self.dataset) - 1)
        self.update_plot()

    def prev(self):
        self.current_idx = max(self.current_idx - 1, 0)
        self.update_plot()

    def jump(self):
        idx, ok = QInputDialog.getInt(self, "Jump", "Enter frame index:", value=self.current_idx)
        if ok and 0 <= idx < len(self.dataset):
            self.current_idx = idx
            self.update_plot()
            
    def frame_slider_func(self, value):
        if 0 <= value < len(self.dataset):
            self.current_idx = value
            self.update_plot()
            
    def change_yscale(self, value):
        self.yscale = value * 0.01
        self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
        self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
        self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
        self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
        
        self.update_plot()
    
    def increase_yscale(self):     
        self.yscale += 0.05

        self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
        self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
        self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
        self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
        
        self.update_plot()
        
    def decrease_yscale(self):
        self.yscale -= 0.05
        
        if self.yscale > 0:
            self.ecog_ylim[0] = (self.ecog_ylim_defaults[0] * self.yscale)
            self.ecog_ylim[1] = (self.ecog_ylim_defaults[1] * self.yscale)
            self.emg_ylim[0] = (self.emg_ylim_defaults[0] * self.yscale)
            self.emg_ylim[1] = (self.emg_ylim_defaults[1] * self.yscale)
            
            self.update_plot()
        
        else:
            pass
    
    def increase_scale(self):
        self.scale += 1
        self.update_plot()

    def decrease_scale(self):
        if self.scale > 1:
            self.scale -= 1
        self.update_plot()
        
        
#another widget to run data chopping on selected data
class ChopWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.chopper = from_Oslo_csv

        layout = QVBoxLayout(self)

        self.label = QLabel("No file selected")
        layout.addWidget(self.label)

        btn = QPushButton("Select file to chop")
        btn.clicked.connect(self.select_file)
        layout.addWidget(btn)

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, caption = "Select file to chop", directory = ".", filter = "CSV files (*.csv)")
        if file_name:
            self.label.setText(file_name)
            if self.chopper:
                print(file_name)
                self.chopper(file_name, sep = '/')