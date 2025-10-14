from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel,
    QFileDialog
)

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
        
        layout = QVBoxLayout(self)

        # Canvas area (for matplotlib plots)
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        # Controls       
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)
        
        btn_load = QPushButton("Load File")
        btn_load.clicked.connect(self.select_dataset)
        control_layout.addWidget(btn_load)

        btn_prev = QPushButton("Prev")
        btn_prev.clicked.connect(self.prev)
        control_layout.addWidget(btn_prev)

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(self.next)
        control_layout.addWidget(btn_next)

        btn_jump = QPushButton("Jump")
        btn_jump.clicked.connect(self.jump)
        control_layout.addWidget(btn_jump)

        btn_plus = QPushButton("+ Scale")
        btn_plus.clicked.connect(self.increase_scale)
        control_layout.addWidget(btn_plus)

        btn_minus = QPushButton("- Scale")
        btn_minus.clicked.connect(self.decrease_scale)
        control_layout.addWidget(btn_minus)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        if self.dataset:
            #set plotting params
            self.ecog_ylim = [dataset.q01_0.item(), dataset.q99_0.item()]
            self.emg_ylim = [dataset.q01_1.item(), dataset.q99_1.item()]
            #start plotting
            self.update_plot()
    
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

        self.status_label.setText(f"Index {self.current_idx}, Scale {self.scale}")

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