import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class SleepGUI(QMainWindow):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.current_idx = 0
        self.scale = 1

        self.setWindowTitle("Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # Canvas area (for matplotlib plots)
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        # Controls
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

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

        self.update_plot()
        
    def get_data(self):
        sample, label = self.dataset[self.current_idx]  
        label = [label.item()]        
        
        if self.scale != 1: 
            selected_data = list(sample)
            for idx in range((self.current_idx * self.scale), (self.current_idx + self.scale) * self.scale, self.scale):
                #address two options: len(self.dataaset[idx][0]) = 1, 2
                if not isinstance(sample, tuple):
                    append_signal = self.dataset[idx][0] 
                    selected_data = torch.cat((selected_data, append_signal), dim = 0)
                else:
                    append_signal = self.dataset[idx][0][0]
                    append_spect = self.dataset[idx][0][1]
                    selected_data[0] = torch.cat((selected_data[0], append_signal), dim = 0)
                    selected_data[1] = torch.cat((selected_data[1], append_spect), dim = 2)
                label.append(self.dataset[idx][1].item())
            sample  = tuple(selected_data)

        return sample, label

    def update_plot(self):

        sample, label = self.get_data()
        
        # Handle tuple (signals, spectrogram)
        if isinstance(sample, tuple):
            signals, spect = sample
            fig = _plot_signals(signals.to("cpu"), label)
            fig2 = _plot_spectrogram(spect.to("cpu"), label)
            figs = [fig, fig2]
            print(spect.size())
        else:
            fig = _plot_signals(sample.to("cpu"), label)
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


if __name__ == "__main__":
    from scorer.data_tools import SleepSignals, _plot_signals, _plot_spectrogram

    app = QApplication(sys.argv)
    dataset = SleepSignals(file_dir="./data/", augment=False, compute_spectrogram=True)
    gui = SleepGUI(dataset)
    gui.show()
    sys.exit(app.exec_())


comment this to stop it
#if plotting sleep state, should handle the timescale correctly
#should add some controls for ylim
#non-norm sum power might be useful, as well as in specific bands