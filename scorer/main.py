
import sys

from PyQt5.QtWidgets import QApplication

from data.loaders import SleepSignals
from gui.main_window import SleepGUI
# from data.preprocessing import from_Oslo_csv

#load and chop data
# path = r'data\raw\trial_1_mouse_b1aqm2.csv'
# from_Oslo_csv(path)

data_path = r'C:\Users\marty\Projects\scorer\data\processed\trial_1_mouse_b1aqm2_X.pkl'
score_path = r'C:\Users\marty\Projects\scorer\data\processed\trial_1_mouse_b1aqm2_y.pkl'

app = QApplication(sys.argv)
dataset = SleepSignals(data_path = data_path, score_path = score_path, augment=False, compute_spectrogram=True)
gui = SleepGUI(dataset)
gui.show()
sys.exit(app.exec_())