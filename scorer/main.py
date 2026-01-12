import sys
import torch
from PyQt5.QtWidgets import QApplication
from scorer.gui.main_window import SleepWindow

#this is because only pre-trained models are supported
torch.set_grad_enabled(False) 

#main loop
app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())