import sys
import torch
from PyQt5.QtWidgets import QApplication
from gui.main_window import SleepWindow

#later enabled in automatic scoring tab
torch.set_grad_enabled(False) 

#main script to run app

app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())

#TODO:
#
#generally save metadata whenever updating 
#
#  add low_memory option which would load only a specified amount of chunks
#
#overwrite doesn't fully overwrite as appending happens in chunk mode. should be fixed.
#
#preprocessing:
#think about filtering - what filter is best? current option distorts a LOT
#bandpows seem very prone to outliers?
#add status bar or label instead of prints
#future development: implement threading to prevent freezing, 
# move path construciton to a separate func,
#think about chunk data saving as it's currently being loaded and joined. Maybe modify loader?
#
#automatic scoring:
#not yet started
#implement at least one model
#remember to run grad back on!
#
#manual labeling:
#fft is very broken???
#fix ylims for all params
#add an option to load and plot multiple scores
#add scorer name to metadata, add option to append name to state file
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#for multilabel, add agreement/confidence 
#
#report:
#not yet started
#maybe export json report?