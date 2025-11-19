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
#BUGS:
#
#fix notch filtering, now it's fucked (bandpass instead of bandstop, but bandstop is not recommended) // or exclude 50Hz from emg somehow 
#somehow channel names after preprocessing end up strange
#
#MISSING FEATURES:
#generally save metadata whenever updating 
#
# add warnings for settings if invalid values are set
# add low_memory option which would load only a specified amount of chunks
#
#preprocessing:
#
# add loading from onebox
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
#add scrolling to viewer?
#fix ylims for all params
#add axis labels
#add an option to load and plot multiple scores
#add scorer name to metadata, add option to append name to state file
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#for multilabel, add agreement/confidence 
#
#report:
#not yet started
#maybe export json report?