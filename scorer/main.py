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
#if calling preprocessing multiple times, metadata gets fucked (ecog, emg repeats etc)
#in loaders, allow folder selection and load all data, because chunked + windowed data is now saved as separate files. 
# This could be a good thing, but then should explicitly be implemented.
#remove warning from notch button
#remove prints from preprocessing
#notch is still weird, should test it more
#
#MISSING FEATURES:
# add warnings for settings if invalid values are set
# add low_memory option which would load only a specified amount of chunks(?)
#
#preprocessing:
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
#remember to turn grad back on if training!
#
#manual labeling:
#make it more pretty - for example, freq plot lims
#decide how single label, multi scorer should be handled
#add scorer name to metadata correctly
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#
#report:
#not yet started
#maybe export json report?
#for multilabel, add agreement/confidence 