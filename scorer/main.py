import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import SleepWindow

#main script to run app

app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())

#TODO:
#
#add user defined optional name in settings to be more readable next to unique ID
#
#overwrite doesn't fully overwrite as appending happens in chunk mode. should be fixed.
#
#preprocessing:
#
#add status bar or label instead of prints
#future development: implement threading to prevent freezing, 
# move path construciton to a separate func,
#think about chunk data saving as it's currently being loaded and joined. Maybe modify loader?
#
#automatic scoring:
#not yet started
#implement at least one model
#
#manual labeling:
#change spectrogram to multi-channel viewing
#add an option to load and plot multiple scores
#add scorer name to metadata setting, add option to append name
#set spectral param of SleepDataset somehow and handle both cases
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#for multilabel, add agreement/confidence 
#
#report:
#not yet started
#maybe export json report?

#automatic labeling:
#implement at least one model
#add "agreement"/"confidence" in plot