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

#NOTE:
#create fresh metadata for each recording
#chop_by_state is depreciated and removed from gui, make sure it works somewhere in script as it is needed for testing of models
#make sure start timestamps match recordings, as they are only updated when loading from csv in preprocessing - or use specific metadata per recording
#
#TODO:
#---PRIORITY:
#missing axis labels
#spectral plots are not very informative, should be depreciated
#
#---TEST:
#if calling preprocessing multiple times, metadata gets * (ecog, emg repeats etc) - @grok is this true?
#notch is still weird, should test it more (or depreciate it)
#
#---MISSING FEATURES/IDEAS:
#
#settings: add default values or specify options
#
#utils:
# could add additional qc metrics for converted obx files
#
#preprocessing:
#future development: prevent freezing? 
#
#automatic scoring:
#not yet started
#aiming for these models:
## rules-based
## CNN
## TimesFM/Moirai pretrained + classifier
## add selection of which channels to use
#remember to turn grad back on if training!
# think about an interface for a pre-trained model
#
#manual labeling:
#add channel toggles(?)
#add autoscale click
#add scorer name to metadata and state save files correctly
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#decide how single label, multi scorer should be handled
#
#report:
#not yet started
#maybe export json report?
#for multilabel, add agreement/confidence 