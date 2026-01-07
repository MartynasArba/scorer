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
#chop_by_state depreciated, make sure it works in script as it is needed for testing of models

#time axis:
    #get recording start time in preprocessing when loading csv;
        #in chunks, get only start time of chunk 0
    #if start/end times are passed, calculate SKIP/PARSE/SKIP datapoints (rows), then reset start time
    #make sure it's overwritten and displayed somewhere (prob print) each time a .csv is loaded

    #implement time display in labeling: start time + idx * win len (datapoints*sr)
    
#notch is still weird, should test it more
#metadata label doesn't update when setting params after loading?

#
#TEST:
#if calling preprocessing multiple times, metadata gets * (ecog, emg repeats etc) - @grok is this true?
#notch is still weird, should test it more (or depreciate it)
#
#BUGS:
#fix fourier lims (or move to a different plot) - is fourier even useful?
#fix plot axis labels
#
#MISSING FEATURES:
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
#add channel toggles
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