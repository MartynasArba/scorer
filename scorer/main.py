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
#TEST:
#if calling preprocessing multiple times, metadata gets fucked (ecog, emg repeats etc) - @grok is this true?
#notch is still weird, should test it more
#
#BUGS:
#fix fourier lims
#fix plot axis labels
#fix powers scaling in standard/infer_ephys modes
#
#MISSING FEATURES:
#utils:
#add "crop to time" for obx csvs to get only night
#add some option to visualize results of obx file validation in utils // "see if it's worth continuing"
# could add additional qc metrics for converted obx files
#
#settings:
# add warnings for settings if invalid values are set
#
#preprocessing:
#might be smart to dirsregard more noise, so scale without top 10% (in bandpows)
#future development: implement threading to prevent freezing? 
#
#automatic scoring:
#not yet started
#implement at least one model, for example, the old CNN
#remember to turn grad back on if training!
#ideas: finetune TimesFM / Moirai + add simple classifier on top
#add interface to use a pre-trained model
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