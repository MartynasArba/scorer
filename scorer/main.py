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
#fix up obx util funcs to take metadata correctly
#why do time values in conv obx csvs start at 3?
# could add additional qc metrics for converted obx files
# #bugfix motion sensor conversiosn - messed up files
#add warning to rec conversion by checking file quality, should be done each chunk as wires might disconnect
#add no-overwrite check
#add pre-scoring utils tab

#now ylim changes dynamically - should be prevented and kept same across samples
#REWORK PLOTS.PY TO PREVENT DYNAMIC YLIMS
#add setting for ylim = infer, standard, infer_ephys
# solution: only pass ylims for ecog/emg files
#might need to zero-center and standardize ylims for signals/sum powers
#ylims also need to be more adjustable, and probably only for ecog/emg, as other values are pretty much standardized
#get back chunk numbers in console (label is frozen), remove notch warning
#pass ylims = infer to metadata to standardize for every channel
#pass ylims = standard to do 0 center, 0.2 spread
#
#
#BUGS:
#if calling preprocessing multiple times, metadata gets fucked (ecog, emg repeats etc) - @grok is this true?
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
#might be smart to dirsregard more noise, so scale without top 10% (in bandpows)
#future development: implement threading to prevent freezing? 
#
#automatic scoring:
#not yet started
#implement at least one model
#remember to turn grad back on if training!
#
#manual labeling:
#make it more pretty - for example, freq plot lims
#decide how single label, multi scorer should be handled
#add scorer name to metadata and state save files correctly
#save progress automatically every N scores? maybe add checkmark, overwrite option?
#jump to next unscored sample
#
#report:
#not yet started
#maybe export json report?
#for multilabel, add agreement/confidence 