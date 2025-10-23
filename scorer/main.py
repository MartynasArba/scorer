import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import SleepWindow

#main script to run app

app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())

#checkmark for whether to overlay new scoring, load alternatives etc
#add option to load other scores, metadata should likely be reflected in file names
#add metadata and settings widget which can return both as dicts, then pass dicts to labeling widget as params in __init__

#track sample rate in metadata, relevant for time axis when plotting

# add fft or specific power options instead of spect, ideally all, then don't generate spectrograms (in loaders.py dataset)

# save progress automatically every N scores? maybe add checkmark? maybe add an overwrite option?
#json saving or other formats

# jump to next unscored sample button
# scale to more ephys channels / do this in dataset 

#automatic labeling:
#implement at least one model
#add "agreement"/"confidence" in plot

#add summary/report widget, option to save plots

#add options for loading similar, but different structures of data

#add cache?? so far not really needed, seems speedy enough

#expand preprocessing with filters etc.
