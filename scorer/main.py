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
#preprocessing:
#calculate sum powers:
#To compare ECoG power across genotypes and different mice, 
# raw data were normalized to the average total power in the 
# 0.5–30 Hz frequency range during NREM sleep per mouse 
# and average power was calculated using the MATLAB bandpower() function.
#calculate band power:
#get abs Hilbert transform after filtering (following Bojarskaite 2020)
#smoothen with Gaussian filter, sigma = .2s
# NREM sleep was defined as high-amplitude delta (0.5–4 Hz) ECoG activity and low EMG activity; 
# IS was defined as an increase in theta (5–9 Hz) and sigma (9–16 Hz) ECoG activity, 
# and a concomitant decrease in delta ECoG activity; 
# REM sleep was defined as low-amplitude theta ECoG activity with theta/delta ratio >0.5 and low EMG activity.
# 
#add status bar or label instead of prints
#make sure the output is compatible with the data loader
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
#these channels can then include sum/band power
#add an option to load and plot multiple scores
#add scorer name to metadata setting, add option to append name
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