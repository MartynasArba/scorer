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
#bugs:
# bandpows seem to work, but extremely slowly
# think about filtering - current option (FFT) distorts a LOT, FIR is extremely slow, should get butterworth instead
#when trying to view (might be a problem with an old file): 
#   File "c:\Users\marty\Projects\scorer\scorer\gui\labeling_widgets.py", line 168, in select_dataset
#     self.dataset = SleepSignals(data_path = self.data_path,
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "c:\Users\marty\Projects\scorer\scorer\data\loaders.py", line 37, in __init__
#     self._load(data_path, score_path)
#   File "c:\Users\marty\Projects\scorer\scorer\data\loaders.py", line 119, in _load
#     self.all_labels = self.all_labels.permute(1, 0, 2)
#                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 1 is not equal to len(dims) = 3
#
#missing features:
#generally save metadata whenever updating 
#
# add warnings for settings if invalid values are set
#  add low_memory option which would load only a specified amount of chunks
#
#preprocessing:
#
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