
import sys
from PyQt5.QtWidgets import QApplication

from gui.main_window import SleepWindow
# from data.preprocessing import from_Oslo_csv

#load and chop data
# path = r'data\raw\trial_1_mouse_b1aqm2.csv'
# from_Oslo_csv(path)

# data_path = r'C:\Users\marty\Projects\scorer\proj_data\processed\trial_1_mouse_b1aqm2_X.pkl'
# score_path = r'C:\Users\marty\Projects\scorer\proj_data\processed\trial_1_mouse_b1aqm2_y.pkl'

app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())

#final workflow should be this:
#1. chop data - added
#2. get predictions by some model
#3. manually go through and confirm/deny
#4. quality of life - autosaving, text file saving, track what scored what, etc

#viewer part:
#split label/slider update into another function
#add reset button
#add fft or specific power options instead of spect, ideally all, then don't generate spectrograms
#add keyboard control
#metadata display
# jump to next unscored sample button
# scale to more channels

#add scoring part:
# choice buttons or dropdown to select sleep stage for current sample - also highlight current sample
# add an option to do scoring export to JSON or text file
# keyboard shortcuts for quick scoring (1-4 for Wake/NREM/IS/REM)

# Save progress automatically every N scores
# Restore last position on app startup
# track which samples have been scored and by whom / optional, generate a new y file per each scorer

#next steps:
#incorporate loading and chopping data first via popup windows, output should be path of chopped data
#add "agreement", do semi-automatic labeling
#add cache
#keep in mind that chopping via labels will LOSE SOME DATA, therefore text saving will not be the same as raw recording!
#claude suggestions
# Multi-scorer Interface: Load multiple scorers' annotations, visualize agreements/disagreements, compute inter-rater reliability
# Statistical Dashboard: Sleep architecture metrics (total time in each stage, fragmentation, etc.)
# Batch Operations: Export scored data in standard formats (EDF+, CSV with stage transitions)
# ML Integration: Show model predictions alongside user scoring; confidence visualizations
# Signal Preprocessing: On-the-fly filtering, artifact detection, reference subtraction
# Export/Analysis: Generate hypnograms, sleep-wake cycles, spectral analysis reports