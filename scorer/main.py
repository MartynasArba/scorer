
import sys
from PyQt5.QtWidgets import QApplication

from gui.main_window import SleepWindow

app = QApplication(sys.argv)
gui = SleepWindow(dataset = None)
gui.show()
sys.exit(app.exec_())

# add a reset settings button
# this should be done in the dataset class:
# add fft or specific power options instead of spect, ideally all, then don't generate spectrograms

# add an option to do metadata
# jump to next unscored sample button
# scale to more channels

# add an option to do scoring export to JSON or text file, or load/save custom annotations generally

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

#change from_Oslo_data when loading
#most relevant when loading other data, but can be adjusted based on my recordings


#final workflow should be this:
#1. chop data - added
#2. get predictions by some model
#3. manually go through and confirm/deny
#4. quality of life - autosaving, text file saving, track what scored what, etc