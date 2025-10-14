# scorer
A custom sleep scorer in development

Aiming for this structure:
scorer/
  ├── data/
  │   ├── loaders.py          # Dataset classes
  │   ├── preprocessing.py    # Signal processing
  │   └── storage.py          # Score persistence
  ├── gui/
  │   ├── main_window.py      # SleepGUI
  │   ├── widgets.py          # Custom controls
  │   └── plots.py            # Plotting utilities
  ├── models/
  │   └── scoring.py          # Annotation data structures
  └── main.py                 # Entry point

data/
  ├── raw/              # Original recordings
  ├── processed/        # Chopped data 
  └── scores/           # Annotations
        ├── scorer1.json
        ├── scorer2.json
        └── consensus.json
