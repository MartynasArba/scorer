# scorer
A custom semi-automated sleep scorer in development

**Notes for use**:
- create fresh metadata for each recording
- chop_by_state is depreciated and removed from gui, make sure it works somewhere in script as it is needed for testing of models
- make sure start timestamps match recordings, as they are only updated when loading from csv in preprocessing - or use specific metadata per recording
- one heuristic-based scorer is added, which not accurate (~40%), but might provide a starting point

## TODO:
### Priority:
- add scorer to GUI
- bug: (in preprocessing) recording duration is calculated by provided hours, so if the recording starts later than the 1st number, the end will also shift

### Test:
- if calling preprocessing multiple times, metadata might be messed up
- notch is still weird, should test it more (or depreciate it)
- spectral plots are not very informative, should be depreciated (or can be left blank in settings)

### Missing features and ideas:

`settings`:

`utils`:
- could add additional qc metrics for converted obx files
- add a "create project folder structure" button

`preprocessing`:
- future development: prevent freezing when it runs? 

`automatic scoring`:
- not yet started
- aiming for these models:
  - heuristic-based
  - CNN
  - TimesFM/Moirai pretrained embeddings + classifier head
- add selection of which channels to use!

*Use pre-trained models only. Pytorch grad is off in this project, and model training is out of scope. If for some reason grad is needed, do `torch.set_grad_enabled(True)`*

`manual labeling`:
- move less common funcs to a separate menu in manual labeling, add screenshot (fig.savefig)
- add channel toggles(?)
- add autoscale click for y
- add scorer name to metadata and state save files correctly
- save progress automatically every N scores? maybe add checkmark, overwrite option?
- jump to next unscored sample
- decide how single label, multi scorer should be handled
- could add mouse click to change state (or better keyboard controlls)

`report`:
- not yet started
- maybe export json report?
- for multilabel, add agreement/confidence 

## Repo structure

scorer/
  - data/
    - loaders.py          | Dataset classes
    - preprocessing.py    | Signal processing
    - storage.py          | Saves and loads everything
    - *_utils.py          | Misc utilities
  - gui/
    - main_window.py      | Main window, tracks global state
    - labeling_widgets.py | Manual labeling tab, display and labeling interface
    - plots.py            | Plotting utilities to be used by labeling window
    - preprocessing_widgets.py | Preprocessing tab
    - settings_widgets.py | Settings tab
    - util_widgets.py | Utilities tab
    - report_widgets.py | Report tab (currently empty)
    - autoscoring_widgets.py | Automatic scoring tab (currently empty)
  - models/
    - scoring.py | Should run selected scorers (currently empty)
    - model1.py | Might need separate scripts for different scorers
  - main.py | GUI entry point

## Expected project structure

**This should probably be per-recording**

- proj_data/
  - meta.json      | Saved metadata
  - raw/              | Original recording
  - quality_plots (optional)  | Recording QC report plots
  - processed/        | Preprocessed tensors ready for scoring
  - scores/           | Annotations
    - scorer1.json    | Scoring report
    - states.np