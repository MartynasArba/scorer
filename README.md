# scorer
A custom semi-automated sleep scorer in development

**Notes for use**:
- create fresh metadata for each recording
- chop_by_state is depreciated and removed from gui, make sure it works somewhere in script as it is needed for testing of models
- make sure start timestamps match recordings, as they are only updated when loading from csv in preprocessing - or use specific metadata per recording
- one heuristic-based scorer is added, which not accurate (~40%), but might provide a starting point

## TODO:
### Bugs or testing needed:
- bug: (in preprocessing) recording duration is calculated by provided hours, so if the recording starts later than the 1st number, the end will also shift
- if calling preprocessing multiple times, metadata might be messed up
- notch is still weird, should test it more (or depreciate it)
- spectral plots are not very informative, should be depreciated (or can be left blank in settings)
- test multiple scorer plotting

### Missing features and ideas:

`settings`:

`utils`:
- could add additional qc metrics for converted obx files
- add a "create project folder structure" button

`preprocessing`:
- future development: prevent freezing when it runs? 

`automatic scoring`:
- aiming for these models:
  - heuristic-based (implemented, low accuracy)
  - CNN (implemented, generalization is not great)
  - Pretrained embeddings

*Use pre-trained models only. Pytorch grad is off in this project, and model training is out of scope, except for the **`training_testing.py`** script, which is not accessible via the GUI. If for some reason grad is needed, do `torch.set_grad_enabled(True)`*

`manual labeling`:
- add autoscale click for y
- add scorer name to metadata and state save files correctly
- save progress automatically every N scores? maybe add checkmark, overwrite option?
- jump to next unscored sample
- decide how single label, multi scorer should be handled

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

## model stats
Often better than they appear, but won't generalize well to all data.
The evaluation is not performed properly and should not be relied upon, but gives some indication. 

**3state_CNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.79      0.88      0.83     42067
           2       0.49      0.92      0.64     21127
           3       0.00      0.00      0.00      6483
           4       0.27      0.89      0.41      2270

    accuracy                           0.62     94024

**3state_ephysCNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.83      0.90      0.86     42067
           2       0.49      0.95      0.64     21127
           3       0.00      0.00      0.00      6483
           4       0.29      0.92      0.44      2270

    accuracy                           0.64     94024

**3state_fftCNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.83      0.88      0.86     42067
           2       0.49      0.93      0.64     21127
           3       0.00      0.00      0.00      6483
           4       0.24      0.93      0.38      2270

    accuracy                           0.63     94024

**4state_CNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.80      0.84      0.82     42067
           2       0.57      0.76      0.65     21127
           3       0.42      0.85      0.56      6483
           4       0.23      0.92      0.37      2270

    accuracy                           0.63     94024

**4state_ephysCNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.81      0.90      0.85     42067
           2       0.56      0.78      0.65     21127
           3       0.43      0.82      0.57      6483
           4       0.32      0.83      0.46      2270

    accuracy                           0.65     94024

**4state_fftCNN_2026-01-27.pt**

              precision    recall  f1-score   support

           0       0.00      0.00      0.00     22077
           1       0.82      0.91      0.86     42067
           2       0.58      0.71      0.64     21127
           3       0.38      0.84      0.52      6483
           4       0.29      0.88      0.44      2270

    accuracy                           0.65     94024