# scorer
A framework for sleep stage classification in rodents, utilizing self-supervised pretraining and temporal sequence modeling.
Automated, manual, and mixed labeling modes are considered. 

## Core architecture
The main (automated scoring) part of this project features a three-stage pipeline designed to maximize accuracy even with limited labeled data:
1.  **SimCLR pretraining**: Unsupervised contrastive learning on large volumes of unlabeled ECoG/EMG data to learn general physiological features
2.  **SupCon pretraining**: Supervised contrastive learning (using labeled data) to refine embeddings for class discriminability
3.  **Sequence training (Bi-GRU)**: A temporal model that takes frozen CNN embeddings across 10-window sequences to capture sleep stage transitions and context

This model relies on a single cortical channel. 

## Other features
- **GUI**: PyQt5-based interface for manual labeling, featuring mouse-drag annotation, macro (10-min) and micro (4+ s) views
- **Data loading**: `BufferedSleepDataset` handles big datasets by chunking data into RAM, while `SequenceSleepDataset` manages temporal windows
- **OneBox support**: conversion utilities for SpikeGLX/OneBox binary formats into downsampled csvs

## Repo structure

```text
scorer/
├── data/
│   ├── loaders.py          # Buffered and Sequence dataset implementations
│   ├── preprocessing.py    # Ephys signal filtering and resampling
│   ├── storage.py          # Unified IO for tensors, pickles, and metadata
│   ├── onebox_utils.py     # SpikeGLX binary conversion tools
│   └── ML_utils.py         # Headless preprocessing for model training
├── gui/
│   ├── labeling_widgets.py # Main interactive labeling interface
│   ├── plots.py            # Matplotlib backends for GUI
│   └── settings_widgets.py # Metadata and project config
├── models/
│   ├── sleep_cnn.py        # SCDSSleepCNN (single channel, dual-stream CNN)
│   ├── sequence_model.py   # ContextAwareSleepScorer (Bi-GRU + Focal Loss)
│   ├── pretraining.py      # SimCLR and SupCon training loops
│   ├── sequence_training.py# sequence model training and validation
│   ├── main_pipeline.py    # full 3-stage training
│   ├── scoring.py          # utils for GUI and actual scoring
│   └── evaluate_model.py   # standalone model evaluation on unseen data
└── tests/                  # Unit tests for training efficiency and sequence logic
```

## Expected project structure
The pipeline expects a organized project directory:

```text
project_folder/
├── raw/                # Downsampled .csv files
├── processed/          # Windowed .pt tensors (X_chunk, y_chunk)
├── scores/             # .pkl files containing manual/auto annotations
└── quality_plots/      # QC reports for signal integrity
```

## Project Status & Notes

### Usage Warnings
- Models use `nn.LazyLinear`. A dummy forward pass is required before defining optimizers or loading weights. Use the provided `load_trained_sequence_model` in `scoring.py`.
- GUI uses 2-channel data flow (ECoG/EMG). Ensure metadata channel indices are correctly mapped in `Settings`.
- Heuristic corrections (e.g., removing REM->Wake transitions) are available in `scoring.py` but are disabled by default for raw model evaluation. They shouldn't be necessasry if all goes well.

---

### Performance metrics


Model performance is tested on unseen data from [Brodersen et al.](https://doi.org/10.5281/zenodo.10200481) Pilot dataset, with F and P channels treated as separate recordings, and EMG channel not used.
Models are trained on our own data for unsupervised learning, and on test, sleep deprivation and optogenetic stimulation datasets from the same authors. 

The full citation of data used is:
> Brodersen et al. Somnotate: A probabilistic sleep stage classifier for studying vigilance state transitions. PLoS Comput Biol. 2024. DOI: 10.1371/journal.pcbi.1011793

The best model to date achieved the following stats:
> - Global Accuracy:   0.9272
> - Macro F1-Score:    0.9070
> - Cohen's Kappa:     0.8735

                  precision    recall  f1-score   support

            Wake       0.95      0.91      0.93     66840
            NREM       0.95      0.94      0.94    121700
            REM       0.79      0.91      0.84     27140

        accuracy                           0.93    215680
      macro avg       0.90      0.92      0.91    215680
    weighted avg       0.93      0.93      0.93    215680

***Best use-case is using the F channel due to slightly higher scores**

F channel:
>- Global Accuracy:   0.9161
>- Macro F1-Score:    0.8869
>- Cohen's Kappa:     0.8527

                  precision    recall  f1-score   support

            Wake       0.92      0.92      0.92     33420
            NREM       0.94      0.94      0.94     60850
            REM       0.80      0.80      0.80     13570

        accuracy                           0.92    107840
      macro avg       0.89      0.89      0.89    107840
    weighted avg       0.92      0.92      0.92    107840

P channel:
>- Global Accuracy:   0.9081
>- Macro F1-Score:    0.8864
>- Cohen's Kappa:     0.8440

                  precision    recall  f1-score   support

            Wake       0.90      0.93      0.92     33420
            NREM       0.97      0.89      0.93     60850
            REM       0.72      0.94      0.81     13570

        accuracy                           0.91    107840
      macro avg       0.86      0.92      0.89    107840
    weighted avg       0.92      0.91      0.91    107840