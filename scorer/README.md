# scorer

Sleep scoring (Wake / NREM / REM) from one cortical channel in 4-second windows.

The package has three parts that share one set of label codes and file formats:

- **Modelling pipeline.** It trains an encoder with contrastive learning (and optional domain alignment), then classifier heads on its embeddings. It evaluates them on held-out datasets. Run it with `python -m scorer.pipeline.run`.
- **Scoring API.** `SleepScorer` scores new recordings with a trained run.
- **GUI.** It converts and preprocesses raw recordings, scores them automatically, and lets you review and label them by hand. It also writes reports and runs training. Start it with `python -m scorer.gui`.

Every module, class and function has a docstring. Use `help(...)` in Python, or your editor's hover, for details this page leaves out. `README_pipeline.md` is a one-page cheat sheet of the pipeline commands.

---

## Contents

| Folder | What is in it |
| --- | --- |
| `config/` | Pipeline configs (`pipeline_128hz.toml`, `pipeline_250hz.toml`). Everything you change for a training run lives here. |
| `data/` | Label codes (`labels.py`) and finding and loading recordings (`recordings.py`). The prepared-data cache (`cache.py`) and training batch streams (`datasets.py`). The GUI's file tools: `storage.py`, `preprocessing.py`, `report.py`, `onebox_utils.py`, `motionsensor_utils.py`. Dataset conversions: `external_dataset_conversions.py`. |
| `models/` | The window encoder (`encoders.py`, `SCDSSleepCNN`), older architectures (`legacy_cnns.py`), heads, losses, gradient reversal, and checkpoint save/load (`checkpoints.py`). |
| `training/` | One module per training stage, plus shared helpers (`common.py`). |
| `evaluation/` | Metrics, reports, figures (`metrics.py`) and domain-shift diagnostics (`diagnostics.py`). |
| `inference/` | `SleepScorer` (`scoring.py`), smoothing and heuristics (`postprocess.py`), and band powers (`spectral.py`). |
| `pipeline/` | Config dataclasses (`config.py`), the stages (`stages.py`) and the command line (`run.py`). |
| `gui/` | The PyQt5 app. `gui/core/` holds the logic and has no Qt import. `gui/widgets/` holds one thin Qt module per tab. |
| `scripts/` | Self-checks that run without a GPU. |

---

## Install

1. Use Python 3.11 or newer.
2. Install PyTorch and torchaudio with the CUDA build for your GPU, using the instructions at https://pytorch.org/get-started/locally/.
3. Install the rest:

   ```bash
   pip install -r scorer/requirements.txt
   ```

4. Run everything from the folder that **contains** `scorer/`, so that `import scorer` works. If you run from elsewhere, add that folder to `PYTHONPATH`.

Check the installation (no GPU needed):

```bash
python -m scorer.scripts.smoke_test --no-torch     # labels, cache, batching, RF, metrics
python -m scorer.scripts.smoke_test                # + every pipeline stage on synthetic data (1-3 min)
python -m scorer.scripts.data_modules_test         # storage, report, OneBox, motion sensors
python -m scorer.scripts.gui_core_test             # GUI logic without Qt
python -m scorer.scripts.gui_widget_smoke          # drives every tab through a stand-in for PyQt5
```

Each prints "passed" or "OK" when all its checks pass, or stops at the first failing check.

---

## Typical workflow

```
raw recording ──► CSV ──► windowed X/y ──► (train) ──► automatic scoring ──► manual review ──► report
  OneBox .bin     utilities  preprocess      training      automatic scoring    manual scoring   report
                  tab        tab             tab / CLI     tab / SleepScorer    tab              tab
```

### 1. Get windowed data

**Your own recordings (GUI):**

1. **Settings tab.** Set the project folder and `sample_rate` (of the raw file), the ECoG/EMG channel numbers and the time column. "Fill defaults" fills empty fields with defaults (sample rate 1000, ECoG 0,1, EMG 2, time column 3).
2. **Utilities tab** (OneBox only). "Convert one OneBox recording" writes a CSV at the settings-tab sample rate. "Quality report" checks every CSV in a folder.
3. **Preprocessing tab.** Select the CSV and press **"preset: windows for the 128 Hz models"**. This resamples to 128 Hz, band-passes ECoG 0.5-49 Hz and saves 512-sample (4 s) windows. Tick "load in chunks" for long recordings, and "only load between" to keep for example 19:00-07:00.

The output goes to `<project>/processed/windowed_<session id><tag><file name>/X_chunk<i>.pt` (the file name only if "add original file name" is ticked) (with `y_chunk<i>.pt`), or `<project>/processed/<name>_X.pt` without chunks. Data saved at 250 Hz with 1000-sample windows can also be scored; it is resampled to the model's rate on the fly.

**External datasets.** Edit and run `data/external_dataset_conversions.py` (`run(...)`). It writes one folder per recording and channel, with `X*.pt` [windows, 1, samples] and `y*.pt` raw codes.

**What the pipeline expects.** A dataset is a folder with one sub-folder per recording. Each recording holds `X*.pt` or `X*.npy` chunks, plus `y*` files with the same names if it is labeled. Chunks are paired by name and read in natural order.

### 2. Train (pipeline)

Copy `config/pipeline_128hz.toml` and set `[run] runs_dir`, `[cache] root` and the `[datasets]` folders. Then run it:

```bash
python -m scorer.pipeline.run --config my.toml --stages inspect     # label counts per dataset: check first
python -m scorer.pipeline.run --config my.toml                      # every enabled stage
python -m scorer.pipeline.run --config my.toml --set run.mode=zeroshot   # same run folder, no alignment
python -m scorer.pipeline.run --config my.toml --stages embed context_mlp evaluate --set context_mlp.epochs=5
python -m scorer.pipeline.run --config my.toml --list               # stage plan + what the run already has
```

The GUI's **model training** tab does the same. It edits the config, checks it, starts the run as a separate process (so a crash cannot close the GUI) and shows progress. "Copy command" gives the exact command line.

| Stage | Does | Main output |
| --- | --- | --- |
| `inspect` | Counts recordings and label codes per dataset; flags unknown codes and missing classes | `inspect/datasets.json` |
| `prepare` | Normalises every recording once into a memory-mapped cache | `<cache root>/<dataset>/` |
| `simclr` | Unsupervised contrastive pretraining | `simclr/encoder_final.pt`, `encoder_best.pt` |
| `supcon` | Supervised contrastive pretraining (usually starts from SimCLR) | `supcon/encoder_final.pt` |
| `adversarial` | Aligns source and target domains; target labels are never read (tuned mode only) | `adversarial/encoder_final.pt` |
| `embed` | Runs the encoder over every dataset the heads and evaluation need | `embeddings/<encoder digest>/<dataset>/` |
| `context_rf`, `context_mlp`, `gru`, `linear_probe` | Train classifier heads on embeddings | `heads/<mode>/<head>.pt` (RF: `.pkl`) |
| `evaluate` | Metrics, confusion matrices, hypnograms, per-recording tables | `evaluate/<mode>/summary.csv` and one folder per head and dataset |

**Modes.** `run.mode = "tuned"` (default) aligns the encoder to the target datasets' unlabeled inputs. Scores on those targets are therefore semi-zero-shot. `run.mode = "zeroshot"` skips alignment. Both modes share `prepare`, `simclr` and `supcon` in one run folder; `embed`, the heads and `evaluate` are kept per mode. So you can run zero-shot first and then tuned without retraining the encoder.

**Resuming.** Every finished stage is recorded in `<runs_dir>/<name>/manifest.json`, and later stages read their inputs from it. Rerun with the same `run.name` and only the stages you need.

**Run folder:**

```
<runs_dir>/<name>/
  pipeline.log                 full log of every session
  manifest.json                finished stages and their outputs ("embed@tuned", "context_mlp@zeroshot", ...)
  config_<time>.toml           the config as used, plus config_<time>_resolved.json with overrides applied
  inspect/  simclr/  supcon/  adversarial/
  embeddings/<encoder digest>/<dataset>/embeddings.npy, index.json
  heads/<mode>/context_mlp.pt, context_rf.pkl, ...
  evaluate/<mode>/summary.csv, pca_<dataset>.png
  evaluate/<mode>/<head>/<dataset>/metrics.json, per_recording.csv, predictions.npz, confusion.png, hypnogram_*.png
```

Every config key is described in the docstrings of `pipeline/config.py` (one dataclass per `[section]`). An unknown key stops the run with the list of valid keys.

### 3. Score recordings

**GUI, automatic-scoring tab:**

1. Choose a pipeline run folder. Every trained head is listed with its mode and encoder, and the encoder and head always come from the same run and mode.
2. Add a recording folder, a folder of recordings, or a single X file.
3. Press **score**.

For each recording it writes:

- `<recording>_<head>_<mode>_states.pkl`
- `..._states_confidence.pkl`
- `..._states_scoring.json` (which model and settings were used)

These go to `<project>/scores/`, or to the folder you choose.

**Python:**

```python
from scorer.inference.scoring import SleepScorer, save_states

scorer = SleepScorer(r"runs\128hz\adversarial\encoder_final.pt", r"runs\128hz\heads\tuned\context_mlp.pt")
res = scorer.score_folder(r"D:\rec\processed\windowed_x", channel=0, heuristics="lite",
                          source_sample_rate=250)        # only if the data is not at the model's rate
save_states(res, r"D:\rec\scores\rec_context_mlp_states.pkl")   # + rec_context_mlp_states_confidence.pkl
res.gui_codes, res.confidence, res.probs                 # per window
```

For many recordings listed in a CSV (`proj_dir`, `channel_selected`), see `inference.scoring.score_from_table`.

**Settings.**

- `heuristics="lite"` fixes isolated single-window states.
- `"full"` also turns REM that follows Wake into Wake.
- `smooth_sigma` smooths the probabilities over that many windows first.

### 4. Review and report (GUI)

**Manual scoring tab.** "Open selected result in manual scoring" (automatic-scoring tab) loads a recording with its scores and confidence. Otherwise use "load data" and then "load states".

| Key | Action |
| --- | --- |
| `0`-`4` | Choose the state to label with (Unlabeled, Wake, NREM, IS, REM) |
| drag on the state axis | Label the windows under the drag |
| `Left` / `Right` | Previous / next window |
| `Shift+Left` / `Shift+Right` | Previous / next state change |
| `Ctrl+Right` | Next window with model confidence below the threshold (default 0.6) |
| `Ctrl+U` | Next unlabeled window |
| `Ctrl+J` | Jump to a window number or a time of day (HH:MM[:SS]) |
| `+` / `-` | Show more / fewer windows |
| `T` | Toggle the 10-minute view |
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Save (a new file in `<project>/scores/`, never overwriting) |

Labels are autosaved every 3 minutes to one autosave file per recording. The window asks before closing with unsaved labels.

**Report tab:**

- **Hypnogram.** Shown with model confidence when a `_confidence.pkl` file is present.
- **State report.** Durations, percentages and bouts, optionally per clock hour, with optional micro-arousal marking.
- **Agreement.** Compares two state files: accuracy, Cohen's kappa, a confusion matrix and per-state agreement.

---

## Labels and files

**Raw codes** are what every file on disk stores (GUI state files, `y` files, scoring output):

| Code | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| State | Unlabeled | Wake | NREM | IS | REM | MA (report tab only) |

**Model classes** come from a fixed label scheme (`data/labels.py`). The default "3state" scheme is 0 Wake, 1 NREM (IS merged in), 2 REM. Unlabeled windows (and MA) are ignored by losses and metrics but kept as temporal context. The mapping never depends on which labels a dataset contains, so class indices mean the same thing everywhere.

**Array layouts.** GUI preprocessing saves X as [channels, windows, samples]; conversions save [windows, channels, samples]. Loaders detect the layout per recording (`layout="auto"`), and you can force it where a "data layout" option is offered. With the GUI's windowed data, channel 0 is the first ECoG channel. Any feature channels come after ECoG and EMG.

**Checkpoints** store their metadata: architecture, sample rate, window length, filter settings, label scheme and, for heads, the digest of the encoder they were trained on. Loading is strict, and a head is refused with a different encoder unless you allow it. Old raw state dicts still load; the assumed settings are logged.

---

## Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| `dataset 'x' is not prepared` | Run the `prepare` stage (or the whole pipeline). |
| `cache for 'x' was prepared with ...` | `[signal]` changed since `prepare`; run `prepare` again (it overwrites). |
| `head ... was trained on encoder ...` | The head and encoder come from different runs or modes. Pick the pair from a run folder, or retrain the head. |
| `... but the model scores 4-s epochs` | The sample rate is wrong (settings tab) or the windows are not 4 s long. Re-window with the 128 Hz preset. |
| Labels look mixed up | Run `--stages inspect` and check `inspect/datasets.json` for unknown codes and missing classes. |
| Accuracy far below chance (negative kappa) on one dataset | Its labels are misaligned, not the model. Run `python -m scorer.scripts.check_label_alignment <evaluate/.../predictions.npz> --cache <cache root> --datasets <names>`: it re-scores with labels shifted by one bout or classes swapped, and shows which label has NREM-like (high delta) power. |
| CUDA out of memory | Lower `batch_size` of the stage that failed; for `context_rf`, lower `max_train_windows`. The `[mem]` line logged after the error shows memory use. |
| Time axis starts at 00:00:00 | `rec_start` is missing from the metadata; set `time_channel` before preprocessing. |
| GUI error dialog | The GUI keeps running. Full details are in `~/.scorer_gui.log`. |
