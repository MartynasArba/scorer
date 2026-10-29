# scorer — pipeline cheat sheet

Start with `README.md`; this page is the short version for the command line.

```
scorer/
  config/        pipeline_128hz.toml, pipeline_250hz.toml   <- everything you change lives here
  data/          labels.py      fixed raw-code -> class mapping (single source of truth)
                 signal.py      fs-aware robust normalisation
                 recordings.py  find + load X/y recordings, pair by name, check lengths
                 cache.py       `prepare`: normalise once -> memory-mapped .npy cache
                 datasets.py    WindowBatchStream for SimCLR / SupCon / adversarial
                 external_dataset_conversions.py; storage, preprocessing, report, onebox_utils,
                 motionsensor_utils (GUI file tools)
  models/        encoders.py (SCDSSleepCNN), legacy_cnns.py, heads.py, losses.py, grl.py,
                 checkpoints.py (self-describing, strict loading)
  training/      common.py, augment.py, contrastive.py, adversarial.py, embeddings.py,
                 context.py (MLP / RF / linear probe), sequence_gru.py
  evaluation/    metrics.py (JSON/CSV/PNG reports), diagnostics.py
  inference/     scoring.py (SleepScorer: GUI + batch API), postprocess.py, spectral.py
  pipeline/      config.py, stages.py, run.py (CLI)
  scripts/       smoke_test.py, data_modules_test.py, gui_core_test.py, gui_widget_smoke.py
  gui/           the PyQt5 app (core/ = logic without Qt, widgets/ = one module per tab)
```

## Run

```bash
python -m scorer.scripts.smoke_test                                    # 1-3 min synthetic end-to-end check
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml --stages inspect
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml                # all enabled stages
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml --stages embed context_mlp evaluate
python -m scorer.pipeline.run --config ... --set context_mlp.epochs=5 --set run.name=debug
python -m scorer.pipeline.run --config ... --list                      # stage plan + what this run already produced
# zero-shot vs tuned on the same run: tuned (default) runs alignment; zeroshot reuses simclr/supcon, skips it
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml --set run.mode=zeroshot
```

Outputs: `<runs_dir>/<run.name>/` → `pipeline.log`, `manifest.json`, `config_*.toml`,
`simclr/ supcon/ adversarial/` (encoders), `embeddings/<encoder digest>/`, `heads/`, `evaluate/`.

## Score a recording

```python
from scorer.inference.scoring import SleepScorer, save_states
scorer = SleepScorer(r"runs\128hz\adversarial\encoder_final.pt", r"runs\128hz\heads\tuned\context_mlp.pt")
res = scorer.score_folder(r"D:\rec\processed\windowed_x", channel=0, heuristics="lite")
save_states(res, r"D:\states\rec_context_mlp_states.pkl")   # GUI codes 1/2/4 + _confidence.pkl
```

Label codes on disk: 0 unlabeled, 1 Wake, 2 NREM, 3 IS, 4 REM. Model classes (3state): 0 Wake, 1 NREM (+IS), 2 REM.

## GUI

```bash
python -m scorer.gui                                   # start the GUI (PyQt5)
python -m scorer.scripts.gui_core_test                 # GUI logic without Qt
python -m scorer.scripts.gui_widget_smoke              # drives the tabs with a stand-in for PyQt5
python -m scorer.scripts.data_modules_test             # storage / report / OneBox / motion-sensor checks
```

Model training tab: open a pipeline config (or start from a template), check it, tick stages, run.
The tab starts `python -u -m scorer.pipeline.run ...` as a separate process (copy the exact command
with "copy command"); a crash or CUDA OOM in training cannot close the GUI, and the run folder can
be handed to automatic scoring with one click.

Workflow: settings → preprocess (or use existing windowed data) → automatic scoring (pick a pipeline
run folder; tuned / zeroshot heads are listed separately) → "open selected result in manual scoring"
→ review with "next uncertain" (Ctrl+Right) → save (Ctrl+S) → report.
Windowed data at 250 Hz (1000-sample, 4-s windows) is resampled to the model's 128 Hz on the fly.
