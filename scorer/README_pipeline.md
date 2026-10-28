# scorer — modelling pipeline

```
scorer/
  config/        pipeline_128hz.toml, pipeline_250hz.toml   <- everything you change lives here
  data/          labels.py      fixed raw-code -> class mapping (single source of truth)
                 signal.py      fs-aware robust normalisation
                 recordings.py  find + load X/y recordings, pair by name, check lengths
                 cache.py       `prepare`: normalise once -> memory-mapped .npy cache
                 datasets.py    WindowBatchStream for SimCLR / SupCon / adversarial
                 external_dataset_conversions.py, preprocessing.py (unchanged)
  models/        encoders.py (SCDSSleepCNN), legacy_cnns.py, heads.py, losses.py, grl.py,
                 checkpoints.py (self-describing, strict loading)
  training/      common.py, augment.py, contrastive.py, adversarial.py, embeddings.py,
                 context.py (MLP / RF / linear probe), sequence_gru.py
  evaluation/    metrics.py (JSON/CSV/PNG reports), diagnostics.py
  inference/     scoring.py (SleepScorer: GUI + batch API), postprocess.py, spectral.py
  pipeline/      config.py, stages.py, run.py (CLI)
  scripts/       smoke_test.py
```

## Run

```bash
python -m scorer.scripts.smoke_test                                    # 1-3 min synthetic end-to-end check
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml --stages inspect
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml                # all enabled stages
python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml --stages embed context_mlp evaluate
python -m scorer.pipeline.run --config ... --set context_mlp.epochs=5 --set run.name=debug
python -m scorer.pipeline.run --config ... --list                      # stage plan + what this run already produced
```

Outputs: `<runs_dir>/<run.name>/` → `pipeline.log`, `manifest.json`, `config_*.toml`,
`simclr/ supcon/ adversarial/` (encoders), `embeddings/<encoder digest>/`, `heads/`, `evaluate/`.

## Score a recording

```python
from scorer.inference.scoring import SleepScorer, save_states
scorer = SleepScorer(r"runs\128hz\adversarial\encoder_final.pt", r"runs\128hz\heads\context_mlp.pt")
res = scorer.score_folder(r"D:\rec\processed\windowed_x", channel=0, heuristics="lite")
save_states(res, r"D:\states\rec_context_mlp_states.pkl")   # GUI codes 1/2/4 + _confidence.pkl
```

Label codes on disk: 0 unlabeled, 1 Wake, 2 NREM, 3 IS, 4 REM. Model classes (3state): 0 Wake, 1 NREM (+IS), 2 REM.
