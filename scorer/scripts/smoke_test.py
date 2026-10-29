"""
End-to-end smoke test on synthetic data (CPU, ~1-3 minutes). Run from the repo root:

    python -m scorer.scripts.smoke_test            # full pipeline incl. torch stages
    python -m scorer.scripts.smoke_test --no-torch # data / label / RF / metrics checks only

The synthetic recordings are built to expose the old bugs:
  * the training set contains unlabeled (0) and artifact windows, the validation set does not
    -> with per-dataset label remapping, train and val would disagree;
  * one training recording has no REM at all;
  * classes have distinct spectra (Wake 20-40 Hz, NREM 1-3 Hz, REM 6-8 Hz), so a working
    pipeline must score well above chance; mixed-up labels would not.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FS, WIN = 128, 512


def synth_recording(rng, n: int, with_unlabeled: bool, with_rem: bool = True):
    """Make one synthetic recording whose classes have distinct spectra.

    Wake is 20-40 Hz, NREM 1-3 Hz and REM 6-8 Hz, in bouts of 5-30 windows, with a random
    gain. Optionally 5% of windows are unlabeled and REM can be left out.

    Returns:
        ``(x [n, 1, 512] float32, y [n] raw codes)``.
    """
    t = np.arange(WIN) / FS
    states = []
    cur = 1
    while len(states) < n:  # bouts of 5-30 windows
        options = [1, 2, 4] if with_rem else [1, 2]
        cur = rng.choice([s for s in options if s != cur])
        states += [cur] * int(rng.integers(5, 30))
    y = np.array(states[:n], dtype=np.int64)
    x = np.empty((n, 1, WIN), dtype=np.float32)
    for i, s in enumerate(y):
        f = {1: rng.uniform(20, 40), 2: rng.uniform(1, 3), 4: rng.uniform(6, 8)}[int(s)]
        amp = {1: 0.5, 2: 2.0, 4: 1.0}[int(s)]
        x[i, 0] = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) + 0.3 * rng.standard_normal(WIN)
    x *= rng.uniform(20, 80)  # arbitrary recording gain (normalisation must remove it)
    if with_unlabeled:
        y[rng.random(n) < 0.05] = 0
    return x, y


def write_dataset(root: Path, name: str, n_recs: int, rng, with_unlabeled: bool, use_pt: bool, no_rem_first=False):
    """Write ``n_recs`` synthetic recordings to ``root/name/rec<i>_ch0`` as .pt or .npy files."""
    for r in range(n_recs):
        d = root / name / f"rec{r}_ch0"
        d.mkdir(parents=True, exist_ok=True)
        x, y = synth_recording(rng, int(rng.integers(300, 500)), with_unlabeled, with_rem=not (no_rem_first and r == 0))
        if use_pt:
            import torch

            torch.save(torch.from_numpy(x), d / "X_ch0.pt")
            torch.save(torch.from_numpy(y), d / "y_ch0.pt")
        else:
            np.save(d / "X_ch0.npy", x)
            np.save(d / "y_ch0.npy", y)


def numpy_checks(tmp: Path) -> None:
    """Check labels, cache, batching, balancing, context, RF and metrics without torch."""
    from scorer.data.cache import DatasetCache, PrepParams, prepare_dataset
    from scorer.data.datasets import build_sources, epoch_selection, iter_window_batches
    from scorer.data.labels import IGNORE, THREE_STATE
    from scorer.evaluation.metrics import compute_metrics
    from scorer.inference.postprocess import heuristics_full, predict_classes
    from scorer.training.context import predict_rf, train_context_rf
    from scorer.training.embeddings import EmbeddingStore, RecordingSpan

    rng = np.random.default_rng(0)
    write_dataset(tmp / "raw", "train", 4, rng, True, use_pt=False, no_rem_first=True)
    write_dataset(tmp / "raw", "val", 2, rng, False, use_pt=False)
    params = PrepParams(FS, WIN, (0,), 49.0, 4.0, "float16")
    for ds in ("train", "val"):
        prepare_dataset(tmp / "raw" / ds, tmp / "cache", ds, params)
    tr, va = DatasetCache(tmp / "cache", "train", params), DatasetCache(tmp / "cache", "val", params)

    # fixed mapping: identical class ids whatever labels a dataset contains
    assert THREE_STATE.to_class(np.array([0, 1, 2, 3, 4, 7])).tolist() == [IGNORE, 0, 1, 1, 2, IGNORE]
    c_tr = np.concatenate([r.classes(THREE_STATE) for r in tr.recordings])
    c_va = np.concatenate([r.classes(THREE_STATE) for r in va.recordings])
    assert set(np.unique(c_tr)) == {IGNORE, 0, 1, 2} and set(np.unique(c_va)) == {0, 1, 2}

    # normalisation removed the recording gain
    x0 = tr.recordings[0].x()
    assert abs(float(np.median(x0))) < 0.2 and float(np.abs(x0).max()) <= 4.0 + 1e-3

    # sampler: every labeled window exactly once per epoch, across 3 workers, batches never short
    sources = build_sources([tr], THREE_STATE)
    sel = epoch_selection(sources, 3, "none", True, 0, 0)
    seen = 0
    for wid in range(3):
        for xb, yb in iter_window_batches(sources, sel, 64, 300, 0, 0, wid, 3, drop_last=False):
            assert xb.shape[1:] == (1, WIN) and (yb != IGNORE).all()
            seen += len(yb)
    assert seen == sum(len(s) for s in sel), (seen, sum(len(s) for s in sel))
    bal = epoch_selection(sources, 3, "undersample", True, 0, 1)
    counts = np.bincount(np.concatenate([s.classes[i] for s, i in zip(sources, bal)]).astype(np.int64), minlength=3)
    assert len(set(counts.tolist())) == 1, counts

    # context never crosses recordings
    def fake_store(cache):
        """Build an EmbeddingStore from simple spectral features of a cache (stands in for an encoder)."""
        feats, cls, spans, off = [], [], [], 0
        for r in cache.recordings:
            x = np.asarray(r.x(), dtype=np.float32)[:, 0]
            spec = np.abs(np.fft.rfft(x, axis=-1))[:, 1:65]
            feats.append(spec.reshape(len(x), 16, 4).mean(-1))
            cls.append(r.classes(THREE_STATE))
            spans.append(RecordingSpan(cache.name, r.name, off, off + len(x)))
            off += len(x)
        return EmbeddingStore([np.concatenate(feats)], np.concatenate(cls), spans, "fake")

    st_tr, st_va = fake_store(tr), fake_store(va)
    ci = st_tr.context_indices(np.array([st_tr.spans[1].start, st_tr.spans[0].end - 1]), 3)
    assert ci[0].min() == st_tr.spans[1].start and ci[1].max() == st_tr.spans[0].end - 1

    rf = train_context_rf(st_tr, THREE_STATE, k=1, n_estimators=50, max_depth=10, max_features=0.3,
                          class_weight="balanced_subsample", max_train_windows=None, n_jobs=1)
    probs = predict_rf(rf, st_va, 1, 3)
    pred = predict_classes(probs, [(s.start, s.end) for s in st_va.spans], THREE_STATE, 0, "lite")
    m = compute_metrics(st_va.classes, pred, THREE_STATE)
    print(f"[numpy checks] RF on synthetic val: acc {m['accuracy']:.3f} macro F1 {m['macro_f1']:.3f}")
    assert m["accuracy"] > 0.85, m
    assert heuristics_full(np.array([0, 2, 2, 1, 1]), THREE_STATE).tolist() == [0, 0, 0, 1, 1]
    print("[numpy checks] OK")


def torch_pipeline(tmp: Path) -> None:
    """Run every pipeline stage in both modes on a small synthetic config, then score with ``SleepScorer``."""
    from scorer.pipeline.run import main

    rng = np.random.default_rng(1)
    write_dataset(tmp / "raw_pt", "train", 4, rng, True, use_pt=True, no_rem_first=True)
    write_dataset(tmp / "raw_pt", "target", 2, rng, True, use_pt=True)
    write_dataset(tmp / "raw_pt", "val", 2, rng, False, use_pt=True)
    cfg = tmp / "smoke.toml"
    root = (tmp / "raw_pt").as_posix()
    cfg.write_text(f"""
[run]
name = "smoke"
runs_dir = '{(tmp / "runs").as_posix()}'
device = "cpu"
num_workers = 0
[signal]
sample_rate = {FS}
win_len = {WIN}
[cache]
root = '{(tmp / "cache_pt").as_posix()}'
[datasets]
train = '{root}/train'
target = '{root}/target'
val = '{root}/val'
[simclr]
datasets = ["train", "target"]
epochs = 2
batch_size = 64
buffer_windows = 512
max_batches_per_epoch = 3
[supcon]
datasets = ["train"]
epochs = 2
batch_size = 64
buffer_windows = 512
max_batches_per_epoch = 3
[adversarial]
source = ["train"]
target = ["target"]
epochs = 1
batch_size = 64
buffer_windows = 512
max_batches_per_epoch = 3
[embed]
batch_size = 256
[context_rf]
train = ["train"]
n_estimators = 30
n_jobs = 1
[context_mlp]
train = ["train"]
val = ["val"]
epochs = 3
batch_size = 256
hidden_dim = 64
[gru]
enabled = true
train = ["train"]
val = ["val"]
epochs = 2
seq_len = 10
stride = 5
lr = 1e-3
[linear_probe]
enabled = true
train = ["train"]
val = ["val"]
epochs = 3
[evaluate]
datasets = ["val"]
heads = ["context_mlp", "context_rf", "gru", "linear_probe"]
""")
    assert main(["--config", str(cfg)]) == 0, "pipeline failed (see log above)"
    print((tmp / "runs" / "smoke" / "evaluate" / "tuned" / "summary.csv").read_text())
    # zero-shot mode on the same run: reuses simclr/supcon, never touches the adversarial encoder
    assert main(["--config", str(cfg), "--set", "run.mode=zeroshot", "--stages", "embed", "context_mlp", "evaluate",
                 "--set", 'evaluate.heads=["context_mlp"]']) == 0, "zero-shot run failed"
    print((tmp / "runs" / "smoke" / "evaluate" / "zeroshot" / "summary.csv").read_text())

    from scorer.inference.scoring import SleepScorer

    man = json.loads((tmp / "runs" / "smoke" / "manifest.json").read_text())
    assert man["stages"]["embed@zeroshot"]["encoder_stage"] == "supcon", man["stages"]["embed@zeroshot"]
    scorer = SleepScorer(man["stages"]["embed@tuned"]["encoder_path"], man["stages"]["context_mlp@tuned"]["head"],
                         device="cpu")
    res = scorer.score_folder(tmp / "raw_pt" / "val" / "rec0_ch0", heuristics="lite")
    assert set(np.unique(res.gui_codes)) <= {1, 2, 4}, np.unique(res.gui_codes)
    print("[torch pipeline] OK; scorer API produced", len(res.gui_codes), "GUI codes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="scorer_smoke_"))
    try:
        numpy_checks(tmp)
        if not args.no_torch:
            torch_pipeline(tmp)
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0)
