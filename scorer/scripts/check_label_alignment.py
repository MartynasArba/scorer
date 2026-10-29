"""
Check how the labels of an evaluated dataset line up with the model's predictions.

Reads evaluate/<mode>/<head>/<dataset>/predictions.npz (written by the evaluate stage) and
re-scores the predictions against re-aligned versions of the true labels:

  as stored                 the labels as converted
  label of previous bout    every bout of equal labels takes the label of the bout BEFORE it
                            (high = stored labels run one bout AHEAD of the signal)
  label of next bout        every bout takes the label of the bout AFTER it
                            (high = stored labels lag one bout BEHIND the signal)
  best class swap           the best of the 6 ways to permute Wake / NREM / REM

If one re-alignment scores far higher than "as stored", the conversion assigned labels to the
wrong interval (or the wrong class). Optional: --cache also prints the delta / theta power of each
raw label per dataset, which shows which label really is NREM (highest delta).

    python -m scorer.scripts.check_label_alignment runs\\128hz\\evaluate\\tuned\\context_mlp\\oxford_test\\predictions.npz
    python -m scorer.scripts.check_label_alignment <predictions.npz> --cache G:\\RAW_EXTERNAL_SLEEP_DATASETS\\128HZ_CACHE
        --datasets mssv oxford_test
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

NAMES = ("Wake", "NREM", "REM")


def runs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and label of every run of equal values."""
    starts = np.concatenate(([0], np.flatnonzero(y[1:] != y[:-1]) + 1))
    return starts, y[starts]


def shift_bouts(y: np.ndarray, direction: int) -> np.ndarray:
    """Give every bout the label of its neighbour (-1 = previous bout, +1 = next bout)."""
    starts, labels = runs(y)
    ends = np.append(starts[1:], len(y))
    out = np.full_like(y, -1)
    for i, (a, b) in enumerate(zip(starts, ends)):
        j = i + direction
        if 0 <= j < len(labels):
            out[a:b] = labels[j]
    return out


def accuracy(true: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    m = true >= 0
    return float((true[m] == pred[m]).mean()) if m.any() else float("nan"), int(m.sum())


def confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    m = true >= 0
    return np.bincount(true[m] * 3 + pred[m], minlength=9).reshape(3, 3)


def check_predictions(path: Path) -> None:
    d = np.load(path)
    true, pred = d["true"].astype(np.int64), d["pred"].astype(np.int64)
    spans = list(zip(d["rec_start"], d["rec_end"]))
    variants = {"as stored": true}
    for name, direction in (("label of previous bout", -1), ("label of next bout", +1)):
        v = np.full_like(true, -1)
        for a, b in spans:
            v[a:b] = shift_bouts(true[a:b], direction)
        variants[name] = v
    print(f"{path}\n{len(spans)} recordings, {len(true)} windows\n")
    print(f"{'labels':<26}{'accuracy':>10}{'windows':>12}")
    for name, v in variants.items():
        acc, n = accuracy(v, pred)
        print(f"{name:<26}{acc:>10.3f}{n:>12}")
    best = max(itertools.permutations(range(3)),
               key=lambda p: accuracy(np.where(true >= 0, np.array(p)[np.clip(true, 0, 2)], -1), pred)[0])
    swapped = np.where(true >= 0, np.array(best)[np.clip(true, 0, 2)], -1)
    print(f"{'best class swap':<26}{accuracy(swapped, pred)[0]:>10.3f}"
          f"   ({', '.join(f'{NAMES[i]}->{NAMES[j]}' for i, j in enumerate(best) if i != j) or 'none'})")
    best_shift = max(("label of previous bout", "label of next bout"), key=lambda k: accuracy(variants[k], pred)[0])
    for name in ("as stored", best_shift):
        print(f"\nconfusion, {name} (rows = label, columns = predicted)")
        cm = confusion(variants[name], pred)
        print(f"{'':<8}" + "".join(f"{n:>10}" for n in NAMES))
        for n, row in zip(NAMES, cm):
            print(f"{n:<8}" + "".join(f"{v:>10}" for v in row))
    print("\nper recording: accuracy as stored | label of previous bout | label of next bout")
    for k, (a, b) in enumerate(spans):
        print(f"  rec {k:2d}: " + " | ".join(f"{accuracy(v[a:b], pred[a:b])[0]:.3f}" for v in variants.values()))


def band_check(cache_root: Path, datasets: list[str], fs: float, per_rec: int, seed: int) -> None:
    """Median relative delta (0.5-4 Hz) and theta (6-9 Hz) power per raw label code."""
    from scipy.signal import welch

    rng = np.random.default_rng(seed)
    print(f"\nrelative band power per raw label (fs = {fs:g} Hz; NREM should have the highest delta,")
    print("REM the highest theta/delta ratio)")
    for ds in datasets:
        recs = sorted(p for p in (cache_root / ds).iterdir() if (p / "X.npy").exists())
        by_code: dict[int, list[np.ndarray]] = {}
        for r in recs:
            x = np.load(r / "X.npy", mmap_mode="r")
            y = np.load(r / "y.npy")
            idx = np.sort(rng.choice(len(y), size=min(per_rec, len(y)), replace=False))
            f, p = welch(np.asarray(x[idx, 0], dtype=np.float64), fs=fs, nperseg=int(fs), axis=-1)
            tot = p[:, (f >= 0.5) & (f <= 45)].sum(1) + 1e-12
            delta = p[:, (f >= 0.5) & (f <= 4)].sum(1) / tot
            theta = p[:, (f >= 6) & (f <= 9)].sum(1) / tot
            for c in np.unique(y[idx]):
                m = y[idx] == c
                by_code.setdefault(int(c), []).append(np.stack([delta[m], theta[m]], 1))
        print(f"\n  {ds} ({len(recs)} recordings)")
        print(f"  {'code':>6}{'windows':>10}{'delta':>9}{'theta':>9}{'theta/delta':>13}")
        for c in sorted(by_code):
            v = np.concatenate(by_code[c])
            dmed, tmed = np.median(v[:, 0]), np.median(v[:, 1])
            print(f"  {c:>6}{len(v):>10}{dmed:>9.3f}{tmed:>9.3f}{tmed / max(dmed, 1e-9):>13.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, help="evaluate/<mode>/<head>/<dataset>/predictions.npz")
    ap.add_argument("--cache", type=Path, help="cache root, to also compare band power per label")
    ap.add_argument("--datasets", nargs="*", default=["mssv", "oxford_test"])
    ap.add_argument("--fs", type=float, default=128.0)
    ap.add_argument("--windows-per-recording", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    check_predictions(args.predictions)
    if args.cache:
        band_check(args.cache, args.datasets, args.fs, args.windows_per_recording, args.seed)


if __name__ == "__main__":
    main()
