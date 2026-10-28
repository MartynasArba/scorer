"""
Finding and loading raw windowed recordings (the X_*.pt / y_*.pt folders produced by
preprocessing and external_dataset_conversions).

Layout handled:
    <dataset_root>/<recording_dir>/X*.pt (+ y*.pt)      one or more chunks per recording
    <dataset_root>/X*.pt (+ y*.pt)                        a dataset root that is itself one recording
.npy files are accepted as well (X*.npy / y*.npy).

Pairing is by NAME (X_ch0_run1.pt <-> y_ch0_run1.pt), never by sort order.
Lengths of X and y are checked per chunk; small mismatches (a partial last epoch) are
trimmed, larger ones raise.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

X_PATTERNS = ("X*.pt", "X*.npy")


def natural_sort_key(path) -> list:
    """X_2.pt sorts before X_10.pt"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def label_name_for(x_path: Path) -> Path:
    """X_chunk3.pt -> y_chunk3.pt (only the leading X is swapped)"""
    return x_path.with_name("y" + x_path.name[1:])


@dataclass
class Recording:
    name: str
    directory: Path
    x_files: list[Path]
    y_files: list[Path | None] = field(default_factory=list)

    @property
    def labeled(self) -> bool:
        return all(y is not None for y in self.y_files)


def _x_files_in(directory: Path) -> list[Path]:
    files: list[Path] = []
    for pat in X_PATTERNS:
        # case-sensitive check: on Windows glob is case-insensitive and would also match x*/y*
        files.extend(p for p in directory.glob(pat) if p.name.startswith("X"))
    return sorted(set(files), key=natural_sort_key)


def discover_recordings(root: str | Path) -> list[Recording]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    candidates = [root] + sorted((d for d in root.iterdir() if d.is_dir()), key=natural_sort_key)
    recordings = []
    for d in candidates:
        xs = _x_files_in(d)
        if not xs:
            continue
        ys = []
        for xf in xs:
            yf = label_name_for(xf)
            ys.append(yf if yf.exists() else None)
        n_missing = sum(y is None for y in ys)
        if 0 < n_missing < len(ys):
            raise RuntimeError(f"{d}: {n_missing}/{len(ys)} X chunks have no matching y file")
        name = d.name if d != root else root.name
        recordings.append(Recording(name=name, directory=d, x_files=xs, y_files=ys))
    if not recordings:
        raise RuntimeError(f"no X*.pt / X*.npy files found under {root}")
    names = [r.name for r in recordings]
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate recording names under {root}")
    return recordings


def load_array(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".pt":
        import torch  # lazy: discovery / cache reading do not need torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    raise ValueError(f"unsupported file type: {path}")


def to_windows_channels_time(x: np.ndarray, path: Path | str = "") -> np.ndarray:
    """
    chunks were saved either as [windows, channels, time] (conversions) or
    [channels, windows, time] (older GUI preprocessing). Channels are always few.
    """
    if x.ndim == 2:  # [windows, time]
        return x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"{path}: expected 2D/3D signal array, got {x.shape}")
    if x.shape[0] < x.shape[1]:
        x = np.transpose(x, (1, 0, 2))
    if x.shape[1] > 16:
        raise ValueError(f"{path}: implausible channel count {x.shape[1]} for shape {x.shape}")
    return x


def labels_per_window(y: np.ndarray, path: Path | str = "") -> np.ndarray:
    """reduce label arrays to one raw code per window: [N], [1,N,T]/[N,1,T] or [N,T] accepted"""
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 3:
        if y.shape[0] == 1:
            y = y[0]  # [N, T]
        elif y.shape[1] == 1:
            y = y[:, 0]  # [N, T]
        else:
            raise ValueError(f"{path}: unexpected 3D label shape {y.shape}")
    if y.ndim == 2:
        if y.shape[0] == 1 or y.shape[1] == 1:
            return y.reshape(-1).astype(np.int64)
        first = y[:, 0]
        varying = int((y != first[:, None]).any(axis=1).sum())
        if varying:
            log.warning("%s: %d windows change label inside the window; using the first sample", path, varying)
        return first.astype(np.int64)
    raise ValueError(f"{path}: unexpected label shape {y.shape}")


def select_channels(x: np.ndarray, channels: list[int], path: Path | str = "") -> np.ndarray:
    c = x.shape[1]
    if max(channels) < c:
        return x[:, channels, :]
    if c == len(channels):
        # file already holds exactly the wanted channels (e.g. conversions save one channel per folder)
        return x
    raise ValueError(f"{path}: wants channels {channels} but file has {c}")


def load_recording(
    rec: Recording, channels: list[int], max_len_mismatch: int = 1
) -> tuple[np.ndarray, np.ndarray | None]:
    """returns X float32 [N, C, T] and raw labels int64 [N] (or None when unlabeled)"""
    xs, ys = [], []
    for xf, yf in zip(rec.x_files, rec.y_files):
        x = select_channels(to_windows_channels_time(load_array(xf), xf), channels, xf).astype(np.float32)
        if yf is not None:
            y = labels_per_window(load_array(yf), yf)
            diff = len(x) - len(y)
            if diff != 0:
                if abs(diff) > max_len_mismatch:
                    raise RuntimeError(
                        f"{xf.name}: {len(x)} windows but {yf.name} has {len(y)} labels "
                        f"(difference {diff} > max_len_mismatch={max_len_mismatch})"
                    )
                n = min(len(x), len(y))
                log.warning("%s: trimming X/y to %d windows (X=%d, y=%d)", xf, n, len(x), len(y))
                x, y = x[:n], y[:n]
            ys.append(y)
        xs.append(x)
    x_all = np.concatenate(xs, axis=0) if len(xs) > 1 else xs[0]
    y_all = (np.concatenate(ys) if len(ys) > 1 else ys[0]) if ys else None
    return x_all, y_all


def scan_labels(rec: Recording) -> np.ndarray | None:
    """labels only (cheap) for inspection"""
    if not rec.labeled:
        return None
    return np.concatenate([labels_per_window(load_array(y), y) for y in rec.y_files])
