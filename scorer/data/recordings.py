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
    """Sort key that orders numbers numerically, so ``X_2.pt`` comes before ``X_10.pt``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def label_name_for(x_path: Path) -> Path:
    """Return the label file name for a data file: ``X_chunk3.pt`` -> ``y_chunk3.pt``."""
    return x_path.with_name("y" + x_path.name[1:])


@dataclass
class Recording:
    """A recording on disk: one folder with one or more X chunks and matching y chunks.

    Attributes:
        name: Recording name (the folder name).
        directory: Folder that holds the chunks.
        x_files: Data chunks in natural order.
        y_files: Label chunk for each X chunk, or None where it is missing.
    """
    name: str
    directory: Path
    x_files: list[Path]
    y_files: list[Path | None] = field(default_factory=list)

    @property
    def labeled(self) -> bool:
        """True if every X chunk has a label chunk."""
        return all(y is not None for y in self.y_files)


def _x_files_in(directory: Path) -> list[Path]:
    """List the X chunks directly inside ``directory`` (case-sensitive, natural order)."""
    files: list[Path] = []
    for pat in X_PATTERNS:
        # case-sensitive check: on Windows glob is case-insensitive and would also match x*/y*
        files.extend(p for p in directory.glob(pat) if p.name.startswith("X"))
    return sorted(set(files), key=natural_sort_key)


def discover_recordings(root: str | Path) -> list[Recording]:
    """Find all recordings under a dataset root.

    The root itself counts as a recording if it holds X files; every sub-folder that
    holds X files is another recording.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        RuntimeError: If no X files are found, recording names repeat, or only some
            chunks of a recording have label files.
    """
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
    """Load a ``.npy`` or ``.pt`` file as a numpy array (torch is imported only for ``.pt``).

    Raises:
        ValueError: For any other file type.
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".pt":
        import torch  # lazy: discovery / cache reading do not need torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    raise ValueError(f"unsupported file type: {path}")


LAYOUTS = ("auto", "nct", "cnt")


def infer_layout(shape: tuple[int, ...]) -> str:
    """Guess the axis order of a 3-D chunk from its shape.

    Returns:
        "cnt" for [channels, windows, time] (GUI preprocessing) when the first axis is the
        shorter one, otherwise "nct" for [windows, channels, time] (dataset conversions).
    """
    if len(shape) != 3:
        return "nct"
    return "cnt" if shape[0] < shape[1] else "nct"


def to_windows_channels_time(x: np.ndarray, path: Path | str = "", layout: str = "auto") -> np.ndarray:
    """Return a chunk as [windows, channels, time].

    Args:
        x: 2-D [windows, time] or 3-D array.
        path: File name, used in error messages only.
        layout: "nct", "cnt" or "auto". "auto" decides from the shape, which is
            ambiguous for chunks with fewer windows than channels; ``load_recording``
            therefore infers the layout once per recording from its largest chunk.

    Raises:
        ValueError: For an unknown layout, an unexpected number of dimensions, or more
            than 16 channels (almost certainly a wrong layout).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if x.ndim == 2:  # [windows, time]
        return x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"{path}: expected 2D/3D signal array, got {x.shape}")
    if (infer_layout(x.shape) if layout == "auto" else layout) == "cnt":
        x = np.transpose(x, (1, 0, 2))
    if x.shape[1] > 16:
        raise ValueError(f"{path}: implausible channel count {x.shape[1]} for shape {x.shape} (wrong layout?)")
    return x


def labels_per_window(y: np.ndarray, path: Path | str = "") -> np.ndarray:
    """Reduce a label array to one raw code per window.

    Accepts [N], [N, T], [1, N, T] and [N, 1, T]. For per-sample labels the first sample
    of each window is used, with a warning if any window changes label inside it.

    Raises:
        ValueError: For any other shape.
    """
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


def select_channels(x: np.ndarray, channels: list[int] | None, path: Path | str = "") -> np.ndarray:
    """Keep only ``channels`` (None keeps all).

    If the file holds exactly as many channels as requested, it is assumed to contain
    just those channels already (conversions store one channel per folder).

    Raises:
        ValueError: If the requested channels do not exist in the file.
    """
    if channels is None:
        return x
    c = x.shape[1]
    if max(channels) < c:
        return x[:, channels, :]
    if c == len(channels):
        # file already holds exactly the wanted channels (e.g. conversions save one channel per folder)
        return x
    raise ValueError(f"{path}: wants channels {channels} but file has {c}")


def load_recording(
    rec: Recording, channels: list[int] | None, max_len_mismatch: int = 1, layout: str = "auto"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load all chunks of a recording into memory.

    Args:
        rec: Recording to load.
        channels: Channel indices to keep, or None for all.
        max_len_mismatch: Largest X/y length difference per chunk that is trimmed
            silently (a partial last window); larger differences raise.
        layout: Axis order of the chunks, see ``to_windows_channels_time``.

    Returns:
        ``(x, y)``: float32 windows [N, C, T] and int64 raw codes [N], or None for ``y``
        when the recording is unlabeled.

    Raises:
        RuntimeError: If X and y lengths differ by more than ``max_len_mismatch``.
    """
    raw = [load_array(xf) for xf in rec.x_files]
    if layout == "auto":
        layout = infer_layout(max(raw, key=lambda a: a.size).shape)
    xs, ys = [], []
    for xf, yf, arr in zip(rec.x_files, rec.y_files, raw):
        x = select_channels(to_windows_channels_time(arr, xf, layout), channels, xf).astype(np.float32)
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
    del raw
    x_all = np.concatenate(xs, axis=0) if len(xs) > 1 else xs[0]
    y_all = (np.concatenate(ys) if len(ys) > 1 else ys[0]) if ys else None
    return x_all, y_all


def scan_labels(rec: Recording) -> np.ndarray | None:
    """Load only the labels of a recording (cheap; used by the ``inspect`` stage). None if unlabeled."""
    if not rec.labeled:
        return None
    return np.concatenate([labels_per_window(load_array(y), y) for y in rec.y_files])
