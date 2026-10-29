"""
Windowed recording for the manual-scoring tab (Qt-free). Replaces loaders.SleepSignals.

Differences from SleepSignals:
  * numpy on the CPU (SleepSignals put the whole recording on 'cuda' by default and the GUI then
    pulled every window back with .cpu().numpy(), one window at a time);
  * the data layout is inferred once per recording (or given), so a small last chunk can no longer
    be transposed the wrong way;
  * a recording without y files opens with all windows unlabeled instead of failing;
  * window ranges are sliced in one operation (`signals(idx, n)`), not looped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scorer.data.recordings import (Recording, discover_recordings, labels_per_window, load_array, load_recording,
                                    to_windows_channels_time, infer_layout)

log = logging.getLogger(__name__)

STATE_NAMES = {0: "Unlabeled", 1: "Wake", 2: "NREM", 3: "IS", 4: "REM"}


def guess_label_file(x_path: str | Path) -> Path | None:
    """Find the label file for a single windowed X file, or None.

    Follows ``storage.save_windowed`` naming: ``<name>_X.pt`` -> ``<name>_y.pt`` and
    ``X_chunk3.pt`` -> ``y_chunk3.pt``. Only the X marker is replaced, so recording
    names that contain an X elsewhere still work.
    """
    p = Path(x_path)
    candidates = []
    for marker in ("_X.", "_X_"):
        i = p.name.rfind(marker)
        if i >= 0:
            candidates.append(p.with_name(p.name[:i] + marker.replace("X", "y") + p.name[i + len(marker):]))
    if p.name.startswith("X"):
        candidates.append(p.with_name("y" + p.name[1:]))
    return next((c for c in candidates if c.exists()), None)


def compute_ylims(x: np.ndarray, mode: str = "standard", n_ephys: int = 2, seed: int = 0) -> list[tuple[float, float]]:
    """Return a ``(center, spread)`` display range per channel of ``x`` [N, C, T].

    Modes (as in the old ``SleepSignals``): "standard" uses fixed ranges; "infer" uses
    the 1-99% quantiles of up to 1000 random windows for every channel; "infer_ephys"
    does that only for the first ``n_ephys`` channels.
    """
    rng = np.random.default_rng(seed)
    n, c, _ = x.shape
    idx = rng.integers(0, n, size=min(1000, n)) if n else np.zeros(0, dtype=int)
    out = []
    for ch in range(c):
        if mode == "standard":
            out.append((0.0, 0.2) if ch < n_ephys else (0.5, 1.0))
        elif mode == "infer" or (mode == "infer_ephys" and ch < n_ephys):
            vals = np.asarray(x[idx, ch, :], dtype=np.float64).reshape(-1)
            if vals.size == 0:
                out.append((0.0, 1.0))
                continue
            lo, med, hi = np.quantile(vals, [0.01, 0.5, 0.99])
            out.append((float(med), float(max(hi - lo, 1e-9))))
        else:
            out.append((0.5, 1.0))
    return out


def robust_display_scale(x: np.ndarray, max_samples: int = 200_000, seed: int = 0) -> np.ndarray:
    """Return ``(x - median) / (IQR / 1.349)`` per channel, for display only (no filter, no clipping)."""
    n, c, t = x.shape
    rng = np.random.default_rng(seed)
    out = np.empty_like(x, dtype=np.float32)
    for ch in range(c):
        flat = x[:, ch, :].reshape(-1)
        sample = flat if flat.size <= max_samples else flat[rng.integers(0, flat.size, max_samples)]
        q1, med, q3 = np.quantile(sample.astype(np.float64), [0.25, 0.5, 0.75])
        out[:, ch, :] = (x[:, ch, :] - med) / max((q3 - q1) / 1.349, 1e-8)
    return out


@dataclass
class RecordingData:
    """One windowed recording held in memory for the manual-scoring tab.

    Attributes:
        x: Windows [N, C, T] float32.
        states: Raw codes from the label files, 0 where there are none.
        source: Folder or file it was loaded from.
        labeled: True if label files were found.
        ylims: Display range per channel (see ``set_ylims``).
    """
    x: np.ndarray                      # [N, C, T] float32
    states: np.ndarray                 # [N] int64 raw codes from y files (0 where none)
    source: str
    labeled: bool
    ylims: list[tuple[float, float]] = field(default_factory=list)

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_folder(cls, folder: str | Path, layout: str = "auto", max_len_mismatch: int = 1) -> "RecordingData":
        """Load the single recording in ``folder``.

        Raises:
            RuntimeError: If the folder holds several recordings.
        """
        recs = discover_recordings(folder)
        if len(recs) != 1:
            raise RuntimeError(f"{folder} holds {len(recs)} recordings ({[r.name for r in recs][:5]}...); "
                               f"open one recording folder")
        return cls._from_recording(recs[0], layout, max_len_mismatch)

    @classmethod
    def from_file(cls, x_path: str | Path, y_path: str | Path | None = "auto", layout: str = "auto") -> "RecordingData":
        """Load a single (non-chunked) X file.

        Args:
            x_path: The X file.
            y_path: Label file; "auto" looks for it with ``guess_label_file``; None loads
                without labels.
            layout: Axis order of the file ("auto", "nct", "cnt").
        """
        x_path = Path(x_path)
        if y_path == "auto":
            y_path = guess_label_file(x_path)
        arr = load_array(x_path)
        x = to_windows_channels_time(arr, x_path, infer_layout(arr.shape) if layout == "auto" else layout)
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = None
        if y_path is not None and Path(y_path).exists():
            y = labels_per_window(load_array(Path(y_path)), y_path)
            if len(y) != len(x):
                log.warning("%s: %d windows but %d labels; trimming/padding labels", x_path, len(x), len(y))
                y = _fit_length(y, len(x))
        return cls(x, y if y is not None else np.zeros(len(x), dtype=np.int64), str(x_path), y is not None)

    @classmethod
    def _from_recording(cls, rec: Recording, layout: str, max_len_mismatch: int) -> "RecordingData":
        """Load every chunk of a discovered recording, all channels."""
        x, y = load_recording(rec, None, max_len_mismatch, layout=layout)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return cls(x, y if y is not None else np.zeros(len(x), dtype=np.int64), str(rec.directory), y is not None)

    # ---- access -------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of windows."""
        return len(self.x)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.x.shape[1])

    @property
    def win_len(self) -> int:
        """Samples per window."""
        return int(self.x.shape[2])

    def signals(self, idx: int, n: int) -> np.ndarray:
        """Return windows ``idx`` to ``idx + n - 1`` joined along time, shape [C, n * T]."""
        block = self.x[idx: min(idx + n, len(self.x))]
        return np.ascontiguousarray(block.transpose(1, 0, 2)).reshape(self.n_channels, -1)

    def set_ylims(self, mode: str, n_ephys: int) -> None:
        """Compute display ranges with ``compute_ylims``."""
        self.ylims = compute_ylims(self.x, mode, n_ephys)

    def normalize_for_display(self) -> None:
        """Replace ``x`` with its robust-scaled version (see ``robust_display_scale``)."""
        self.x = robust_display_scale(self.x)


def _fit_length(arr: np.ndarray, n: int, fill: int = 0) -> np.ndarray:
    """Trim or pad ``arr`` to ``n`` entries with ``fill``."""
    arr = np.asarray(arr)
    if len(arr) >= n:
        return arr[:n].copy()
    return np.concatenate([arr, np.full(n - len(arr), fill, dtype=arr.dtype)])
