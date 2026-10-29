"""
Prepared-data cache: the `prepare` stage normalises every recording ONCE and stores it as
memory-mappable .npy files. All training stages read from here.

    <cache_root>/<dataset>/index.json
    <cache_root>/<dataset>/<recording>/X.npy     [N, C, T] float16|float32, normalised
    <cache_root>/<dataset>/<recording>/y.npy     [N] int8 RAW codes (-1 if the recording is unlabeled)
    <cache_root>/<dataset>/<recording>/meta.json

Raw codes are stored (not class indices) so changing the label scheme never requires
re-preparing. Reading uses np.load(mmap_mode='r'): nothing is held in RAM until indexed.

Why: the old loaders re-ran torch.load + float64 filtfilt (twice) on every file every epoch
(200 epochs x all files), and held whole datasets on the GPU in the RF/MLP stages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scorer.data.labels import IGNORE, LabelScheme, raw_counts
from scorer.data.recordings import discover_recordings, load_recording
from scorer.data.signal import robust_normalize

log = logging.getLogger(__name__)

CACHE_FORMAT = "scorer-cache-v1"


@dataclass(frozen=True)
class PrepParams:
    """Settings that decide how recordings are prepared; part of every cache's identity.

    Changing any field changes ``digest()``, and ``DatasetCache`` then refuses the old
    cache until ``prepare`` is re-run.

    Attributes:
        sample_rate: Sample rate of the source data in Hz (used by the low-pass filter).
        win_len: Expected samples per window; a mismatch stops ``prepare`` with an error.
        channels: Channel indices to keep from the source arrays.
        lowpass_hz: Low-pass cut-off applied before normalising, or None for no filter.
        clip: Normalised values are clipped to ``[-clip, clip]``.
        dtype: Storage dtype of ``X.npy`` ("float16" halves disk and RAM use).
        max_len_mismatch: Largest allowed difference in windows between X and y files.
    """
    sample_rate: float
    win_len: int
    channels: tuple[int, ...]
    lowpass_hz: float | None
    clip: float
    dtype: str = "float16"
    max_len_mismatch: int = 1

    def to_dict(self) -> dict:
        """Return the parameters as a JSON-safe dict."""
        d = dict(self.__dict__)
        d["channels"] = list(self.channels)
        return d

    def digest(self) -> str:
        """Return a 12-character hash of the parameters, used to detect stale caches."""
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def prepare_dataset(source_root: str | Path, cache_root: str | Path, name: str, params: PrepParams,
                    force: bool = False, seed: int = 0) -> Path:
    """Normalise every recording under ``source_root`` and store it in ``cache_root/name``.

    Recordings whose ``meta.json`` already carries the same parameter digest are skipped,
    so an interrupted run resumes where it stopped.

    Args:
        source_root: Folder searched for recordings (see ``discover_recordings``).
        cache_root: Root of all caches.
        name: Dataset name; becomes the sub-folder.
        params: Preparation settings.
        force: Re-prepare every recording even if it is up to date.
        seed: Base seed for the random subsample used to estimate normalisation statistics.

    Returns:
        The dataset's cache folder.

    Raises:
        ValueError: If a recording's window length differs from ``params.win_len``.
    """
    out_dir = Path(cache_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = discover_recordings(source_root)
    entries = []
    t0 = time.time()
    for i, rec in enumerate(recordings):
        rec_dir = out_dir / rec.name
        meta_path = rec_dir / "meta.json"
        if not force and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("params_digest") == params.digest():
                entries.append(meta)
                continue
        x, y = load_recording(rec, list(params.channels), params.max_len_mismatch)
        if x.shape[-1] != params.win_len:
            raise ValueError(
                f"{rec.directory}: window length {x.shape[-1]} != configured win_len {params.win_len} "
                f"(is this dataset sampled at {params.sample_rate} Hz?)"
            )
        x, stats = robust_normalize(x, params.sample_rate, params.lowpass_hz, params.clip, seed=seed + i)
        rec_dir.mkdir(parents=True, exist_ok=True)
        np.save(rec_dir / "X.npy", x.astype(params.dtype, copy=False))
        y_store = (y if y is not None else np.full(len(x), IGNORE)).astype(np.int8)
        np.save(rec_dir / "y.npy", y_store)
        meta = {
            "format": CACHE_FORMAT,
            "recording": rec.name,
            "source_dir": str(rec.directory),
            "source_files": [str(p) for p in rec.x_files],
            "n_windows": int(len(x)),
            "n_channels": int(x.shape[1]),
            "win_len": int(x.shape[2]),
            "labeled": y is not None,
            "raw_label_counts": raw_counts(y) if y is not None else {},
            "norm_stats": stats,
            "params": params.to_dict(),
            "params_digest": params.digest(),
        }
        meta_path.write_text(json.dumps(meta, indent=1))
        entries.append(meta)
        log.info("[%s] %d/%d %s: %d windows, labels %s", name, i + 1, len(recordings), rec.name,
                 len(x), meta["raw_label_counts"])
        del x, y
    index = {
        "format": CACHE_FORMAT,
        "dataset": name,
        "source_root": str(source_root),
        "params": params.to_dict(),
        "params_digest": params.digest(),
        "recordings": [{"name": e["recording"], "n_windows": e["n_windows"], "labeled": e["labeled"],
                        "raw_label_counts": e["raw_label_counts"]} for e in entries],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log.info("[%s] prepared %d recordings in %.0fs -> %s", name, len(entries), time.time() - t0, out_dir)
    return out_dir


@dataclass
class CachedRecording:
    """One prepared recording inside a dataset cache.

    Attributes:
        dataset: Name of the dataset it belongs to.
        name: Recording name (folder name).
        path: Folder holding ``X.npy``, ``y.npy`` and ``meta.json``.
        n_windows: Number of windows.
        labeled: False if the source had no label file; its labels are then all ``IGNORE``.
    """
    dataset: str
    name: str
    path: Path
    n_windows: int
    labeled: bool

    def x(self, mmap: bool = True) -> np.ndarray:
        """Load the normalised windows, shape [N, C, T].

        Args:
            mmap: Memory-map the file (default) so nothing is read until indexed.
        """
        return np.load(self.path / "X.npy", mmap_mode="r" if mmap else None)

    def raw_labels(self) -> np.ndarray:
        """Load the raw label codes as int64, shape [N]."""
        return np.load(self.path / "y.npy").astype(np.int64)

    def classes(self, scheme: LabelScheme, unlabeled_as_wake: bool = False) -> np.ndarray:
        """Return class indices for ``scheme`` (all ``IGNORE`` if the recording is unlabeled).

        Args:
            scheme: Label scheme to map raw codes with.
            unlabeled_as_wake: See ``LabelScheme.to_class``.
        """
        raw = self.raw_labels()
        if not self.labeled:
            return np.full(len(raw), IGNORE, dtype=np.int64)
        return scheme.to_class(raw, unlabeled_as_wake)


class DatasetCache:
    """Read-only view of one prepared dataset.

    Args:
        cache_root: Root of all caches.
        name: Dataset name.
        expect: If given, the cache must have been prepared with exactly these parameters.

    Raises:
        FileNotFoundError: If the dataset has not been prepared.
        RuntimeError: If ``expect`` differs from the parameters the cache was built with.
    """

    def __init__(self, cache_root: str | Path, name: str, expect: PrepParams | None = None):
        """Open the dataset's index and check its parameters; see the class docstring."""
        self.name = name
        self.root = Path(cache_root) / name
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"dataset {name!r} is not prepared ({index_path} missing); run the 'prepare' stage")
        self.index = json.loads(index_path.read_text())
        if expect is not None and self.index["params_digest"] != expect.digest():
            raise RuntimeError(
                f"cache for {name!r} was prepared with {self.index['params']}, "
                f"but the config now asks for {expect.to_dict()}; re-run 'prepare' (it overwrites)"
            )
        self.params = self.index["params"]
        self.recordings = [
            CachedRecording(name, r["name"], self.root / r["name"], r["n_windows"], r["labeled"])
            for r in self.index["recordings"]
        ]

    @property
    def n_windows(self) -> int:
        """Total number of windows over all recordings."""
        return sum(r.n_windows for r in self.recordings)

    def __repr__(self) -> str:
        """Return a short summary, e.g. ``DatasetCache('oxford', 12 recordings, 40000 windows)``."""
        return f"DatasetCache({self.name!r}, {len(self.recordings)} recordings, {self.n_windows} windows)"
