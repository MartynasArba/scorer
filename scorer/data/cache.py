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
    sample_rate: float
    win_len: int
    channels: tuple[int, ...]
    lowpass_hz: float | None
    clip: float
    dtype: str = "float16"
    max_len_mismatch: int = 1

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["channels"] = list(self.channels)
        return d

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def prepare_dataset(source_root: str | Path, cache_root: str | Path, name: str, params: PrepParams,
                    force: bool = False, seed: int = 0) -> Path:
    """normalise all recordings under source_root into cache_root/name. Idempotent per recording."""
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
    dataset: str
    name: str
    path: Path
    n_windows: int
    labeled: bool

    def x(self, mmap: bool = True) -> np.ndarray:
        return np.load(self.path / "X.npy", mmap_mode="r" if mmap else None)

    def raw_labels(self) -> np.ndarray:
        return np.load(self.path / "y.npy").astype(np.int64)

    def classes(self, scheme: LabelScheme, unlabeled_as_wake: bool = False) -> np.ndarray:
        raw = self.raw_labels()
        if not self.labeled:
            return np.full(len(raw), IGNORE, dtype=np.int64)
        return scheme.to_class(raw, unlabeled_as_wake)


class DatasetCache:
    """read-only view of one prepared dataset"""

    def __init__(self, cache_root: str | Path, name: str, expect: PrepParams | None = None):
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
        return sum(r.n_windows for r in self.recordings)

    def __repr__(self) -> str:
        return f"DatasetCache({self.name!r}, {len(self.recordings)} recordings, {self.n_windows} windows)"
