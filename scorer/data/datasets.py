"""
Streaming window batches from prepared caches, for SimCLR / SupCon / adversarial training.

Replaces BufferedSleepDataset. The core sampler is plain numpy (testable without torch);
WindowBatchStream wraps it as a torch IterableDataset that yields whole BATCHES
(use DataLoader(..., batch_size=None)).

Fixes vs BufferedSleepDataset:
  * buffer size is counted in WINDOWS, not files -> RAM per worker is bounded
    (buffer_windows * C * T * 4 bytes, e.g. 100k x 512 x 4 B = 205 MB);
  * the recording order is drawn from a seed shared by all workers and then sharded,
    so every recording is seen exactly once per epoch (before, each worker shuffled
    independently -> files duplicated / skipped);
  * balancing happens on CLASS indices (after IS->NREM merging), fresh every epoch;
  * labels come from the fixed scheme (no per-dataset remapping);
  * no per-epoch filtering/normalisation (done once in `prepare`);
  * drop_last semantics: no batch smaller than batch_size reaches BatchNorm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from scorer.data.cache import CachedRecording, DatasetCache
from scorer.data.labels import IGNORE, LabelScheme

log = logging.getLogger(__name__)

BALANCE_MODES = ("none", "undersample", "oversample")


@dataclass
class WindowSource:
    recording: CachedRecording
    classes: np.ndarray  # int8 class index per window, IGNORE for unlabeled


def build_sources(caches: Sequence[DatasetCache], scheme: LabelScheme,
                  unlabeled_as_wake: bool = False) -> list[WindowSource]:
    sources = []
    for cache in caches:
        for rec in cache.recordings:
            sources.append(WindowSource(rec, rec.classes(scheme, unlabeled_as_wake).astype(np.int8)))
    return sources


def epoch_selection(sources: Sequence[WindowSource], n_classes: int, balance: str, drop_unlabeled: bool,
                    seed: int, epoch: int) -> list[np.ndarray]:
    """local window indices to use from every source in this epoch (sorted, may repeat when oversampling)"""
    if balance not in BALANCE_MODES:
        raise ValueError(f"balance must be one of {BALANCE_MODES}, got {balance!r}")
    if balance == "none":
        return [np.flatnonzero(s.classes != IGNORE) if drop_unlabeled else np.arange(len(s.classes))
                for s in sources]

    rng = np.random.default_rng([seed, epoch, 7])
    offsets = np.cumsum([0] + [len(s.classes) for s in sources])
    all_cls = np.concatenate([s.classes for s in sources]).astype(np.int64)
    per_class = [np.flatnonzero(all_cls == c) for c in range(n_classes)]
    counts = {c: len(ix) for c, ix in enumerate(per_class)}
    present = [c for c in range(n_classes) if counts[c] > 0]
    if len(present) < n_classes:
        log.warning("balancing: classes %s have no windows; balancing over %s only",
                    [c for c in range(n_classes) if counts[c] == 0], present)
    if not present:
        raise RuntimeError("balancing requested but no labeled windows found")
    target = min(counts[c] for c in present) if balance == "undersample" else max(counts[c] for c in present)
    picked = []
    for c in present:
        ix = per_class[c]
        if balance == "undersample":
            picked.append(rng.choice(ix, size=target, replace=False))
        else:
            picked.append(rng.choice(ix, size=target, replace=True))
    picked = np.sort(np.concatenate(picked))
    src_of = np.searchsorted(offsets, picked, side="right") - 1
    return [picked[src_of == i] - offsets[i] for i in range(len(sources))]


def legacy_loader_augment(x: np.ndarray, y: np.ndarray, wake_index: int | None, rng: np.random.Generator) -> np.ndarray:
    """
    vectorised port of BufferedSleepDataset._augment (applied per window on the CPU):
    noise (p=.7, sd .01), amplitude scale 0.5-1.5 (p=.5), circular shift <=2% (p=.5),
    large noise on Wake windows (p=.1, sd .5)
    """
    n, c, t = x.shape
    x = x.astype(np.float32, copy=True)
    noisy = rng.random(n) < 0.7
    x[noisy] += rng.standard_normal((int(noisy.sum()), c, t), dtype=np.float32) * 0.01
    scaled = rng.random(n) < 0.5
    x[scaled] *= (0.5 + rng.random(int(scaled.sum()), dtype=np.float32))[:, None, None]
    max_shift = int(t * 0.02)
    if max_shift > 0:
        shifts = rng.integers(-max_shift, max_shift + 1, size=n) * (rng.random(n) < 0.5)
        cols = (np.arange(t)[None, :] - shifts[:, None]) % t
        x = np.take_along_axis(x, np.broadcast_to(cols[:, None, :], x.shape), axis=2)
    if wake_index is not None:
        wake = (y == wake_index) & (rng.random(n) < 0.1)
        x[wake] += rng.standard_normal((int(wake.sum()), c, t), dtype=np.float32) * 0.5
    return x


def iter_window_batches(sources: Sequence[WindowSource], selection: Sequence[np.ndarray], batch_size: int,
                        buffer_windows: int, seed: int, epoch: int, worker_id: int = 0, num_workers: int = 1,
                        drop_last: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    yields (x [B, C, T] in cache dtype, y [B] int64) batches.
    Recording order: one permutation per epoch shared by all workers, then sharded by worker.
    Within a worker: fill a RAM buffer up to `buffer_windows`, shuffle it, emit full batches,
    carry the remainder into the next buffer.
    """
    order = np.random.default_rng([seed, epoch, 11]).permutation(len(sources))
    shard = order[worker_id::num_workers]
    rng = np.random.default_rng([seed, epoch, 13, worker_id])
    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    buf_n = 0

    def flush(final: bool):
        nonlocal buf_x, buf_y, buf_n
        if buf_n == 0:
            return
        x = np.concatenate(buf_x) if len(buf_x) > 1 else buf_x[0]
        y = np.concatenate(buf_y) if len(buf_y) > 1 else buf_y[0]
        perm = rng.permutation(len(y))
        n_full = (len(y) // batch_size) * batch_size
        for s in range(0, n_full, batch_size):
            ix = perm[s : s + batch_size]
            yield x[ix], y[ix]
        rest = perm[n_full:]
        if final:
            if len(rest) and not drop_last:
                yield x[rest], y[rest]
            buf_x, buf_y, buf_n = [], [], 0
        else:
            buf_x, buf_y, buf_n = [x[rest]], [y[rest]], len(rest)

    for si in shard:
        idx = selection[si]
        if len(idx) == 0:
            continue
        src = sources[si]
        xmm = src.recording.x(mmap=True)
        pos = 0
        while pos < len(idx):
            take = idx[pos : pos + max(1, buffer_windows - buf_n)]
            pos += len(take)
            buf_x.append(np.asarray(xmm[take]))
            buf_y.append(src.classes[take].astype(np.int64))
            buf_n += len(take)
            if buf_n >= buffer_windows:
                yield from flush(final=False)
    yield from flush(final=True)


try:  # torch wrapper (the numpy core above is importable without torch)
    import torch
    from torch.utils.data import IterableDataset, get_worker_info

    class WindowBatchStream(IterableDataset):
        """
        IterableDataset yielding (x float32 [B, C, T], y int64 [B]) batches.
        Call set_epoch(e) before each epoch; wrap with DataLoader(ds, batch_size=None, num_workers=...).
        """

        def __init__(self, caches: Sequence[DatasetCache], scheme: LabelScheme, batch_size: int,
                     buffer_windows: int = 100_000, balance: str = "none", drop_unlabeled: bool = True,
                     seed: int = 0, drop_last: bool = True, loader_augment: bool = False,
                     unlabeled_as_wake: bool = False):
            super().__init__()
            self.sources = build_sources(caches, scheme, unlabeled_as_wake)
            self.scheme = scheme
            self.batch_size = int(batch_size)
            self.buffer_windows = max(int(buffer_windows), self.batch_size)
            self.balance = balance
            self.drop_unlabeled = drop_unlabeled
            self.seed = seed
            self.drop_last = drop_last
            self.loader_augment = loader_augment
            self.epoch = 0
            if balance != "none" and not drop_unlabeled:
                raise ValueError("balancing needs labels: set drop_unlabeled=True")
            n_labeled = sum(int((s.classes != IGNORE).sum()) for s in self.sources)
            if drop_unlabeled and n_labeled == 0:
                raise RuntimeError("drop_unlabeled=True but the datasets contain no labeled windows")
            self._len = sum(len(i) for i in self._selection(0)) // self.batch_size

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def _selection(self, epoch: int) -> list[np.ndarray]:
            return epoch_selection(self.sources, self.scheme.n_classes, self.balance, self.drop_unlabeled,
                                   self.seed, epoch)

        def __len__(self) -> int:  # approximate number of batches per epoch (informational)
            return self._len

        def class_distribution(self, epoch: int = 0) -> dict[str, int]:
            sel = self._selection(epoch)
            counts = np.zeros(self.scheme.n_classes + 1, dtype=np.int64)
            for s, ix in zip(self.sources, sel):
                c = s.classes[ix].astype(np.int64)
                counts += np.bincount(np.where(c == IGNORE, self.scheme.n_classes, c),
                                      minlength=self.scheme.n_classes + 1)
            out = {n: int(counts[i]) for i, n in enumerate(self.scheme.class_names)}
            out["unlabeled"] = int(counts[-1])
            return out

        def __iter__(self):
            info = get_worker_info()
            wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
            selection = self._selection(self.epoch)
            aug_rng = np.random.default_rng([self.seed, self.epoch, 17, wid])
            wake = self.scheme.index_of("Wake")
            for x, y in iter_window_batches(self.sources, selection, self.batch_size, self.buffer_windows,
                                            self.seed, self.epoch, wid, nw, self.drop_last):
                if self.loader_augment:
                    x = legacy_loader_augment(x, y, wake, aug_rng)
                yield torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(y)

except ImportError:  # pragma: no cover
    WindowBatchStream = None  # type: ignore
