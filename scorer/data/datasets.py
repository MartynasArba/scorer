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
    """One cached recording plus its per-window class labels for the current scheme.

    Attributes:
        recording: The cached recording (windows are read lazily from its memory map).
        classes: int8 class index per window; ``IGNORE`` for unlabeled windows.
    """
    recording: CachedRecording
    classes: np.ndarray  # int8 class index per window, IGNORE for unlabeled


def build_sources(caches: Sequence[DatasetCache], scheme: LabelScheme,
                  unlabeled_as_wake: bool = False) -> list[WindowSource]:
    """Wrap every recording of ``caches`` as a ``WindowSource`` with labels mapped by ``scheme``.

    Args:
        caches: Prepared datasets to draw from.
        scheme: Label scheme that turns raw codes into class indices.
        unlabeled_as_wake: See ``LabelScheme.to_class``.
    """
    sources = []
    for cache in caches:
        for rec in cache.recordings:
            sources.append(WindowSource(rec, rec.classes(scheme, unlabeled_as_wake).astype(np.int8)))
    return sources


def epoch_selection(sources: Sequence[WindowSource], n_classes: int, balance: str, drop_unlabeled: bool,
                    seed: int, epoch: int) -> list[np.ndarray]:
    """Choose which windows of every source are used in one epoch.

    Args:
        sources: Recordings with their class labels.
        n_classes: Number of classes in the scheme.
        balance: "none" keeps every window; "undersample" draws the size of the
            smallest class from each class; "oversample" draws the size of the largest,
            with replacement.
        drop_unlabeled: With ``balance="none"``, drop ``IGNORE`` windows.
        seed: Run seed.
        epoch: Epoch number; balanced selections differ per epoch but are reproducible.

    Returns:
        One sorted array of local window indices per source (indices may repeat when
        oversampling).

    Raises:
        ValueError: If ``balance`` is not one of ``BALANCE_MODES``.
        RuntimeError: If balancing is requested but no window has a label.
    """
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
    """Apply the old loader's CPU augmentations to a batch (vectorised port of
    ``BufferedSleepDataset._augment``).

    Per window: Gaussian noise (p=0.7, sd 0.01); amplitude scale 0.5-1.5 (p=0.5);
    circular shift up to 2% of the window (p=0.5); strong noise on Wake windows
    (p=0.1, sd 0.5).

    Args:
        x: Batch [B, C, T].
        y: Class index per window [B].
        wake_index: Class index of Wake, or None to skip the Wake-noise step.
        rng: Random generator (seeded per worker and epoch).

    Returns:
        Augmented float32 copy of ``x``.
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
    """Yield shuffled batches ``(x [B, C, T], y [B] int64)`` for one worker and one epoch.

    The recording order is one permutation per epoch, shared by all workers and then
    sharded, so each recording is read by exactly one worker. Inside a worker, windows
    fill a RAM buffer of ``buffer_windows``; the buffer is shuffled, full batches are
    emitted, and the remainder is carried into the next buffer.

    Args:
        sources: Recordings with labels.
        selection: Windows to use per source (from ``epoch_selection``).
        batch_size: Windows per batch.
        buffer_windows: Shuffle-buffer size in windows; bounds RAM per worker.
        seed: Run seed.
        epoch: Epoch number.
        worker_id: Index of this DataLoader worker.
        num_workers: Total number of workers.
        drop_last: Drop the final partial batch (avoids tiny BatchNorm batches).
    """
    order = np.random.default_rng([seed, epoch, 11]).permutation(len(sources))
    shard = order[worker_id::num_workers]
    rng = np.random.default_rng([seed, epoch, 13, worker_id])
    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    buf_n = 0

    def flush(final: bool):
        """Shuffle the buffer and yield its full batches; keep or (if final) emit the rest."""
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
        """Torch ``IterableDataset`` that yields whole batches ``(x float32 [B, C, T], y int64 [B])``.

        Use it with ``DataLoader(ds, batch_size=None, num_workers=...)`` and call
        ``set_epoch(e)`` before each epoch.

        Args:
            caches: Prepared datasets to stream.
            scheme: Label scheme.
            batch_size: Windows per batch.
            buffer_windows: Shuffle-buffer size per worker (at least ``batch_size``).
            balance: Class balancing mode, see ``epoch_selection``.
            drop_unlabeled: Skip windows without a label.
            seed: Run seed.
            drop_last: Drop the final partial batch.
            loader_augment: Apply ``legacy_loader_augment`` on the CPU.
            unlabeled_as_wake: See ``LabelScheme.to_class``.

        Raises:
            ValueError: If balancing is requested with ``drop_unlabeled=False``.
            RuntimeError: If ``drop_unlabeled`` is set but no window has a label.
        """

        def __init__(self, caches: Sequence[DatasetCache], scheme: LabelScheme, batch_size: int,
                     buffer_windows: int = 100_000, balance: str = "none", drop_unlabeled: bool = True,
                     seed: int = 0, drop_last: bool = True, loader_augment: bool = False,
                     unlabeled_as_wake: bool = False):
            """Collect the sources and check the settings; see the class docstring."""
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
            """Select the epoch whose window selection and shuffling the next iteration uses."""
            self.epoch = int(epoch)

        def _selection(self, epoch: int) -> list[np.ndarray]:
            """Return the window selection for ``epoch``."""
            return epoch_selection(self.sources, self.scheme.n_classes, self.balance, self.drop_unlabeled,
                                   self.seed, epoch)

        def __len__(self) -> int:  # approximate number of batches per epoch (informational)
            """Approximate number of batches per epoch (for progress logs only)."""
            return self._len

        def class_distribution(self, epoch: int = 0) -> dict[str, int]:
            """Count windows per class (and 'unlabeled') in the selection of ``epoch``."""
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
            """Yield the batches of the current epoch for this DataLoader worker."""
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
