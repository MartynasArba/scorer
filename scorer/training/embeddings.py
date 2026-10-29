"""
Embedding stage + the in-memory/memory-mapped store every downstream head trains on.

embed_dataset():  encoder(prepared windows) -> <emb_root>/<dataset>/embeddings.npy [N, D]
                  streamed recording by recording, batch by batch; only one batch of raw
                  windows is ever on the GPU and results move to CPU immediately.
                  (Old code: SequenceSleepDataset(device='cuda') put whole raw train AND val sets
                  on the GPU -> the CUDA OOM in the RF / MLP stages.)
EmbeddingStore:   embeddings + class labels + recording boundaries for one or more datasets.
                  Context windows [t-k..t+k] and sequences are built per batch by index,
                  clamped INSIDE each recording (never across recordings), with every window
                  kept: unlabeled windows are still context, just excluded from loss/metrics.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scorer.data.cache import DatasetCache
from scorer.data.labels import IGNORE, LabelScheme

log = logging.getLogger(__name__)


def embed_dataset(encoder, cache: DatasetCache, out_root: str | Path, encoder_meta: dict, device,
                  batch_size: int = 2048, dtype: str = "float16", force: bool = False) -> Path:
    """Run the encoder over every window of a prepared dataset and store the embeddings.

    Windows are streamed recording by recording, one batch at a time; each batch's
    output moves to the CPU at once, so GPU memory never holds more than one batch.
    The result is ``<out_root>/<dataset>/embeddings.npy`` [N, D] plus ``index.json`` with
    recording boundaries and the encoder digest.

    Args:
        encoder: Trained encoder (``get_features`` is used if present).
        cache: Prepared dataset.
        out_root: Root folder for embeddings.
        encoder_meta: Encoder metadata; its "digest" decides whether existing output is
            up to date.
        device: Device to run the encoder on.
        batch_size: Windows per forward pass.
        dtype: Storage dtype of the embeddings.
        force: Recompute even if up to date.

    Returns:
        The dataset's embedding folder.
    """
    import torch

    out_dir = Path(out_root) / cache.name
    index_path = out_dir / "index.json"
    if not force and index_path.exists():
        idx = json.loads(index_path.read_text())
        if idx.get("encoder_digest") == encoder_meta.get("digest") and idx.get("n_windows") == cache.n_windows:
            log.info("embeddings for %s already exist for this encoder, skipping", cache.name)
            return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = encoder.to(device).eval()
    feat_fn = getattr(encoder, "get_features", encoder)
    total = cache.n_windows
    t0 = time.time()
    store = None
    recs = []
    pos = 0
    with torch.inference_mode():
        for rec in cache.recordings:
            x = rec.x(mmap=True)
            for s in range(0, rec.n_windows, batch_size):
                xb = torch.from_numpy(np.asarray(x[s : s + batch_size], dtype=np.float32)).to(device, non_blocking=True)
                emb = feat_fn(xb).float().cpu().numpy()
                if store is None:
                    store = np.lib.format.open_memmap(out_dir / "embeddings.npy", mode="w+", dtype=dtype,
                                                      shape=(total, emb.shape[1]))
                store[pos + s : pos + s + len(emb)] = emb
            recs.append({"name": rec.name, "start": pos, "end": pos + rec.n_windows})
            pos += rec.n_windows
    store.flush()
    dim = int(store.shape[1])
    del store
    index = {"dataset": cache.name, "n_windows": total, "dim": dim, "dtype": dtype, "recordings": recs,
             "encoder_path": encoder_meta.get("path"), "encoder_digest": encoder_meta.get("digest"),
             "encoder_stage": encoder_meta.get("stage")}
    index_path.write_text(json.dumps(index, indent=1))
    log.info("embedded %s: %d windows x %d dims in %.0fs", cache.name, total, dim, time.time() - t0)
    return out_dir


@dataclass
class RecordingSpan:
    """Where one recording sits inside an ``EmbeddingStore``.

    Attributes:
        dataset: Dataset name.
        name: Recording name.
        start: First row (inclusive).
        end: Last row (exclusive).
    """
    dataset: str
    name: str
    start: int
    end: int


class EmbeddingStore:
    """Embeddings [N, D], class labels [N] and recording boundaries for one or more datasets.

    Heads read from here. Context windows and sequences are built on demand by index and
    never cross a recording boundary. Unlabeled windows stay in the store as context;
    they are only left out of losses and metrics.

    Args:
        parts: One embedding array per dataset (in memory or memory-mapped).
        classes: Class index per row (``IGNORE`` for unlabeled).
        spans: Recording boundaries in store row numbers.
        encoder_digest: Digest of the encoder that produced the embeddings.

    Raises:
        RuntimeError: If labels and embeddings differ in length.
    """

    def __init__(self, parts: list[np.ndarray], classes: np.ndarray, spans: list[RecordingSpan], encoder_digest: str | None):
        """Index the rows by recording; see the class docstring."""
        self.parts = parts
        self.part_offsets = np.cumsum([0] + [len(p) for p in parts])
        self.classes = classes.astype(np.int64)
        self.spans = spans
        self.encoder_digest = encoder_digest
        n = int(self.part_offsets[-1])
        rec_id = np.empty(n, dtype=np.int32)
        for i, sp in enumerate(spans):
            rec_id[sp.start : sp.end] = i
        self.rec_id = rec_id
        self.rec_start = np.array([sp.start for sp in spans], dtype=np.int64)
        self.rec_end = np.array([sp.end for sp in spans], dtype=np.int64)
        if len(self.classes) != n:
            raise RuntimeError(f"labels ({len(self.classes)}) and embeddings ({n}) differ in length")

    @classmethod
    def load(cls, emb_root: str | Path, caches: Sequence[DatasetCache], scheme: LabelScheme,
             in_memory: bool = True, expect_digest: str | None = None, unlabeled_as_wake: bool = False) -> "EmbeddingStore":
        """Load the embeddings of several datasets into one store.

        Args:
            emb_root: Root folder written by ``embed_dataset``.
            caches: Prepared datasets, for labels and recording order.
            scheme: Label scheme.
            in_memory: Read into RAM (default) instead of memory-mapping.
            expect_digest: Required encoder digest, or None to accept any (but one) encoder.
            unlabeled_as_wake: See ``LabelScheme.to_class``.

        Raises:
            FileNotFoundError: If a dataset has not been embedded.
            RuntimeError: If the encoder digest differs from ``expect_digest`` or between
                datasets, or a recording's size differs from the cache.
        """
        parts, classes, spans = [], [], []
        offset = 0
        digest = None
        for cache in caches:
            d = Path(emb_root) / cache.name
            index_path = d / "index.json"
            if not index_path.exists():
                raise FileNotFoundError(f"no embeddings for {cache.name} under {emb_root}; run the 'embed' stage")
            index = json.loads(index_path.read_text())
            if expect_digest is not None and index["encoder_digest"] != expect_digest:
                raise RuntimeError(f"embeddings of {cache.name} were made with encoder {index['encoder_digest']}, "
                                   f"expected {expect_digest}; re-run 'embed'")
            if digest is not None and index["encoder_digest"] != digest:
                raise RuntimeError("datasets in one store were embedded with different encoders")
            digest = index["encoder_digest"]
            arr = np.load(d / "embeddings.npy", mmap_mode=None if in_memory else "r")
            by_name = {r.name: r for r in cache.recordings}
            for r in index["recordings"]:
                rec = by_name[r["name"]]
                if r["end"] - r["start"] != rec.n_windows:
                    raise RuntimeError(f"{cache.name}/{rec.name}: embedding count differs from cache; re-run 'embed'")
                classes.append(rec.classes(scheme, unlabeled_as_wake))
                spans.append(RecordingSpan(cache.name, rec.name, offset + r["start"], offset + r["end"]))
            parts.append(arr)
            offset += len(arr)
        return cls(parts, np.concatenate(classes), spans, digest)

    # ---- basic access -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Total number of rows (windows)."""
        return int(self.part_offsets[-1])

    @property
    def dim(self) -> int:
        """Embedding size D."""
        return int(self.parts[0].shape[1])

    def rows(self, idx: np.ndarray) -> np.ndarray:
        """Return float32 embeddings for the given row indices, shape [len(idx), D]."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(self.parts) == 1:
            return np.asarray(self.parts[0][idx], dtype=np.float32)
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        part = np.searchsorted(self.part_offsets, idx, side="right") - 1
        for p in np.unique(part):
            m = part == p
            out[m] = self.parts[p][idx[m] - self.part_offsets[p]]
        return out

    def labeled_indices(self) -> np.ndarray:
        """Return the row indices that have a class label."""
        return np.flatnonzero(self.classes != IGNORE)

    # ---- context windows ----------------------------------------------------------------------
    def context_indices(self, centers: np.ndarray, k: int) -> np.ndarray:
        """Return the neighbour rows of each centre, shape [B, 2k + 1].

        Neighbours outside the centre's recording are replaced by the recording's first or
        last row (edge replication), so context never mixes recordings.
        """
        centers = np.asarray(centers, dtype=np.int64)
        offs = np.arange(-k, k + 1, dtype=np.int64)
        r = self.rec_id[centers]
        return np.clip(centers[:, None] + offs[None, :], self.rec_start[r][:, None], self.rec_end[r][:, None] - 1)

    def context_features(self, centers: np.ndarray, k: int) -> np.ndarray:
        """Return flattened context vectors [B, (2k + 1) * D] ordered t-k ... t ... t+k.

        Same layout as the old ``create_context_features``, so old MLP/RF heads still apply.
        """
        idx = self.context_indices(centers, k)
        return self.rows(idx.reshape(-1)).reshape(len(centers), -1)

    # ---- sequences ----------------------------------------------------------------------------
    def sequence_starts(self, seq_len: int, stride: int, max_unlabeled_fraction: float = 1.0,
                        cover_tail: bool = True) -> np.ndarray:
        """Return the start rows of all ``seq_len``-long runs that stay inside one recording.

        Args:
            seq_len: Run length in windows. Recordings shorter than this are skipped (logged).
            stride: Step between starts.
            max_unlabeled_fraction: Drop runs with a larger share of unlabeled windows.
            cover_tail: Add a final start so the last windows of each recording are covered.
        """
        starts = []
        short = 0
        for s, e in zip(self.rec_start, self.rec_end):
            if e - s < seq_len:
                short += 1
                continue
            st = np.arange(s, e - seq_len + 1, stride)
            if cover_tail and st[-1] != e - seq_len:
                st = np.append(st, e - seq_len)
            starts.append(st)
        if short:
            log.warning("%d recordings shorter than seq_len=%d were skipped", short, seq_len)
        starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
        if max_unlabeled_fraction < 1.0 and len(starts):
            lab = (self.classes != IGNORE).astype(np.float64)
            csum = np.concatenate([[0.0], np.cumsum(lab)])
            frac_labeled = (csum[starts + seq_len] - csum[starts]) / seq_len
            starts = starts[1.0 - frac_labeled <= max_unlabeled_fraction]
        return starts

    def sequences(self, starts: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x [S, seq_len, D], classes [S, seq_len])`` for the given start rows."""
        idx = np.asarray(starts)[:, None] + np.arange(seq_len)[None, :]
        x = self.rows(idx.reshape(-1)).reshape(len(starts), seq_len, -1)
        return x, self.classes[idx]

    def describe(self, scheme: LabelScheme) -> str:
        """Return a one-line summary: recordings, windows and counts per class."""
        counts = np.bincount(np.where(self.classes == IGNORE, scheme.n_classes, self.classes),
                             minlength=scheme.n_classes + 1)
        parts = ", ".join(f"{n}={counts[i]}" for i, n in enumerate(scheme.class_names))
        return f"{len(self.spans)} recordings, {len(self)} windows ({parts}, unlabeled={counts[-1]})"
