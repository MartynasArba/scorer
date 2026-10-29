"""
Scoring API (clean break from scoring.score_signal). The GUI and batch scoring use SleepScorer;
the pipeline's evaluate stage uses the same head loading and post-processing, so validation
numbers come from the code that scores data.

    scorer = SleepScorer("runs/128hz/adversarial/encoder_final.pt", "runs/128hz/heads/tuned/context_mlp.pt")
    result = scorer.score_folder(r"D:\\rec\\processed\\windowed_x", channel=0, heuristics="lite")
    result.gui_codes      # int array, 1=Wake 2=NREM 4=REM (raw codes, see data/labels.py)
    result.confidence     # max class probability per window
    save_states(result, r"D:\\states\\rec_states.pkl")

Preprocessing (sample rate, window length, low-pass, clip) is read from the encoder checkpoint,
so GUI scoring can no longer drift from what the model was trained on. Data at another sample
rate is accepted when you pass source_sample_rate and its windows have the model's duration
(e.g. 1000 samples at 250 Hz = 4 s): the windows are joined into one continuous signal,
resampled (polyphase) and cut again, so states stay one per window.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scorer.data.labels import IGNORE, LabelScheme, scheme_from_meta
from scorer.data.recordings import discover_recordings, load_recording
from scorer.data.signal import resample_windows, robust_normalize
from scorer.inference.postprocess import predict_classes
from scorer.training.embeddings import EmbeddingStore, RecordingSpan

log = logging.getLogger(__name__)

HEAD_KINDS = ("context_mlp", "linear_probe", "gru", "context_rf")


@dataclass
class Head:
    """A loaded classifier head and what it needs to run.

    Attributes:
        kind: "context_mlp", "linear_probe", "gru" or "context_rf".
        model: The torch module or the fitted forest.
        meta: Checkpoint metadata (class names, context size or sequence length, ...).
        path: File it was loaded from.
    """
    kind: str
    model: Any
    meta: dict
    path: str

    def predict_probs(self, store: EmbeddingStore, device) -> np.ndarray:
        """Return probabilities [N, n_classes] for every window in ``store``."""
        n_classes = len(self.meta["class_names"])
        if self.kind in ("context_mlp", "linear_probe"):
            from scorer.training.context import predict_window_head

            return predict_window_head(self.model, store, int(self.meta.get("context_k", 0)), device)
        if self.kind == "gru":
            from scorer.training.sequence_gru import predict_gru

            return predict_gru(self.model, store, int(self.meta["seq_len"]), device)
        if self.kind == "context_rf":
            from scorer.training.context import predict_rf

            return predict_rf(self.model, store, int(self.meta.get("context_k", 1)), n_classes)
        raise ValueError(self.kind)


def load_head(path: str | Path, device="cpu", legacy_meta: dict | None = None) -> Head:
    """Load any head checkpoint, new or old format.

    ``.pkl`` files are random-forest bundles. Old raw state dicts are recognised by
    their keys (an old GRU model or ContextResMLP) and given assumed metadata, which is
    logged; ``legacy_head_meta`` overrides it.

    Args:
        path: Checkpoint file.
        device: Device for torch heads.
        legacy_meta: Metadata for old checkpoints that cannot be inferred.

    Raises:
        RuntimeError: If the checkpoint is not a head or does not match its architecture.
    """
    path = Path(path)
    if path.suffix in (".pkl", ".pickle"):
        from scorer.models.checkpoints import load_rf_bundle

        rf, meta = load_rf_bundle(path, legacy_meta)
        return Head("context_rf", rf, meta, str(path))
    from scorer.models.checkpoints import read_checkpoint, strict_load
    from scorer.models.heads import ContextResMLP, GRUSequenceHead, LinearProbe, load_legacy_gru_state

    state, meta, kind, legacy = read_checkpoint(path)
    if legacy:
        if any(k.startswith("rnn.") for k in state):  # old ContextAwareSleepScorer (encoder + GRU in one file)
            _, state = load_legacy_gru_state(state)
            kind = "gru"
            meta = {"arch": {"embedding_dim": 512, "hidden_dim": 64, "num_classes": 3, "num_layers": 2},
                    "seq_len": 10, "label_scheme": "3state", "class_names": ["Wake", "NREM", "REM"], **(legacy_meta or {})}
        elif any(k.startswith("input_proj.") for k in state):  # old ContextResMLP weights
            kind = "context_mlp"
            in_dim = int(state["input_proj.0.weight"].shape[1])
            hidden = int(state["input_proj.0.weight"].shape[0])
            n_blocks = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
            meta = {"arch": {"input_dim": in_dim, "hidden_dim": hidden, "num_classes": 3, "num_blocks": n_blocks,
                             "dropout": 0.2},
                    "context_k": (in_dim // 512 - 1) // 2, "label_scheme": "3state",
                    "class_names": ["Wake", "NREM", "REM"], **(legacy_meta or {})}
        else:
            raise RuntimeError(f"{path}: cannot tell which head this old checkpoint belongs to")
        log.warning("%s: old-format %s checkpoint, assuming %s", path, kind,
                    {k: meta[k] for k in meta if k != "arch"})
    builders = {"context_mlp": ContextResMLP, "linear_probe": LinearProbe, "gru": GRUSequenceHead}
    if kind not in builders:
        raise RuntimeError(f"{path}: checkpoint kind {kind!r} is not a classifier head ({HEAD_KINDS})")
    model = builders[kind](**meta["arch"])
    strict_load(model, state, source=str(path))
    return Head(kind, model.to(device).eval(), meta, str(path))


@dataclass
class ScoringResult:
    """Output of ``SleepScorer`` for one recording.

    Attributes:
        classes: Class index per window, in scheme order.
        gui_codes: Raw codes for the GUI (1 Wake, 2 NREM, 3 IS, 4 REM).
        probs: Probabilities [N, n_classes], after optional smoothing.
        confidence: Highest probability per window.
        class_names: Class names in index order.
    """
    classes: np.ndarray       # class index per window (scheme order)
    gui_codes: np.ndarray     # raw codes 1/2/(3)/4 for the GUI
    probs: np.ndarray         # [N, n_classes] after optional smoothing
    confidence: np.ndarray    # max probability per window
    class_names: tuple[str, ...]


class SleepScorer:
    """Score recordings with a trained encoder and head.

    The GUI and batch scoring use this class. The pipeline's ``evaluate`` stage uses the
    same head loading and post-processing (``load_head``, ``predict_classes``) on its
    precomputed embeddings, so validation numbers come from the same code.

    Preprocessing settings (sample rate, window length, low-pass, clip) come from the
    encoder checkpoint, so scoring cannot drift from training.

    Args:
        encoder_path: Encoder checkpoint.
        head_path: Head checkpoint (``.pt`` or random-forest ``.pkl``).
        device: "auto", "cuda" or "cpu". Random forests always run on the CPU.
        legacy_encoder_meta: Metadata for an old-format encoder, see ``load_encoder``.
        legacy_head_meta: Metadata for an old-format head, see ``load_head``.
        allow_encoder_mismatch: Only warn (instead of failing) when the head was trained
            on a different encoder file.

    Raises:
        RuntimeError: If the head was trained on a different encoder and
            ``allow_encoder_mismatch`` is False.
    """
    def __init__(self, encoder_path: str | Path, head_path: str | Path, device: str = "auto",
                 legacy_encoder_meta: dict | None = None, legacy_head_meta: dict | None = None,
                 allow_encoder_mismatch: bool = False):
        """Load both checkpoints and check that they belong together; see the class docstring."""
        from scorer.models.checkpoints import load_encoder
        from scorer.training.common import resolve_device

        self.device = resolve_device(device)
        self.encoder, self.enc_meta = load_encoder(encoder_path, self.device, legacy_meta=legacy_encoder_meta)
        self.head = load_head(head_path, self.device if not str(head_path).endswith(".pkl") else "cpu",
                              legacy_head_meta)
        trained_on = self.head.meta.get("encoder_digest")
        if trained_on and trained_on != self.enc_meta["digest"]:
            msg = (f"head {head_path} was trained on encoder {self.head.meta.get('encoder_path')} "
                   f"(digest {trained_on}), but {encoder_path} has digest {self.enc_meta['digest']}")
            if not allow_encoder_mismatch:
                raise RuntimeError(msg)
            log.warning(msg)
        self.scheme: LabelScheme = scheme_from_meta(self.head.meta)

    # ---------------------------------------------------------------------------------------
    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz the encoder was trained at."""
        return float(self.enc_meta["sample_rate"])

    @property
    def win_len(self) -> int:
        """Window length in samples the encoder expects."""
        return int(self.enc_meta["win_len"])

    @property
    def epoch_seconds(self) -> float:
        """Window duration in seconds (``win_len / sample_rate``)."""
        return self.win_len / self.sample_rate

    def describe(self) -> dict:
        """Return a dict describing the model pair (head, paths, classes, rate, training data) for display."""
        return {"head": self.head.kind, "head_path": self.head.path, "encoder_path": self.enc_meta.get("path"),
                "encoder_stage": self.enc_meta.get("stage"), "mode": self.head.meta.get("mode"),
                "classes": list(self.scheme.class_names), "sample_rate": self.sample_rate, "win_len": self.win_len,
                "trained_on": self.head.meta.get("datasets"), "context_k": self.head.meta.get("context_k")}

    def preprocess(self, x_raw: np.ndarray, source_sample_rate: float | None = None) -> np.ndarray:
        """Resample (if needed) and normalise raw windows the way the encoder was trained.

        Args:
            x_raw: Windows [N, T] or [N, 1, T] of one channel of one continuous recording.
            source_sample_rate: Rate of ``x_raw`` if it differs from the model's. The
                windows must last as long as the model's (e.g. 1000 samples at 250 Hz = 4 s);
                they are joined, resampled and cut again.

        Returns:
            Normalised float32 windows [N, 1, win_len].

        Raises:
            ValueError: If the window duration or length does not fit the model.
        """
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if x_raw.ndim == 2:
            x_raw = x_raw[:, None, :]
        if source_sample_rate and float(source_sample_rate) != self.sample_rate:
            src_epoch = x_raw.shape[-1] / float(source_sample_rate)
            if abs(src_epoch - self.epoch_seconds) > 1e-6:
                raise ValueError(f"windows last {src_epoch:g} s ({x_raw.shape[-1]} samples at {source_sample_rate} Hz) "
                                 f"but the model scores {self.epoch_seconds:g}-s epochs")
            x_raw = resample_windows(x_raw, float(source_sample_rate), self.sample_rate, self.win_len)
        if x_raw.shape[-1] != self.win_len:
            raise ValueError(f"windows have {x_raw.shape[-1]} samples, model expects {self.win_len} "
                             f"({self.sample_rate:g} Hz x {self.epoch_seconds:g} s): pass source_sample_rate "
                             f"to resample, or preprocess at {self.sample_rate:g} Hz")
        x, _ = robust_normalize(x_raw, self.enc_meta["sample_rate"], self.enc_meta.get("lowpass_hz"),
                                self.enc_meta.get("clip", 4.0))
        return x

    def embed(self, x_norm: np.ndarray, batch_size: int = 2048) -> EmbeddingStore:
        """Encode normalised windows into an ``EmbeddingStore`` (one unlabeled recording)."""
        import torch

        out = []
        with torch.inference_mode():
            for s in range(0, len(x_norm), batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(x_norm[s:s + batch_size], dtype=np.float32)).to(self.device)
                out.append(self.encoder(xb).float().cpu().numpy())
        emb = np.concatenate(out)
        n = len(emb)
        return EmbeddingStore([emb], np.full(n, IGNORE), [RecordingSpan("input", "input", 0, n)],
                              self.enc_meta["digest"])

    def score_windows(self, x_raw: np.ndarray, normalize: bool = True, smooth_sigma: float = 0.0,
                      heuristics: str = "none", source_sample_rate: float | None = None) -> ScoringResult:
        """Score the windows of one channel of one continuous recording.

        Args:
            x_raw: Windows [N, T] or [N, 1, T].
            normalize: Run ``preprocess`` first; set False only for already normalised input.
            smooth_sigma: Gaussian smoothing of probabilities, in windows.
            heuristics: "none", "lite" or "full", see ``inference.postprocess``.
            source_sample_rate: See ``preprocess``.

        Returns:
            A ``ScoringResult`` with one state per input window.
        """
        if normalize:
            x = self.preprocess(x_raw, source_sample_rate)
        else:
            x = np.asarray(x_raw, dtype=np.float32)
            if x.ndim == 2:
                x = x[:, None, :]
        store = self.embed(x)
        probs = self.head.predict_probs(store, self.device)
        classes = predict_classes(probs, [(0, len(probs))], self.scheme, smooth_sigma, heuristics)
        from scorer.inference.postprocess import smooth_probs

        probs_out = smooth_probs(probs, smooth_sigma)
        return ScoringResult(classes, self.scheme.to_raw(classes), probs_out, probs_out.max(1),
                             self.scheme.class_names)

    def score_folder(self, folder: str | Path, channel: int = 0, max_len_mismatch: int = 1, layout: str = "auto",
                     **kwargs) -> ScoringResult:
        """Score one recording folder (X chunks; label files are ignored).

        Args:
            folder: Folder of one recording.
            channel: Channel to score.
            max_len_mismatch: Passed to ``load_recording``.
            layout: Chunk axis order, see ``to_windows_channels_time``.
            **kwargs: Passed to ``score_windows`` (smoothing, heuristics, source rate).

        Raises:
            RuntimeError: If the folder holds more than one recording.
        """
        recs = discover_recordings(folder)
        if len(recs) != 1:
            raise RuntimeError(f"{folder} contains {len(recs)} recordings; score them one at a time")
        x, _ = load_recording(recs[0], [channel], max_len_mismatch, layout=layout)
        return self.score_windows(x, **kwargs)


def save_states(result: ScoringResult, path: str | Path, with_confidence: bool = True) -> Path:
    """Pickle the GUI codes to ``path`` (same content as the old ``*_states.pkl``).

    With ``with_confidence``, the confidence array goes to ``<stem>_confidence.pkl``
    next to it. Returns ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result.gui_codes, f)
    if with_confidence:
        with open(path.with_name(path.stem + "_confidence.pkl"), "wb") as f:
            pickle.dump(result.confidence, f)
    return path


def _legacy_folder_key(p: Path) -> str:
    """Sort key of the old batch scorer: a fixed slice of the path that holds the folder's timestamp."""
    return str(p)[-44:-22]


def score_from_table(meta_csv: str | Path, scorer: SleepScorer, out_dir: str | Path, tag: str = "",
                     overwrite: bool = False, folder_key=_legacy_folder_key, **score_kwargs) -> list[dict]:
    """Score every project listed in a metadata CSV (port of ``score_multiple_signals``).

    CSV columns: "proj_dir", "channel_selected" ("f" = channel 0, "p" = channel 1) and an
    optional "state_file" with an earlier scoring; if present, Cohen's kappa against it is
    reported. In each project the ``processed/`` folder with the largest ``folder_key`` is
    scored.

    Args:
        meta_csv: The CSV.
        scorer: Loaded scorer.
        out_dir: Folder for ``<project>[_tag]_states.pkl``.
        tag: Optional suffix for output names.
        overwrite: Re-score projects whose output exists.
        folder_key: Sort key choosing the folder to score.
        **score_kwargs: Passed to ``score_folder``.

    Returns:
        One dict per scored project (windows, output path, optional kappa).
    """
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    out_dir = Path(out_dir)
    rows = []
    df = pd.read_csv(meta_csv)
    for _, row in df.iterrows():
        proj = Path(row["proj_dir"])
        ch = {"f": 0, "p": 1}.get(str(row["channel_selected"]).strip().lower())
        if ch is None:
            log.error("%s: cannot parse channel_selected=%r", proj, row["channel_selected"])
            continue
        out_path = out_dir / f"{proj.name}{('_' + tag) if tag else ''}_states.pkl"
        if out_path.exists() and not overwrite:
            log.info("%s exists, skipping", out_path)
            continue
        processed = sorted((p for p in (proj / "processed").iterdir() if p.is_dir()), key=folder_key)
        if not processed:
            log.error("%s: no processed folders", proj)
            continue
        result = scorer.score_folder(processed[-1], channel=ch, **score_kwargs)
        save_states(result, out_path)
        entry = {"recording": proj.name, "windows": len(result.classes), "out": str(out_path)}
        old = row.get("state_file")
        if isinstance(old, str) and Path(old).is_file():
            with open(old, "rb") as f:
                old_states = np.asarray(pickle.load(f))
            n = min(len(old_states), len(result.gui_codes))
            entry["kappa_vs_previous"] = float(cohen_kappa_score(old_states[:n], result.gui_codes[:n]))
        rows.append(entry)
        log.info("scored %s", entry)
    kappas = [r["kappa_vs_previous"] for r in rows if "kappa_vs_previous" in r]
    if kappas:
        log.info("mean kappa vs previous scoring: %.3f over %d recordings", float(np.mean(kappas)), len(kappas))
    return rows
