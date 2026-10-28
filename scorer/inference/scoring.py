"""
Scoring API (clean break from scoring.score_signal). Used by the GUI, batch scoring AND the
pipeline's evaluate stage, so validation numbers come from exactly the code that scores data.

    scorer = SleepScorer("runs/128hz/adversarial/encoder_final.pt", "runs/128hz/heads/context_mlp.pt")
    result = scorer.score_folder(r"D:\\rec\\processed\\windowed_x", channel=0, heuristics="lite")
    result.gui_codes      # int array, 1=Wake 2=NREM 4=REM (raw codes, see data/labels.py)
    result.confidence     # max class probability per window
    save_states(result, r"D:\\states\\rec_states.pkl")

Preprocessing (sample rate, window length, low-pass, clip) is read from the encoder checkpoint,
so GUI scoring can no longer drift from what the model was trained on. Windows must already be
at the model's sample rate (e.g. 512 samples = 4 s at 128 Hz) - mismatches raise.
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
from scorer.data.signal import robust_normalize
from scorer.inference.postprocess import predict_classes
from scorer.training.embeddings import EmbeddingStore, RecordingSpan

log = logging.getLogger(__name__)

HEAD_KINDS = ("context_mlp", "linear_probe", "gru", "context_rf")


@dataclass
class Head:
    kind: str
    model: Any
    meta: dict
    path: str

    def predict_probs(self, store: EmbeddingStore, device) -> np.ndarray:
        """probabilities [N, n_classes] for every window in the store"""
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
    classes: np.ndarray       # class index per window (scheme order)
    gui_codes: np.ndarray     # raw codes 1/2/(3)/4 for the GUI
    probs: np.ndarray         # [N, n_classes] after optional smoothing
    confidence: np.ndarray    # max probability per window
    class_names: tuple[str, ...]


class SleepScorer:
    def __init__(self, encoder_path: str | Path, head_path: str | Path, device: str = "auto",
                 legacy_encoder_meta: dict | None = None, legacy_head_meta: dict | None = None,
                 allow_encoder_mismatch: bool = False):
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
    def preprocess(self, x_raw: np.ndarray) -> np.ndarray:
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if x_raw.ndim == 2:
            x_raw = x_raw[:, None, :]
        if x_raw.shape[-1] != int(self.enc_meta["win_len"]):
            raise ValueError(f"windows have {x_raw.shape[-1]} samples, model expects {self.enc_meta['win_len']} "
                             f"({self.enc_meta['sample_rate']} Hz x 4 s): resample the recording first")
        x, _ = robust_normalize(x_raw, self.enc_meta["sample_rate"], self.enc_meta.get("lowpass_hz"),
                                self.enc_meta.get("clip", 4.0))
        return x

    def embed(self, x_norm: np.ndarray, batch_size: int = 2048) -> EmbeddingStore:
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
                      heuristics: str = "none") -> ScoringResult:
        """x_raw: [N, T] or [N, 1, T] windows of ONE channel of ONE continuous recording"""
        x = self.preprocess(x_raw) if normalize else np.asarray(x_raw, dtype=np.float32)
        store = self.embed(x)
        probs = self.head.predict_probs(store, self.device)
        classes = predict_classes(probs, [(0, len(probs))], self.scheme, smooth_sigma, heuristics)
        from scorer.inference.postprocess import smooth_probs

        probs_out = smooth_probs(probs, smooth_sigma)
        return ScoringResult(classes, self.scheme.to_raw(classes), probs_out, probs_out.max(1),
                             self.scheme.class_names)

    def score_folder(self, folder: str | Path, channel: int = 0, max_len_mismatch: int = 1, **kwargs) -> ScoringResult:
        """folder = one recording's windowed data (X_*.pt chunks, labels optional)"""
        recs = discover_recordings(folder)
        if len(recs) != 1:
            raise RuntimeError(f"{folder} contains {len(recs)} recordings; score them one at a time")
        x, _ = load_recording(recs[0], [channel], max_len_mismatch)
        return self.score_windows(x, **kwargs)


def save_states(result: ScoringResult, path: str | Path, with_confidence: bool = True) -> Path:
    """pickles the GUI codes (same content type as the old *_states.pkl); confidence alongside"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result.gui_codes, f)
    if with_confidence:
        with open(path.with_name(path.stem + "_confidence.pkl"), "wb") as f:
            pickle.dump(result.confidence, f)
    return path


def _legacy_folder_key(p: Path) -> str:
    """old selection rule: max over a fixed slice of the path string (the timestamp in the folder name)"""
    return str(p)[-44:-22]


def score_from_table(meta_csv: str | Path, scorer: SleepScorer, out_dir: str | Path, tag: str = "",
                     overwrite: bool = False, folder_key=_legacy_folder_key, **score_kwargs) -> list[dict]:
    """
    port of scoring.score_multiple_signals. CSV columns: proj_dir, channel_selected ('f' -> 0, 'p' -> 1),
    optional state_file (previous scoring, for Cohen's kappa). Scores the folder in <proj_dir>/processed
    with the largest folder_key (default: the old timestamp slice of the path).
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


def compute_ylims(x: np.ndarray, mode: str = "standard", n_ephys: int = 2, seed: int = 0) -> list[tuple[float, float]]:
    """port of SleepSignals.compute_ylims for GUI display: (center, spread) per channel. x: [N, C, T]"""
    rng = np.random.default_rng(seed)
    n, c, _ = x.shape
    idx = rng.integers(0, n, size=min(1000, n))
    out = []
    for ch in range(c):
        if mode == "standard":
            out.append((0.0, 0.2) if ch < n_ephys else (0.5, 1.0))
        elif mode == "infer" or (mode == "infer_ephys" and ch < n_ephys):
            vals = np.asarray(x[idx, ch, :]).reshape(-1)
            lo, med, hi = np.quantile(vals, [0.01, 0.5, 0.99])
            out.append((float(med), float(max(hi - lo, 1e-9))))
        else:
            out.append((0.5, 1.0))
    return out
