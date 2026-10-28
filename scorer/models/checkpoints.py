"""
Checkpoints that describe themselves.

New checkpoints are dicts:
    {"format": "scorer-ckpt-v1", "kind": "encoder" | "context_mlp" | "gru" | "linear_probe" | ...,
     "state_dict": {...}, "meta": {...}}
meta always carries what is needed to use the weights correctly: architecture, sample_rate,
win_len, low-pass / clip used in preprocessing, label scheme + class names and, for heads,
the encoder they were trained on (path + content hash).

Loading is STRICT: missing or unexpected keys raise (the old init_encoder used strict=False
and would silently run a randomly initialised encoder if the wrong file was passed).
Old raw state dicts still load; their meta is inferred (window length from the Hann buffer)
and the preprocessing defaults to what the old code actually did (see legacy_equivalent_lowpass).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from scorer.data.signal import legacy_equivalent_lowpass
from scorer.models.encoders import SCDSSleepCNN
from scorer.models.legacy_cnns import DualStreamSleepCNN, EphysSleepCNN, SleepCNN

log = logging.getLogger(__name__)

FORMAT = "scorer-ckpt-v1"
RF_FORMAT = "scorer-rf-v1"

ENCODERS = {
    "SCDSSleepCNN": SCDSSleepCNN,
    "SleepCNN": SleepCNN,
    "EphysSleepCNN": EphysSleepCNN,
    "DualStreamSleepCNN": DualStreamSleepCNN,
}
# buffers that are deterministic and may be absent from old checkpoints
_OPTIONAL_KEYS = ("hann_window",)


def file_digest(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def save_checkpoint(path: str | Path, model: nn.Module | dict, kind: str, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict() if isinstance(model, nn.Module) else model
    state = {k: v.detach().cpu() for k, v in state.items()}
    meta = {**meta, "saved": now()}
    torch.save({"format": FORMAT, "kind": kind, "state_dict": state, "meta": meta}, path)
    return path


def read_checkpoint(path: str | Path) -> tuple[dict, dict, str | None, bool]:
    """returns (state_dict, meta, kind, is_legacy)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # whole pickled nn.Module (old linear_train_eval saved those)
        raise RuntimeError(
            f"{path} is not a plain state dict (probably a pickled nn.Module from torch.save(model)). "
            f"Convert it once with: torch.save(torch.load(p, weights_only=False).state_dict(), p_new)"
        ) from e
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        return obj["state_dict"], dict(obj["meta"]), obj.get("kind"), False
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj, {}, None, True
    raise RuntimeError(f"{path}: unrecognised checkpoint content ({type(obj)})")


def strict_load(model: nn.Module, state: dict, ignore_unexpected: tuple[str, ...] = (), source: str = "") -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(_OPTIONAL_KEYS)]
    unexpected = [k for k in unexpected if not k.startswith(ignore_unexpected)]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {source} does not match {type(model).__name__}: "
            f"{len(missing)} missing (e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]})"
        )


def build_encoder(meta: dict) -> nn.Module:
    arch = meta.get("arch", "SCDSSleepCNN")
    n_classes = len(meta.get("class_names", ("Wake", "NREM", "REM")))
    if arch == "SCDSSleepCNN":
        return SCDSSleepCNN(num_classes=n_classes, sampling_rate=meta["sample_rate"], win_len=meta["win_len"])
    if arch == "EphysSleepCNN":
        return EphysSleepCNN(num_classes=n_classes, win_len=meta["win_len"])
    if arch == "SleepCNN":
        return SleepCNN(num_classes=n_classes, in_channels=meta.get("in_channels", 7))
    if arch == "DualStreamSleepCNN":
        return DualStreamSleepCNN(num_classes=n_classes)
    raise KeyError(f"unknown encoder architecture {arch!r}; known: {sorted(ENCODERS)}")


def encoder_meta(cfg_signal: Any, scheme, arch: str = "SCDSSleepCNN", **extra) -> dict:
    """meta for a freshly built encoder, from the [signal] config section"""
    return {
        "arch": arch,
        "sample_rate": float(cfg_signal.sample_rate),
        "win_len": int(cfg_signal.win_len),
        "channels": list(cfg_signal.channels),
        "lowpass_hz": cfg_signal.lowpass_hz,
        "clip": float(cfg_signal.clip),
        "label_scheme": scheme.name,
        "class_names": list(scheme.class_names),
        **extra,
    }


def load_encoder(path: str | Path, device: str | torch.device = "cpu", expect: dict | None = None,
                 legacy_meta: dict | None = None) -> tuple[nn.Module, dict]:
    """
    load an encoder checkpoint (new or old format) in eval mode.
    expect: e.g. {"sample_rate": 128, "win_len": 512} -> raise if the checkpoint disagrees.
    legacy_meta: facts about OLD raw state dicts that cannot be inferred (arch, sample_rate, lowpass_hz).
    """
    state, meta, kind, legacy = read_checkpoint(path)
    if kind not in (None, "encoder"):
        raise RuntimeError(f"{path} is a {kind!r} checkpoint, not an encoder")
    if legacy:
        if any(k.startswith("encoder.") for k in state):
            # wrapper checkpoints: SupConSleepCNN (encoder.* + projection_head.*) or
            # ContextAwareSleepScorer (encoder.* + rnn.* + classifier.*) -> keep the encoder part
            state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        meta = {"arch": "SCDSSleepCNN", "label_scheme": "3state", "class_names": ["Wake", "NREM", "REM"],
                "legacy_checkpoint": True}
        if "hann_window" in state:
            meta["win_len"] = int(state["hann_window"].shape[0])
            meta["sample_rate"] = meta["win_len"] / 4.0
        meta.update(legacy_meta or {})
        if "sample_rate" not in meta or "win_len" not in meta:
            raise RuntimeError(f"{path}: old checkpoint without Hann buffer; pass legacy_meta with sample_rate and win_len")
        meta.setdefault("lowpass_hz", legacy_equivalent_lowpass(meta["sample_rate"]))
        meta.setdefault("clip", 4.0)
        log.warning("%s: old-format checkpoint, assuming %s", path,
                    {k: meta[k] for k in ("arch", "sample_rate", "win_len", "lowpass_hz")})
    for key, value in (expect or {}).items():
        if key in meta and meta[key] is not None and value is not None and float(meta[key]) != float(value):
            raise RuntimeError(f"{path}: checkpoint {key}={meta[key]} but expected {value}")
    model = build_encoder(meta)
    strict_load(model, state, ignore_unexpected=("projection_head.",), source=str(path))
    meta["path"] = str(path)
    meta["digest"] = file_digest(path)
    return model.to(device).eval(), meta


def save_rf_bundle(path: str | Path, model, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"format": RF_FORMAT, "kind": "context_rf", "model": model, "meta": {**meta, "saved": now()}}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_rf_bundle(path: str | Path, legacy_meta: dict | None = None) -> tuple[Any, dict]:
    """pickle files are code: only load bundles you created yourself"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and obj.get("format") == RF_FORMAT:
        return obj["model"], dict(obj["meta"])
    log.warning("%s: old-format RF pickle; assuming context k=1 and the 3-state scheme", path)
    return obj, {"context_k": 1, "label_scheme": "3state", "class_names": ["Wake", "NREM", "REM"],
                 "classes_": [int(c) for c in getattr(obj, "classes_", [0, 1, 2])], **(legacy_meta or {})}
