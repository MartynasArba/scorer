"""
Typed pipeline configuration loaded from a TOML file (stdlib tomllib, Python >= 3.11).

Every section is a dataclass with defaults; unknown keys raise immediately (a typo in the
config no longer silently falls back to a default buried in some function).
Command-line overrides: --set section.key=value (value parsed as TOML, e.g. 5, 1e-3, "x", [1, 2]).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunCfg:
    name: str = "run"
    runs_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    label_scheme: str = "3state"
    num_workers: int = 2
    unlabeled_as_wake: bool = False  # port of merge_unknown_into_wake


@dataclass
class SignalCfg:
    sample_rate: float = 128.0
    win_len: int = 512
    channels: list[int] = field(default_factory=lambda: [0])
    lowpass_hz: float | None = 49.0  # old code effectively used 25.088 at 128 Hz (see data/signal.py)
    clip: float = 4.0
    cache_dtype: str = "float16"
    max_len_mismatch: int = 1


@dataclass
class CacheCfg:
    root: str = "cache"


@dataclass
class ContrastiveCfg:
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    init: str = ""  # "" = random | "simclr" | "supcon" | "adversarial" (this run) | path to a checkpoint
    use: str = "final"  # which checkpoint of the init stage: final | best
    epochs: int = 200
    batch_size: int = 2048
    lr: float = 5e-5
    temperature: float = 0.15
    balance: str = "none"
    drop_unlabeled: bool = True
    loader_augment: bool = False
    buffer_windows: int = 100_000
    patience: int = 5
    min_delta: float = 1e-3
    snapshot_every: int = 20
    max_batches_per_epoch: int = 0  # 0 = full epochs (set small for smoke tests)


@dataclass
class AdversarialCfg:
    enabled: bool = True
    source: list[str] = field(default_factory=list)
    target: list[str] = field(default_factory=list)
    init: str = "supcon"
    use: str = "final"
    epochs: int = 10
    batch_size: int = 1024
    lr_encoder: float = 2e-6
    lr_freq_flatten: float = 1e-5
    lr_heads: float = 1e-3
    warmup_fraction: float = 0.2
    buffer_windows: int = 100_000
    target_drop_unlabeled: bool = False  # True = old behaviour (used the target's labels to drop windows)
    max_batches_per_epoch: int = 0


@dataclass
class EmbedCfg:
    encoder: str = "auto"  # auto = adversarial -> supcon -> simclr output of this run, or a path
    datasets: list[str] = field(default_factory=list)  # empty = every dataset used by enabled heads / evaluate
    batch_size: int = 2048
    dtype: str = "float16"
    in_memory: bool = True  # False = memory-map embeddings (very large datasets)
    legacy_sample_rate: float = 0.0  # only for old raw state dicts without a Hann buffer
    legacy_lowpass_hz: float = -1.0  # -1 = assume the old code's effective cut-off


@dataclass
class ContextMLPCfg:
    enabled: bool = True
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    k: int = 3
    epochs: int = 20
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    hidden_dim: int = 1024
    num_blocks: int = 2
    dropout: float = 0.2
    class_weighting: Any = "inv_sqrt"  # "inv_sqrt" | "inv" | "none" | [w0, w1, w2]
    max_train_windows: int = 0


@dataclass
class ContextRFCfg:
    enabled: bool = True
    train: list[str] = field(default_factory=list)
    k: int = 1
    n_estimators: int = 150
    max_depth: int = 20
    max_features: float = 0.15
    class_weight: str = "balanced_subsample"
    max_train_windows: int = 300_000
    n_jobs: int = -1


@dataclass
class GRUCfg:
    enabled: bool = False
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    seq_len: int = 20
    stride: int = 1
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_layers: int = 2
    focal_gamma: float = 1.5
    max_unlabeled_fraction: float = 0.5
    grad_clip: float = 0.1
    patience: int = 3


@dataclass
class LinearProbeCfg:
    enabled: bool = False
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 5e-5
    class_weighting: Any = "inv_sqrt"


@dataclass
class EvaluateCfg:
    datasets: list[str] = field(default_factory=list)
    heads: list[str] = field(default_factory=lambda: ["context_mlp", "context_rf"])
    smooth_sigma: float = 0.0
    heuristics: str = "lite"  # none | lite | full
    hypnogram_windows: int = 900
    hypnograms_per_dataset: int = 3
    pca: bool = True


@dataclass
class PipelineConfig:
    run: RunCfg = field(default_factory=RunCfg)
    signal: SignalCfg = field(default_factory=SignalCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)
    datasets: dict[str, str] = field(default_factory=dict)
    simclr: ContrastiveCfg = field(default_factory=lambda: ContrastiveCfg(loader_augment=True))
    supcon: ContrastiveCfg = field(default_factory=lambda: ContrastiveCfg(init="simclr", lr=1e-5, temperature=0.2,
                                                                          balance="undersample"))
    adversarial: AdversarialCfg = field(default_factory=AdversarialCfg)
    embed: EmbedCfg = field(default_factory=EmbedCfg)
    context_mlp: ContextMLPCfg = field(default_factory=ContextMLPCfg)
    context_rf: ContextRFCfg = field(default_factory=ContextRFCfg)
    gru: GRUCfg = field(default_factory=GRUCfg)
    linear_probe: LinearProbeCfg = field(default_factory=LinearProbeCfg)
    evaluate: EvaluateCfg = field(default_factory=EvaluateCfg)
    source_path: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        d = self.to_dict()
        d.pop("source_path", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def dataset_path(self, name: str) -> Path:
        if name not in self.datasets:
            raise KeyError(f"dataset {name!r} is not defined in [datasets] (known: {sorted(self.datasets)})")
        return Path(self.datasets[name])

    def warnings(self) -> list[str]:
        """configuration smells worth shouting about (not errors)"""
        out = []
        reported = set(self.evaluate.datasets) | set(self.context_mlp.val) | set(self.gru.val)
        if self.adversarial.enabled:
            leak = reported & set(self.adversarial.target)
            if leak:
                out.append(f"adversarial target domain {sorted(leak)} is also used for validation/evaluation: "
                           f"those scores are transductive, not zero-shot")
            overlap = set(self.adversarial.source) & set(self.adversarial.target)
            if overlap:
                out.append(f"adversarial source and target overlap: {sorted(overlap)}")
        for head in ("context_mlp", "gru", "linear_probe"):
            h = getattr(self, head)
            if h.enabled and set(h.train) & set(h.val):
                out.append(f"{head}: train and val datasets overlap: {sorted(set(h.train) & set(h.val))}")
        for head in ("context_mlp", "context_rf", "gru", "linear_probe"):
            h = getattr(self, head)
            if h.enabled and set(h.train) & set(self.evaluate.datasets):
                out.append(f"{head}: training datasets are also evaluated: {sorted(set(h.train) & set(self.evaluate.datasets))}")
        return out


def _build(cls, data: dict, where: str):
    if not dataclasses.is_dataclass(cls):
        return data
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown key(s) {sorted(unknown)} in [{where}]; valid: {sorted(names)}")
    kwargs = {}
    for k, v in data.items():
        f = names[k]
        ftype = f.type if not isinstance(f.type, str) else None
        default_obj = None
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default_obj = f.default_factory()  # type: ignore[misc]
        if dataclasses.is_dataclass(default_obj) and isinstance(v, dict):
            base = dataclasses.asdict(default_obj)
            base.update(v)
            kwargs[k] = _build(type(default_obj), base, f"{where}.{k}" if where else k)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_override(expr: str) -> tuple[str, Any]:
    if "=" not in expr:
        raise ValueError(f"override must look like section.key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return key.strip(), value


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for expr in overrides or []:
        k, v = parse_override(expr)
        _set_path(data, k, v)
    cfg = _build(PipelineConfig, {k: v for k, v in data.items()}, "")
    cfg.source_path = str(path)
    for name in ("simclr", "supcon", "adversarial", "embed", "context_mlp", "context_rf", "gru", "linear_probe",
                 "evaluate"):
        section = getattr(cfg, name)
        for attr in ("datasets", "train", "val", "source", "target"):
            for ds in getattr(section, attr, []) or []:
                cfg.dataset_path(ds)  # raises on undefined names
    if cfg.evaluate.heuristics not in ("none", "lite", "full"):
        raise ValueError("evaluate.heuristics must be none | lite | full")
    return cfg
