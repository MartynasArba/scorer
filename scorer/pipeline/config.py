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
    """``[run]``: where results go and how the run behaves.

    Attributes:
        name: Run folder name under ``runs_dir``.
        mode: "tuned" aligns the encoder to the target's unlabeled inputs (labels never
            read); "zeroshot" skips alignment. Both modes share prepare/simclr/supcon in one
            run folder; embed, heads and evaluate are stored per mode.
        runs_dir: Root folder of all runs.
        seed: Seed for every random choice.
        device: "auto", "cuda" or "cpu".
        label_scheme: "3state" or "4state".
        num_workers: DataLoader worker processes.
        unlabeled_as_wake: Treat unscored windows as Wake (old ``merge_unknown_into_wake``).
    """
    name: str = "run"
    # tuned    = run domain alignment on the target's unlabeled inputs (its labels are never read)
    # zeroshot = skip alignment; heads are trained on the SupCon/SimCLR encoder. Both modes share
    #            prepare / simclr / supcon in one run folder; embed, heads and evaluate are kept per mode.
    mode: str = "tuned"
    runs_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    label_scheme: str = "3state"
    num_workers: int = 2
    unlabeled_as_wake: bool = False  # port of merge_unknown_into_wake


@dataclass
class SignalCfg:
    """``[signal]``: how windows are read and normalised (part of the cache identity).

    Attributes:
        sample_rate: Sample rate of the datasets in Hz.
        win_len: Samples per window (512 = 4 s at 128 Hz).
        channels: Channel indices to keep.
        lowpass_hz: Low-pass cut-off, or None. The old code effectively used 25.088 Hz at
            128 Hz (see ``data.signal.legacy_equivalent_lowpass``).
        clip: Clip limit in robust SDs.
        cache_dtype: Storage dtype of prepared windows.
        max_len_mismatch: Largest X/y length difference trimmed without error.
    """
    sample_rate: float = 128.0
    win_len: int = 512
    channels: list[int] = field(default_factory=lambda: [0])
    lowpass_hz: float | None = 49.0  # old code effectively used 25.088 at 128 Hz (see data/signal.py)
    clip: float = 4.0
    cache_dtype: str = "float16"
    max_len_mismatch: int = 1


@dataclass
class CacheCfg:
    """``[cache]``: where prepared data is stored.

    Attributes:
        root: Cache root folder (relative paths are relative to the working directory).
    """
    root: str = "cache"


@dataclass
class ContrastiveCfg:
    """``[simclr]`` and ``[supcon]``: contrastive pretraining of the encoder.

    Attributes:
        enabled: Run this stage.
        datasets: Datasets to train on.
        init: Starting weights: "" (random), "simclr", "supcon", "adversarial" (output of
            this run) or a checkpoint path.
        use: Which checkpoint of the init stage: "final" or "best".
        epochs, batch_size, lr, temperature: Training settings.
        balance: "none", "undersample" or "oversample" (see ``epoch_selection``).
        drop_unlabeled: Skip windows without labels.
        loader_augment: Apply the old loader's CPU augmentations as well.
        buffer_windows: Shuffle-buffer size per worker, in windows.
        patience, min_delta: Early stopping on the epoch loss.
        snapshot_every: Save a numbered checkpoint every N epochs (0 = never).
        max_batches_per_epoch: Cap for quick tests (0 = full epochs).
    """
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
    """``[adversarial]``: domain alignment of the encoder (tuned mode only).

    Attributes:
        enabled: Run this stage (ignored in zero-shot mode).
        source: Labeled source datasets.
        target: Target datasets; only their inputs are used.
        init: Starting encoder ("supcon", "simclr" or a path).
        use: "final" or "best" checkpoint of ``init``.
        epochs, batch_size: Training settings.
        lr_encoder, lr_freq_flatten, lr_heads: Learning rates per parameter group.
        warmup_fraction: Share of steps before gradient reversal starts.
        buffer_windows: Shuffle-buffer size per worker.
        target_drop_unlabeled: Old behaviour that used target labels to drop windows;
            refused if a target is also evaluated.
        max_batches_per_epoch: Cap for quick tests (0 = full epochs).
    """
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
    """``[embed]``: which encoder embeds which datasets.

    Attributes:
        encoder: "auto" (adversarial -> supcon -> simclr output of this run, per mode) or
            a checkpoint path.
        datasets: Datasets to embed; empty means every dataset used by enabled heads and
            ``evaluate``.
        batch_size: Windows per forward pass.
        dtype: Storage dtype of embeddings.
        in_memory: Load embeddings into RAM for the heads (False memory-maps them).
        legacy_sample_rate: Sample rate of an old encoder without a Hann buffer (0 = infer).
        legacy_lowpass_hz: Low-pass of an old encoder (-1 = the old code's effective cut-off).
    """
    encoder: str = "auto"  # auto = adversarial -> supcon -> simclr output of this run, or a path
    datasets: list[str] = field(default_factory=list)  # empty = every dataset used by enabled heads / evaluate
    batch_size: int = 2048
    dtype: str = "float16"
    in_memory: bool = True  # False = memory-map embeddings (very large datasets)
    legacy_sample_rate: float = 0.0  # only for old raw state dicts without a Hann buffer
    legacy_lowpass_hz: float = -1.0  # -1 = assume the old code's effective cut-off


@dataclass
class ContextMLPCfg:
    """``[context_mlp]``: residual MLP on windows t-k .. t+k.

    Attributes:
        enabled: Train this head.
        train, val: Datasets for training and model selection.
        k: Context half-width.
        epochs, batch_size, lr, weight_decay, label_smoothing: Training settings.
        hidden_dim, num_blocks, dropout: Architecture.
        class_weighting: "inv_sqrt", "inv", "none" or a list of weights.
        max_train_windows: Stratified cap (0 = all).
    """
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
    """``[context_rf]``: random forest on windows t-k .. t+k (CPU).

    Attributes:
        enabled: Train this head.
        train: Training datasets.
        k: Context half-width.
        n_estimators, max_depth, max_features, class_weight: Forest settings.
        max_train_windows: Stratified cap; bounds RAM, since features are held in memory.
        n_jobs: CPU cores (-1 = all).
    """
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
    """``[gru]``: bidirectional GRU over embedding sequences.

    Attributes:
        enabled: Train this head.
        train, val: Datasets for training and model selection.
        seq_len, stride: Sequence length and step, in windows.
        epochs, batch_size, lr, weight_decay: Training settings.
        hidden_dim, num_layers: GRU size.
        focal_gamma: Focal-loss gamma.
        max_unlabeled_fraction: Skip sequences with more unlabeled windows than this.
        grad_clip: Gradient-norm limit.
        patience: Early-stopping patience in epochs.
    """
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
    """``[linear_probe]``: one linear layer on single-window embeddings.

    Attributes:
        enabled: Train this head.
        train, val: Datasets for training and model selection.
        epochs, batch_size, lr, weight_decay: Training settings.
        class_weighting: "inv_sqrt", "inv", "none" or a list of weights.
    """
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
    """``[evaluate]``: scoring the evaluation datasets with the trained heads.

    Attributes:
        datasets: Datasets to evaluate.
        heads: Heads to evaluate (each must be trained in this mode).
        smooth_sigma: Probability smoothing in windows (0 = off).
        heuristics: "none", "lite" or "full".
        hypnogram_windows: Windows shown per hypnogram figure.
        hypnograms_per_dataset: Recordings plotted per dataset.
        pca: Save an embedding PCA plot per dataset.
    """
    datasets: list[str] = field(default_factory=list)
    heads: list[str] = field(default_factory=lambda: ["context_mlp", "context_rf"])
    smooth_sigma: float = 0.0
    heuristics: str = "lite"  # none | lite | full
    hypnogram_windows: int = 900
    hypnograms_per_dataset: int = 3
    pca: bool = True


@dataclass
class PipelineConfig:
    """The whole TOML config: one attribute per section, plus the ``[datasets]`` table.

    Attributes:
        datasets: Dataset name -> folder of windowed recordings.
        source_path: The TOML file it was loaded from (not part of ``digest``).
    """
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
        """Return the config as nested dicts (for logs and the manifest)."""
        return dataclasses.asdict(self)

    def digest(self) -> str:
        """Return a 10-character hash of the settings, recorded with every stage's outputs."""
        d = self.to_dict()
        d.pop("source_path", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def dataset_path(self, name: str) -> Path:
        """Return the folder of dataset ``name``.

        Raises:
            KeyError: If ``name`` is not defined in ``[datasets]``.
        """
        if name not in self.datasets:
            raise KeyError(f"dataset {name!r} is not defined in [datasets] (known: {sorted(self.datasets)})")
        return Path(self.datasets[name])

    def warnings(self) -> list[str]:
        """List settings that are allowed but probably unintended.

        Examples: tuned mode aligning to a dataset that is also evaluated (then the numbers
        are semi-zero-shot), overlapping source and target, or heads trained on evaluated
        datasets.
        """
        out = []
        reported = set(self.evaluate.datasets) | set(self.context_mlp.val) | set(self.gru.val)
        if self.adversarial.enabled and self.run.mode == "tuned":
            leak = reported & set(self.adversarial.target)
            if leak:
                out.append(f"tuned mode: alignment sees the unlabeled inputs of {sorted(leak)}, which are also "
                           f"evaluated -> report these as 'tuned' (semi-zero-shot), not zero-shot; "
                           f"use run.mode=zeroshot for zero-shot numbers")
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
    """Build dataclass ``cls`` from a dict, recursing into nested sections.

    Raises:
        KeyError: For any key the dataclass does not define (typos fail loudly).
    """
    if not dataclasses.is_dataclass(cls):
        return data
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown key(s) {sorted(unknown)} in [{where}]; valid: {sorted(names)}")
    kwargs = {}
    for k, v in data.items():
        f = names[k]
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
    """Set ``d[a][b][c] = value`` for ``dotted = 'a.b.c'``, creating missing levels."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_override(expr: str) -> tuple[str, Any]:
    """Split a ``section.key=value`` override; the value is parsed as TOML (5, 1e-3, "x",
    [1, 2], true), falling back to the raw string.

    Raises:
        ValueError: If there is no "=".
    """
    if "=" not in expr:
        raise ValueError(f"override must look like section.key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return key.strip(), value


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    """Load and check a pipeline config.

    Args:
        path: TOML file.
        overrides: ``section.key=value`` strings applied before validation.

    Returns:
        The typed config.

    Raises:
        KeyError: For unknown keys or undefined dataset names.
        ValueError: For an invalid mode or heuristics value, or a setting that would read
            labels of an evaluated target dataset.
    """
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
    if cfg.run.mode not in ("tuned", "zeroshot"):
        raise ValueError("run.mode must be 'tuned' or 'zeroshot'")
    reported = set(cfg.evaluate.datasets) | set(cfg.context_mlp.val) | set(cfg.gru.val) | set(cfg.linear_probe.val)
    if cfg.adversarial.target_drop_unlabeled and reported & set(cfg.adversarial.target):
        raise ValueError("adversarial.target_drop_unlabeled=true reads the labels of an evaluated target dataset; "
                         "keep it false so test labels are never seen")
    if cfg.evaluate.heuristics not in ("none", "lite", "full"):
        raise ValueError("evaluate.heuristics must be none | lite | full")
    return cfg
