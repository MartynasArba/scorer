"""
Pipeline stages. Each stage reads its inputs from the run's manifest (or explicit config paths),
writes its outputs under <runs_dir>/<run name>/<stage>/ and records them in manifest.json.
Stages can therefore be run one at a time, re-run, or resumed in a later session.

    inspect      label / recording overview of every configured dataset (no training)
    prepare      normalise recordings once -> memory-mapped cache
    simclr       unsupervised contrastive pretraining
    supcon       supervised contrastive pretraining
    adversarial  domain-adversarial alignment
    embed        encoder -> per-window embeddings for every dataset the heads need
    context_rf | context_mlp | gru | linear_probe   heads on embeddings
    evaluate     metrics, confusion matrices, hypnograms, per-recording tables
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from scorer.data.cache import DatasetCache, PrepParams, prepare_dataset
from scorer.data.labels import LabelScheme, get_scheme, raw_counts
from scorer.data.recordings import discover_recordings, scan_labels
from scorer.pipeline.config import PipelineConfig

log = logging.getLogger("scorer.pipeline")

STAGE_ORDER = ["inspect", "prepare", "simclr", "supcon", "adversarial", "embed", "context_rf", "context_mlp", "gru",
               "linear_probe", "evaluate"]
HEAD_STAGES = ["context_rf", "context_mlp", "gru", "linear_probe"]
ENCODER_STAGES = ["simclr", "supcon", "adversarial"]
# stages whose outputs depend on run.mode; their manifest entries and folders are kept per mode
MODE_SCOPED = {"embed", *HEAD_STAGES, "evaluate", "heads"}
MODES = ("tuned", "zeroshot")


@dataclass
class RunContext:
    """Everything a stage needs: the config, the run folder and its manifest.

    The manifest (``manifest.json``) records each finished stage's outputs, so later
    stages find their inputs there and a run can be resumed in a new session.
    Mode-dependent stages are stored as ``"<stage>@<mode>"``.

    Attributes:
        cfg: The loaded config.
        run_dir: ``<runs_dir>/<run name>``.
        manifest: Parsed ``manifest.json``.
    """
    cfg: PipelineConfig
    run_dir: Path
    manifest: dict = field(default_factory=dict)
    _device: object = None

    @classmethod
    def open(cls, cfg: PipelineConfig) -> "RunContext":
        """Create the run folder if needed and load its manifest."""
        run_dir = Path(cfg.run.runs_dir) / cfg.run.name
        run_dir.mkdir(parents=True, exist_ok=True)
        mpath = run_dir / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {"stages": {}}
        return cls(cfg, run_dir, manifest)

    # -- shared objects ----------------------------------------------------------------------
    @property
    def scheme(self) -> LabelScheme:
        """The label scheme named in ``[run]``."""
        return get_scheme(self.cfg.run.label_scheme)

    @property
    def device(self):
        """The torch device (resolved once, on first use)."""
        if self._device is None:
            from scorer.training.common import resolve_device

            self._device = resolve_device(self.cfg.run.device)
        return self._device

    def prep_params(self) -> PrepParams:
        """Return the cache parameters from ``[signal]``."""
        s = self.cfg.signal
        return PrepParams(sample_rate=float(s.sample_rate), win_len=int(s.win_len), channels=tuple(s.channels),
                          lowpass_hz=s.lowpass_hz, clip=float(s.clip), dtype=s.cache_dtype,
                          max_len_mismatch=int(s.max_len_mismatch))

    def caches(self, names: list[str]) -> list[DatasetCache]:
        """Open the prepared caches of ``names``, checking they match the current ``[signal]``.

        Raises:
            ValueError: If ``names`` is empty (a stage with no datasets configured).
        """
        if not names:
            raise ValueError("no datasets given for this stage (check the config)")
        return [DatasetCache(self.cfg.cache.root, n, expect=self.prep_params()) for n in names]

    # -- manifest ------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """The run mode, 'tuned' or 'zeroshot'."""
        return self.cfg.run.mode

    def key(self, stage: str) -> str:
        """Return the manifest key: ``embed@zeroshot``, ``context_mlp@tuned``; shared stages keep their name."""
        return f"{stage}@{self.mode}" if stage in MODE_SCOPED else stage

    def stage_dir(self, stage: str) -> Path:
        """Return the output folder of a stage; mode-scoped stages get a ``<mode>`` sub-folder."""
        return self.run_dir / stage / self.mode if stage in MODE_SCOPED else self.run_dir / stage

    def record(self, stage: str, **outputs) -> None:
        """Store a stage's outputs in the manifest with a timestamp and the config digest, and save it."""
        stage = self.key(stage)
        self.manifest["stages"][stage] = {**outputs, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "config_digest": self.cfg.digest()}
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1, default=str))

    def output(self, stage: str, key: str) -> str:
        """Return output ``key`` of an earlier stage in this mode.

        Raises:
            RuntimeError: If that stage has not run in this mode (the message says what to run).
        """
        stage = self.key(stage)
        entry = self.manifest["stages"].get(stage)
        if not entry or key not in entry:
            raise RuntimeError(f"run '{self.run_dir.name}' has no '{key}' from stage '{stage}' "
                               f"(run that stage first in run.mode={self.mode!r}, or point the config at a "
                               f"checkpoint path)")
        return entry[key]

    def resolve_encoder(self, ref: str, use: str = "final") -> str | None:
        """Turn an encoder reference from the config into a checkpoint path.

        Args:
            ref: "" (none: train from scratch), "simclr", "supcon", "adversarial" (this run's
                output), "auto" (the latest encoder stage this mode may use) or a path.
            use: "final" or "best" checkpoint of contrastive stages (adversarial always "final").

        Returns:
            The path, or None for "".

        Raises:
            RuntimeError: For the adversarial encoder in zero-shot mode, or "auto" with no
                encoder stage done yet.
            FileNotFoundError: For a path that does not exist.
        """
        if not ref:
            return None
        if ref == "adversarial" and self.mode == "zeroshot":
            raise RuntimeError("run.mode='zeroshot' must not use the adversarially aligned encoder")
        if ref == "auto":
            usable = ENCODER_STAGES if self.mode == "tuned" else [s for s in ENCODER_STAGES if s != "adversarial"]
            for stage in reversed(usable):
                if stage in self.manifest["stages"]:
                    key = "final" if stage == "adversarial" else use
                    return self.output(stage, key)
            raise RuntimeError("embed.encoder='auto' but this run has no encoder stage output yet")
        if ref in ENCODER_STAGES:
            return self.output(ref, "final" if ref == "adversarial" else use)
        if not Path(ref).exists():
            raise FileNotFoundError(f"encoder checkpoint not found: {ref}")
        return ref

    def encoder_expect(self) -> dict:
        """Return the sample rate and window length a loaded encoder must match."""
        return {"sample_rate": self.cfg.signal.sample_rate, "win_len": self.cfg.signal.win_len}

    def embedding_store(self, names: list[str]):
        """Load the embeddings of ``names`` made by this mode's ``embed`` stage (encoder digest checked)."""
        from scorer.training.embeddings import EmbeddingStore

        emb_root = self.output("embed", "emb_root")
        digest = self.output("embed", "encoder_digest")
        store = EmbeddingStore.load(emb_root, self.caches(names), self.scheme, in_memory=self.cfg.embed.in_memory,
                                    expect_digest=digest, unlabeled_as_wake=self.cfg.run.unlabeled_as_wake)
        log.info("embeddings %s: %s", "+".join(names), store.describe(self.scheme))
        return store

    def head_meta(self, datasets: dict) -> dict:
        """Return the metadata stored with every head.

        It holds the mode, label scheme, encoder path and digest, signal settings and datasets.
        """
        s = self.cfg.signal
        return {"mode": self.mode, "label_scheme": self.scheme.name, "class_names": list(self.scheme.class_names),
                "encoder_path": self.output("embed", "encoder_path"),
                "encoder_digest": self.output("embed", "encoder_digest"),
                "sample_rate": s.sample_rate, "win_len": s.win_len, "lowpass_hz": s.lowpass_hz, "clip": s.clip,
                "datasets": datasets, "run": self.cfg.run.name}

    def referenced_datasets(self) -> list[str]:
        """Return every dataset an enabled stage uses, without duplicates, in config order."""
        c = self.cfg
        names: list[str] = []
        for sec in (c.simclr, c.supcon):
            if sec.enabled:
                names += sec.datasets
        if c.adversarial.enabled:
            names += c.adversarial.source + c.adversarial.target
        for head in (c.context_mlp, c.gru, c.linear_probe):
            if head.enabled:
                names += head.train + head.val
        if c.context_rf.enabled:
            names += c.context_rf.train
        names += c.evaluate.datasets + c.embed.datasets
        return list(dict.fromkeys(names))  # unique, order kept


# =============================================================================================
def stage_inspect(ctx: RunContext) -> None:
    """Stage ``inspect``: count recordings, windows and label codes per dataset.

    Runs before any training and writes ``inspect/datasets.json``. Unknown codes and
    classes with no windows are logged; this is the first place to look when labels
    seem mixed up.
    """
    scheme = ctx.scheme
    report = {}
    for name, path in ctx.cfg.datasets.items():
        try:
            recs = discover_recordings(path)
        except Exception as e:  # report and continue
            log.error("[inspect] %s: %s", name, e)
            report[name] = {"error": str(e)}
            continue
        totals: dict[int, int] = {}
        unlabeled = []
        per_rec = []
        for r in recs:
            y = scan_labels(r)
            if y is None:
                unlabeled.append(r.name)
                continue
            rc = raw_counts(y)
            for k, v in rc.items():
                totals[k] = totals.get(k, 0) + v
            per_rec.append({"recording": r.name, "windows": int(len(y)), "raw_counts": rc})
        unknown_codes = sorted(set(totals) - set(scheme.raw_to_class) - {0})
        mapped = {n: sum(v for k, v in totals.items() if scheme.raw_to_class.get(k) == i)
                  for i, n in enumerate(scheme.class_names)}
        mapped["ignored"] = sum(v for k, v in totals.items() if k not in scheme.raw_to_class)
        missing = [n for n in scheme.class_names if mapped[n] == 0]
        report[name] = {"path": str(path), "recordings": len(recs), "unlabeled_recordings": unlabeled,
                        "raw_label_counts": totals, "class_counts": mapped, "unknown_raw_codes": unknown_codes,
                        "classes_missing": missing, "per_recording": per_rec}
        log.info("[inspect] %-14s %4d recordings | raw %s | %s -> %s%s%s", name, len(recs), totals, scheme.name,
                 mapped, f" | UNKNOWN CODES {unknown_codes}" if unknown_codes else "",
                 f" | MISSING {missing}" if missing else "")
        if unlabeled:
            log.info("[inspect] %s: %d recordings without y files", name, len(unlabeled))
    out = ctx.run_dir / "inspect" / "datasets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for w in ctx.cfg.warnings():
        log.warning("[config] %s", w)
    ctx.record("inspect", report=str(out))


def stage_prepare(ctx: RunContext, force: bool = False) -> None:
    """Stage ``prepare``: normalise every referenced dataset into the cache.

    Recordings that are already up to date are skipped unless ``force`` is set.
    """
    params = ctx.prep_params()
    done = {}
    for name in ctx.referenced_datasets():
        done[name] = str(prepare_dataset(ctx.cfg.dataset_path(name), ctx.cfg.cache.root, name, params, force=force,
                                         seed=ctx.cfg.run.seed))
    ctx.record("prepare", datasets=done, params=params.to_dict())


def _encoder_for_training(ctx: RunContext, init: str, use: str):
    """Return ``(encoder, init_path)`` for a training stage.

    Loads the checkpoint named by ``init``/``use`` (see ``RunContext.resolve_encoder``),
    or builds a fresh encoder from ``[signal]`` when ``init`` is empty.
    """
    from scorer.models.checkpoints import build_encoder, encoder_meta, load_encoder

    path = ctx.resolve_encoder(init, use)
    if path:
        encoder, meta = load_encoder(path, ctx.device, expect=ctx.encoder_expect())
        log.info("initialised encoder from %s", path)
        encoder.train()
        return encoder, path
    meta = encoder_meta(ctx.cfg.signal, ctx.scheme)
    return build_encoder(meta).to(ctx.device), None


def _contrastive_stage(ctx: RunContext, mode: str) -> None:
    """Shared body of the ``simclr`` and ``supcon`` stages.

    Continues from the previous stage's projection head when ``init`` names a
    contrastive stage, as the old pipeline did. Records best/final encoder and
    projection-head paths.
    """
    import torch

    from scorer.data.datasets import WindowBatchStream
    from scorer.models.checkpoints import encoder_meta, read_checkpoint, save_checkpoint
    from scorer.models.heads import ContrastiveModel
    from scorer.training.common import set_seed
    from scorer.training.contrastive import train_contrastive

    sec = getattr(ctx.cfg, mode)
    set_seed(ctx.cfg.run.seed)
    encoder, init_path = _encoder_for_training(ctx, sec.init, sec.use)
    model = ContrastiveModel(encoder).to(ctx.device)
    if sec.init in ("simclr", "supcon"):  # continue with the previous stage's projection head (as the old pipeline did)
        head_path = Path(ctx.output(sec.init, "projection_head"))
        if head_path.exists():
            state, _, _, _ = read_checkpoint(head_path)
            model.projection_head.load_state_dict(state)
    stream = WindowBatchStream(ctx.caches(sec.datasets), ctx.scheme, sec.batch_size, sec.buffer_windows,
                               balance=sec.balance, drop_unlabeled=sec.drop_unlabeled, seed=ctx.cfg.run.seed,
                               loader_augment=sec.loader_augment, unlabeled_as_wake=ctx.cfg.run.unlabeled_as_wake)
    meta = encoder_meta(ctx.cfg.signal, ctx.scheme, init=init_path, datasets=sec.datasets)
    out_dir = ctx.run_dir / mode
    res = train_contrastive(model, stream, mode, epochs=sec.epochs, lr=sec.lr, temperature=sec.temperature,
                            sample_rate=ctx.cfg.signal.sample_rate, device=ctx.device, out_dir=out_dir, enc_meta=meta,
                            num_workers=ctx.cfg.run.num_workers, patience=sec.patience, min_delta=sec.min_delta,
                            snapshot_every=sec.snapshot_every, max_batches_per_epoch=sec.max_batches_per_epoch or None)
    head_path = save_checkpoint(out_dir / "projection_head_final.pt", model.projection_head, "projection_head", {})
    (out_dir / "history.json").write_text(json.dumps(res["history"], indent=1))
    ctx.record(mode, best=res["best"], final=res["final"], projection_head=str(head_path), best_epoch=res["best_epoch"])
    del model, encoder, stream
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_simclr(ctx: RunContext) -> None:
    """Stage ``simclr``: unsupervised contrastive pretraining."""
    _contrastive_stage(ctx, "simclr")


def stage_supcon(ctx: RunContext) -> None:
    """Stage ``supcon``: supervised contrastive pretraining."""
    _contrastive_stage(ctx, "supcon")


def stage_adversarial(ctx: RunContext) -> None:
    """Stage ``adversarial``: align source and target domains (tuned mode only).

    Raises:
        RuntimeError: In zero-shot mode.
    """
    from scorer.data.datasets import WindowBatchStream
    from scorer.models.checkpoints import encoder_meta
    from scorer.training.adversarial import train_adversarial
    from scorer.training.common import set_seed

    sec = ctx.cfg.adversarial
    if ctx.mode == "zeroshot":
        raise RuntimeError("run.mode='zeroshot' skips domain alignment; use run.mode='tuned' to run 'adversarial'")
    set_seed(ctx.cfg.run.seed)
    encoder, init_path = _encoder_for_training(ctx, sec.init, sec.use)
    src = WindowBatchStream(ctx.caches(sec.source), ctx.scheme, sec.batch_size, sec.buffer_windows,
                            drop_unlabeled=True, seed=ctx.cfg.run.seed, unlabeled_as_wake=ctx.cfg.run.unlabeled_as_wake)
    tgt = WindowBatchStream(ctx.caches(sec.target), ctx.scheme, sec.batch_size, sec.buffer_windows,
                            drop_unlabeled=sec.target_drop_unlabeled, seed=ctx.cfg.run.seed + 1)
    meta = encoder_meta(ctx.cfg.signal, ctx.scheme, init=init_path, source=sec.source, target=sec.target)
    res = train_adversarial(encoder, src, tgt, epochs=sec.epochs, device=ctx.device, out_dir=ctx.run_dir / "adversarial",
                            enc_meta=meta, lr_encoder=sec.lr_encoder, lr_freq_flatten=sec.lr_freq_flatten,
                            lr_heads=sec.lr_heads, warmup_fraction=sec.warmup_fraction,
                            num_workers=ctx.cfg.run.num_workers, max_batches_per_epoch=sec.max_batches_per_epoch or None)
    (ctx.run_dir / "adversarial" / "history.json").write_text(json.dumps(res["history"], indent=1))
    ctx.record("adversarial", final=res["final"])


def stage_embed(ctx: RunContext) -> None:
    """Stage ``embed``: run the chosen encoder over every dataset the heads and ``evaluate`` need.

    Embeddings go to ``embeddings/<encoder digest>/<dataset>``; a warning is logged if the
    encoder's low-pass differs from the cache's.
    """
    from scorer.models.checkpoints import load_encoder
    from scorer.training.common import log_memory
    from scorer.training.embeddings import embed_dataset

    sec = ctx.cfg.embed
    path = ctx.resolve_encoder(sec.encoder)
    legacy = {}
    if sec.legacy_sample_rate:
        legacy["sample_rate"] = sec.legacy_sample_rate
    if sec.legacy_lowpass_hz >= 0:
        legacy["lowpass_hz"] = sec.legacy_lowpass_hz
    encoder, meta = load_encoder(path, ctx.device, expect=ctx.encoder_expect(), legacy_meta=legacy)
    if meta.get("lowpass_hz") != ctx.cfg.signal.lowpass_hz:
        log.warning("encoder was trained with lowpass_hz=%s but the cache uses %s: inputs differ from training",
                    meta.get("lowpass_hz"), ctx.cfg.signal.lowpass_hz)
    names = sec.datasets or [n for n in ctx.referenced_datasets()
                             if n in set(ctx.cfg.evaluate.datasets) | _head_datasets(ctx)]
    emb_root = ctx.run_dir / "embeddings" / meta["digest"]
    for name in names:
        embed_dataset(encoder, ctx.caches([name])[0], emb_root, meta, ctx.device, sec.batch_size, sec.dtype)
    log_memory(log, "after embed")
    ctx.record("embed", emb_root=str(emb_root), encoder_path=path, encoder_digest=meta["digest"],
               encoder_stage=meta.get("stage"), datasets=names)


def _head_datasets(ctx: RunContext) -> set[str]:
    """Return the training and validation datasets of all enabled heads."""
    c = ctx.cfg
    out = set()
    for head in (c.context_mlp, c.gru, c.linear_probe):
        if head.enabled:
            out |= set(head.train) | set(head.val)
    if c.context_rf.enabled:
        out |= set(c.context_rf.train)
    return out


def stage_context_rf(ctx: RunContext) -> None:
    """Stage ``context_rf``: fit the random forest and save it as ``heads/<mode>/context_rf.pkl``."""
    from scorer.models.checkpoints import save_rf_bundle
    from scorer.training.context import train_context_rf

    sec = ctx.cfg.context_rf
    store = ctx.embedding_store(sec.train)
    rf = train_context_rf(store, ctx.scheme, k=sec.k, n_estimators=sec.n_estimators, max_depth=sec.max_depth or None,
                          max_features=sec.max_features, class_weight=sec.class_weight or None,
                          max_train_windows=sec.max_train_windows or None, n_jobs=sec.n_jobs, seed=ctx.cfg.run.seed)
    meta = {**ctx.head_meta({"train": sec.train}), "context_k": sec.k, "classes_": [int(c) for c in rf.classes_]}
    path = save_rf_bundle(ctx.stage_dir("heads") / "context_rf.pkl", rf, meta)
    ctx.record("context_rf", head=str(path))


def stage_context_mlp(ctx: RunContext) -> None:
    """Stage ``context_mlp``: train the context MLP and save it with its training history."""
    from scorer.training.common import set_seed
    from scorer.training.context import train_context_mlp

    sec = ctx.cfg.context_mlp
    set_seed(ctx.cfg.run.seed)
    train = ctx.embedding_store(sec.train)
    val = ctx.embedding_store(sec.val) if sec.val else None
    path = ctx.stage_dir("heads") / "context_mlp.pt"
    _, info = train_context_mlp(train, val, ctx.scheme, k=sec.k, epochs=sec.epochs, batch_size=sec.batch_size, lr=sec.lr,
                                weight_decay=sec.weight_decay, label_smoothing=sec.label_smoothing,
                                hidden_dim=sec.hidden_dim, num_blocks=sec.num_blocks, dropout=sec.dropout,
                                class_weighting=sec.class_weighting, device=ctx.device, out_path=path,
                                meta=ctx.head_meta({"train": sec.train, "val": sec.val}),
                                max_train_windows=sec.max_train_windows or None, seed=ctx.cfg.run.seed)
    (ctx.stage_dir("heads") / "context_mlp_history.json").write_text(json.dumps(info["history"], indent=1))
    ctx.record("context_mlp", head=str(path), best_epoch=info["best_epoch"])


def stage_gru(ctx: RunContext) -> None:
    """Stage ``gru``: train the GRU sequence head."""
    from scorer.training.common import set_seed
    from scorer.training.sequence_gru import train_gru

    sec = ctx.cfg.gru
    set_seed(ctx.cfg.run.seed)
    train = ctx.embedding_store(sec.train)
    val = ctx.embedding_store(sec.val) if sec.val else None
    path = ctx.stage_dir("heads") / "gru.pt"
    _, info = train_gru(train, val, ctx.scheme, seq_len=sec.seq_len, stride=sec.stride, epochs=sec.epochs,
                        batch_size=sec.batch_size, lr=sec.lr, weight_decay=sec.weight_decay, hidden_dim=sec.hidden_dim,
                        num_layers=sec.num_layers, focal_gamma=sec.focal_gamma,
                        max_unlabeled_fraction=sec.max_unlabeled_fraction, grad_clip=sec.grad_clip,
                        patience=sec.patience, device=ctx.device, out_path=path,
                        meta=ctx.head_meta({"train": sec.train, "val": sec.val}), seed=ctx.cfg.run.seed)
    ctx.record("gru", head=str(path), best_epoch=info["best_epoch"])


def stage_linear_probe(ctx: RunContext) -> None:
    """Stage ``linear_probe``: train the linear probe."""
    from scorer.training.context import train_linear_probe

    sec = ctx.cfg.linear_probe
    train = ctx.embedding_store(sec.train)
    val = ctx.embedding_store(sec.val) if sec.val else None
    path = ctx.stage_dir("heads") / "linear_probe.pt"
    _, info = train_linear_probe(train, val, ctx.scheme, epochs=sec.epochs, batch_size=sec.batch_size, lr=sec.lr,
                                 weight_decay=sec.weight_decay, class_weighting=sec.class_weighting, device=ctx.device,
                                 out_path=path, meta=ctx.head_meta({"train": sec.train, "val": sec.val}),
                                 seed=ctx.cfg.run.seed)
    ctx.record("linear_probe", head=str(path), best_epoch=info["best_epoch"])


def stage_evaluate(ctx: RunContext) -> None:
    """Stage ``evaluate``: score every evaluation dataset with every listed head.

    Per head and dataset it writes ``metrics.json`` (post-processed and argmax-only),
    ``per_recording.csv``, ``predictions.npz``, a confusion matrix and hypnograms; per
    mode it writes ``summary.csv`` and optional embedding PCA plots.

    Raises:
        RuntimeError: If a head was trained on different embeddings or class names.
    """
    from scorer.evaluation.metrics import (compute_metrics, per_recording_metrics, save_confusion_png,
                                           save_embedding_pca_png, save_hypnogram_png, text_report, write_csv,
                                           write_json)
    from scorer.inference.postprocess import predict_classes
    from scorer.inference.scoring import load_head

    sec = ctx.cfg.evaluate
    scheme = ctx.scheme
    out_root = ctx.stage_dir("evaluate")
    summary = []
    stores = {name: ctx.embedding_store([name]) for name in sec.datasets}
    for head_name in sec.heads:
        head_path = ctx.output(head_name, "head")
        head = load_head(head_path, ctx.device if not head_path.endswith(".pkl") else "cpu")
        if head.meta.get("encoder_digest") != ctx.output("embed", "encoder_digest"):
            raise RuntimeError(f"{head_name} was trained on encoder {head.meta.get('encoder_digest')}, but the current "
                               f"embeddings come from {ctx.output('embed', 'encoder_digest')}: re-train the head")
        if tuple(head.meta["class_names"]) != scheme.class_names:
            raise RuntimeError(f"{head_name} class names {head.meta['class_names']} != scheme {scheme.class_names}")
        for ds, store in stores.items():
            out = out_root / head_name / ds
            out.mkdir(parents=True, exist_ok=True)
            probs = head.predict_probs(store, ctx.device)
            bounds = [(sp.start, sp.end) for sp in store.spans]
            raw_pred = predict_classes(probs, bounds, scheme)
            pred = predict_classes(probs, bounds, scheme, sec.smooth_sigma, sec.heuristics)
            m_raw = compute_metrics(store.classes, raw_pred, scheme)
            m = compute_metrics(store.classes, pred, scheme)
            write_json({"postprocessed": m, "argmax_only": m_raw,
                        "postprocess": {"smooth_sigma": sec.smooth_sigma, "heuristics": sec.heuristics},
                        "head": head_path, "encoder_digest": head.meta.get("encoder_digest"),
                        "mode": ctx.mode, "encoder_stage": ctx.output("embed", "encoder_stage")}, out / "metrics.json")
            write_csv(per_recording_metrics(store.classes, pred, store.spans), out / "per_recording.csv")
            np.savez_compressed(out / "predictions.npz", probs=probs.astype(np.float16), pred=pred,
                                true=store.classes, rec_start=store.rec_start, rec_end=store.rec_end)
            if m.get("n_scored"):
                save_confusion_png(m, out / "confusion.png", f"{head_name} on {ds}")
            for sp in store.spans[: sec.hypnograms_per_dataset]:
                save_hypnogram_png(store.classes[sp.start:sp.end], pred[sp.start:sp.end], scheme.class_names,
                                   out / f"hypnogram_{sp.name}.png", f"{head_name} | {ds} / {sp.name}",
                                   sec.hypnogram_windows)
            log.info("[evaluate] %s on %s: acc %.4f | macro F1 %.4f | kappa %.4f (argmax only: acc %.4f)\n%s",
                     head_name, ds, m.get("accuracy", float("nan")), m.get("macro_f1", float("nan")),
                     m.get("kappa", float("nan")), m_raw.get("accuracy", float("nan")),
                     text_report(store.classes, pred, scheme))
            summary.append({"mode": ctx.mode, "head": head_name, "dataset": ds, "accuracy": m.get("accuracy"),
                            "macro_f1": m.get("macro_f1"), "kappa": m.get("kappa"),
                            "accuracy_argmax_only": m_raw.get("accuracy"), "scored_windows": m.get("n_scored")})
    if sec.pca:
        for ds, store in stores.items():
            idx = np.arange(len(store))
            save_embedding_pca_png(store.rows(idx[:: max(1, len(idx) // 20000)]), store.classes[:: max(1, len(idx) // 20000)],
                                   scheme.class_names, out_root / f"pca_{ds}.png", f"embeddings: {ds}")
    write_csv(summary, out_root / "summary.csv")
    ctx.record("evaluate", summary=str(out_root / "summary.csv"))


STAGES: dict[str, Callable[[RunContext], None]] = {
    "inspect": stage_inspect,
    "prepare": stage_prepare,
    "simclr": stage_simclr,
    "supcon": stage_supcon,
    "adversarial": stage_adversarial,
    "embed": stage_embed,
    "context_rf": stage_context_rf,
    "context_mlp": stage_context_mlp,
    "gru": stage_gru,
    "linear_probe": stage_linear_probe,
    "evaluate": stage_evaluate,
}


def enabled_stages(cfg: PipelineConfig) -> list[str]:
    """Return the default stage list: inspect, prepare, enabled encoder stages, embed, enabled heads, evaluate."""
    out = ["inspect", "prepare"]
    out += [s for s in ENCODER_STAGES if getattr(cfg, s).enabled
            and not (s == "adversarial" and cfg.run.mode == "zeroshot")]
    out.append("embed")
    out += [s for s in HEAD_STAGES if getattr(cfg, s).enabled]
    out.append("evaluate")
    return out
