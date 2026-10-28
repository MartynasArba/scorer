"""
Window-level heads on frozen embeddings, with temporal context [t-k .. t+k]:
  * context MLP   (ContextResMLP, port of classifier_training.train_context_net)
  * linear probe  (k = 0, port of linear_train_eval)
  * context RF    (port of rf_training.train_context_rf_model)

Label handling: targets are fixed-scheme class indices; IGNORE windows are never trained on
or scored, but they are still used as neighbours. Class weights are computed with
bincount(minlength=n_classes), so a class missing from a dataset cannot shift indices.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from scorer.data.labels import IGNORE, LabelScheme
from scorer.training.embeddings import EmbeddingStore

log = logging.getLogger(__name__)


def class_weight_vector(classes: np.ndarray, n_classes: int, mode: str | list | None) -> np.ndarray | None:
    if mode in (None, "none"):
        return None
    if isinstance(mode, (list, tuple)):
        if len(mode) != n_classes:
            raise ValueError(f"class_weights has {len(mode)} entries for {n_classes} classes")
        return np.asarray(mode, dtype=np.float32)
    counts = np.bincount(classes[classes != IGNORE], minlength=n_classes).astype(np.float64)
    if mode == "inv_sqrt":
        w = 1.0 / np.sqrt(counts + 1e-6)
    elif mode == "inv":
        w = 1.0 / (counts + 1e-6)
    else:
        raise ValueError(f"unknown class weighting {mode!r}")
    present = counts > 0
    w = w / w[present].mean()
    w[~present] = 0.0
    return w.astype(np.float32)


def subsample_stratified(idx: np.ndarray, classes: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    """keep at most max_n indices, preserving class proportions"""
    if not max_n or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    frac = max_n / len(idx)
    keep = []
    for c in np.unique(classes[idx]):
        ic = idx[classes[idx] == c]
        keep.append(rng.choice(ic, size=max(1, int(round(len(ic) * frac))), replace=False))
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------------------------------------
# neural heads (MLP / linear)
# ---------------------------------------------------------------------------------------------
def predict_window_head(model, store: EmbeddingStore, k: int, device, centers: np.ndarray | None = None,
                        batch_size: int = 8192) -> np.ndarray:
    """softmax probabilities [len(centers), C]; all windows when centers is None"""
    import torch

    centers = np.arange(len(store)) if centers is None else np.asarray(centers)
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(centers), batch_size):
            xb = torch.from_numpy(store.context_features(centers[s : s + batch_size], k)).to(device)
            out.append(torch.softmax(model(xb).float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def fit_window_head(model, train: EmbeddingStore, val: EmbeddingStore | None, scheme: LabelScheme, k: int, *,
                    epochs: int, batch_size: int, lr: float, weight_decay: float, label_smoothing: float,
                    class_weighting, device, optimizer: str = "adamw", max_train_windows: int | None = None,
                    seed: int = 0, tag: str = "head") -> dict:
    """generic training loop; restores the best-val-macro-F1 weights (deep copy) at the end"""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    from scorer.training.common import BestState

    rng = np.random.default_rng(seed)
    centers = subsample_stratified(train.labeled_indices(), train.classes, max_train_windows, seed)
    if len(centers) == 0:
        raise RuntimeError(f"{tag}: training store has no labeled windows")
    w = class_weight_vector(train.classes, scheme.n_classes, class_weighting)
    criterion = nn.CrossEntropyLoss(weight=None if w is None else torch.from_numpy(w).to(device),
                                    label_smoothing=label_smoothing)
    opt_cls = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = BestState(mode="max")
    val_centers = val.labeled_indices() if val is not None else None
    log.info("%s: %d training windows, k=%d, input %d, class weights %s", tag, len(centers), k,
             (2 * k + 1) * train.dim, None if w is None else np.round(w, 3).tolist())
    history = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(centers)
        loss_sum, n_seen = 0.0, 0
        for s in range(0, len(perm), batch_size):
            c = perm[s : s + batch_size]
            if len(c) < 2:
                continue
            xb = torch.from_numpy(train.context_features(c, k)).to(device, non_blocking=True)
            yb = torch.from_numpy(train.classes[c]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(c)
            n_seen += len(c)
        sched.step()
        stats = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "seconds": round(time.time() - t0, 1)}
        if val is not None and len(val_centers):
            probs = predict_window_head(model, val, k, device, val_centers)
            pred = probs.argmax(1)
            yv = val.classes[val_centers]
            stats["val_acc"] = float((pred == yv).mean())
            stats["val_macro_f1"] = float(f1_score(yv, pred, average="macro", labels=list(range(scheme.n_classes)),
                                                   zero_division=0))
            score = stats["val_macro_f1"]
        else:
            score = -stats["train_loss"]
        if best.update(score, model, epoch):
            stats["best"] = True
        history.append(stats)
        log.info("%s epoch %d/%d: train loss %.4f%s (%.0fs)%s", tag, epoch + 1, epochs, stats["train_loss"],
                 f" | val acc {stats['val_acc']:.4f} macro F1 {stats['val_macro_f1']:.4f}" if "val_acc" in stats else "",
                 stats["seconds"], " *" if stats.get("best") else "")
    best.restore(model)
    return {"history": history, "best_epoch": best.epoch, "best_score": best.value}


def train_context_mlp(train: EmbeddingStore, val: EmbeddingStore | None, scheme: LabelScheme, *, k: int, epochs: int,
                      batch_size: int, lr: float, weight_decay: float, label_smoothing: float, hidden_dim: int,
                      num_blocks: int, dropout: float, class_weighting, device, out_path: Path, meta: dict,
                      max_train_windows: int | None = None, seed: int = 0):
    from scorer.models.checkpoints import save_checkpoint
    from scorer.models.heads import ContextResMLP

    arch = {"input_dim": (2 * k + 1) * train.dim, "hidden_dim": hidden_dim, "num_classes": scheme.n_classes,
            "num_blocks": num_blocks, "dropout": dropout}
    model = ContextResMLP(**arch)
    info = fit_window_head(model, train, val, scheme, k, epochs=epochs, batch_size=batch_size, lr=lr,
                           weight_decay=weight_decay, label_smoothing=label_smoothing, class_weighting=class_weighting,
                           device=device, max_train_windows=max_train_windows, seed=seed, tag="context_mlp")
    save_checkpoint(out_path, model, "context_mlp", {**meta, "arch": arch, "context_k": k,
                                                     "best_epoch": info["best_epoch"], "best_score": info["best_score"]})
    return model, info


def train_linear_probe(train: EmbeddingStore, val: EmbeddingStore | None, scheme: LabelScheme, *, epochs: int,
                       batch_size: int, lr: float, weight_decay: float, class_weighting, device, out_path: Path,
                       meta: dict, seed: int = 0):
    from scorer.models.checkpoints import save_checkpoint
    from scorer.models.heads import LinearProbe

    arch = {"input_dim": train.dim, "num_classes": scheme.n_classes}
    model = LinearProbe(**arch)
    info = fit_window_head(model, train, val, scheme, 0, epochs=epochs, batch_size=batch_size, lr=lr,
                           weight_decay=weight_decay, label_smoothing=0.0, class_weighting=class_weighting,
                           device=device, optimizer="adam", seed=seed, tag="linear_probe")
    save_checkpoint(out_path, model, "linear_probe", {**meta, "arch": arch, "context_k": 0,
                                                      "best_epoch": info["best_epoch"], "best_score": info["best_score"]})
    return model, info


# ---------------------------------------------------------------------------------------------
# random forest
# ---------------------------------------------------------------------------------------------
def train_context_rf(train: EmbeddingStore, scheme: LabelScheme, *, k: int, n_estimators: int, max_depth: int | None,
                     max_features, class_weight: str | None, max_train_windows: int | None, n_jobs: int, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    centers = subsample_stratified(train.labeled_indices(), train.classes, max_train_windows, seed)
    x = train.context_features(centers, k)
    y = train.classes[centers]
    log.info("context_rf: fitting on %d windows x %d features (%.2f GB)", x.shape[0], x.shape[1], x.nbytes / 1e9)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                class_weight=class_weight, n_jobs=n_jobs, random_state=seed)
    rf.fit(x, y)
    del x
    return rf


def predict_rf(rf, store: EmbeddingStore, k: int, n_classes: int, centers: np.ndarray | None = None,
               chunk: int = 50_000) -> np.ndarray:
    """probabilities with columns for ALL classes (rf.classes_ may miss a class absent from training)"""
    centers = np.arange(len(store)) if centers is None else np.asarray(centers)
    cols = np.asarray(rf.classes_, dtype=np.int64)
    out = np.zeros((len(centers), n_classes), dtype=np.float32)
    for s in range(0, len(centers), chunk):
        c = centers[s : s + chunk]
        out[s : s + len(c)][:, cols] = rf.predict_proba(store.context_features(c, k))
    return out
