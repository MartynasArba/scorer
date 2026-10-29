"""
Bi-GRU sequence head on precomputed embeddings (port of sequence_training + scoring.run_sequence_inference).

Differences from the old ContextAwareSleepScorer training:
  * the frozen encoder is not re-run for every sequence; embeddings come from the `embed` stage
    (identical outputs, since the encoder was frozen and in eval mode; much faster);
  * signal-space augmentation of sequences is therefore not available (it only perturbed the
    frozen encoder's input);
  * windows with unknown labels stay in the sequence and are ignored by the loss
    (old code dropped every sequence containing a single unlabeled window);
  * inference averages overlapping sequences and also covers recording tails and recordings
    shorter than seq_len.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from scorer.data.labels import IGNORE, LabelScheme
from scorer.training.embeddings import EmbeddingStore

log = logging.getLogger(__name__)


def predict_gru(model, store: EmbeddingStore, seq_len: int, device, stride: int | None = None,
                batch_size: int = 256) -> np.ndarray:
    """Return per-window probabilities [N, C] from a GRU head.

    Overlapping sequences (default stride ``seq_len // 2``) are averaged; the last
    windows of each recording are covered, and recordings shorter than ``seq_len`` are
    scored as one shorter sequence.
    """
    import torch

    stride = stride or max(1, seq_len // 2)
    model.eval()
    n = len(store)
    sums = None
    counts = np.zeros(n, dtype=np.float32)
    starts = store.sequence_starts(seq_len, stride, cover_tail=True)
    with torch.inference_mode():
        for s in range(0, len(starts), batch_size):
            st = starts[s : s + batch_size]
            x, _ = store.sequences(st, seq_len)
            probs = torch.softmax(model(torch.from_numpy(x).to(device)).float(), dim=-1).cpu().numpy()
            if sums is None:
                sums = np.zeros((n, probs.shape[-1]), dtype=np.float32)
            idx = (st[:, None] + np.arange(seq_len)[None, :]).reshape(-1)
            np.add.at(sums, idx, probs.reshape(-1, probs.shape[-1]))
            np.add.at(counts, idx, 1.0)
        # recordings shorter than seq_len: one variable-length sequence each
        for a, b in zip(store.rec_start, store.rec_end):
            if b - a < seq_len and b > a:
                x = store.rows(np.arange(a, b))[None]
                probs = torch.softmax(model(torch.from_numpy(x).to(device)).float(), dim=-1).cpu().numpy()[0]
                if sums is None:
                    sums = np.zeros((n, probs.shape[-1]), dtype=np.float32)
                sums[a:b] += probs
                counts[a:b] += 1
    return sums / np.maximum(counts, 1)[:, None]


def train_gru(train: EmbeddingStore, val: EmbeddingStore | None, scheme: LabelScheme, *, seq_len: int, stride: int,
              epochs: int, batch_size: int, lr: float, weight_decay: float, hidden_dim: int, num_layers: int,
              focal_gamma: float, max_unlabeled_fraction: float, grad_clip: float, patience: int, device,
              out_path: Path, meta: dict, seed: int = 0):
    """Train a ``GRUSequenceHead`` on embedding sequences and save it as ``out_path``.

    Sequences stay inside one recording. Unlabeled windows stay in the sequence and are
    ignored by the focal loss. The best epoch (validation macro F1, or training loss
    without ``val``) is restored before saving.

    Args:
        train: Training embeddings.
        val: Validation embeddings, or None.
        scheme: Label scheme.
        seq_len: Windows per sequence.
        stride: Step between training sequences.
        epochs, batch_size, lr, weight_decay: Optimisation settings (AdamW, cosine schedule).
        hidden_dim, num_layers: GRU size.
        focal_gamma: Focal-loss gamma.
        max_unlabeled_fraction: Skip sequences with more unlabeled windows than this.
        grad_clip: Gradient-norm limit.
        patience: Early-stopping patience in epochs.
        device: Training device.
        out_path: Checkpoint path.
        meta: Metadata stored in the checkpoint.
        seed: Shuffling seed.

    Returns:
        ``(model, info)`` with the best weights loaded.

    Raises:
        RuntimeError: If no training sequence qualifies.
    """
    import torch
    from sklearn.metrics import f1_score

    from scorer.models.checkpoints import save_checkpoint
    from scorer.models.heads import GRUSequenceHead
    from scorer.models.losses import FocalLoss
    from scorer.training.common import BestState, EarlyStopping

    rng = np.random.default_rng(seed)
    starts = train.sequence_starts(seq_len, stride, max_unlabeled_fraction)
    if len(starts) == 0:
        raise RuntimeError("gru: no training sequences (seq_len too long or too many unlabeled windows)")
    arch = {"embedding_dim": train.dim, "hidden_dim": hidden_dim, "num_classes": scheme.n_classes,
            "num_layers": num_layers}
    model = GRUSequenceHead(**arch).to(device)
    criterion = FocalLoss(gamma=focal_gamma, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * int(np.ceil(len(starts) / batch_size)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    best = BestState(mode="max")
    stopper = EarlyStopping(patience=patience, min_delta=0.0, mode="max")
    log.info("gru: %d sequences of %d windows (stride %d)", len(starts), seq_len, stride)
    history = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(starts)
        loss_sum, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            x, y = train.sequences(perm[s : s + batch_size], seq_len)
            xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb).permute(0, 2, 1), yb)  # [B, C, L] vs [B, L]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            nb += 1
        stats = {"epoch": epoch, "train_loss": loss_sum / max(nb, 1), "seconds": round(time.time() - t0, 1)}
        score = -stats["train_loss"]
        if val is not None:
            lab = val.labeled_indices()
            pred = predict_gru(model, val, seq_len, device)[lab].argmax(1)
            yv = val.classes[lab]
            stats["val_acc"] = float((pred == yv).mean())
            stats["val_macro_f1"] = float(f1_score(yv, pred, average="macro", labels=list(range(scheme.n_classes)),
                                                   zero_division=0))
            score = stats["val_macro_f1"]
        best.update(score, model, epoch)
        history.append(stats)
        log.info("gru epoch %d/%d: train loss %.4f%s (%.0fs)", epoch + 1, epochs, stats["train_loss"],
                 f" | val acc {stats['val_acc']:.4f} macro F1 {stats['val_macro_f1']:.4f}" if "val_acc" in stats else "",
                 stats["seconds"])
        if stopper.step(score):
            log.info("gru: early stop at epoch %d", epoch + 1)
            break
    best.restore(model)
    save_checkpoint(out_path, model, "gru", {**meta, "arch": arch, "seq_len": seq_len, "best_epoch": best.epoch,
                                             "best_score": best.value})
    return model, {"history": history, "best_epoch": best.epoch, "best_score": best.value}
