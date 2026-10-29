"""
SimCLR (unsupervised) and SupCon (supervised) pretraining of the window encoder.
One loop for both (the old train_unsupervised / train_supcon were 90% duplicated).
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.optim as optim

from scorer.models.checkpoints import save_checkpoint
from scorer.models.heads import ContrastiveModel
from scorer.models.losses import SimCLRLoss, SupConLoss
from scorer.training.augment import AugmentConfig, batch_augment
from scorer.training.common import BestState, EarlyStopping, log_memory, make_loader

log = logging.getLogger(__name__)


def train_contrastive(model: ContrastiveModel, stream, mode: str, *, epochs: int, lr: float, temperature: float,
                      sample_rate: float, device: torch.device, out_dir: Path, enc_meta: dict,
                      num_workers: int = 2, patience: int = 5, min_delta: float = 1e-3, snapshot_every: int = 20,
                      aug_cfg: AugmentConfig | None = None, max_batches_per_epoch: int | None = None) -> dict:
    """Pretrain the window encoder with SimCLR or SupCon.

    Each batch is augmented twice. SimCLR pulls the two views of a window together;
    SupCon pulls together all views of the same class. Training stops early when the
    epoch loss stops improving.

    Args:
        model: Encoder plus projection head.
        stream: ``WindowBatchStream`` of training windows (labels are used by SupCon only).
        mode: "simclr" or "supcon".
        epochs: Maximum number of epochs.
        lr: Adam learning rate.
        temperature: Contrastive temperature.
        sample_rate: Sample rate in Hz, for the augmentations.
        device: Training device.
        out_dir: Folder for checkpoints.
        enc_meta: Encoder metadata stored in every checkpoint.
        num_workers: DataLoader workers.
        patience: Early-stopping patience in epochs (0 disables).
        min_delta: Smallest loss decrease that counts as improvement.
        snapshot_every: Also save ``encoder_epochNNN.pt`` every this many epochs (0 = never).
        aug_cfg: Augmentation settings.
        max_batches_per_epoch: Cap per epoch, for quick test runs.

    Returns:
        ``{"best", "final", "history", "best_epoch", "best_loss"}``. Checkpoints:
        ``encoder_best.pt`` (lowest epoch loss) and ``encoder_final.pt``.

    Raises:
        ValueError: For an unknown ``mode``.
        FloatingPointError: If the loss becomes NaN or infinite.
        RuntimeError: If the stream yields no batches.
    """
    if mode not in ("simclr", "supcon"):
        raise ValueError(mode)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = SimCLRLoss(temperature) if mode == "simclr" else SupConLoss(temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
    best = BestState(mode="min")
    loader = make_loader(stream, num_workers=num_workers, pin_memory=device.type == "cuda")
    log.info("%s: %d epochs, ~%d batches/epoch of %d, class distribution (epoch 0) %s", mode, epochs, len(stream),
             stream.batch_size, stream.class_distribution(0))
    history = []

    for epoch in range(epochs):
        stream.set_epoch(epoch)
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "pos": 0.0, "neg": 0.0, "feat_sd": 0.0}
        n = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            v1 = batch_augment(x, sample_rate, aug_cfg)
            v2 = batch_augment(x, sample_rate, aug_cfg)
            optimizer.zero_grad(set_to_none=True)
            if mode == "simclr":
                z1, z2 = model(v1), model(v2)
                loss = criterion(z1, z2)
                feats = z1
            else:
                feats = model(torch.cat([v1, v2]))
                labels = torch.cat([y, y])
                loss = criterion(feats, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{mode}: non-finite loss at epoch {epoch}, batch {n}")
            loss.backward()
            optimizer.step()

            sums["loss"] += loss.item()
            with torch.no_grad():
                sums["feat_sd"] += feats.std(dim=0).mean().item()
                if mode == "supcon":
                    sim = feats @ feats.T
                    same = (labels[:, None] == labels[None, :]).float()
                    sums["pos"] += ((sim * same).sum() / same.sum()).item()
                    sums["neg"] += ((sim * (1 - same)).sum() / (1 - same).sum().clamp(min=1)).item()
            n += 1
            if max_batches_per_epoch and n >= max_batches_per_epoch:
                break
        if n == 0:
            raise RuntimeError(f"{mode}: the data stream produced no batches (batch_size larger than the data?)")
        stats = {k: v / n for k, v in sums.items()}
        stats.update(epoch=epoch, batches=n, seconds=round(time.time() - t0, 1))
        history.append(stats)
        extra = f" | pos sim {stats['pos']:.3f} neg sim {stats['neg']:.3f}" if mode == "supcon" else ""
        log.info("%s epoch %d/%d: loss %.4f | feature sd %.4f%s | %d batches in %.0fs", mode, epoch + 1, epochs,
                 stats["loss"], stats["feat_sd"], extra, n, stats["seconds"])
        if stats["feat_sd"] < 1e-4:
            log.warning("%s: feature sd %.2e -> representation may be collapsing", mode, stats["feat_sd"])
        if epoch == 0:
            log_memory(log, f"{mode} after first epoch")

        if best.update(stats["loss"], model.encoder, epoch):
            save_checkpoint(out_dir / "encoder_best.pt", model.encoder, "encoder",
                            {**enc_meta, "stage": mode, "epoch": epoch, "loss": stats["loss"]})
        if snapshot_every and epoch % snapshot_every == 0:
            save_checkpoint(out_dir / f"encoder_epoch{epoch:03d}.pt", model.encoder, "encoder",
                            {**enc_meta, "stage": mode, "epoch": epoch, "loss": stats["loss"]})
        if stopper.step(stats["loss"]):
            log.info("%s: early stop at epoch %d (no improvement > %g for %d epochs)", mode, epoch + 1, min_delta,
                     patience)
            break

    final = save_checkpoint(out_dir / "encoder_final.pt", model.encoder, "encoder",
                            {**enc_meta, "stage": mode, "epoch": history[-1]["epoch"], "loss": history[-1]["loss"]})
    return {"best": str(out_dir / "encoder_best.pt"), "final": str(final), "history": history,
            "best_epoch": best.epoch, "best_loss": None if best.value is None or math.isnan(best.value) else best.value}
