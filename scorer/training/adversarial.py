"""
Domain-adversarial fine-tuning (DANN): keep sleep-state information (state classifier on the
labeled SOURCE domain) while making SOURCE and TARGET embeddings indistinguishable
(domain classifier behind a gradient-reversal layer).

Changes vs adversarial_training.train_adversarial_domain:
  * learning rates come from config (encoder groups small, heads large, as tuned before);
  * domain accuracy is logged: it should drift towards 0.5 as alignment works;
  * the target domain should NOT be a dataset you report as validation/test (the pipeline
    warns): otherwise the "zero-shot" score has seen the test recordings.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from scorer.models.checkpoints import save_checkpoint
from scorer.training.common import log_memory, make_loader

log = logging.getLogger(__name__)


def alpha_schedule(step: int, total: int, warmup_fraction: float) -> float:
    """Return the gradient-reversal strength for a training step.

    Zero during the first ``warmup_fraction`` of training, then rising linearly to 1.
    """
    p = step / max(total, 1)
    return 0.0 if p < warmup_fraction else (p - warmup_fraction) / (1.0 - warmup_fraction)


def train_adversarial(encoder: nn.Module, source_stream, target_stream, *, epochs: int, device: torch.device,
                      out_dir: Path, enc_meta: dict, lr_encoder: float = 2e-6, lr_freq_flatten: float = 1e-5,
                      lr_heads: float = 1e-3, warmup_fraction: float = 0.2, num_workers: int = 2,
                      max_batches_per_epoch: int | None = None) -> dict:
    """Fine-tune the encoder so source and target embeddings become indistinguishable (DANN).

    A state classifier is trained on the labeled source; a domain classifier behind
    gradient reversal tries to tell source from target, and the encoder learns to
    defeat it. Target labels are never read. Domain accuracy near 0.5 means the domains
    are aligned.

    Args:
        encoder: ``SCDSSleepCNN`` with ``state_classifier`` and ``domain_classifier``.
        source_stream: Labeled source windows.
        target_stream: Target windows (labels ignored).
        epochs: Number of epochs.
        device: Training device.
        out_dir: Folder for ``encoder_final.pt``.
        enc_meta: Encoder metadata stored in the checkpoint.
        lr_encoder: Learning rate of the conv streams (small, to keep pretraining).
        lr_freq_flatten: Learning rate of the frequency projection.
        lr_heads: Learning rate of both classifiers.
        warmup_fraction: Share of steps before gradient reversal starts.
        num_workers: DataLoader workers per stream.
        max_batches_per_epoch: Cap per epoch, for quick test runs.

    Returns:
        ``{"final": checkpoint path, "history": per-epoch stats}``.

    Raises:
        RuntimeError: If either stream yields no batches.
    """
    out_dir = Path(out_dir)
    encoder.to(device).train()
    optimizer = optim.Adam([
        {"params": encoder.time_conv.parameters(), "lr": lr_encoder},
        {"params": encoder.freq_conv.parameters(), "lr": lr_encoder},
        {"params": encoder.freq_flatten.parameters(), "lr": lr_freq_flatten},
        {"params": encoder.domain_classifier.parameters(), "lr": lr_heads},
        {"params": encoder.state_classifier.parameters(), "lr": lr_heads},
    ])
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    pin = device.type == "cuda"
    src_loader = make_loader(source_stream, num_workers, pin)
    tgt_loader = make_loader(target_stream, num_workers, pin)
    steps_per_epoch = min(len(source_stream), len(target_stream))
    if max_batches_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_batches_per_epoch)
    total = max(1, epochs * steps_per_epoch)
    # the target's labels are never used (not even logged): only its inputs enter the domain loss
    log.info("adversarial: %d epochs x ~%d steps; source %s / target ~%d batches (labels unused)", epochs,
             steps_per_epoch, source_stream.class_distribution(0), len(target_stream))
    history = []
    step = 0
    for epoch in range(epochs):
        source_stream.set_epoch(epoch)
        target_stream.set_epoch(epoch)
        encoder.train()
        t0 = time.time()
        sums = {"state_loss": 0.0, "domain_loss": 0.0, "state_acc": 0.0, "domain_acc": 0.0}
        n = 0
        for (xs, ys), (xt, _) in zip(src_loader, tgt_loader):
            xs, ys, xt = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True), xt.to(device, non_blocking=True)
            alpha = alpha_schedule(step, total, warmup_fraction)
            optimizer.zero_grad(set_to_none=True)
            feats_s, dom_s = encoder.forward_domain(xs, alpha=alpha)
            _, dom_t = encoder.forward_domain(xt, alpha=alpha)
            state_logits = encoder.state_classifier(feats_s)
            loss_state = ce(state_logits, ys)
            loss_dom = 0.5 * (bce(dom_s, torch.zeros_like(dom_s)) + bce(dom_t, torch.ones_like(dom_t)))
            (loss_state + loss_dom).backward()
            optimizer.step()
            with torch.no_grad():
                sums["state_loss"] += loss_state.item()
                sums["domain_loss"] += loss_dom.item()
                sums["state_acc"] += (state_logits.argmax(1) == ys).float().mean().item()
                sums["domain_acc"] += 0.5 * ((dom_s < 0).float().mean() + (dom_t > 0).float().mean()).item()
            n += 1
            step += 1
            if max_batches_per_epoch and n >= max_batches_per_epoch:
                break
        if n == 0:
            raise RuntimeError("adversarial: no batches (check batch_size vs. dataset sizes)")
        stats = {k: v / n for k, v in sums.items()}
        stats.update(epoch=epoch, alpha=alpha, seconds=round(time.time() - t0, 1))
        history.append(stats)
        log.info("adversarial epoch %d/%d: state loss %.3f acc %.3f | domain loss %.3f acc %.3f (0.5 = aligned) | "
                 "alpha %.2f | %.0fs", epoch + 1, epochs, stats["state_loss"], stats["state_acc"],
                 stats["domain_loss"], stats["domain_acc"], alpha, stats["seconds"])
        if epoch == 0:
            log_memory(log, "adversarial after first epoch")
    path = save_checkpoint(out_dir / "encoder_final.pt", encoder, "encoder",
                           {**enc_meta, "stage": "adversarial", "epochs": len(history)})
    return {"final": str(path), "history": history}
