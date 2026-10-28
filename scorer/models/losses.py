"""Losses: supervised contrastive, SimCLR (NT-Xent) and focal loss (with ignore_index)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss, https://arxiv.org/abs/2004.11362. features must be L2-normalised."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        positives = torch.eq(labels, labels.T).float()
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        not_self = 1.0 - torch.eye(n, device=features.device)
        positives = positives * not_self
        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        pos_counts = positives.sum(1).clamp(min=1.0)
        return -((positives * log_prob).sum(1) / pos_counts).mean()


class SimCLRLoss(nn.Module):
    """NT-Xent, https://arxiv.org/abs/2002.05709. z1, z2: two views [N, D], L2-normalised."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n = z1.size(0)
        z = torch.cat((z1, z2), dim=0)
        sim = z @ z.T / self.temperature
        sim.fill_diagonal_(float("-inf"))
        targets = torch.arange(n, device=z.device)
        targets = torch.cat([targets + n, targets])
        return F.cross_entropy(sim, targets)


class FocalLoss(nn.Module):
    """
    focal loss; works for logits [B, C] or [B, C, L]. Targets equal to ignore_index are
    excluded from the mean (the old version averaged their zero loss in, diluting it).
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = -1,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none", ignore_index=self.ignore_index,
                             label_smoothing=self.label_smoothing)
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss[valid].mean()
