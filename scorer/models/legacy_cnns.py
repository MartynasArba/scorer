"""
Earlier multi-channel 250 Hz architectures, ported so they can be trained / evaluated
through the same pipeline (prepare -> embed -> heads) as SCDSSleepCNN.

Fixes vs scorer.models.sleep_cnn:
  * EphysSleepCNN: the projection size (256 * 31) was hard-coded for 1000-sample windows;
    it is now derived from win_len (state-dict compatible at win_len=1000).
  * DualStreamSleepCNN: torch.cat was computed twice (harmless but misleading comment).
  * all models expose get_features() and `embedding_dim`, so they work with the embed stage.
  * the per-batch quantile scaling is kept (it is part of these models' design), but
    torch.quantile has a 16M-element limit: inputs larger than that are subsampled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_QUANTILE_LIMIT = 16_000_000


def _batch_robust_scale(x: torch.Tensor) -> torch.Tensor:
    """Scale ``x`` [B, C, T] per channel by the batch median and IQR.

    Part of the legacy models' design. ``torch.quantile`` accepts at most 16M elements,
    so larger batches are randomly subsampled for the statistics.
    """
    b, c, t = x.shape
    flat = x.transpose(0, 1).reshape(c, -1)
    if flat.shape[1] > _QUANTILE_LIMIT // max(c, 1):
        idx = torch.randint(0, flat.shape[1], (_QUANTILE_LIMIT // max(c, 1),), device=x.device)
        flat = flat[:, idx]
    q = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)  # [3, C]
    med = q[1].view(1, c, 1)
    iqr = (q[2] - q[0]).view(1, c, 1) + 1e-8
    return (x - med) / iqr


class SleepCNN(nn.Module):
    """Oldest model: 4 conv blocks over 7 input channels (ephys + derived), global average pooling.

    Args:
        num_classes: Number of output classes.
        mean_std: Optional ``(mean [C], std [C])`` tensors to standardise inputs with.
        in_channels: Number of input channels.

    Attributes:
        embedding_dim: Size of ``get_features`` output (128).
    """
    embedding_dim = 128

    def __init__(self, num_classes: int, mean_std=None, in_channels: int = 7):
        """Build the layers; see the class docstring."""
        super().__init__()
        self.mean_std = mean_std
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def get_features(self, x):
        """Return 128-d features for windows ``x`` [B, C, T] (used by the embed stage)."""
        if self.mean_std is not None:
            mean, std = self.mean_std
            x = (x - mean[None, :, None]) / std[None, :, None]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = self.pool(F.relu(conv(x)))
        return self.global_pool(x).flatten(1)

    def forward(self, x):
        """Return class logits for windows ``x``."""
        x = self.dropout(self.get_features(x))
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class EphysSleepCNN(nn.Module):
    """Two-channel (ECoG + EMG) CNN with batch-median scaling.

    Args:
        num_classes: Number of output classes.
        win_len: Window length in samples; sets the projection size (``256 * win_len // 32``).
            At 1000 samples the state dict matches the old model.

    Attributes:
        embedding_dim: Size of ``get_features`` output (128).
    """
    embedding_dim = 128

    def __init__(self, num_classes: int, win_len: int = 1000):
        """Build the layers; see the class docstring."""
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.pool5 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.feature_proj = nn.Linear(256 * (win_len // 32), 128)  # 5 pools of 2
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def get_features(self, x):
        """Return 128-d features from the first two channels of ``x`` [B, C, T]."""
        x = _batch_robust_scale(x[:, :2, :])
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(x).flatten(1)
        return F.relu(self.feature_proj(x))

    def forward(self, x):
        """Return class logits for windows ``x``."""
        x = self.dropout(self.get_features(x))
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class DualStreamSleepCNN(nn.Module):
    """Two-channel model with a time stream and an FFT log-power stream.

    Args:
        num_classes: Number of output classes.

    Attributes:
        embedding_dim: Size of ``get_features`` output (128).
    """
    embedding_dim = 128

    def __init__(self, num_classes: int):
        """Build both streams and the classifier."""
        super().__init__()

        def block(cin, cout, k):
            """Return one conv-BatchNorm-ReLU-MaxPool block as a list of layers."""
            return [nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]

        self.time_conv = nn.Sequential(*block(2, 32, 11), *block(32, 64, 7), *block(64, 128, 5), *block(128, 256, 3),
                                       nn.AdaptiveAvgPool1d(1))
        self.freq_conv = nn.Sequential(*block(2, 64, 7), *block(64, 128, 5), *block(128, 256, 3), nn.AdaptiveAvgPool1d(1))
        self.dropout = nn.Dropout(0.5)
        self.feature_proj = nn.Linear(512, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    @staticmethod
    def _freqtransform(x):
        """Return ``log1p`` power of the real FFT of ``x``, DC bin dropped."""
        spec = torch.fft.rfft(x, dim=-1)
        return torch.log1p((spec.real.square() + spec.imag.square())[..., 1:])

    def get_features(self, x):
        """Return 128-d features from the first two channels of ``x`` [B, C, T]."""
        x = x[:, :2, :]
        feat_time = self.time_conv(_batch_robust_scale(x)).squeeze(-1)
        feat_freq = self.freq_conv(self._freqtransform(x)).squeeze(-1)
        return F.relu(self.feature_proj(torch.cat([feat_time, feat_freq], dim=1)))

    def forward(self, x):
        """Return class logits for windows ``x``."""
        out = self.dropout(self.get_features(x))
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)
