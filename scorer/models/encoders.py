"""
Window encoder used by the 128 Hz pipeline (single-channel dual-stream CNN).

State-dict compatible with the old scorer.models.sleep_cnn.SCDSSleepCNN, so existing
weights load unchanged. Changes:
  * LazyLinear -> Linear(256 * 16, 256): the input size is fixed by AdaptiveAvgPool1d(16),
    so no dummy forward pass is needed before loading weights or building optimizers;
  * win_len is explicit; the Hann buffer is no longer re-registered during forward()
    (that silently changed the state_dict when a different window length was passed);
  * `embedding_dim` attribute for downstream heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scorer.models.grl import grad_reverse


class SCDSSleepCNN(nn.Module):
    """Single-channel dual-stream CNN that turns a 4-s window into a 512-d embedding.

    The time stream (4 conv blocks, wide first kernel for delta) gives 256 features; the
    frequency stream (3 conv blocks on the log-power spectrum) gives another 256. The
    two are concatenated. The state dict matches the old ``sleep_cnn.SCDSSleepCNN``, so
    old weights load unchanged.

    Args:
        num_classes: Outputs of ``state_classifier`` (used by adversarial training only).
        sampling_rate: Sample rate in Hz; sets the default window length (4 s).
        win_len: Window length in samples; defaults to ``round(4 * sampling_rate)``.
        in_channel: Which input channel to encode when batches carry several.

    Attributes:
        embedding_dim: Size of the output embedding (512).
    """
    embedding_dim = 512

    def __init__(self, num_classes: int = 3, sampling_rate: float = 128, win_len: int | None = None,
                 in_channel: int = 0):
        """Build both streams and the adversarial heads; see the class docstring for arguments."""
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.window_size = int(win_len if win_len is not None else round(sampling_rate * 4))
        self.in_channel = in_channel

        # time stream: wide first kernel to capture delta
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=32), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        # frequency stream on the log-power spectrum
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.freq_flatten = nn.Sequential(
            nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(256 * 16, 256), nn.ReLU(),
        )
        # heads used only by adversarial training (kept here for checkpoint compatibility)
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, 1),
        )
        self.state_classifier = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
        self.register_buffer("hann_window", torch.hann_window(self.window_size))

    def _freqtransform(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-power spectrum ``log1p(|rfft(x * hann)|^2)`` without the DC bin.

        The stored Hann window is used when the length matches; otherwise a temporary one
        is made, so the state dict never changes during ``forward``.
        """
        t = x.shape[-1]
        window = self.hann_window if self.hann_window.shape[0] == t else torch.hann_window(t, device=x.device, dtype=x.dtype)
        spec = torch.fft.rfft(x * window, dim=-1, norm="ortho")
        power = spec.real.square() + spec.imag.square()
        return torch.log1p(power[..., 1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows ``x`` [B, C, T] into embeddings [B, 512]."""
        x = x[:, self.in_channel : self.in_channel + 1, :]
        feat_time = self.time_conv(x).squeeze(-1)
        feat_freq = self.freq_flatten(self.freq_conv(self._freqtransform(x)))
        return torch.cat([feat_time, feat_freq], dim=1)

    get_features = forward

    def forward_domain(self, x: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` and predict its domain through gradient reversal.

        Args:
            x: Windows [B, C, T].
            alpha: Strength of the reversed gradient reaching the encoder.

        Returns:
            ``(embeddings [B, 512], domain logits [B, 1])``.
        """
        features = self.forward(x)
        return features, self.domain_classifier(grad_reverse(features, alpha))
