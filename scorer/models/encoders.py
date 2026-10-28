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
    embedding_dim = 512

    def __init__(self, num_classes: int = 3, sampling_rate: float = 128, win_len: int | None = None,
                 in_channel: int = 0):
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
        """log-power spectrum of the Hann-windowed signal, DC bin dropped"""
        t = x.shape[-1]
        window = self.hann_window if self.hann_window.shape[0] == t else torch.hann_window(t, device=x.device, dtype=x.dtype)
        spec = torch.fft.rfft(x * window, dim=-1, norm="ortho")
        power = spec.real.square() + spec.imag.square()
        return torch.log1p(power[..., 1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, C, T] -> embedding [B, 512]"""
        x = x[:, self.in_channel : self.in_channel + 1, :]
        feat_time = self.time_conv(x).squeeze(-1)
        feat_freq = self.freq_flatten(self.freq_conv(self._freqtransform(x)))
        return torch.cat([feat_time, feat_freq], dim=1)

    get_features = forward

    def forward_domain(self, x: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward(x)
        return features, self.domain_classifier(grad_reverse(features, alpha))
