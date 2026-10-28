"""
GPU batch augmentations for contrastive pretraining (port of pretraining.batch_augment).

Fix: every frequency is specified in Hz and converted with the REAL sample rate.
The old function defaulted to sample_rate=250 and was never given the rate, so at 128 Hz
its "theta protection" covered 3-4.5 Hz instead of 6-9 Hz and the minimum low-pass
cut-off was ~15 Hz instead of 30 Hz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    noise_p: float = 0.7
    noise_sd: float = 0.01
    scale_p: float = 0.7
    scale_log10_range: float = 1.0  # scales drawn log-uniformly in 10^[-r, r]
    dc_p: float = 0.5
    dc_range: float = 0.7  # shift drawn uniformly in [-dc_range, dc_range]
    freq_mask_p: float = 0.3
    protect_band_hz: tuple[float, float] = (6.0, 9.0)  # theta is never masked
    min_lowpass_hz: float = 30.0
    bandstop_width_hz: float = 10.0
    shift_p: float = 0.5
    max_shift_frac: float = 0.02
    channel_dropout_p: float = 0.3  # only applies with >1 channel


def batch_augment(x: torch.Tensor, sample_rate: float, cfg: AugmentConfig | None = None) -> torch.Tensor:
    """x [B, C, T] -> augmented copy (independent random draws per sample)"""
    cfg = cfg or AugmentConfig()
    x = x.clone()
    bsz, n_ch, t = x.shape
    dev = x.device

    def coin(p):
        return (torch.rand(bsz, 1, 1, device=dev) < p).float()

    x = x + coin(cfg.noise_p) * torch.randn_like(x) * cfg.noise_sd

    apply_scale = coin(cfg.scale_p)
    exps = (torch.rand(bsz, n_ch, 1, device=dev) * 2.0 - 1.0) * cfg.scale_log10_range
    x = x * ((1 - apply_scale) + apply_scale * torch.exp(exps * math.log(10.0)))

    shifts = torch.rand(bsz, n_ch, 1, device=dev) * 2 * cfg.dc_range - cfg.dc_range
    x = x + coin(cfg.dc_p) * shifts

    apply_mask = torch.rand(bsz, 1, 1, device=dev) < cfg.freq_mask_p
    if apply_mask.any():
        spec = torch.fft.rfft(x, dim=-1)
        n_bins = spec.shape[-1]
        hz_per_bin = sample_rate / t
        to_bin = lambda hz: int(round(hz / hz_per_bin))  # noqa: E731
        freq_idx = torch.arange(n_bins, device=dev).view(1, 1, n_bins)
        th0, th1 = to_bin(cfg.protect_band_hz[0]), to_bin(cfg.protect_band_hz[1])
        is_lowpass = torch.rand(bsz, 1, 1, device=dev) < 0.5
        min_cut = min(to_bin(cfg.min_lowpass_hz), n_bins - 1)
        cutoffs = torch.randint(min_cut, n_bins, (bsz, 1, 1), device=dev)
        lp_keep = freq_idx < cutoffs
        width = max(1, to_bin(cfg.bandstop_width_hz))
        bs_lo = th1 + 5
        bs_hi = max(bs_lo + 1, n_bins - width)
        bs_start = torch.randint(bs_lo, bs_hi, (bsz, 1, 1), device=dev)
        bs_keep = ~((freq_idx >= bs_start) & (freq_idx < bs_start + width))
        keep = torch.where(is_lowpass, lp_keep, bs_keep) | ((freq_idx >= th0) & (freq_idx < th1))
        keep = torch.where(apply_mask, keep, torch.ones_like(keep))
        x = torch.fft.irfft(spec * keep.float(), n=t, dim=-1)

    max_shift = int(t * cfg.max_shift_frac)
    if max_shift > 0:
        do_shift = torch.rand(bsz, device=dev) < cfg.shift_p
        s = torch.randint(-max_shift, max_shift + 1, (bsz,), device=dev) * do_shift.long()
        idx = (torch.arange(t, device=dev).view(1, 1, t) - s.view(bsz, 1, 1)) % t
        x = torch.gather(x, 2, idx.expand(bsz, n_ch, t))

    if n_ch > 1:
        drop = coin(cfg.channel_dropout_p)
        noise = torch.randn(bsz, 1, t, device=dev) * (torch.rand(bsz, 1, 1, device=dev) * 1e-4)
        x[:, 1:2, :] = (1 - drop) * x[:, 1:2, :] + drop * noise
    return x
