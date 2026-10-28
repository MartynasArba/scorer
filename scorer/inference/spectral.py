"""
Per-window band powers (port of scoring.extract_spectral_features / welch_pytorch_batched /
compute_all_spectral_features). Fixes: the sample rate is a parameter everywhere (Welch segment
= 1 s at the real rate), CUDA is optional, and output folders are arguments.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from scorer.data.recordings import discover_recordings, load_recording
from scorer.data.signal import robust_normalize

log = logging.getLogger(__name__)

BANDS = {"delta": (0.5, 4.0), "sigma": (10.0, 15.0), "theta": (5.0, 9.0), "gamma": (30.0, 49.0), "total": (0.5, 49.0)}


def welch_batched(windows, fs: float, nperseg: int | None = None, device: str = "cpu"):
    """windows [N, T] -> (freqs [F], psd [N, F]); mean |STFT|^2 with 50 % overlap (torch, CPU or CUDA)"""
    import torch

    x = torch.as_tensor(np.asarray(windows), dtype=torch.float32, device=device)
    nperseg = nperseg or int(fs)
    win = torch.hann_window(nperseg, device=device)
    stft = torch.stft(x, n_fft=nperseg, hop_length=nperseg // 2, win_length=nperseg, window=win, return_complex=True,
                      center=False)
    psd = (stft.abs() ** 2).mean(dim=-1)
    freqs = torch.fft.rfftfreq(nperseg, 1 / fs)
    return freqs.numpy(), psd.cpu().numpy()


def band_powers(windows: np.ndarray, fs: float, bands: dict = BANDS, device: str = "cpu") -> dict[str, np.ndarray]:
    freqs, psd = welch_batched(windows, fs, device=device)
    hi = fs / 2
    return {name: psd[:, (freqs >= lo) & (freqs <= min(up, hi))].sum(1) for name, (lo, up) in bands.items()}


def extract_spectral_features(folder, fs: float, channel: int = 0, lowpass_hz: float | None = 49.0,
                              device: str = "cpu") -> np.ndarray:
    """one recording folder -> array [n_bands, N] in BANDS order (as the old function returned)"""
    recs = discover_recordings(folder)
    x, _ = load_recording(recs[0], [channel])
    x, _ = robust_normalize(x, fs, lowpass_hz)
    bp = band_powers(x[:, 0], fs, device=device)
    return np.stack([bp[k] for k in BANDS])


def compute_all_spectral_features(meta_csv, out_dir, fs: float, lowpass_hz: float | None = 49.0, device: str = "cpu"):
    """CSV with proj_dir + channel_selected ('f'/'p') -> <out_dir>/<recording>_spectral_features.npy"""
    import pandas as pd

    from scorer.inference.scoring import _legacy_folder_key

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pd.read_csv(meta_csv).iterrows():
        proj = Path(row["proj_dir"])
        ch = {"f": 0, "p": 1}.get(str(row["channel_selected"]).strip().lower())
        if ch is None:
            log.error("%s: bad channel_selected %r", proj, row["channel_selected"])
            continue
        folders = sorted((p for p in (proj / "processed").iterdir() if p.is_dir()), key=_legacy_folder_key)
        feats = extract_spectral_features(folders[-1], fs, ch, lowpass_hz, device)
        np.save(out_dir / f"{proj.name}_spectral_features.npy", feats)
        log.info("%s: %s", proj.name, feats.shape)
