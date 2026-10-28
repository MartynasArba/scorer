"""
Per-recording signal normalization (numpy / scipy, CPU).

Replaces loaders.robust_normalize / clean_signal. Differences:
  * the sampling rate is a parameter. The old code always assumed 250 Hz, so on 128 Hz
    data its "49 Hz" low-pass actually cut at ~25 Hz. To reproduce that exactly, set
    lowpass_hz = 49 * fs / 250 (25.088 at 128 Hz).
  * filtering runs in float64 on bounded chunks of windows and returns float32
    (the old version materialised the whole tensor in float64 several times).
  * statistics subsampling uses a seeded generator, so caches are reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

LEGACY_FS = 250.0


def legacy_equivalent_lowpass(fs: float, nominal_hz: float = 49.0) -> float:
    """cut-off the old code effectively applied when fed data sampled at `fs`"""
    return min(nominal_hz, LEGACY_FS / 2.0 - 1.0) * fs / LEGACY_FS


def lowpass_windows(
    x: np.ndarray, fs: float, cutoff_hz: float | None, order: int = 4, chunk_windows: int = 8192
) -> np.ndarray:
    """
    zero-phase Butterworth low-pass applied independently to every window (last axis),
    exactly like the old clean_signal (filtfilt per window, default odd padding).
    x: [N, C, T] float. returns float32 array of the same shape.
    """
    x = np.asarray(x)
    if cutoff_hz is None or cutoff_hz <= 0:
        return x.astype(np.float32, copy=False)
    nyq = fs / 2.0
    cutoff = min(float(cutoff_hz), nyq - 1.0)
    b, a = butter(order, cutoff / nyq, btype="low")
    out = np.empty(x.shape, dtype=np.float32)
    for s in range(0, x.shape[0], chunk_windows):
        out[s : s + chunk_windows] = filtfilt(b, a, x[s : s + chunk_windows].astype(np.float64), axis=-1)
    return out


def robust_stats(
    x: np.ndarray, max_samples: int = 100_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """per-channel median and IQR-based sd (iqr / 1.349) from a random subsample. x: [N, C, T]"""
    n, c, t = x.shape
    total = n * t
    rng = np.random.default_rng(seed)
    medians = np.empty(c, dtype=np.float64)
    sds = np.empty(c, dtype=np.float64)
    for ch in range(c):
        if total > max_samples:
            flat_idx = rng.integers(0, total, size=max_samples)
            vals = x[flat_idx // t, ch, flat_idx % t].astype(np.float64)
        else:
            vals = x[:, ch, :].reshape(-1).astype(np.float64)
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        medians[ch] = med
        sds[ch] = max((q3 - q1) / 1.349, 1e-8)
    return medians, sds


def robust_normalize(
    x: np.ndarray,
    fs: float,
    lowpass_hz: float | None = 49.0,
    clip: float = 4.0,
    scale: float = 1.0,
    scale_offset: float = 0.7,
    max_stat_samples: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    median/IQR normalisation per channel over a whole recording, then clipping.
    Mirrors the old pipeline: low-pass -> stats -> (x - median) / sd -> optional GUI scale
    -> clip -> low-pass again (removes the square edges created by clipping).
    returns (float32 [N, C, T], stats dict)
    """
    if x.ndim != 3:
        raise ValueError(f"expected [windows, channels, time], got shape {x.shape}")
    x = lowpass_windows(x, fs, lowpass_hz)
    med, sd = robust_stats(x, max_stat_samples, seed)
    x -= med[None, :, None].astype(np.float32)
    x /= sd[None, :, None].astype(np.float32)
    if scale != 1.0:  # legacy GUI display option
        x *= scale
        x += scale_offset
    np.clip(x, -clip, clip, out=x)
    x = lowpass_windows(x, fs, lowpass_hz)
    stats = {"median": med.tolist(), "sd": sd.tolist(), "lowpass_hz": lowpass_hz, "clip": clip, "fs": fs}
    return x, stats
