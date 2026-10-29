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
    """Return the cut-off the old code actually applied to data sampled at ``fs``.

    The old code assumed 250 Hz, so its nominal 49 Hz filter cut at 49 * fs / 250 on
    other rates (25.1 Hz at 128 Hz). Use this to reproduce old results exactly.
    """
    return min(nominal_hz, LEGACY_FS / 2.0 - 1.0) * fs / LEGACY_FS


def lowpass_windows(
    x: np.ndarray, fs: float, cutoff_hz: float | None, order: int = 4, chunk_windows: int = 8192
) -> np.ndarray:
    """Zero-phase Butterworth low-pass applied to every window separately (last axis).

    Matches the old ``clean_signal`` (``filtfilt`` per window with default padding), but
    filters in float64 on chunks of ``chunk_windows`` to bound memory.

    Args:
        x: Windows [N, C, T].
        fs: Sample rate in Hz.
        cutoff_hz: Cut-off in Hz; None or <= 0 skips filtering. Capped at Nyquist - 1.
        order: Filter order.
        chunk_windows: Windows filtered per step.

    Returns:
        float32 array of the same shape.
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
    """Estimate per-channel median and robust SD (IQR / 1.349) of a recording.

    Args:
        x: Windows [N, C, T].
        max_samples: Random samples per channel used when the recording is larger.
        seed: Seed for that subsample, so caches are reproducible.

    Returns:
        ``(medians, sds)``, each float64 [C]; SDs are at least 1e-8.
    """
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
    """Normalise a whole recording per channel, as the old pipeline did.

    Steps: low-pass, median/IQR statistics, ``(x - median) / sd``, optional legacy GUI
    scaling, clipping to ``[-clip, clip]``, then low-pass again to round off the edges
    that clipping creates.

    Args:
        x: Windows [N, C, T].
        fs: Sample rate in Hz.
        lowpass_hz: Low-pass cut-off, or None for no filtering.
        clip: Clip limit in robust SDs.
        scale: Legacy GUI display gain (1.0 = off).
        scale_offset: Offset added together with ``scale``.
        max_stat_samples: Samples per channel used for the statistics.
        seed: Seed for the statistics subsample.

    Returns:
        ``(x_norm, stats)``: float32 [N, C, T] and a dict with median, sd and settings.

    Raises:
        ValueError: If ``x`` is not 3-D.
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


def resample_windows(x: np.ndarray, fs_in: float, fs_out: float, win_out: int) -> np.ndarray:
    """Resample contiguous windows [N, C, T_in] to [N, C, win_out].

    The windows are joined into one signal, resampled with a polyphase filter and cut
    again. This avoids the edge artefacts of resampling each window separately. The
    rate ratio is approximated with a denominator of at most 1000; a short tail is
    padded with the last value.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    n, c, _ = x.shape
    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    out = np.empty((n, c, win_out), dtype=np.float32)
    for ch in range(c):
        y = resample_poly(x[:, ch, :].reshape(-1).astype(np.float64), ratio.numerator, ratio.denominator)
        need = n * win_out
        if len(y) < need:
            y = np.pad(y, (0, need - len(y)), mode="edge")
        out[:, ch, :] = y[:need].reshape(n, win_out)
    return out
