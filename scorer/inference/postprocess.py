"""
Temporal post-processing of predictions, always applied PER RECORDING
(the old validation applied it across concatenated sequences from different recordings).

Class indices come from the label scheme (no more '3_class' / 'gui' / '5_class' mapping strings).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from scorer.data.labels import LabelScheme


def smooth_probs(probs: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth probabilities [N, C] along time with a Gaussian of ``sigma`` windows (0 or None: unchanged)."""
    if not sigma or sigma <= 0 or len(probs) < 2:
        return probs
    return gaussian_filter1d(probs, sigma=sigma, axis=0, mode="nearest")


def _single_window_cleanup(states: np.ndarray, wake: int, passes: int = 2) -> np.ndarray:
    """Replace isolated single windows by the state of their two (equal) neighbours.

    Wake windows are never replaced, so single-window micro-arousals survive. Runs
    ``passes`` times.
    """
    s = states.copy()
    for _ in range(passes):
        for i in range(1, len(s) - 1):
            if s[i] != wake and s[i - 1] == s[i + 1] and s[i] != s[i - 1]:
                s[i] = s[i - 1]
    return s


def heuristics_full(states: np.ndarray, scheme: LabelScheme) -> np.ndarray:
    """Apply the old ``apply_heuristics`` rules to a class sequence.

    1. Wake -> REM is not allowed: a REM window right after Wake becomes Wake. Because
       the rule runs left to right, a whole REM run that starts after Wake becomes Wake.
    2. Single-window cleanup, twice (see ``_single_window_cleanup``).
    """
    wake, rem = scheme.index_of("Wake"), scheme.index_of("REM")
    s = np.asarray(states).copy()
    if len(s) < 3:
        return s
    for i in range(1, len(s)):
        if s[i] == rem and s[i - 1] == wake:
            s[i] = wake
    return _single_window_cleanup(s, wake)


def heuristics_lite(states: np.ndarray, scheme: LabelScheme) -> np.ndarray:
    """Apply only the single-window cleanup (port of ``apply_heuristics_lite``)."""
    s = np.asarray(states)
    if len(s) < 3:
        return s.copy()
    return _single_window_cleanup(s, scheme.index_of("Wake"))


def predict_classes(probs: np.ndarray, bounds: list[tuple[int, int]], scheme: LabelScheme,
                    smooth_sigma: float = 0.0, heuristics: str = "none") -> np.ndarray:
    """Turn probabilities into classes, recording by recording.

    Args:
        probs: Probabilities [N, C].
        bounds: ``(start, end)`` row range of every recording; smoothing and
            heuristics never cross these boundaries.
        scheme: Label scheme (to find Wake and REM).
        smooth_sigma: Gaussian smoothing in windows (0 = off).
        heuristics: "none", "lite" or "full".

    Returns:
        int64 classes [N].

    Raises:
        ValueError: For an unknown ``heuristics`` value.
    """
    if heuristics not in ("none", "lite", "full"):
        raise ValueError(f"heuristics must be none|lite|full, got {heuristics!r}")
    out = np.empty(len(probs), dtype=np.int64)
    for a, b in bounds:
        p = smooth_probs(probs[a:b], smooth_sigma)
        c = p.argmax(1)
        if heuristics == "full":
            c = heuristics_full(c, scheme)
        elif heuristics == "lite":
            c = heuristics_lite(c, scheme)
        out[a:b] = c
    return out
