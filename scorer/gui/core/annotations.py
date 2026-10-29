"""
Editable state track for manual scoring (Qt-free): labels, undo, unsaved-change tracking,
navigation by state change or by low model confidence, and state-file IO with length checks.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scorer.data.labels import RAW_NAMES


def load_states_file(path: str | Path) -> np.ndarray:
    """Load a pickled state file (list, array or tensor of raw codes) as int64 [N]; None entries become 0."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "detach"):  # a torch tensor
        obj = obj.detach().cpu().numpy()
    arr = np.asarray([0 if v is None else v for v in obj] if isinstance(obj, (list, tuple)) else obj)
    return arr.astype(np.int64).reshape(-1)


def confidence_path_for(states_path: str | Path) -> Path:
    """Return the confidence file that belongs to a state file: ``<stem>_confidence.pkl``."""
    p = Path(states_path)
    return p.with_name(p.stem + "_confidence.pkl")


def load_confidence_for(states_path: str | Path) -> np.ndarray | None:
    """Load the confidence array saved next to a state file, or None if there is none."""
    p = confidence_path_for(states_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float32).reshape(-1)


@dataclass
class Edit:
    """One undo step: the labels a range had before it was changed.

    Attributes:
        start: First window of the range.
        old: Previous labels of the range.
    """
    start: int
    old: np.ndarray


@dataclass
class StateTrack:
    """The labels being edited in the manual-scoring tab, with undo and change tracking.

    Attributes:
        states: Raw code per window (int64).
        confidence: Model confidence per window, or None for manual-only scoring.
        dirty: True when there are changes not yet saved.
        version: Increases on every change; autosave compares it to skip unchanged saves.
        max_undo: Oldest undo steps are dropped beyond this count.
    """
    states: np.ndarray
    confidence: np.ndarray | None = None
    dirty: bool = False
    version: int = 0  # increases on every change (autosave compares it)
    max_undo: int = 200
    _undo: list[Edit] = field(default_factory=list)

    @classmethod
    def empty(cls, n: int) -> "StateTrack":
        """Return a track of ``n`` unlabeled windows."""
        return cls(np.zeros(n, dtype=np.int64))

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.states)

    # ---- editing ------------------------------------------------------------------------------
    def set_range(self, i0: int, i1: int, value: int) -> bool:
        """Label windows ``i0`` to ``i1`` (inclusive, any order, clipped) with ``value``; True if anything changed."""
        if len(self.states) == 0:
            return False
        i0, i1 = sorted((int(i0), int(i1)))
        i0, i1 = max(0, i0), min(len(self.states) - 1, i1)
        if i1 < i0:
            return False
        old = self.states[i0:i1 + 1].copy()
        if np.all(old == value):
            return False
        self._undo.append(Edit(i0, old))
        if len(self._undo) > self.max_undo:
            self._undo.pop(0)
        self.states[i0:i1 + 1] = value
        self.dirty = True
        self.version += 1
        return True

    def undo(self) -> tuple[int, int] | None:
        """Revert the last edit; return the restored ``(start, end)`` range, or None if nothing to undo."""
        if not self._undo:
            return None
        e = self._undo.pop()
        self.states[e.start:e.start + len(e.old)] = e.old
        self.dirty = True
        self.version += 1
        return e.start, e.start + len(e.old) - 1

    @property
    def can_undo(self) -> bool:
        """True if there is an edit to undo."""
        return bool(self._undo)

    def replace(self, states: np.ndarray, confidence: np.ndarray | None = None) -> None:
        """Replace all labels (and confidence) at once, e.g. after loading a file; clears undo and the unsaved flag."""
        self.states = np.asarray(states, dtype=np.int64).copy()
        self.confidence = confidence
        self._undo.clear()
        self.dirty = False
        self.version += 1

    def mark_saved(self) -> None:
        """Clear the unsaved flag after a successful save."""
        self.dirty = False

    # ---- navigation ---------------------------------------------------------------------------
    def segment_starts(self) -> np.ndarray:
        """Return the first window of every run of equal labels (always starts with 0)."""
        s = self.states
        if len(s) == 0:
            return np.array([0])
        return np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1))

    def next_change(self, idx: int) -> int | None:
        """Return the start of the next run after ``idx``, or None."""
        starts = self.segment_starts()
        after = starts[starts > idx]
        return int(after[0]) if after.size else None

    def prev_change(self, idx: int) -> int | None:
        """Return the start of the nearest run before ``idx``, or None."""
        starts = self.segment_starts()
        before = starts[starts < idx]
        return int(before[-1]) if before.size else None

    def next_uncertain(self, idx: int, threshold: float, skip: int = 1) -> int | None:
        """Return the first window at or after ``idx + skip`` whose confidence is below ``threshold``.

        None if there is no such window or no confidence.
        """
        if self.confidence is None:
            return None
        n = min(len(self.confidence), len(self.states))
        low = np.flatnonzero(self.confidence[:n] < threshold)
        after = low[low >= idx + skip]
        return int(after[0]) if after.size else None

    def next_unlabeled(self, idx: int) -> int | None:
        """Return the next unlabeled window (code 0) after ``idx``, or None."""
        hits = np.flatnonzero(self.states == 0)
        after = hits[hits > idx]
        return int(after[0]) if after.size else None

    # ---- summaries ----------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Return the number of windows per state name."""
        vals, cnt = np.unique(self.states, return_counts=True)
        return {RAW_NAMES.get(int(v), str(int(v))): int(c) for v, c in zip(vals, cnt)}


def fit_states_to(states: np.ndarray, n: int) -> tuple[np.ndarray, str]:
    """Trim or pad (with 0 = unlabeled) a state array to ``n`` windows.

    Returns:
        ``(array, message)``; the message explains a change and is "" when none was needed.
    """
    states = np.asarray(states, dtype=np.int64)
    if len(states) == n:
        return states, ""
    if len(states) > n:
        return states[:n].copy(), f"{len(states) - n} trailing windows dropped ({len(states)} states, {n} windows)"
    return (np.concatenate([states, np.zeros(n - len(states), dtype=np.int64)]),
            f"{n - len(states)} missing windows set to unlabeled ({len(states)} states, {n} windows)")


def fit_array(arr: np.ndarray, n: int, fill=0) -> np.ndarray:
    """Trim or pad any 1-D array to ``n`` entries with ``fill``, keeping its dtype (float confidence stays float)."""
    arr = np.asarray(arr)
    if len(arr) >= n:
        return arr[:n].copy()
    return np.concatenate([arr, np.full(n - len(arr), fill, dtype=np.result_type(arr, np.asarray(fill)))])
