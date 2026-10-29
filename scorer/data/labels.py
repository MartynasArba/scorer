"""
Single source of truth for sleep-state codes.

Every file on disk (converted datasets, GUI state files) uses the RAW codes below.
Every model, loss and metric uses CLASS indices 0..n_classes-1 from a LabelScheme.
Windows whose raw code has no class (unlabeled / artifact / unknown) map to IGNORE (-1)
and are excluded from losses and metrics, but are kept as temporal context.

The mapping is FIXED per scheme. It never depends on which labels happen to be present
in a dataset, so train / val / test / inference always agree (this was the root cause of
the old label mixups: each dataset was remapped "to contiguous" independently).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# raw codes (as written by preprocessing, conversions and the GUI)
RAW_UNLABELED = 0
RAW_WAKE = 1
RAW_NREM = 2
RAW_IS = 3
RAW_REM = 4
RAW_NAMES = {0: "Unlabeled", 1: "Wake", 2: "NREM", 3: "IS", 4: "REM"}

IGNORE = -1


@dataclass(frozen=True)
class LabelScheme:
    """A fixed mapping between raw on-disk sleep codes and model class indices.

    A scheme never depends on which labels a dataset happens to contain, so the same
    class index means the same state in training, validation, testing and inference.

    Attributes:
        name: Registry key, e.g. ``"3state"``.
        class_names: Class names in class-index order.
        raw_to_class: Raw code -> class index. Raw codes not listed map to ``IGNORE``.
        class_to_raw: Class index -> raw code written back to disk (GUI state files).
        description: One line for logs and reports.
    """
    name: str
    class_names: tuple[str, ...]
    raw_to_class: dict[int, int]
    class_to_raw: dict[int, int]  # what predictions are written back as (GUI codes)
    description: str = ""

    @property
    def n_classes(self) -> int:
        """Number of classes the models predict."""
        return len(self.class_names)

    def index_of(self, class_name: str) -> int | None:
        """Return the class index for ``class_name``, or None if the scheme has no such class.

        Example: ``THREE_STATE.index_of("REM") == 2``.
        """
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return None

    def to_class(self, raw: np.ndarray, unlabeled_as_wake: bool = False) -> np.ndarray:
        """Convert raw codes to class indices.

        Args:
            raw: Array of raw codes (any integer dtype, any shape).
            unlabeled_as_wake: Treat ``RAW_UNLABELED`` (0) as Wake instead of ignoring it.
                Only for datasets where unscored time is known to be wake.

        Returns:
            int64 array of the same shape; codes without a class become ``IGNORE`` (-1).
        """
        raw = np.asarray(raw)
        lut = np.full(256, IGNORE, dtype=np.int64)
        for r, c in self.raw_to_class.items():
            lut[r] = c
        if unlabeled_as_wake:
            lut[RAW_UNLABELED] = self.raw_to_class[RAW_WAKE]
        out = np.full(raw.shape, IGNORE, dtype=np.int64)
        in_range = (raw >= 0) & (raw < 256)
        out[in_range] = lut[raw[in_range].astype(np.int64)]
        return out

    def to_raw(self, classes: np.ndarray) -> np.ndarray:
        """Convert class indices back to raw codes, e.g. to save predictions for the GUI.

        ``IGNORE`` and any index outside the scheme become ``RAW_UNLABELED`` (0).
        """
        classes = np.asarray(classes)
        out = np.full(classes.shape, RAW_UNLABELED, dtype=np.int64)
        for c, r in self.class_to_raw.items():
            out[classes == c] = r
        return out

    def to_dict(self) -> dict:
        """Return a JSON-safe description, stored in checkpoint metadata."""
        return {
            "name": self.name,
            "class_names": list(self.class_names),
            "raw_to_class": {str(k): v for k, v in self.raw_to_class.items()},
            "class_to_raw": {str(k): v for k, v in self.class_to_raw.items()},
        }


THREE_STATE = LabelScheme(
    name="3state",
    class_names=("Wake", "NREM", "REM"),
    raw_to_class={RAW_WAKE: 0, RAW_NREM: 1, RAW_IS: 1, RAW_REM: 2},  # IS merged into NREM
    class_to_raw={0: RAW_WAKE, 1: RAW_NREM, 2: RAW_REM},
    description="Wake / NREM (+IS) / REM; unlabeled ignored",
)

FOUR_STATE = LabelScheme(
    name="4state",
    class_names=("Wake", "NREM", "IS", "REM"),
    raw_to_class={RAW_WAKE: 0, RAW_NREM: 1, RAW_IS: 2, RAW_REM: 3},
    class_to_raw={0: RAW_WAKE, 1: RAW_NREM, 2: RAW_IS, 3: RAW_REM},
    description="Wake / NREM / IS / REM; unlabeled ignored",
)

SCHEMES: dict[str, LabelScheme] = {s.name: s for s in (THREE_STATE, FOUR_STATE)}


def get_scheme(name: str) -> LabelScheme:
    """Look up a registered scheme by name.

    Raises:
        KeyError: If ``name`` is not one of ``SCHEMES`` ("3state", "4state").
    """
    if name not in SCHEMES:
        raise KeyError(f"unknown label scheme {name!r}; options: {sorted(SCHEMES)}")
    return SCHEMES[name]


def scheme_from_meta(meta: dict) -> LabelScheme:
    """Rebuild the label scheme recorded in checkpoint metadata.

    Raises:
        ValueError: If the stored class names differ from the registered scheme, which
            would mean the checkpoint was trained with a different class order.
    """
    name = meta.get("label_scheme", THREE_STATE.name)
    scheme = get_scheme(name)
    names = meta.get("class_names")
    if names is not None and tuple(names) != scheme.class_names:
        raise ValueError(
            f"checkpoint class names {names} do not match scheme {scheme.name} {scheme.class_names}"
        )
    return scheme


def raw_counts(raw: np.ndarray) -> dict[int, int]:
    """Count windows per raw code, e.g. ``{1: 812, 2: 1430, 4: 210}``."""
    vals, counts = np.unique(np.asarray(raw), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def class_counts(classes: np.ndarray, scheme: LabelScheme) -> dict[str, int]:
    """Count windows per class name for ``scheme``, plus an 'ignored' entry for ``IGNORE``."""
    classes = np.asarray(classes)
    out = {name: int((classes == i).sum()) for i, name in enumerate(scheme.class_names)}
    out["ignored"] = int((classes == IGNORE).sum())
    return out
