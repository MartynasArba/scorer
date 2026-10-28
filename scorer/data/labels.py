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

from dataclasses import dataclass, field

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
    name: str
    class_names: tuple[str, ...]
    raw_to_class: dict[int, int]
    class_to_raw: dict[int, int]  # what predictions are written back as (GUI codes)
    description: str = ""

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def index_of(self, class_name: str) -> int | None:
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return None

    def to_class(self, raw: np.ndarray, unlabeled_as_wake: bool = False) -> np.ndarray:
        """raw codes -> class indices (int64), unknown codes -> IGNORE"""
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
        """class indices -> raw/GUI codes (IGNORE -> RAW_UNLABELED)"""
        classes = np.asarray(classes)
        out = np.full(classes.shape, RAW_UNLABELED, dtype=np.int64)
        for c, r in self.class_to_raw.items():
            out[classes == c] = r
        return out

    def to_dict(self) -> dict:
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
    if name not in SCHEMES:
        raise KeyError(f"unknown label scheme {name!r}; options: {sorted(SCHEMES)}")
    return SCHEMES[name]


def scheme_from_meta(meta: dict) -> LabelScheme:
    """rebuild a scheme from checkpoint metadata and verify it matches the registered one"""
    name = meta.get("label_scheme", THREE_STATE.name)
    scheme = get_scheme(name)
    names = meta.get("class_names")
    if names is not None and tuple(names) != scheme.class_names:
        raise ValueError(
            f"checkpoint class names {names} do not match scheme {scheme.name} {scheme.class_names}"
        )
    return scheme


def raw_counts(raw: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(np.asarray(raw), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def class_counts(classes: np.ndarray, scheme: LabelScheme) -> dict[str, int]:
    classes = np.asarray(classes)
    out = {name: int((classes == i).sum()) for i, name in enumerate(scheme.class_names)}
    out["ignored"] = int((classes == IGNORE).sum())
    return out
