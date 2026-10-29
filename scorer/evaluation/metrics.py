"""
Metrics, reports and figures. Everything is written to disk (PNG / JSON / CSV) with the Agg
backend: no plt.show() blocking an unattended pipeline.

All metrics use labels=range(n_classes) explicitly and drop IGNORE windows, so a class that is
absent from a dataset cannot shift rows/columns or names (classification_report previously
inferred labels from the data and the confusion-matrix helper was called with arguments swapped).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from scorer.data.labels import IGNORE, LabelScheme

log = logging.getLogger(__name__)


def _plt():
    """Import pyplot with the non-interactive Agg backend (no windows, safe without a display)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scheme: LabelScheme) -> dict:
    """Score predictions against true labels; ``IGNORE`` windows are left out.

    All metrics use ``labels=range(n_classes)``, so a class missing from the data still
    has its own row and column.

    Returns:
        A dict with "n_scored", "n_ignored", "accuracy", "macro_f1", "kappa",
        "per_class" (precision, recall, f1, support), "confusion" (rows true, columns
        predicted), "class_names" and "predicted_distribution". Only the two counts if
        nothing is scored.

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} differ")
    m = y_true != IGNORE
    yt, yp = y_true[m], y_pred[m]
    labels = list(range(scheme.n_classes))
    if len(yt) == 0:
        return {"n_scored": 0, "n_ignored": int((~m).sum())}
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {
        "n_scored": int(len(yt)),
        "n_ignored": int((~m).sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(yt, yp, labels=labels)) if len(np.unique(np.r_[yt, yp])) > 1 else 0.0,
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(scheme.class_names)},
        "confusion": confusion_matrix(yt, yp, labels=labels).tolist(),  # rows = true, cols = predicted
        "class_names": list(scheme.class_names),
        "predicted_distribution": {n: int((yp == i).sum()) for i, n in enumerate(scheme.class_names)},
    }


def text_report(y_true: np.ndarray, y_pred: np.ndarray, scheme: LabelScheme) -> str:
    """Return sklearn's classification report for the labeled windows, with the scheme's class names."""
    m = np.asarray(y_true) != IGNORE
    return classification_report(np.asarray(y_true)[m], np.asarray(y_pred)[m], labels=list(range(scheme.n_classes)),
                                 target_names=list(scheme.class_names), digits=4, zero_division=0)


def per_recording_metrics(y_true: np.ndarray, y_pred: np.ndarray, spans) -> list[dict]:
    """Return accuracy and kappa per recording.

    Args:
        y_true, y_pred: Class indices for the whole store.
        spans: ``RecordingSpan`` list giving each recording's rows.

    Returns:
        One dict per recording; NaN where a recording has no labeled windows.
    """
    rows = []
    for sp in spans:
        yt, yp = y_true[sp.start:sp.end], y_pred[sp.start:sp.end]
        m = yt != IGNORE
        n = int(m.sum())
        acc = float((yt[m] == yp[m]).mean()) if n else float("nan")
        kappa = float(cohen_kappa_score(yt[m], yp[m])) if n and len(np.unique(np.r_[yt[m], yp[m]])) > 1 else float("nan")
        rows.append({"dataset": sp.dataset, "recording": sp.name, "windows": sp.end - sp.start, "scored": n,
                     "accuracy": acc, "kappa": kappa})
    return rows


def write_json(obj, path: str | Path) -> None:
    """Write ``obj`` as indented JSON, creating parent folders; numpy numbers become floats."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Write a list of dicts as CSV with the first row's keys as header (nothing for an empty list)."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_confusion_png(metrics: dict, path: str | Path, title: str) -> None:
    """Save a row-normalised confusion matrix with counts and the headline metrics in the title."""
    plt = _plt()
    cm = np.asarray(metrics["confusion"], dtype=float)
    names = metrics["class_names"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nacc {metrics['accuracy']:.3f} | macro F1 {metrics['macro_f1']:.3f} | kappa {metrics['kappa']:.3f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_hypnogram_png(y_true: np.ndarray, y_pred: np.ndarray, class_names, path: str | Path, title: str,
                       max_windows: int = 900) -> None:
    """Save true and predicted hypnograms of the first ``max_windows`` windows as step plots."""
    plt = _plt()
    n = min(max_windows, len(y_true))
    t = np.arange(n)
    yt = np.where(y_true[:n] == IGNORE, np.nan, y_true[:n]).astype(float)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(t, yt, where="post", color="black", lw=1.2, label="true")
    ax.step(t, y_pred[:n] + 0.08, where="post", color="crimson", lw=1.0, alpha=0.8, label="predicted")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("window (4 s)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_embedding_pca_png(emb: np.ndarray, classes: np.ndarray, class_names, path: str | Path, title: str,
                           sample: int = 3000, seed: int = 0) -> None:
    """Save a 2-D PCA scatter of up to ``sample`` labeled embeddings, coloured by class."""
    from sklearn.decomposition import PCA

    plt = _plt()
    m = np.flatnonzero(classes != IGNORE)
    if len(m) > sample:
        m = np.sort(np.random.default_rng(seed).choice(m, sample, replace=False))
    xy = PCA(n_components=2, random_state=seed).fit_transform(np.asarray(emb[m], dtype=np.float32))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, n in enumerate(class_names):
        sel = classes[m] == i
        ax.scatter(xy[sel, 0], xy[sel, 1], s=6, alpha=0.5, label=n)
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
