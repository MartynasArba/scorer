"""
Agreement between two state files (Qt-free). Replaces the report tab's call
compare_scorers(a, b, num_classes=max(len(unique(a)), len(unique(b)))): that passed a COUNT of
states, so with codes {1, 2, 4} 'num_classes=3' covered 0..2 and REM (4) was never compared.
Here the confusion matrix runs over the actual raw codes present in either file.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from scorer.data.labels import RAW_NAMES


def compare_states(a: np.ndarray, b: np.ndarray, ignore_unlabeled: bool = True, merge_is: bool = False) -> dict:
    """Compare two state sequences window by window.

    Args:
        a, b: Raw state codes. If lengths differ, the first ``min`` windows are compared
            and a note says so.
        ignore_unlabeled: Skip windows that either side left unlabeled (code 0).
        merge_is: Count IS (3) as NREM (2) on both sides.

    Returns:
        A dict with "windows_compared", "windows_excluded", "note", "codes", "names",
        "accuracy", "kappa", "confusion" (rows A, columns B) and "per_state" (window
        counts, agreement each way and Dice per state); "error" instead when nothing is
        left to compare.
    """
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    note = ""
    if len(a) != len(b):
        n = min(len(a), len(b))
        note = f"lengths differ ({len(a)} vs {len(b)}); compared the first {n} windows"
        a, b = a[:n], b[:n]
    if merge_is:
        a, b = np.where(a == 3, 2, a), np.where(b == 3, 2, b)
    mask = np.ones(len(a), dtype=bool)
    if ignore_unlabeled:
        mask = (a != 0) & (b != 0)
    a_m, b_m = a[mask], b[mask]
    codes = sorted(set(np.unique(a_m)) | set(np.unique(b_m)))
    out = {"windows_compared": int(mask.sum()), "windows_excluded": int((~mask).sum()), "note": note,
           "codes": [int(c) for c in codes], "names": [RAW_NAMES.get(int(c), str(c)) for c in codes]}
    if not len(a_m):
        out["error"] = "no windows to compare"
        return out
    cm = (confusion_matrix(a_m, b_m, labels=codes) if len(codes) > 1
          else np.array([[int((a_m == b_m).sum())]]))
    out["accuracy"] = float((a_m == b_m).mean())
    out["kappa"] = float(cohen_kappa_score(a_m, b_m, labels=codes)) if len(codes) > 1 else 1.0
    out["confusion"] = cm.tolist()  # rows = A, columns = B
    per = {}
    for i, c in enumerate(codes):
        na, nb = int(cm[i].sum()), int(cm[:, i].sum())
        both = int(cm[i, i])
        per[RAW_NAMES.get(int(c), str(c))] = {
            "A_windows": na, "B_windows": nb,
            "B_agrees_with_A": both / na if na else None,  # recall of B against A
            "A_agrees_with_B": both / nb if nb else None,
            "dice": 2 * both / (na + nb) if na + nb else None,
        }
    out["per_state"] = per
    return out


def format_agreement(res: dict, name_a: str = "A", name_b: str = "B") -> str:
    """Format a ``compare_states`` result as a fixed-width text report for the agreement dialog."""
    if "error" in res:
        return res["error"]
    lines = [f"A = {name_a}", f"B = {name_b}", ""]
    if res["note"]:
        lines += [f"NOTE: {res['note']}", ""]
    lines.append(f"windows compared: {res['windows_compared']} (excluded: {res['windows_excluded']})")
    lines.append(f"accuracy: {res['accuracy']:.4f}    Cohen's kappa: {res['kappa']:.4f}")
    lines.append("")
    w = max(8, max(len(n) for n in res["names"]) + 2)
    lines.append("confusion (rows = A, columns = B)")
    lines.append(" " * w + "".join(f"{n:>{w}}" for n in res["names"]))
    for n, row in zip(res["names"], res["confusion"]):
        lines.append(f"{n:<{w}}" + "".join(f"{v:>{w}}" for v in row))
    lines.append("")
    lines.append(f"{'state':<{w}}{'A n':>9}{'B n':>9}{'B~A':>8}{'A~B':>8}{'dice':>8}")
    fmt = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"  # noqa: E731
    for n, d in res["per_state"].items():
        lines.append(f"{n:<{w}}{d['A_windows']:>9}{d['B_windows']:>9}{fmt(d['B_agrees_with_A'])}"
                     f"{fmt(d['A_agrees_with_B'])}{fmt(d['dice'])}")
    return "\n".join(lines)
