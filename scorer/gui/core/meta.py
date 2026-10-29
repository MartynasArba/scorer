"""
Typed access to the shared GUI metadata dict (Qt-free, unit-testable).

The metadata dict is shared by every tab and saved to JSON by scorer.data.storage. Its values
can be None (unset), str (typed into the settings tab) or int/float (written by preprocessing),
and the old widgets mixed `params.get(key, default)` (returns None when the key exists but is
unset) with `int(...)` / `.isdigit()` (crash on None, floats or ints). These helpers make every
read tolerant and every problem explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# keys shown in the settings tab, in display order (same names as before: storage.py relies on them)
PARAM_KEYS = [
    "scoring_started", "project_path", "scorer", "animal_id", "group", "trial", "sample_rate", "time_channel",
    "ecog_channels", "emg_channels", "ylim", "device", "optional_tag",
]
YLIM_MODES = ("infer", "infer_ephys", "standard")
DEVICES = ("cuda", "cpu", "auto")

DEFAULTS = {
    "animal_id": "0",
    "sample_rate": "1000",
    "ecog_channels": "0,1",
    "emg_channels": "2",
    "time_channel": "3",
    "ylim": "infer",
    "device": "auto",
    "optional_tag": "",
}


def new_metadata() -> dict:
    """Return a fresh metadata dict: every ``PARAM_KEYS`` entry unset, a new session id, today's date, project '.'."""
    meta = {k: None for k in PARAM_KEYS}
    meta["scoring_started"] = timestamp()
    meta["project_path"] = "."
    meta["date"] = datetime.now().strftime("%Y_%m_%d")
    return meta


def timestamp() -> str:
    """Return the current time as ``YYYYMMDDHHMMSS`` (used as the session id)."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_str(meta: dict, key: str, default: str = "") -> str:
    """Return ``meta[key]`` as stripped text; None, missing or blank gives ``default``."""
    v = meta.get(key)
    return default if v is None or (isinstance(v, str) and not v.strip()) else str(v).strip()


def get_float(meta: dict, key: str, default: float | None = None) -> float | None:
    """Return ``meta[key]`` as a float; None, blank or unparseable gives ``default``."""
    v = meta.get(key)
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except (TypeError, ValueError):
        return default


def get_int(meta: dict, key: str, default: int | None = None) -> int | None:
    """Return ``meta[key]`` as an int if it is a whole number, else ``default``."""
    f = get_float(meta, key)
    return default if f is None or f != int(f) else int(f)


def parse_channels(value) -> list[int]:
    """Parse a channel list: "0,1", "0, 1", [0, 1] and 2 all work; None or blank gives [].

    Raises:
        ValueError: If any entry is not a non-negative integer.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part.lstrip("-").isdigit() or int(part) < 0:
            raise ValueError(f"channel list {value!r}: {part!r} is not a non-negative integer")
        out.append(int(part))
    return out


def channel_counts(meta: dict) -> tuple[int, int]:
    """Return ``(number of ECoG channels, number of EMG channels)``; unparseable lists count as 0."""
    try:
        n_ecog = len(parse_channels(meta.get("ecog_channels")))
    except ValueError:
        n_ecog = 0
    try:
        n_emg = len(parse_channels(meta.get("emg_channels")))
    except ValueError:
        n_emg = 0
    return n_ecog, n_emg


def project_dir(meta: dict) -> str:
    """Return the project folder (default '.')."""
    return get_str(meta, "project_path", ".")


def sample_rate(meta: dict, default: float = 250.0) -> float:
    """Return the sample rate in Hz, or ``default`` if unset."""
    return get_float(meta, "sample_rate", default)


def device(meta: dict) -> str:
    """Return 'cuda' if the setting allows it (cuda or auto) and a GPU is available, else 'cpu'."""
    want = get_str(meta, "device", "auto").lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Problem:
    """One finding of ``validate``.

    Attributes:
        key: Metadata key concerned.
        message: What is wrong.
        severity: "error" (blocks work) or "warning" (e.g. a value not set).
    """
    key: str
    message: str
    severity: str = "error"  # error | warning

    def __str__(self) -> str:
        """Return ``'<severity>: <key>: <message>'``."""
        return f"{self.severity}: {self.key}: {self.message}"


def validate(meta: dict) -> list[Problem]:
    """Check the metadata and return every problem found (empty list = all good).

    Checks: unset keys (warnings), a positive whole-Hz sample rate, valid and
    non-overlapping channel lists, and known ``ylim`` and ``device`` values. IDs like
    "4F" are accepted.
    """
    problems: list[Problem] = []
    for key in PARAM_KEYS:
        v = meta.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            problems.append(Problem(key, "not set", "warning"))
    sr = get_float(meta, "sample_rate")
    if meta.get("sample_rate") not in (None, "") and (sr is None or sr <= 0):
        problems.append(Problem("sample_rate", f"{meta.get('sample_rate')!r} is not a positive number"))
    elif sr is not None and sr != int(sr):  # storage, report and conversion use whole-Hz rates
        problems.append(Problem("sample_rate", f"{meta.get('sample_rate')!r} must be a whole number of Hz"))
    chans = {}
    for key in ("ecog_channels", "emg_channels", "time_channel"):
        if meta.get(key) in (None, ""):
            continue
        try:
            chans[key] = parse_channels(meta.get(key))
        except ValueError as e:
            problems.append(Problem(key, str(e)))
    seen = {}
    for key, chs in chans.items():
        for c in chs:
            if c in seen:
                problems.append(Problem(key, f"channel {c} is also used by {seen[c]}"))
            seen[c] = key
    if meta.get("ylim") not in (None, "") and meta.get("ylim") not in YLIM_MODES:
        problems.append(Problem("ylim", f"{meta.get('ylim')!r} is not one of {YLIM_MODES}"))
    if meta.get("device") not in (None, "") and str(meta.get("device")).lower() not in DEVICES:
        problems.append(Problem("device", f"{meta.get('device')!r} is not one of {DEVICES}"))
    return problems


def normalise_value(key: str, value: str) -> str:
    """Tidy a typed value: '250.0' -> '250' for sample_rate, '0, 1' -> '0,1' for channel lists."""
    value = value.strip()
    if key == "sample_rate":
        f = get_float({"v": value}, "v")
        if f is not None and f == int(f):
            return str(int(f))
    if key in ("ecog_channels", "emg_channels"):
        try:
            return ",".join(str(c) for c in parse_channels(value))
        except ValueError:
            return value
    return value
