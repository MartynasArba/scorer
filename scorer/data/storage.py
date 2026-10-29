"""
File IO for the GUI: state files, metadata JSON, raw/preprocessed/windowed tensors, csv loading.

Changes in this revision (behaviour kept unless it was a bug):
  * torch is imported lazily (only the tensor functions need it), so the GUI starts without
    loading torch and the time/metadata helpers can be tested without it;
  * metadata values that are None (e.g. an unset optional_tag) no longer become the text
    "None" inside file names;
  * construct_paths raises FileNotFoundError for a missing path (it returned None, which
    callers then failed to unpack);
  * save_metadata writes non-JSON values as text and raises on failure (it printed and
    carried on, so a failed save looked successful);
  * save_tensor checked for an empty tensor list AFTER min() over it (ValueError), and both
    savers reported length mismatches only after truncating (never printed); both now return
    the written path(s), or None when a file already existed;
  * csv time-chopping: a start time after midnight on a recording that began the evening before
    was treated as "before the recording" and read from the first sample (with rec_start kept
    on the old date); a start before the recording began made the window end late by the gap.
    (skiprows=<int> also turned the row before the window into the header; data were still
    right because columns are read by position - now the real header is kept);
  * channel-parsing failures raise instead of returning (None, None, None) / an empty
    generator, which made preprocessing "succeed" with nothing processed;
  * device 'auto' (GUI default) is resolved; sample rates like '250.0' are accepted;
  * get_timearray_for_states always returns exactly one timestamp per state (it rounded the
    window length to whole seconds, so 2.048-s windows gave the wrong number of timestamps)
    and tolerates rec_start with trailing whitespace or missing.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SECONDS_IN_DAY = 24 * 60 * 60


def _m(metadata: dict, key: str, default: str = "") -> str:
    """Return a metadata value as text for file names; None or missing gives ``default``."""
    v = metadata.get(key) if metadata else None
    return default if v is None else str(v)


def _sample_rate(metadata: dict, default: float) -> int:
    """Return ``metadata['sample_rate']`` as whole Hz ('250', 250.0 and '250.0' all work).

    Raises:
        ValueError: If the value is not a number or not a whole number of Hz.
    """
    v = metadata.get("sample_rate") if metadata else None
    try:
        sr = float(v) if v not in (None, "") else float(default)
    except (TypeError, ValueError):
        raise ValueError(f"sample_rate {v!r} is not a number") from None
    if sr != int(sr):
        raise ValueError(f"sample_rate must be a whole number of Hz, got {v!r}")
    return int(sr)


def _device(metadata: dict) -> str:
    """Return the torch device to load data on: 'cpu' when asked or when no GPU exists, else 'cuda'."""
    import torch

    want = str((metadata or {}).get("device") or "cuda").lower()
    if want == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return "cuda" if want in ("auto", "cuda") else want


def _channels(value, name: str) -> list[int]:
    """Parse a channel list such as "0,1" into integers.

    Raises:
        ValueError: If the text is not comma-separated integers.
    """
    try:
        return [int(ch) for ch in str(value).split(",")]
    except (TypeError, ValueError):
        raise ValueError(f"{name} {value!r} must be comma-separated channel numbers, e.g. 0,1") from None


# ---------------------------------------------------------------------------------------------
# state files / metadata
# ---------------------------------------------------------------------------------------------
def construct_paths(data_path: str, format: str = ".pkl", add="", **metadata) -> tuple[Path, Path]:
    """Return the label path for a data file and a new state-file path.

    The state file goes to ``<project>/scores/`` (two levels above the data) and is named
    from the metadata (session id, tag, scorer, date, animal, trial). A repetition
    counter makes sure an existing file is never overwritten.

    Args:
        data_path: Data file (its label file is the same name with X replaced by y) or
            a windowed-data folder (returned as is).
        format: File extension of the state file.
        add: Extra text before the extension, e.g. "autosave".
        **metadata: The GUI's metadata.

    Returns:
        ``(label_path, state_file_path)``.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"construct_paths: {data_path} does not exist")
    if data_path.is_dir():
        y_file_path = data_path
    else:
        y_file_path = data_path.with_name(data_path.name.replace("X", "y"))
    scorer = metadata.get("scorer") or (data_path.stem[:-2] if data_path.suffix else data_path.stem)
    unique_id = _m(metadata, "scoring_started", "noID")
    date, animal, trial, tag = (_m(metadata, k) for k in ("date", "animal", "trial", "optional_tag"))
    score_folder = data_path.parent.parent / "scores"
    os.makedirs(score_folder, exist_ok=True)
    repetition_id = 0
    while True:  # never overwrite an existing state file
        path = score_folder / f"{unique_id}{tag}_scores_{scorer}_{date}_{animal}_{trial}_{repetition_id}_frame{add}{format}"
        if not path.exists():
            return y_file_path, path
        repetition_id += 1


def save_pickled_states(states: list, path: str) -> None:
    """Pickle a state sequence to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(states, f)


def load_pickled_states(path: str) -> np.ndarray:
    """Load a pickled state sequence from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(states: list, path: str) -> None:
    """Not implemented; kept so old imports keep working. Raises NotImplementedError."""
    raise NotImplementedError("save_json is not implemented")


def save_metadata(path: str, metadata: dict) -> None:
    """Save metadata as indented JSON; values JSON cannot store (numpy numbers, paths) become text.

    Raises:
        Exception: Whatever the write raised (logged first), so a failed save is visible.
    """
    try:
        with open(path, "wt") as f:
            json.dump(metadata, f, indent=1, default=str)
    except Exception as e:
        log.error("save_metadata failed for %s: %s", path, e)
        raise


def load_metadata(path: str) -> dict:
    """Load metadata JSON from ``path``."""
    with open(path, "rt") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------------------------
def _align(tensors, where: str):
    """Keep only tensors and trim them to a common length along dimension 1 (time).

    A length mismatch is logged with the function name ``where``.

    Returns:
        ``(tensors, min_len)``, or ``([], 0)`` when no tensor was given.
    """
    import torch

    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    if not tensors:
        return [], 0
    lengths = [t.size(1) for t in tensors]
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        log.warning("length mismatch in %s: %s, truncating to %d", where, lengths, min_len)
    return [t[:, :min_len] for t in tensors], min_len


def _base_name(metadata: dict) -> str:
    """Return the base file name: session id + optional tag + original file name."""
    return f"{_m(metadata, 'scoring_started', 'noID')}{_m(metadata, 'optional_tag')}{_m(metadata, 'filename')}"


def save_tensor(tensor_seq: tuple = (), metadata: dict = {}, overwrite: bool = False, chunk: int = None,
                raw: bool = False) -> Path | None:
    """Save tensors concatenated along channels to ``<project>/raw`` or ``<project>/processed``.

    Args:
        tensor_seq: Tensors [channels, time]; non-tensors (e.g. None) are skipped.
        metadata: The GUI's metadata (project path, names).
        overwrite: Replace an existing file.
        chunk: Chunk number when preprocessing in chunks; the file then goes into a
            per-recording sub-folder.
        raw: Save as raw instead of preprocessed.

    Returns:
        The path written, or None if the file existed or there was nothing to save.
    """
    import torch

    proj_path = Path(_m(metadata, "project_path", ".") or ".")
    kind = "raw" if raw else "preprocessed"
    save_folder = proj_path / ("raw" if raw else "processed")
    os.makedirs(save_folder, exist_ok=True)
    if isinstance(chunk, int):
        chunk_folder = save_folder / f"{kind}_{_base_name(metadata)}"
        os.makedirs(chunk_folder, exist_ok=True)
        save_path = chunk_folder / f"chunk{chunk}_{kind}.pt"
    else:
        save_path = save_folder / f"{kind}_{_base_name(metadata)}.pt"
    if not overwrite and save_path.exists():
        log.warning("%s already exists, not saving (tick overwrite)", save_path)
        return None
    tensors, _ = _align(tensor_seq, "save_tensor")
    if not tensors:
        log.warning("save_tensor: no tensors to save")
        return None
    with open(save_path, "wb") as f:
        torch.save(torch.cat(tensors, dim=0), f)
    return save_path


def save_windowed(tensors: tuple, states=None, metadata: dict = {}, win_len: int = 1000, chunked: bool = True,
                  chunk_id: int = None, overwrite: bool = False, testing: bool = False):
    """Cut [channels, time] tensors into windows and save X and y files.

    X is saved as [C, N, win_len] and y as [1, N, win_len] in ``<project>/processed``
    (chunked: in a ``windowed_<name>`` sub-folder as ``X_chunk<i>.pt`` / ``y_chunk<i>.pt``).

    Args:
        tensors: Tensors [channels, time]; non-tensors are skipped.
        states: Per-sample labels [time] or [1, time]; None saves zeros (unlabeled).
        metadata: The GUI's metadata.
        win_len: Window length in samples.
        chunked: Save into the chunk sub-folder.
        chunk_id: Chunk number for the file names.
        overwrite: Replace existing files.
        testing: Cut only windows that do not cross a state change.

    Returns:
        ``(x_path, y_path)``, or None if skipped or no full window fits.
    """
    import torch

    proj_path = Path(_m(metadata, "project_path", ".") or ".")
    save_folder = proj_path / "processed"
    os.makedirs(save_folder, exist_ok=True)
    if not chunked:
        save_path = save_folder / f"{_base_name(metadata)}_X.pt"
        states_path = save_folder / f"{_base_name(metadata)}_y.pt"
    else:
        chunk_folder = save_folder / f"windowed_{_base_name(metadata)}"
        os.makedirs(chunk_folder, exist_ok=True)
        save_path = chunk_folder / f"X_chunk{chunk_id}.pt"
        states_path = chunk_folder / f"y_chunk{chunk_id}.pt"
    if not overwrite and save_path.exists():
        log.warning("%s already exists, not saving (tick overwrite)", save_path)
        return None
    tensors, _ = _align(tensors, "save_windowed")
    if not tensors:
        log.warning("save_windowed: no tensors to save")
        return None
    to_save = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
    if states is None:
        states = torch.zeros(size=(1, to_save.size()[-1]), dtype=torch.long, device=to_save.device)
    elif states.dim() != 2:
        states = states.unsqueeze(0)
    if not testing:
        to_save = _chop(to_save, win_len=win_len)
        states = _chop(states, win_len=win_len)
    else:
        to_save, states = _chop_by_state(states, to_save, win_len)
    if to_save.size(1) == 0:
        log.warning("save_windowed: no complete window of %d samples - nothing saved", win_len)
        return None
    with open(save_path, "wb") as f:
        torch.save(to_save, f)
    with open(states_path, "wb") as f:
        torch.save(states, f)
    return save_path, states_path


def save_windowed_for_testing(tensors: tuple, save_folder: str, file_name: str, states=None, win_len: int = 1000,
                              chunked: bool = True, chunk_id: int = None, overwrite: bool = False):
    """Cut data into windows and save them with the most common label of each window.

    Returns:
        ``(x_path, y_path)``, or None if skipped or no full window fits.
    """
    import torch

    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    if not chunked:
        save_path = save_folder / file_name
        states_path = save_folder / f"{file_name}_y.pt"
    else:
        chunk_folder = save_folder / f"windowed_{file_name}"
        os.makedirs(chunk_folder, exist_ok=True)
        save_path = chunk_folder / f"X_{file_name}_chunk{chunk_id}.pt"
        states_path = chunk_folder / f"y_{file_name}_chunk{chunk_id}.pt"
    if not overwrite and save_path.exists():
        log.warning("%s already exists, not saving", save_path)
        return None
    tensors, _ = _align(tensors, "save_windowed_for_testing")
    if not tensors:
        return None
    to_save = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
    if states is None:
        states = torch.zeros(size=(1, to_save.size()[-1]), dtype=torch.long, device=to_save.device)
    elif states.dim() != 2:
        states = states.unsqueeze(0)
    to_save, states = _chop_continuous(states, to_save, win_len)
    if to_save.size(1) == 0:
        return None
    with open(save_path, "wb") as f:
        torch.save(to_save, f)
    with open(states_path, "wb") as f:
        torch.save(states, f)
    return save_path, states_path


def _chop(values, win_len: int):
    """Reshape [C, T] into [C, T // win_len, win_len]; leftover samples are dropped."""
    n_channels, length = values.size()[0], values.size()[1]
    n_samples = length // win_len
    return values[:, : win_len * n_samples].reshape(n_channels, n_samples, win_len)


def _chop_continuous(states, values, win_len: int = 1000):
    """Cut into windows and give each window its most common label.

    Returns:
        ``(values [C, N, win_len], states [S, N, 1])``.

    Raises:
        ValueError: For wrong shapes or a length mismatch between states and values.
    """
    import torch

    if values.dim() != 2:
        raise ValueError(f"values must be [C, L], got {tuple(values.shape)}")
    C, L = values.shape
    states_ = states.unsqueeze(0) if states.dim() == 1 else states
    if states_.dim() != 2:
        raise ValueError(f"states must be [L] or [S, L], got {tuple(states.shape)}")
    S, Ls = states_.shape
    if Ls != L:
        raise ValueError(f"states length ({Ls}) must match values length ({L})")
    n = L // win_len
    values_out = values[:, : win_len * n].reshape(C, n, win_len)
    states_majority, _ = torch.mode(states_[:, : win_len * n].reshape(S, n, win_len), dim=2)
    return values_out, states_majority.unsqueeze(2)


def _chop_by_state(states, values, win_len: int):
    """Cut only windows that lie entirely inside one state (the state never changes inside).

    Returns:
        ``(values [C, N, win_len], states [S, N, win_len])``.

    Raises:
        ValueError: For wrong shapes or a length mismatch between states and values.
    """
    import torch

    if values.dim() != 2:
        raise ValueError(f"values must be [C, L], got {tuple(values.shape)}")
    C, L = values.shape
    states_ = states.unsqueeze(0) if states.dim() == 1 else states
    if states_.dim() != 2:
        raise ValueError(f"states must be [L] or [S, L], got {tuple(states.shape)}")
    S, Ls = states_.shape
    if Ls != L:
        raise ValueError(f"states length ({Ls}) must match values length ({L})")
    st0 = states_[0].flatten()
    change_idx = torch.nonzero(torch.diff(st0) != 0).flatten() + 1
    boundaries = torch.cat([torch.tensor([0], device=st0.device), change_idx.to(st0.device),
                            torch.tensor([L], device=st0.device)])
    xs, ss = [], []
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        i = start
        while i + win_len <= end:
            xs.append(values[:, i:i + win_len].unsqueeze(1))
            ss.append(states_[:, i:i + win_len].unsqueeze(1))
            i += win_len
    if not xs:
        return values.new_empty((C, 0, win_len)), states_.new_empty((S, 0, win_len))
    return torch.cat(xs, dim=1), torch.cat(ss, dim=1)


# ---------------------------------------------------------------------------------------------
# csv loading
# ---------------------------------------------------------------------------------------------
def _get_start_time(path, time_channel) -> str | None:
    """Return the recording start from the time column of the first data row, or None if unknown."""
    if time_channel in (None, ""):
        log.warning("no time column specified: recording start unknown")
        return None
    try:
        with open(path, "rt") as f:
            f.readline()  # header
            start_time = f.readline().split(",")[int(time_channel)].strip()
        log.info("recording start time: %s", start_time)
        return start_time
    except Exception as e:
        log.warning("could not read the start time from %s (%s): recording start unknown", path, e)
        return None


def _seconds_since_midnight(t) -> float:
    """Return the seconds since midnight of a ``datetime.time``."""
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def resolve_time_window(start_time: str | None, times: tuple[str, str], sr: int) -> tuple[int, int, str | None]:
    """Find the rows to read for a HH:MM:SS start/end window.

    A start before the recording began is read from the first row if it is less than
    12 h earlier (19:00 for a recording that began at 19:16); otherwise it is taken as
    the next day (01:00 for a recording that began at 19:16 starts 5 h 44 min in). An end
    time before the start wraps past midnight.

    Args:
        start_time: Recording start, e.g. "2025-11-27 19:16:00".
        times: ``(start, end)`` as "HH:MM:SS".
        sr: Sample rate in Hz (one row per sample).

    Returns:
        ``(first data row, end row (exclusive), new rec_start or None)``.

    Raises:
        ValueError: If the start time is unknown or a time cannot be parsed.
    """
    if start_time is None:
        raise ValueError("cannot cut by time: the recording start time is unknown (set time_channel)")
    try:
        rec_dt = pd.to_datetime(start_time)
        recstart = _seconds_since_midnight(rec_dt.time())
        t0 = _seconds_since_midnight(pd.to_datetime(times[0]).time())
        t1 = _seconds_since_midnight(pd.to_datetime(times[1]).time())
    except Exception as e:
        raise ValueError(f"could not parse times {times} / recording start {start_time!r}: {e}") from None
    offset = (t0 - recstart) % SECONDS_IN_DAY
    if offset > SECONDS_IN_DAY / 2:  # start shortly before the recording began: read from the beginning
        offset, new_start = 0.0, None
        duration = (t1 - recstart) % SECONDS_IN_DAY or SECONDS_IN_DAY
    else:
        new_start = str(rec_dt + pd.Timedelta(seconds=offset))
        duration = (t1 - t0) % SECONDS_IN_DAY or SECONDS_IN_DAY
    line_start = int(round(offset * sr))
    return line_start, line_start + int(round(duration * sr)), new_start


def _read_csv_window(path, line_start: int, line_end: int, chunksize=None):
    """Read data rows ``[line_start, line_end)`` and keep the header row (optionally as chunks)."""
    skip = range(1, line_start + 1) if line_start > 0 else None
    return pd.read_csv(path, skiprows=skip, nrows=line_end - line_start, chunksize=chunksize)


def load_from_csv(path: str, metadata: dict = None, states: int = None, times: tuple[str, str] = (None, None)):
    """Load ECoG, EMG and optionally a state column from a CSV file.

    Sets ``metadata['rec_start']`` from the time column (moved forward if a time window
    starts later).

    Args:
        path: CSV file.
        metadata: The GUI's metadata (channels, time column, sample rate, device).
        states: Column index of per-sample states, or None.
        times: ``(start, end)`` HH:MM:SS window, or ``(None, None)`` for everything.

    Returns:
        ``(ecog [time, channels], emg [time, channels], states or None)`` as tensors.

    Raises:
        ValueError: For bad channel lists, sample rates or times.
    """
    import torch

    sample_rate = _sample_rate(metadata, 1000)
    device = _device(metadata)
    ecog_channels = _channels(metadata.get("ecog_channels"), "ecog_channels")
    emg_channels = _channels(metadata.get("emg_channels"), "emg_channels")
    start_time = _get_start_time(path, metadata.get("time_channel"))
    metadata["rec_start"] = start_time
    if times[0] is not None and times[1] is not None:
        line_start, line_end, new_start = resolve_time_window(start_time, times, sample_rate)
        if new_start is not None:
            metadata["rec_start"] = new_start
            log.info("updated rec_start: %s", new_start)
        data = _read_csv_window(path, line_start, line_end)
    else:
        data = pd.read_csv(path)
    ecog = torch.tensor(data.iloc[:, ecog_channels].values, device=device, dtype=torch.float32)
    emg = torch.tensor(data.iloc[:, emg_channels].values, device=device, dtype=torch.float32)
    if states is not None:
        states = torch.tensor(data.iloc[:, states].values, device=device, dtype=torch.float32)
    del data
    return ecog, emg, states


def load_from_csv_in_chunks(path: str, metadata: dict = None, states: int = None, chunk_size: int = None,
                            times: tuple[str, str] = (None, None)):
    """Like ``load_from_csv`` but yields ``(ecog, emg, states)`` chunks of ``chunk_size`` rows.

    Errors in the settings raise when the first chunk is requested.
    """
    import torch

    sample_rate = _sample_rate(metadata, 1000)
    device = _device(metadata)
    ecog_channels = _channels(metadata.get("ecog_channels"), "ecog_channels")
    emg_channels = _channels(metadata.get("emg_channels"), "emg_channels")
    start_time = _get_start_time(path, metadata.get("time_channel"))
    metadata["rec_start"] = start_time
    if times[0] is not None and times[1] is not None:
        line_start, line_end, new_start = resolve_time_window(start_time, times, sample_rate)
        if new_start is not None:
            metadata["rec_start"] = new_start
            log.info("updated rec_start: %s", new_start)
        reader = _read_csv_window(path, line_start, line_end, chunksize=chunk_size)
    else:
        reader = pd.read_csv(path, chunksize=chunk_size)
    for chunk in reader:
        ecog_chunk = torch.tensor(chunk.iloc[:, ecog_channels].values, device=device)
        emg_chunk = torch.tensor(chunk.iloc[:, emg_channels].values, device=device)
        states_chunk = torch.tensor(chunk.iloc[:, states].values, device=device) if states is not None else None
        yield ecog_chunk, emg_chunk, states_chunk


def get_timearray_for_states(states, win_len: int = 1000, metadata: dict = {}) -> pd.DatetimeIndex:
    """Return one timestamp per state window, starting at ``metadata['rec_start']``.

    Times have millisecond precision, so windows such as 2.048 s stay exact. Without a
    start time the times start at today 00:00 (logged).
    """
    n = len(states)
    win_len_s = win_len / _sample_rate(metadata, 250)
    rec_start = metadata.get("rec_start") if metadata else None
    if rec_start in (None, ""):
        log.warning("rec_start not set: times start at today 00:00")
        start = pd.Timestamp.today().normalize()
    else:
        start = pd.to_datetime(str(rec_start).strip())
    offsets_ms = np.round(np.arange(n) * win_len_s * 1000).astype("int64")
    return pd.DatetimeIndex(start + pd.to_timedelta(offsets_ms, unit="ms"))
