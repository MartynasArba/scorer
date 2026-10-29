"""
Convert external datasets into 4-s windowed tensors (X [N, 1, win], y [N] raw codes 0/1/2/4).

Fixes vs. the previous version (re-run the conversion, then `prepare --force-prepare`):
  * Oxford .hyp: epoch starts were np.linspace(0, duration, n) -> spacing duration/(n-1) > 4 s,
    so labels drifted up to one epoch late by the end of the file. Now start_i = 4 * i.
  * Oxford .hyp: 'sleep' -> NREM was tested BEFORE 'rem', so a REM label containing the word
    'sleep' (e.g. "REM sleep", "paradoxical sleep") became NREM. Matching is now ordered
    NREM-patterns -> REM -> Wake -> generic sleep, and every distinct annotation string is
    logged with the code it received (check this table once!). An explicit label_map overrides.
  * Oxford .hyp: the time on each row is where that bout ENDS, not where it starts. Reading it
    as a start time gave every bout the label of the bout before it (checked on oxford_test:
    accuracy 6% as converted, 93% with labels moved one bout later; Wake-labelled windows had
    the most delta power). hyp_to_tensor now reads bout ends (time_column="end").
  * Re-running the conversion no longer duplicates recordings: scan_and_move replaced every
    existing folder name with "<name>0", "<name>1", ... so each run added another copy.
    It now overwrites files of the same name and warns about folders this run did not write.
  * MSSV events.tsv: the duration assert indexed a ROW (states[1]) instead of the duration column.
  * X and y lengths are reconciled (trimmed to the shorter, logged) instead of crashing on
    reshape or silently misaligning; tensors are saved on the CPU.
Paths are no longer hard-coded inside functions: edit CONVERSIONS / run(...) arguments.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

EPOCH_S = 4.0


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------
def _window_and_align(signal_1d: torch.Tensor, y: torch.Tensor, win: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a 1-D signal into windows of ``win`` samples and align it with its labels.

    If the number of windows and labels differ, both are trimmed to the shorter and a
    warning names the recording. Returns CPU tensors ``(x [N, 1, win], y [N] int64)``.
    """
    n_x = signal_1d.numel() // win
    n = min(n_x, int(y.numel()))
    if n_x != y.numel():
        log.warning("%s: %d signal windows vs %d labels -> keeping %d", name, n_x, y.numel(), n)
    return signal_1d[: n * win].reshape(n, 1, win).cpu().contiguous(), y[:n].cpu().long()


def _resample_filter(x: torch.Tensor, fs_in: float, fs_out: float, device: str) -> torch.Tensor:
    """Resample ``x`` [C, T] from ``fs_in`` to ``fs_out`` and band-pass it to 0.5-49 Hz."""
    from torchaudio.functional import resample

    from scorer.data.preprocessing import bandpass_filter

    x = resample(x.to(device), int(round(fs_in)), int(round(fs_out)))
    return bandpass_filter(x, sr=fs_out, freqs=(0.5, 49.0), device=device)


# ---------------------------------------------------------------------------------------------
# MLS-Net (https://doi.org/10.3390/bios14080406)
# ---------------------------------------------------------------------------------------------
def load_mlsnet_npz(path, new_sr: float = 128.0, device: str = "cuda", n_rows: int = 7):
    """Convert the MLS-Net validation file to windows at ``new_sr``.

    ``obj['x'][:, :, 0:2000]`` holds 4-s ECoG epochs at 500 Hz. The stream is cut into
    ``n_rows`` contiguous rows before resampling, as in the original script (the meaning
    of the row count is not documented; check it). Labels 0 NREM, 1 REM, 2 Wake become
    raw codes 2, 4, 1.

    Returns:
        ``(x [N, 1, win], y [N])`` CPU tensors.

    Raises:
        ValueError: If the samples cannot be split evenly into ``n_rows`` rows.
    """
    obj = np.load(path)
    x = obj["x"][:, :, 0:2000]
    log.info("mlsnet x %s, y %s", obj["x"].shape, obj["y"].shape)
    n_epochs = x.shape[0] * x.shape[1]
    if (n_epochs * 2000) % n_rows:
        raise ValueError(f"{n_epochs} epochs x 2000 samples is not divisible into {n_rows} rows")
    sig = torch.tensor(x, dtype=torch.float32).reshape(n_rows, -1)
    sig = _resample_filter(sig, 500, new_sr, device).reshape(-1)
    y_src = torch.tensor(obj["y"].reshape(-1), dtype=torch.long)
    y = torch.zeros_like(y_src)
    y[y_src == 0] = 2
    y[y_src == 1] = 4
    y[y_src == 2] = 1
    return _window_and_align(sig, y, int(new_sr * EPOCH_S), str(path))


# ---------------------------------------------------------------------------------------------
# MSSV (OpenNeuro ds006366)
# ---------------------------------------------------------------------------------------------
def parse_channel_tsv(path) -> dict:
    """Read a BIDS ``channels.tsv`` into ``{column name: list of values}``."""
    with open(path) as f:
        lines = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    header = lines[0]
    return {key: [row[i] for row in lines[1:]] for i, key in enumerate(header)}


def parse_state_tsv(path) -> np.ndarray:
    """Read an MSSV ``events.tsv`` and return the stage code of every 4-s epoch.

    Raises:
        ValueError: If any epoch other than the last is not 4 s long, the first onset is
            not 0, or the onsets are not contiguous 4-s steps.
    """
    states = np.loadtxt(path, skiprows=1, delimiter="\t", ndmin=2)
    durations = states[:, 1]
    if np.any(durations[:-1] != EPOCH_S):
        raise ValueError(f"{path}: {int(np.sum(durations[:-1] != EPOCH_S))} non-4-s epochs before the last one")
    if states[0, 0] != 0:
        raise ValueError(f"{path}: first onset is {states[0, 0]}, expected 0")
    if not np.allclose(states[:, 0], np.arange(len(states)) * EPOCH_S):
        raise ValueError(f"{path}: onsets are not contiguous 4-s steps")
    return states[:, -1].copy()


def parse_edf(edf_path, channel_names: list[str] | None = None):
    """Read selected channels from an EDF file.

    Args:
        edf_path: EDF file.
        channel_names: Substrings to match channel labels; None picks every channel
            whose label contains "eeg".

    Returns:
        ``(signals [T, C] float32 tensor, sample rate in Hz)``.

    Raises:
        ValueError: If no channel matches or the chosen channels have different rates.
    """
    from pyedflib import highlevel

    signals, headers, _ = highlevel.read_edf(str(edf_path))
    chans = [(i, h["label"]) for i, h in enumerate(headers)]
    if channel_names:
        picked = [c for c in chans if any(n in c[1] for n in channel_names)]
    else:
        picked = [c for c in chans if "eeg" in c[1].lower()]
    if not picked:
        raise ValueError(f"{edf_path}: no channels matched {channel_names or 'EEG'}; available {[c[1] for c in chans]}")
    rates = {float(headers[i]["sample_frequency"]) for i, _ in picked}
    if len(rates) != 1:
        raise ValueError(f"{edf_path}: selected channels have different sample rates {rates}")
    log.info("%s: channels %s @ %s Hz", Path(edf_path).name, [c[1] for c in picked], rates)
    return torch.stack([torch.tensor(signals[i], dtype=torch.float32) for i, _ in picked], dim=1), rates.pop()


def parse_mssv_paths(dir_path) -> dict:
    """Sort the files of one MSSV ``eeg`` folder into EDF, events, channels and unknown lists."""
    paths = {"eeg_paths": [], "state_paths": [], "channel_paths": [], "unknown_paths": []}
    for p in sorted(glob.glob(f"{dir_path}/*")):
        if p.endswith(".edf"):
            paths["eeg_paths"].append(p)
        elif p.endswith(".tsv") and "channels" in p:
            paths["channel_paths"].append(p)
        elif p.endswith(".tsv") and "events" in p:
            paths["state_paths"].append(p)
        else:
            paths["unknown_paths"].append(p)
    return paths


def load_mssv_rec(dir_path, new_sr: float = 128.0, device: str = "cuda") -> None:
    """Convert one MSSV ``eeg`` folder; writes one sub-folder per run and channel.

    MSSV stages 1 Wake, 2 NREM, 3 REM, 4 artifact become raw codes 1, 2, 4, 0.
    Events and EDF files are paired by sort order; a warning is logged if their names
    do not match.

    Raises:
        RuntimeError: If the folder has different numbers of event and EDF files.
    """
    paths = parse_mssv_paths(dir_path)
    if len(paths["state_paths"]) != len(paths["eeg_paths"]):
        raise RuntimeError(f"{dir_path}: {len(paths['state_paths'])} event files vs {len(paths['eeg_paths'])} EDFs")
    if paths["unknown_paths"]:
        log.info("%s: ignoring %d unknown files", dir_path, len(paths["unknown_paths"]))
    win = int(new_sr * EPOCH_S)
    for run, (state_path, edf_path) in enumerate(zip(paths["state_paths"], paths["eeg_paths"])):
        # make sure events and EDF belong to the same run (sorted lists could otherwise pair wrongly)
        key = lambda p: re.sub(r"_(events|eeg)\.(tsv|edf)$", "", Path(p).name)  # noqa: E731
        if key(state_path) != key(edf_path):
            log.warning("CHECK PAIRING: events %s paired with %s by sort order", Path(state_path).name,
                        Path(edf_path).name)
        stages = parse_state_tsv(state_path)
        raw = np.zeros_like(stages, dtype=np.int64)
        raw[stages == 1] = 1
        raw[stages == 2] = 2
        raw[stages == 3] = 4
        y = torch.tensor(raw)
        sig, fs = parse_edf(edf_path)
        sig = _resample_filter(sig.T, fs, new_sr, device)  # [C, T]
        state_path = Path(state_path)
        for ch in range(sig.shape[0]):
            x_ch, y_ch = _window_and_align(sig[ch], y, win, f"{state_path.name} ch{ch}")
            out = state_path.parent / f"{state_path.stem}_ch{ch}"
            out.mkdir(exist_ok=True)
            torch.save(x_ch, out / f"X_ch{ch}_run{run}.pt")
            torch.save(y_ch, out / f"y_ch{ch}_run{run}.pt")


# ---------------------------------------------------------------------------------------------
# Oxford (https://doi.org/10.5281/zenodo.10200482)
# ---------------------------------------------------------------------------------------------
_NREM_PAT = re.compile(r"\b(n[\s_-]?rem|non[\s_-]?rem|sws|slow[\s_-]?wave)\b")
_REM_PAT = re.compile(r"\b(rem|paradox\w*)\b")
_WAKE_PAT = re.compile(r"\b(wake|awake|w)\b")


def annotation_to_code(text: str, label_map: dict[str, int] | None = None) -> int:
    """Map one Oxford hypnogram annotation string to a raw code.

    Matching order: an explicit ``label_map`` entry, then NREM patterns, REM, Wake, and
    finally a generic "sleep" (NREM). Anything else is 0 (unlabeled). The order matters:
    "REM sleep" must become REM, not NREM.
    """
    t = text.strip().lower()
    if label_map and text.strip() in label_map:
        return int(label_map[text.strip()])
    if _NREM_PAT.search(t) or t in ("nr", "n"):
        return 2
    if _REM_PAT.search(t) or t == "r":
        return 4
    if _WAKE_PAT.search(t):
        return 1
    if "sleep" in t:  # generic 'sleep' that is not REM
        return 2
    return 0


def hyp_to_tensor(hypnogram_path, label_map: dict[str, int] | None = None,
                  time_column: str = "end") -> tuple[torch.Tensor, dict]:
    """Convert an Oxford ``.hyp`` file to one raw code per 4-s epoch.

    Epoch i starts at 4 * i seconds (the old ``linspace`` version drifted late) and gets the
    state of the bout that contains its start.

    Args:
        hypnogram_path: The ``.hyp`` file (``Duration_sec`` header, then label / time rows).
        label_map: Explicit annotation -> raw code overrides.
        time_column: What the time on each row means. "end" (the Oxford files): the bout
            with that label ends there, so row k covers [time[k-1], time[k]) and the first
            row starts at 0. "start": the bout starts there (the old reading, which shifted
            every label one bout early on oxford_test).

    Returns:
        ``(codes, table)``: a long tensor of raw codes and
        ``{annotation string: (code, number of segments)}`` for checking the mapping.

    Raises:
        ValueError: If the ``Duration_sec`` header is missing, times are unsorted, or
            ``time_column`` is not "end" or "start".
    """
    if time_column not in ("end", "start"):
        raise ValueError(f"time_column must be 'end' or 'start', got {time_column!r}")
    duration = None
    with open(hypnogram_path) as f:
        row = [v for v in re.split(r"[\t ]+", f.readline().strip()) if v]
        if row and "Duration_sec" in row[0]:
            duration = float(row[1])
    if not duration:
        raise ValueError(f"{hypnogram_path}: no Duration_sec header")
    annotations = np.loadtxt(hypnogram_path, skiprows=2, usecols=0, dtype=str, delimiter="\t", ndmin=1)
    onsets = np.loadtxt(hypnogram_path, skiprows=2, usecols=1, dtype=float, delimiter="\t", ndmin=1)
    if np.any(np.diff(onsets) < 0):
        raise ValueError(f"{hypnogram_path}: onsets are not sorted")
    n_epochs = int(duration // EPOCH_S)
    starts = np.arange(n_epochs) * EPOCH_S  # FIX: was np.linspace(0, duration, n) -> drifting epochs
    if time_column == "end":
        # first bout whose end lies after the epoch start; epochs after the last end stay unlabeled
        seg = np.searchsorted(onsets, starts, side="right")
        valid = seg < len(onsets)
        if abs(onsets[-1] - duration) > EPOCH_S:
            log.warning("%s: last bout ends at %.1f s but Duration_sec is %.1f s; check time_column",
                        Path(hypnogram_path).name, onsets[-1], duration)
    else:
        seg = np.searchsorted(onsets, starts, side="right") - 1
        valid = seg >= 0
    codes_per_segment = np.array([annotation_to_code(a, label_map) for a in annotations])
    codes = np.where(valid, codes_per_segment[np.clip(seg, 0, len(onsets) - 1)], 0)
    table = {}
    for a, c in zip(annotations, codes_per_segment):
        code, n = table.get(str(a), (int(c), 0))
        table[str(a)] = (code, n + 1)
    return torch.tensor(codes, dtype=torch.long), table


def load_oxford_rec(edf_path, hypnogram_path, new_sr: float = 128.0, device: str = "cuda",
                    channels: list[str] | None = None, label_map: dict[str, int] | None = None) -> dict:
    """Convert one Oxford recording (EDF + hypnogram); writes one sub-folder per channel.

    Returns:
        The annotation table from ``hyp_to_tensor``, so the caller can log every string
        and the code it received.
    """
    edf_path, hypnogram_path = Path(edf_path), Path(hypnogram_path)
    y, table = hyp_to_tensor(hypnogram_path, label_map)
    log.info("%s annotation -> code: %s", hypnogram_path.name, table)
    sig, fs = parse_edf(edf_path, channels)
    sig = _resample_filter(sig.T, fs, new_sr, device)
    win = int(new_sr * EPOCH_S)
    for ch in range(sig.shape[0]):
        x_ch, y_ch = _window_and_align(sig[ch], y, win, f"{edf_path.name} ch{ch}")
        out = edf_path.parent / f"{edf_path.stem}_ch{ch}"
        out.mkdir(exist_ok=True)
        torch.save(x_ch, out / f"X_ch{ch}.pt")
        torch.save(y_ch, out / f"y_ch{ch}.pt")
    return table


def _pair_by_stem(edf_paths, hyp_paths) -> list[tuple[str, str]]:
    """Pair EDF files with hypnograms by recording name.

    Falls back to sort order (the old behaviour) with a warning listing the pairs if any
    EDF has zero or several name matches.

    Raises:
        RuntimeError: If the two lists have different lengths.
    """
    if len(edf_paths) != len(hyp_paths):
        raise RuntimeError(f"{len(edf_paths)} EDF files but {len(hyp_paths)} hypnograms")
    pairs = []
    for e in edf_paths:
        match = [h for h in hyp_paths if Path(h).name.startswith(Path(e).stem)]
        if len(match) != 1:
            pairs = list(zip(edf_paths, hyp_paths))
            log.warning("could not pair files by name; using sort order - CHECK: %s",
                        [(Path(a).name, Path(b).name) for a, b in pairs])
            return pairs
        pairs.append((e, match[0]))
    return pairs


def scan_and_move(initial_dir, dest_dir=".") -> set[str]:
    """Move every ``.pt`` file under ``initial_dir`` to ``dest_dir/<recording>/``.

    A file with the same name from an earlier run is replaced, so re-running a conversion
    updates the recordings instead of adding "<name>0", "<name>1" copies. Folders in
    ``dest_dir`` that this call did not write (for example from an older converter) are
    listed in a warning: delete them, or the pipeline will treat them as extra recordings.

    Returns:
        The names of the recording folders written.
    """
    dest_dir = Path(dest_dir)
    files = [Path(f) for f in Path(initial_dir).rglob("*.pt")]
    written, replaced = set(), 0
    for file in files:
        rec_id = file.parent.stem
        target = dest_dir / rec_id / file.name
        replaced += target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(file, target)
        written.add(rec_id)
    log.info("moved %d files from %s to %s (%d replaced files from an earlier run)", len(files), initial_dir,
             dest_dir, replaced)
    existing = dest_dir.iterdir() if dest_dir.exists() else []
    stale = sorted(d.name for d in existing if d.is_dir() and d.name not in written)
    if written and stale:
        log.warning("%s: %d folders were not written by this run and will be read as extra recordings "
                    "(older conversions?) - delete them: %s", dest_dir, len(stale), stale[:20])
    return written


# ---------------------------------------------------------------------------------------------
RAW_ROOT = r"G:\RAW_EXTERNAL_SLEEP_DATASETS"
OXFORD_SETS = [  # (sub-folder, hypnogram glob suffix, EDF channel names, output name)
    ("optogenetic_stimulation", "*_TY.hyp", ["Signal 0", "Signal 1"], "oxford_opto"),
    ("pilot", "*_consensus.hyp", ["Signal 0", "Signal 1"], "oxford_pilot"),
    ("test", "*_consensus_state_annotation.hyp", ["Signal 2", "Signal 3"], "oxford_test"),
    ("sleep_deprivation", "*_CBD.hyp", ["Signal 16", "Signal 17"], "oxford_deprivation"),
]


def run(raw_root: str = RAW_ROOT, new_sr: float = 128.0, device: str = "cuda", out_name: str = "128HZ_CONVERTED",
        label_map: dict[str, int] | None = None) -> None:
    """Convert MLS-Net, MSSV and the four Oxford sets, then gather the outputs.

    Args:
        raw_root: Folder that holds the downloaded datasets.
        new_sr: Output sample rate in Hz.
        device: Device for resampling and filtering.
        out_name: Output folder name under ``raw_root``.
        label_map: Explicit Oxford annotation -> raw code overrides.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw_root = Path(raw_root)
    out_root = raw_root / out_name

    mls = raw_root / "labeled-val-mlsnet" / "sleepnet_fea10_xy.npz"
    x, y = load_mlsnet_npz(mls, new_sr, device)
    (mls.parent / mls.stem).mkdir(exist_ok=True)
    torch.save(x, mls.parent / mls.stem / "X.pt")
    torch.save(y, mls.parent / mls.stem / "y.pt")

    for p in sorted(glob.glob(str(raw_root / "mssv_128hz" / "sub-*" / "eeg"))):
        load_mssv_rec(p, new_sr, device)

    all_tables = {}
    for sub, hyp_glob, chans, _ in OXFORD_SETS:
        base = raw_root / "Oxford" / sub / sub
        edfs = sorted(glob.glob(str(base / "recordings" / "*.edf")))
        hyps = sorted(glob.glob(str(base / "annotations" / hyp_glob)))
        for edf, hyp in _pair_by_stem(edfs, hyps):
            for k, v in load_oxford_rec(edf, hyp, new_sr, device, chans, label_map).items():
                code, n = all_tables.get(k, (v[0], 0))
                all_tables[k] = (code, n + v[1])
    log.info("ALL Oxford annotation strings -> raw code (count of segments): %s", all_tables)

    moves = [(raw_root / "Oxford" / sub, out_root / name) for sub, _, _, name in OXFORD_SETS]
    moves += [(raw_root / "mssv_128hz", out_root / "mssv"), (raw_root / "labeled-val-mlsnet", out_root / "mlsnet")]
    for src, dst in moves:
        scan_and_move(src, dst)


if __name__ == "__main__":
    run()
