"""
Preprocessing run for the GUI (Qt-free). The logic is the old PreprocessWidget.run_preprocessing /
_preprocess, moved out of the widget so it can run in a worker thread and be tested.
Storage and filtering are still done by scorer.data.storage / scorer.data.preprocessing, with the
same calls and arguments as before.

Fixed on the way:
  * 'add original file name' was tested with `if self.add_filename_check:` (the widget, always
    truthy), so the file name was always written into the metadata;
  * filter parameters were appended under the wrong key test ('filter_freqs' vs 'filter_params'),
    so EMG band limits overwrote the ECoG ones;
  * when resampling, Watson prescoring ran on the resampled signal with the OLD sample rate;
  * cancelling the memory warning left `bands` undefined (NameError at return);
  * the chunk/window divisibility warning popped up once per chunk;
  * with no metadata_path, metadata was "saved" to the project FOLDER path;
  * 'cuda' was used even on machines without a GPU;
  * with resampling, only ECoG and EMG were resampled: sum-power / band / ratio feature channels
    stayed at the original rate and storage.save_* then cut every channel to the shortest one, so
    the feature channels held a time-compressed copy of the first part of the recording
    (e.g. at 1000 -> 250 Hz, the first quarter, squeezed into the whole file);
  * Watson prescoring only works on one ECoG and one EMG channel; with two ECoG channels the
    spectrogram ran over the channel axis and failed - the first channel of each is now used;
  * raw tensors were passed to storage.save_tensor as [time, channels] (the csv loaders' layout)
    while it expects [channels, time]: it truncated "length" to the channel count and stacked ECoG
    and EMG along time, so 'save raw tensors' wrote a [2*time, 1] array; now transposed first;
  * unset optional_tag / filename became the text "None" in every output file name;
  * existing files skipped by storage (overwrite off) were reported as saved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scorer.gui.core import meta as M

log = logging.getLogger(__name__)

BANDS = {"delta": (0.5, 4), "theta": (5, 9), "alpha": (8, 13), "sigma": (12, 15)}


@dataclass
class PreprocessOptions:
    """All choices of the preprocessing tab, in one object (Qt-free, so it can be tested).

    Attributes:
        file: CSV file to process.
        chunked, chunk_size: Read the file in chunks of this many rows.
        timechop, time_start, time_end: Only read between these times of day (HH:MM:SS).
        resample, new_sr: Resample every channel to this rate (after filtering).
        ecog_filter, ecog_band: Band-pass the ECoG channels (Hz).
        emg_filter, emg_band: Band-pass the EMG channels (Hz).
        notch, notch_freq: Notch filter at this frequency (Hz).
        sum_power, band_power, band_signals, ratio_features: Feature channels to add (at
            most one of the last three).
        smooth, smooth_sigma: Gaussian smoothing of power features (s).
        prescore: Add a Watson (2016) prescoring as labels.
        save_raw, save_preprocessed, save_windowed: What to write.
        win_len: Window length in samples of the output rate.
        overwrite: Replace existing files.
        add_filename: Put the CSV file name into output names.
    """
    file: str = ""
    chunked: bool = False
    chunk_size: int = 1_000_000
    timechop: bool = False
    time_start: str = "19:00:00"
    time_end: str = "07:00:00"
    resample: bool = False
    new_sr: int = 250
    ecog_filter: bool = False
    ecog_band: tuple[float, float] = (0.5, 49.0)
    emg_filter: bool = False
    emg_band: tuple[float, float] = (10.0, 100.0)
    notch: bool = False
    notch_freq: float = 50.0
    sum_power: bool = False
    band_power: bool = False
    band_signals: bool = False
    ratio_features: bool = False
    smooth: bool = False
    smooth_sigma: float = 0.25
    prescore: bool = False
    save_raw: bool = False
    save_preprocessed: bool = False
    save_windowed: bool = False
    win_len: int = 1000
    overwrite: bool = False
    add_filename: bool = False

    @property
    def heavy(self) -> bool:
        """True if a filter or power feature is selected (these need the most memory)."""
        return any((self.ecog_filter, self.emg_filter, self.notch, self.band_power, self.sum_power))

    def output_rate(self, meta: dict) -> float:
        """Return the sample rate of the saved data: ``new_sr`` when resampling, else the raw rate."""
        return float(self.new_sr) if self.resample else M.sample_rate(meta)


def check_options(opts: PreprocessOptions, meta: dict, model_epoch_s: float = 4.0) -> list[tuple[str, str]]:
    """Check the options against the metadata before anything runs.

    Args:
        opts: The options.
        meta: The GUI's metadata (sample rate, channels).
        model_epoch_s: Epoch length the models score; windows of another duration cannot
            be auto-scored.

    Returns:
        ``(level, message)`` pairs: "error" blocks the run, "confirm" needs a yes,
        "info" is only shown.
    """
    out = []
    if not opts.file or not Path(opts.file).exists():
        out.append(("error", "select a .csv file to preprocess"))
    if M.get_float(meta, "sample_rate") is None:
        out.append(("error", "set sample_rate (of the raw file) in the settings tab"))
    for key in ("ecog_channels", "emg_channels"):
        try:
            if not M.parse_channels(meta.get(key)):
                out.append(("error", f"set {key} in the settings tab"))
        except ValueError as e:
            out.append(("error", str(e)))
    n_band = sum((opts.band_power, opts.band_signals, opts.ratio_features))
    if n_band > 1:
        out.append(("error", "select only one of band power / band signals / ratio features"))
    if opts.save_windowed and opts.win_len <= 0:
        out.append(("error", "window length must be positive"))
    if opts.chunked and opts.save_windowed and opts.chunk_size % opts.win_len:
        out.append(("confirm", f"chunk size {opts.chunk_size} is not a multiple of the window length {opts.win_len}: "
                               f"the remainder of every chunk is dropped from the windowed data. Continue?"))
    if opts.chunked and opts.heavy:
        out.append(("info", "filters run per chunk: expect small edge effects at chunk boundaries"))
    if not opts.chunked and opts.heavy:
        out.append(("confirm", "chunk loading is off, so filtering a long recording may run out of memory. Continue?"))
    if opts.save_windowed and M.get_float(meta, "sample_rate"):
        sr = opts.output_rate(meta)
        dur = opts.win_len / sr
        if abs(dur - model_epoch_s) > 1e-6:
            out.append(("confirm", f"windows will last {dur:g} s ({opts.win_len} samples at {sr:g} Hz); the models "
                                   f"score {model_epoch_s:g}-s epochs, so these windows cannot be auto-scored. "
                                   f"Continue?"))
    raw_sr = M.get_float(meta, "sample_rate")
    if raw_sr is not None and raw_sr != int(raw_sr):
        out.append(("error", f"sample_rate must be a whole number of Hz (got {meta.get('sample_rate')})"))
    if raw_sr:
        for on, band, name in ((opts.ecog_filter, opts.ecog_band, "ECoG"), (opts.emg_filter, opts.emg_band, "EMG")):
            if on and not (0 < band[0] < band[1] < raw_sr / 2):
                out.append(("error", f"{name} band {band[0]}-{band[1]} Hz must satisfy 0 < low < high < "
                                     f"{raw_sr / 2:g} Hz (Nyquist of the raw {raw_sr:g} Hz data)"))
        if opts.notch and not (1 < opts.notch_freq < raw_sr / 2 - 1):
            out.append(("error", f"notch frequency must lie inside 1 .. {raw_sr / 2 - 1:g} Hz"))
    if opts.resample and opts.ecog_filter and opts.ecog_band[1] >= opts.new_sr / 2:
        out.append(("error", f"ECoG high cut {opts.ecog_band[1]} Hz is above the new Nyquist {opts.new_sr / 2} Hz"))
    if opts.prescore and opts.chunked:
        out.append(("info", "prescoring runs per chunk: its thresholds are fitted to each chunk separately"))
    try:
        if opts.prescore and (len(M.parse_channels(meta.get("ecog_channels"))) > 1
                              or len(M.parse_channels(meta.get("emg_channels"))) > 1):
            out.append(("info", "prescoring uses the first ECoG and the first EMG channel"))
    except ValueError:
        pass
    return out


def model_preset(meta: dict, model_rate: float = 128.0, epoch_s: float = 4.0) -> dict:
    """Return option values for windows the 128 Hz models can score directly.

    That is: resample to 128 Hz, band-pass ECoG 0.5-49 Hz, save 512-sample windows.
    """
    return {"resample": True, "new_sr": int(model_rate), "ecog_filter": True, "ecog_band": (0.5, 49.0),
            "save_windowed": True, "win_len": int(model_rate * epoch_s)}


def _preprocess(ecog, emg, opts: PreprocessOptions, params: dict, progress: Callable[[str], None]):
    """Filter, add features and resample one block of data (the old widget's ``_preprocess``).

    Records every step in ``params`` ("preprocessing", "filter_params",
    "channels_after_preprocessing"). Resampling runs last and covers every channel,
    features included, so all channels keep the same length.

    Args:
        ecog, emg: Data [time, channels] as returned by the CSV loaders.
        opts: The options.
        params: The GUI's metadata (updated in place).
        progress: Called with one line per step.

    Returns:
        ``(ecog, emg, ecog_power, emg_power, *band_features)`` as [channels, time] tensors
        (None where not computed), with one trailing None when no band option is chosen,
        as ``storage.save_*`` expects.
    """
    from scorer.data.preprocessing import band_powers, band_signals, bandpass_filter, notch_filter, ratio_signals, sum_power

    sr = M.sample_rate(params)
    sr_arg = int(sr) if float(sr).is_integer() else sr
    dev = M.device(params)
    n_ecog, n_emg = M.channel_counts(params)
    params["channels_after_preprocessing"] = ["ecog"] * n_ecog + ["emg"] * n_emg
    params["preprocessing"] = []
    params["filter_params"] = []
    ecog, emg = ecog.T, emg.T  # channels x time

    if opts.ecog_filter:
        params["preprocessing"].append("bandpass")
        params["filter_params"].append(list(opts.ecog_band))
        ecog = bandpass_filter(ecog, sr=sr_arg, freqs=tuple(opts.ecog_band), device=dev)
        progress("ECoG filtered")
    if opts.emg_filter:
        params["preprocessing"].append("bandpass")
        params["filter_params"].append(list(opts.emg_band))
        emg = bandpass_filter(emg, sr=sr_arg, freqs=tuple(opts.emg_band), device=dev)
        progress("EMG filtered")
    if opts.notch:
        params["preprocessing"].append("notch")
        params["notch_freq"] = [opts.notch_freq]
        ecog = notch_filter(ecog, sr=sr_arg, freq=opts.notch_freq, device=dev)
        emg = notch_filter(emg, sr=sr_arg, freq=opts.notch_freq, device=dev)
        progress("notch filter applied")

    smoothing = opts.smooth_sigma if opts.smooth else None
    ecog_power = emg_power = None
    if opts.sum_power:
        ecog_power = sum_power(ecog, smoothing=0.2, sr=sr_arg, device=dev, normalize=True, gaussian_smoothen=smoothing)
        emg_power = sum_power(emg, smoothing=0.2, sr=sr_arg, device=dev, normalize=True, gaussian_smoothen=smoothing)
        params["preprocessing"].append("sum_pows_calculated")
        params["channels_after_preprocessing"] += ["ecog_sum_power"] * n_ecog + ["emg_sum_power"] * n_emg
        progress("sum powers calculated")

    bands: dict = {}
    if opts.band_power:
        bands = band_powers(signal=ecog, bands=BANDS, sr=sr_arg, device=dev, smoothen=smoothing)
        params["preprocessing"].append("band_pows_calculated")
    elif opts.band_signals:
        bands = band_signals(signal=ecog, bands=BANDS, sr=sr_arg, device=dev)
        params["preprocessing"].append("band_signals_calculated")
    elif opts.ratio_features:
        bands = ratio_signals(signal=ecog, emg=emg, sr=sr_arg, device=dev)
        params["preprocessing"].append("ratio_signals_calculated")
    if bands:
        params["channels_after_preprocessing"] += list(bands.keys())
        progress(f"{params['preprocessing'][-1].replace('_', ' ')}")

    if opts.resample and float(opts.new_sr) != float(sr):  # last, so every filter above used the original rate
        import torch
        from torchaudio.functional import resample

        def rs(t):  # every [channels, time] tensor at the original rate, features included
            """Resample a [channels, time] tensor to ``new_sr``; anything else is returned unchanged."""
            return resample(t.contiguous().float(), int(round(sr)), int(opts.new_sr)) if isinstance(t, torch.Tensor) else t

        ecog, emg, ecog_power, emg_power = rs(ecog), rs(emg), rs(ecog_power), rs(emg_power)
        bands = {k: rs(v) for k, v in bands.items()}
        progress(f"resampled every channel {sr:g} -> {opts.new_sr} Hz")
    # exactly the old tuple shape: (ecog, emg, ecog_power, emg_power, *bands), with one trailing
    # None when no band option is selected (the old `bands = {None: None}`), which storage.save_* expects
    return (ecog, emg, ecog_power, emg_power) + (tuple(bands.values()) if bands else (None,))


def run_preprocessing(opts: PreprocessOptions, params: dict, progress: Callable[[str], None] = lambda s: None,
                      cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Run the whole preprocessing job: read, preprocess, optionally prescore, save, write metadata.

    Args:
        opts: The options (checked with ``check_options`` beforehand).
        params: The GUI's metadata; updated with the steps, the new sample rate and the
            metadata path.
        progress: Called with one line per step.
        cancelled: Checked between chunks.

    Returns:
        ``{"chunks", "saved", "skipped", "metadata_path"}``.
    """
    from scorer.data.preprocessing import prescore_watson
    from scorer.data.storage import load_from_csv, load_from_csv_in_chunks, save_metadata, save_tensor, save_windowed

    if opts.add_filename:
        params["filename"] = Path(opts.file).stem
    for key in ("optional_tag", "filename"):  # storage builds file names from these
        if params.get(key) is None:
            params[key] = ""
    times = (opts.time_start, opts.time_end) if opts.timechop else (None, None)
    sr_out = opts.output_rate(params)
    dev = M.device(params)
    summary = {"chunks": 0, "saved": [], "skipped": []}

    def note(kind, result):
        """Record a save result as saved or skipped (storage returns None when a file exists)."""
        if result is None:
            summary["skipped"].append(kind)
            progress(f"{kind}: file exists or nothing to save - skipped (tick overwrite to replace)")
        else:
            summary["saved"].append(str(result[0] if isinstance(result, tuple) else result))

    def handle(ecog, emg, states, chunk):
        """Process and save one block (the whole file, or one chunk)."""
        tensor_seq = (ecog.T, emg.T)  # storage expects [channels, time]; csv loaders return [time, channels]
        if opts.save_raw:
            note("raw", save_tensor(tensor_seq=tensor_seq, metadata=params, overwrite=opts.overwrite, chunk=chunk,
                                    raw=True))
        tensor_seq = _preprocess(ecog, emg, opts, params, progress)
        if opts.prescore:
            progress("prescoring (Watson 2016)")
            # [1, time] each: the first ECoG and the first EMG channel
            states = prescore_watson(tensor_seq[0][:1], tensor_seq[1][:1], opts.win_len,
                                     sample_rate=int(round(sr_out)), device=dev)
        if opts.save_preprocessed:
            note("preprocessed", save_tensor(tensor_seq=tensor_seq, metadata=params, overwrite=opts.overwrite,
                                             chunk=chunk, raw=False))
        if opts.save_windowed:
            note("windowed", save_windowed(tensors=tensor_seq, states=states, metadata=params, win_len=opts.win_len,
                                           chunked=chunk is not None,
                                           **({"chunk_id": chunk} if chunk is not None else {}),
                                           overwrite=opts.overwrite, testing=False))

    if not opts.chunked:
        progress("reading file")
        raw_ecog, raw_emg, states = load_from_csv(opts.file, metadata=params, states=None, times=times)
        handle(raw_ecog, raw_emg, states, None)
        summary["chunks"] = 1
    else:
        for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(
                load_from_csv_in_chunks(opts.file, metadata=params, states=None, chunk_size=opts.chunk_size,
                                        times=times)):
            if cancelled():
                progress(f"cancelled after {i} chunks")
                break
            progress(f"chunk {i}: read")
            handle(ecog_chunk, emg_chunk, states_chunk, i)
            summary["chunks"] = i + 1

    if opts.resample and float(opts.new_sr) != M.sample_rate(params):
        params["preprocessing"] = list(params.get("preprocessing") or []) + ["resampling"]
        params["old_sample_rate"] = params["sample_rate"]
        params["sample_rate"] = int(opts.new_sr)

    meta_path = params.get("metadata_path")
    if not meta_path or Path(meta_path).is_dir():
        meta_path = str(Path(M.project_dir(params)) / f"{M.get_str(params, 'scoring_started', M.timestamp())}_meta.json")
        params["metadata_path"] = meta_path
    save_metadata(meta_path, params)
    summary["metadata_path"] = meta_path
    progress(f"done: {summary['chunks']} chunk(s), metadata saved to {meta_path}")
    return summary
