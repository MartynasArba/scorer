"""
Automatic scoring for the GUI (Qt-free): find trained models, decide which folders to score,
run SleepScorer on each, write states + confidence + a provenance JSON. Replaces the GUI's use of
scoring.score_signal and its hard-coded weight paths.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from scorer.data.recordings import discover_recordings

log = logging.getLogger(__name__)

HEADS = ("context_mlp", "context_rf", "gru", "linear_probe")


@dataclass
class ModelChoice:
    """One encoder + head pair offered in the automatic-scoring tab.

    Attributes:
        label: Text shown in the model list, e.g. "context_mlp · tuned · adversarial".
        encoder: Encoder checkpoint path.
        head: Head checkpoint path.
        mode: "tuned", "zeroshot" or "" for runs made before modes existed.
        run: Run folder name.
    """
    label: str
    encoder: str
    head: str
    mode: str = ""
    run: str = ""

    def exists(self) -> bool:
        """True if both checkpoint files exist."""
        return Path(self.encoder).exists() and Path(self.head).exists()


def discover_models(run_dir: str | Path) -> list[ModelChoice]:
    """List every (encoder, head) pair a pipeline run produced, from its ``manifest.json``.

    Each head is paired with the encoder of the ``embed`` stage of the same mode, so a
    head is never matched with the wrong encoder. Zero-shot pairs are listed first.

    Raises:
        FileNotFoundError: If the folder has no manifest (not a pipeline run folder).
    """
    run_dir = Path(run_dir)
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"{run_dir} has no manifest.json (select a run folder made by scorer.pipeline.run)")
    stages = json.loads(mpath.read_text()).get("stages", {})
    out = []
    for key, entry in stages.items():
        stage, _, mode = key.partition("@")
        if stage not in HEADS or "head" not in entry:
            continue
        embed = stages.get(f"embed@{mode}" if mode else "embed")
        if not embed:
            continue
        label = f"{stage}" + (f" · {mode}" if mode else "") + f" · {embed.get('encoder_stage') or 'encoder'}"
        out.append(ModelChoice(label, embed["encoder_path"], entry["head"], mode, run_dir.name))
    order = {h: i for i, h in enumerate(HEADS)}
    out.sort(key=lambda m: (m.mode != "zeroshot", order.get(m.label.split(" ")[0], 9)))
    return out


def expand_targets(folder: str | Path) -> list[Path]:
    """Turn a selection into recordings to score.

    A windowed X file gives itself; a recording folder gives itself; a folder of
    recording folders gives each of them.
    """
    if Path(folder).is_file():
        return [Path(folder)]
    return [r.directory for r in discover_recordings(folder)]


def target_name(target: Path) -> str:
    """Return the name used for output files: the folder name, or the X file's stem without ``_X``."""
    if target.is_file():
        stem = target.stem
        return stem[:-2] if stem.endswith("_X") else stem
    return target.name


def load_target(target: Path, channel: int, layout: str = "auto"):
    """Load the windows of one channel as [N, 1, T] from a recording folder or a single X file.

    Raises:
        ValueError: If the channel does not exist in a single file.
        RuntimeError: If a folder holds more than one recording.
    """
    from scorer.data.recordings import load_recording

    if target.is_file():
        from scorer.gui.core.recording import RecordingData

        x = RecordingData.from_file(target, None, layout=layout).x
        if channel >= x.shape[1]:
            raise ValueError(f"{target.name} has {x.shape[1]} channels; channel {channel} does not exist")
        return x[:, channel:channel + 1]
    recs = discover_recordings(target)
    if len(recs) != 1:
        raise RuntimeError(f"{len(recs)} recordings in folder")
    x, _ = load_recording(recs[0], [channel], layout=layout)
    return x


@dataclass
class ScoreRequest:
    """Everything the automatic-scoring tab asks for, in one object (also saved as provenance).

    Attributes:
        targets: Recording folders or X files to score.
        out_dir: Folder for state files.
        channel: Channel to score.
        heuristics: "none", "lite" or "full".
        smooth_sigma: Probability smoothing in windows.
        layout: Chunk axis order ("auto", "nct", "cnt").
        metadata_sample_rate: Sample rate of the windowed data from the settings tab, or
            None to infer it from the window length.
        tag: Optional suffix for output names.
        overwrite: Replace existing outputs.
        device: Device for the models.
    """
    targets: list[str]
    out_dir: str
    channel: int = 0
    heuristics: str = "lite"
    smooth_sigma: float = 0.0
    layout: str = "auto"
    metadata_sample_rate: float | None = None  # sample rate of the windowed data, from the settings tab
    tag: str = ""
    overwrite: bool = False
    device: str = "auto"


@dataclass
class ScoreOutcome:
    """Result of scoring one target.

    Attributes:
        target: The folder or file scored.
        ok: True on success.
        states_path: Output state file (also set when skipped because it exists).
        message: Error, skip reason or a note (e.g. an assumed sample rate).
        windows: Number of windows scored.
        counts: Windows per raw code.
        mean_confidence: Mean model confidence.
        source_sample_rate: Sample rate the data was taken to have.
    """
    target: str
    ok: bool
    states_path: str = ""
    message: str = ""
    windows: int = 0
    counts: dict = field(default_factory=dict)
    mean_confidence: float | None = None
    source_sample_rate: float | None = None


def source_rate_for(win_samples: int, model_epoch_s: float, metadata_rate: float | None) -> tuple[float, str]:
    """Work out the sample rate of the data to score.

    Windows must be epochs of the model's length (4 s), so the rate follows from the
    window length. If the settings tab gives a rate, it must agree.

    Returns:
        ``(rate, note)``; the note explains an inferred rate and is "" otherwise.

    Raises:
        ValueError: If the settings-tab rate gives a different window duration.
    """
    inferred = win_samples / model_epoch_s
    if metadata_rate:
        if abs(win_samples / metadata_rate - model_epoch_s) > 1e-6:
            raise ValueError(f"windows of {win_samples} samples at {metadata_rate:g} Hz last "
                             f"{win_samples / metadata_rate:g} s, but the model scores {model_epoch_s:g}-s epochs. "
                             f"Check sample_rate in the settings tab or re-window the data.")
        return float(metadata_rate), ""
    return inferred, (f"sample_rate not set; assumed {inferred:g} Hz from {win_samples}-sample windows "
                      f"of {model_epoch_s:g} s")


def output_path(out_dir: str | Path, target: Path, head_kind: str, mode: str | None, tag: str) -> Path:
    """Return ``<out_dir>/<target>_<head>[_<mode>][_<tag>]_states.pkl``."""
    parts = [target_name(Path(target)), head_kind] + ([mode] if mode else []) + ([tag] if tag else [])
    return Path(out_dir) / ("_".join(parts) + "_states.pkl")


def score_targets(scorer, req: ScoreRequest, progress: Callable[[str], None] = lambda s: None,
                  cancelled: Callable[[], bool] = lambda: False) -> list[ScoreOutcome]:
    """Score every target of ``req`` with ``scorer`` (a ``SleepScorer``).

    For each target it writes the state file, ``<stem>_confidence.pkl`` and
    ``<stem>_scoring.json`` (model, request and outcome, for provenance). A failure is
    recorded and the next target continues.

    Args:
        scorer: Loaded ``SleepScorer``.
        req: The request.
        progress: Called with one line per step.
        cancelled: Checked before each target.

    Returns:
        One ``ScoreOutcome`` per target handled.
    """
    import numpy as np

    from scorer.inference.scoring import save_states

    outcomes = []
    info = scorer.describe()
    for i, target in enumerate(map(Path, req.targets)):
        if cancelled():
            outcomes.append(ScoreOutcome(str(target), False, message="cancelled"))
            break
        out = output_path(req.out_dir, target, info["head"], info.get("mode"), req.tag)
        if out.exists() and not req.overwrite:
            outcomes.append(ScoreOutcome(str(target), False, str(out), "exists, skipped (tick overwrite)"))
            progress(f"[{i + 1}/{len(req.targets)}] {target.name}: exists, skipped")
            continue
        t0 = time.time()
        try:
            progress(f"[{i + 1}/{len(req.targets)}] {target.name}: loading")
            x = load_target(target, req.channel, req.layout)
            rate, note = source_rate_for(x.shape[-1], scorer.epoch_seconds, req.metadata_sample_rate)
            progress(f"[{i + 1}/{len(req.targets)}] {target.name}: scoring {len(x)} windows")
            res = scorer.score_windows(x, smooth_sigma=req.smooth_sigma, heuristics=req.heuristics,
                                       source_sample_rate=rate)
            save_states(res, out)
            vals, cnt = np.unique(res.gui_codes, return_counts=True)
            oc = ScoreOutcome(str(target), True, str(out), note, int(len(res.gui_codes)),
                              {int(v): int(c) for v, c in zip(vals, cnt)}, float(res.confidence.mean()), rate)
            provenance = {"scored": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
                          "model": info, "request": {k: v for k, v in asdict(req).items() if k != "targets"},
                          "outcome": asdict(oc)}
            out.with_name(out.stem + "_scoring.json").write_text(json.dumps(provenance, indent=1, default=str))
            progress(f"[{i + 1}/{len(req.targets)}] {target.name}: done, mean confidence {oc.mean_confidence:.3f}")
        except Exception as e:  # keep going with the next recording
            log.exception("scoring %s failed", target)
            oc = ScoreOutcome(str(target), False, message=f"{type(e).__name__}: {e}")
            progress(f"[{i + 1}/{len(req.targets)}] {target.name}: FAILED ({e})")
        outcomes.append(oc)
    return outcomes
