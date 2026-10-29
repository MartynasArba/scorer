"""
Model training from the GUI (Qt-free part).

The training tab never runs training code inside the GUI process. It starts the same command
you would type,

    python -u -m scorer.pipeline.run --config <file> --stages ... --set key=value ...

as a child process, so a CUDA out-of-memory error or a crash cannot take the GUI (or unsaved
labels) down with it, GPU memory is returned when the run ends, DataLoader worker processes are
spawned from the pipeline's own entry point (safe on Windows), and a run started from the GUI is
reproducible on the command line. This module builds and checks that command, parses its log
output into progress, and reads the run folder afterwards. Nothing here changes the pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HEAD_SECTIONS = ("context_rf", "context_mlp", "gru", "linear_probe")
ENCODER_SECTIONS = ("simclr", "supcon", "adversarial")


def package_root() -> Path:
    """Return the folder that contains the ``scorer`` package; the training process runs there."""
    import scorer

    return Path(scorer.__file__).resolve().parent.parent


def config_dir() -> Path:
    """Return ``scorer/config``, where the config templates live."""
    return package_root() / "scorer" / "config"


def templates() -> dict[str, Path]:
    """Return the bundled config templates as ``{name: path}`` (e.g. ``pipeline_128hz``)."""
    return {p.stem: p for p in sorted(config_dir().glob("*.toml"))}


def resolve(path: str | Path) -> Path:
    """Return ``path`` as is if absolute, else relative to ``package_root()`` (as the child process sees it)."""
    p = Path(path)
    return p if p.is_absolute() else package_root() / p


def child_env(base: dict | None = None) -> dict:
    """Return the environment for the training process.

    ``PYTHONPATH`` gets the package root first (so ``scorer`` imports from this copy), and
    output is unbuffered UTF-8 so log lines arrive as they are written.
    """
    env = dict(os.environ if base is None else base)
    root = str(package_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def toml_value(value) -> str:
    """Turn a Python value into a TOML literal for ``--set`` (strings are quoted, so 'debug-1' stays a string)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value))  # JSON string escaping is valid TOML basic-string escaping


def build_overrides(run_name: str = "", mode: str = "", device: str = "", extra: str = "") -> list[str]:
    """Turn the training tab's fields into ``--set`` overrides.

    Args:
        run_name: ``run.name``; blank leaves the config's value.
        mode: "tuned" or "zeroshot"; anything else leaves the config's value.
        device: "auto", "cuda" or "cpu"; anything else leaves the config's value.
        extra: Further ``section.key=value`` items separated by ";".

    Returns:
        Override strings such as ``'run.name="debug"'``.

    Raises:
        ValueError: If an extra item does not look like ``section.key=value``.
    """
    out = []
    if run_name.strip():
        out.append(f"run.name={toml_value(run_name.strip())}")
    if mode in ("tuned", "zeroshot"):
        out.append(f"run.mode={toml_value(mode)}")
    if device in ("auto", "cuda", "cpu"):
        out.append(f"run.device={toml_value(device)}")
    for item in (extra or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"extra override {item!r} must look like section.key=value")
        out.append(item)
    return out


def build_command(config_path: str | Path, stages: list[str], overrides: list[str], force_prepare: bool = False,
                  python: str | None = None) -> list[str]:
    """Return the training command as an argument list.

    The command is ``python -u -m scorer.pipeline.run --config <file> --stages ...
    --set ...``, the same one you would type, so a GUI run can be repeated on the
    command line.

    Args:
        config_path: TOML config.
        stages: Stages to run; empty runs every enabled stage.
        overrides: ``section.key=value`` strings.
        force_prepare: Add ``--force-prepare``.
        python: Interpreter; defaults to the one running the GUI.
    """
    cmd = [python or sys.executable, "-u", "-m", "scorer.pipeline.run", "--config", str(config_path)]
    if stages:
        cmd += ["--stages", *stages]
    for o in overrides:
        cmd += ["--set", o]
    if force_prepare:
        cmd.append("--force-prepare")
    return cmd


def command_text(cmd: list[str]) -> str:
    """Return the command as one copy-pasteable line, quoted for the current OS."""
    import shlex

    return " ".join(shlex.quote(c) if os.name != "nt" else (f'"{c}"' if " " in c else c) for c in cmd)


# ---------------------------------------------------------------------------------------------
# config checks
# ---------------------------------------------------------------------------------------------
@dataclass
class ConfigCheck:
    """Result of ``check_config``.

    Attributes:
        ok: True if there are no errors.
        errors: Problems that block a run.
        warnings: Settings that are allowed but probably unintended.
        default_stages: Stages the pipeline would run by default.
        run_dir: Run folder the config points at.
        mode: Run mode after overrides.
        cfg: The loaded config, or None if it failed to load.
    """
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    default_stages: list[str] = field(default_factory=list)
    run_dir: Path | None = None
    mode: str = ""
    cfg: object = None


def _referenced(cfg) -> set[str]:
    """Return every dataset name an enabled section of the config uses."""
    names = set()
    for sec in (cfg.simclr, cfg.supcon):
        if sec.enabled:
            names |= set(sec.datasets)
    if cfg.adversarial.enabled and cfg.run.mode == "tuned":
        names |= set(cfg.adversarial.source) | set(cfg.adversarial.target)
    for head in (cfg.context_mlp, cfg.gru, cfg.linear_probe):
        if head.enabled:
            names |= set(head.train) | set(head.val)
    if cfg.context_rf.enabled:
        names |= set(cfg.context_rf.train)
    return names | set(cfg.evaluate.datasets) | set(cfg.embed.datasets)


def check_config(path: str | Path, overrides: list[str] | None = None) -> ConfigCheck:
    """Load the config exactly as the pipeline will (no torch needed) and list problems.

    Checks that it parses, that every referenced dataset folder exists, and collects
    the config's own warnings.

    Args:
        path: TOML config.
        overrides: ``section.key=value`` strings, applied as on the command line.
    """
    import tomllib

    from scorer.pipeline.config import load_config
    from scorer.pipeline.stages import enabled_stages

    try:
        cfg = load_config(path, overrides or [])
    except (OSError, tomllib.TOMLDecodeError, KeyError, ValueError, TypeError) as e:
        return ConfigCheck(False, [f"{type(e).__name__}: {e}"])
    res = ConfigCheck(True, cfg=cfg, mode=cfg.run.mode)
    res.warnings += list(cfg.warnings())
    for name in sorted(_referenced(cfg)):
        if not resolve(cfg.datasets[name]).exists():
            res.errors.append(f"dataset {name!r}: folder not found: {cfg.datasets[name]}")
    if not cfg.datasets:
        res.errors.append("no [datasets] defined")
    res.default_stages = enabled_stages(cfg)
    res.run_dir = resolve(cfg.run.runs_dir) / cfg.run.name
    cache = resolve(cfg.cache.root)
    if not cache.exists():
        res.warnings.append(f"cache folder {cfg.cache.root} does not exist yet; 'prepare' will create it")
    res.ok = not res.errors
    return res


def stage_problems(cfg, stages: list[str], manifest: dict | None = None) -> list[tuple[str, str]]:
    """Check a stage selection before starting it.

    Args:
        cfg: Loaded config.
        stages: Stages ticked in the tab.
        manifest: The run folder's manifest, so stages done earlier count as available.

    Returns:
        ``(level, message)`` pairs: "error" blocks the run (a disabled section, a missing
        prerequisite such as embeddings or an init encoder, adversarial in zero-shot
        mode); "warning" asks to confirm.
    """
    out = []
    manifest = manifest or {}
    done = set((manifest.get("stages") or {}).keys())
    mode = cfg.run.mode
    for s in stages:
        if s in ENCODER_SECTIONS + HEAD_SECTIONS and not getattr(cfg, s).enabled:
            out.append(("error", f"[{s}] has enabled = false in the config; enable it or untick the stage"))
    if "adversarial" in stages and mode == "zeroshot":
        out.append(("error", "run.mode = zeroshot skips domain alignment: untick 'adversarial' or use mode tuned"))
    # a stage needs the outputs of an earlier one: selected now, or already in this run's manifest
    have = lambda st: st in stages or st in done or f"{st}@{mode}" in done  # noqa: E731
    if any(s in stages for s in ("simclr", "supcon", "adversarial", "embed")) and not have("prepare"):
        out.append(("warning", "'prepare' has not run for this run folder; the cache must already exist"))
    for s in HEAD_SECTIONS + ("evaluate",):
        if s in stages and not have("embed"):
            out.append(("error", f"'{s}' needs embeddings: tick 'embed' (none recorded for mode {mode})"))
    if "evaluate" in stages:
        for h in cfg.evaluate.heads:
            if not have(h):
                out.append(("error", f"evaluate uses head '{h}', which is neither ticked nor trained in mode {mode}"))
    if "supcon" in stages and cfg.supcon.init in ENCODER_SECTIONS and not have(cfg.supcon.init):
        out.append(("error", f"supcon starts from '{cfg.supcon.init}', which is neither ticked nor done"))
    if "adversarial" in stages and cfg.adversarial.init in ENCODER_SECTIONS and not have(cfg.adversarial.init):
        out.append(("error", f"adversarial starts from '{cfg.adversarial.init}', which is neither ticked nor done"))
    return out


# ---------------------------------------------------------------------------------------------
# progress from the log
# ---------------------------------------------------------------------------------------------
_STAGE = re.compile(r"=+ stage (\w+) =+")
_DONE = re.compile(r"stage (\w+) done in (\d+)s")
_FAILED = re.compile(r"stage (\w+) FAILED")
_EPOCH = re.compile(r"\b([a-z_]+) epoch (\d+)/(\d+)")
_PREP = re.compile(r"\[([\w-]+)\] (\d+)/(\d+) ")
_ALL = re.compile(r"all stages finished")


@dataclass
class Progress:
    """Training progress parsed from the pipeline's log lines.

    Attributes:
        stages: Stages of this run, for the "(i/n)" counter.
        current: Stage now running.
        done: Stages finished.
        failed: Stage that failed, or "".
        step: ``(i, n)`` epoch or recording counter inside the current stage.
        step_label: What ``step`` counts, e.g. "simclr epoch".
        finished: True after "all stages finished".
    """
    stages: list[str] = field(default_factory=list)
    current: str = ""
    done: list[str] = field(default_factory=list)
    failed: str = ""
    step: tuple[int, int] = (0, 0)  # epoch / recording counter inside the current stage
    step_label: str = ""
    finished: bool = False

    def feed(self, line: str) -> bool:
        """Update from one log line; return True if anything changed."""
        if m := _STAGE.search(line):
            self.current, self.step, self.step_label = m.group(1), (0, 0), ""
        elif m := _DONE.search(line):
            if m.group(1) not in self.done:
                self.done.append(m.group(1))
            self.step = (0, 0)
        elif m := _FAILED.search(line):
            self.failed = m.group(1)
        elif m := _EPOCH.search(line):
            self.step, self.step_label = (int(m.group(2)), int(m.group(3))), f"{m.group(1)} epoch"
        elif (m := _PREP.search(line)) and self.current == "prepare":
            self.step, self.step_label = (int(m.group(2)), int(m.group(3))), f"{m.group(1)} recordings"
        elif _ALL.search(line):
            self.finished = True
        else:
            return False
        return True

    def summary(self) -> str:
        """Return a short status such as 'stage supcon (3/7)', 'finished 7 stage(s)' or 'FAILED in stage embed'."""
        if self.failed:
            return f"FAILED in stage {self.failed}"
        if self.finished:
            return f"finished {len(self.done)} stage(s)"
        if not self.current:
            return "starting"
        pos = f" ({len(self.done) + 1}/{len(self.stages)})" if self.stages else ""
        return f"stage {self.current}{pos}"


# ---------------------------------------------------------------------------------------------
# run folder
# ---------------------------------------------------------------------------------------------
def read_manifest(run_dir: str | Path) -> dict:
    """Load ``manifest.json`` of a run folder (an empty manifest if there is none)."""
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"stages": {}}


def manifest_rows(manifest: dict) -> list[dict]:
    """Return one row per finished stage (stage, time, main output, all outputs as JSON), in pipeline order."""
    rows = []
    for key, entry in (manifest.get("stages") or {}).items():
        outputs = {k: v for k, v in entry.items() if k not in ("finished", "config_digest")}
        main = next((outputs[k] for k in ("head", "final", "summary", "emb_root", "report") if k in outputs), "")
        rows.append({"stage": key, "finished": entry.get("finished", ""), "output": str(main),
                     "details": json.dumps(outputs, default=str)})
    order = ["inspect", "prepare", "simclr", "supcon", "adversarial", "embed", "context_rf", "context_mlp", "gru",
             "linear_probe", "evaluate"]
    rows.sort(key=lambda r: (order.index(r["stage"].split("@")[0]) if r["stage"].split("@")[0] in order else 99,
                             r["stage"]))
    return rows


def read_summaries(run_dir: str | Path) -> list[dict]:
    """Return the rows of every ``evaluate/<mode>/summary.csv`` in the run (older runs: ``evaluate/summary.csv``)."""
    out = []
    for p in sorted(Path(run_dir, "evaluate").glob("**/summary.csv")):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                row.setdefault("mode", p.parent.name if p.parent.name != "evaluate" else "")
                out.append(row)
    return out


def tail_lines(path: str | Path, n: int = 40) -> list[str]:
    """Return the last ``n`` lines of a text file (reads at most the last 64 kB); [] if it does not exist."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64_000))
        return f.read().decode("utf-8", "replace").splitlines()[-n:]
