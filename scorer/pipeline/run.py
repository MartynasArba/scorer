"""
Command-line entry point.

    python -m scorer.pipeline.run --config scorer/config/pipeline_128hz.toml                 # all enabled stages
    python -m scorer.pipeline.run --config ... --stages inspect                              # label sanity report only
    python -m scorer.pipeline.run --config ... --stages embed context_mlp evaluate           # resume from an encoder
    python -m scorer.pipeline.run --config ... --stages context_mlp --set context_mlp.epochs=5 --set run.name=debug
    python -m scorer.pipeline.run --config ... --list                                        # show stages + manifest

Unlike the old pipeline, a failing stage stops the run with a non-zero exit code
(the old try/except logged the error and exited 0).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback

from scorer.pipeline.config import load_config
from scorer.pipeline.stages import STAGE_ORDER, STAGES, RunContext, enabled_stages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scorer.pipeline.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="TOML config file")
    ap.add_argument("--stages", nargs="*", default=None, help=f"subset of {STAGE_ORDER} (default: all enabled)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config value (TOML syntax), repeatable")
    ap.add_argument("--force-prepare", action="store_true", help="re-normalise recordings even if cached")
    ap.add_argument("--list", action="store_true", help="print the stage plan and the run manifest, then exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    from scorer.training.common import setup_logging  # imports torch

    ctx = RunContext.open(cfg)
    logger = setup_logging(ctx.run_dir / "pipeline.log")
    stages = args.stages or enabled_stages(cfg)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; choose from {STAGE_ORDER}")
    stages = sorted(stages, key=STAGE_ORDER.index)

    if args.list:
        print("stages to run:", " -> ".join(stages))
        print(json.dumps(ctx.manifest, indent=1))
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(args.config, ctx.run_dir / f"config_{stamp}.toml")
    (ctx.run_dir / f"config_{stamp}_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str))
    logger.info("run dir %s | stages %s | config %s (digest %s) | overrides %s", ctx.run_dir, stages, args.config,
                cfg.digest(), args.overrides)
    _log_system(logger)
    for w in cfg.warnings():
        logger.warning("[config] %s", w)

    from scorer.training.common import log_memory, release_cuda

    for stage in stages:
        t0 = time.time()
        logger.info("=" * 20 + f" stage {stage} " + "=" * 20)
        try:
            if stage == "prepare":
                STAGES[stage](ctx, force=args.force_prepare)  # type: ignore[call-arg]
            else:
                STAGES[stage](ctx)
        except Exception:
            logger.error("stage %s FAILED after %.0fs\n%s", stage, time.time() - t0, traceback.format_exc())
            log_memory(logger, f"at failure of {stage}")
            return 1
        logger.info("stage %s done in %.0fs", stage, time.time() - t0)
        release_cuda()
    logger.info("all stages finished: %s", stages)
    return 0


def _log_system(logger) -> None:
    import os
    import platform

    import torch

    logger.info("python %s | torch %s | cuda %s%s | cwd %s", platform.python_version(), torch.__version__,
                torch.cuda.is_available(),
                f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
                if torch.cuda.is_available() else "", os.getcwd())


if __name__ == "__main__":
    sys.exit(main())
