"""Small shared helpers for training loops: logging, seeding, devices, best-state tracking, memory logs."""
from __future__ import annotations

import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """configure the root logger once (file + stdout). Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(LOG_FORMAT)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in root.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger("scorer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def log_memory(logger: logging.Logger, tag: str) -> None:
    msg = f"[mem] {tag}"
    if torch.cuda.is_available():
        msg += (f" | cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                f"reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB, "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    try:
        import psutil  # optional

        msg += f" | process RSS {psutil.Process().memory_info().rss / 1e9:.2f} GB"
    except ImportError:
        pass
    logger.info(msg)


def release_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class BestState:
    """keeps a DEEP copy (on CPU) of the best weights. The old code used state_dict().copy(),
    a shallow copy whose tensors kept changing -> 'best' was always the last epoch."""

    def __init__(self, mode: str = "max"):
        self.mode = mode
        self.value: float | None = None
        self.epoch: int | None = None
        self.state: dict | None = None

    def update(self, value: float, model: torch.nn.Module, epoch: int) -> bool:
        better = self.value is None or (value > self.value if self.mode == "max" else value < self.value)
        if better:
            self.value, self.epoch = value, epoch
            self.state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        return better

    def restore(self, model: torch.nn.Module) -> None:
        if self.state is not None:
            model.load_state_dict(copy.deepcopy(self.state))


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-3, mode: str = "min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """returns True when training should stop"""
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


def make_loader(dataset, num_workers: int, pin_memory: bool, prefetch_factor: int = 2):
    """DataLoader for a WindowBatchStream (which already yields batches)"""
    from torch.utils.data import DataLoader

    kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=False)
    return DataLoader(dataset, **kwargs)
