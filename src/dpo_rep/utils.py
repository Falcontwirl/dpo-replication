"""Seeding, device/precision handling, run directories, logging and timing."""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, precision: str):
    """bf16 autocast on CUDA only. Weights stay fp32 so that tiny updates (lr=1e-6) are not rounded away.
    MPS/CPU run in plain fp32."""
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def gpu_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "apple-mps"
    return "cpu"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Accumulates wall-clock seconds per named section. Synchronizes the device at section
    boundaries so asynchronous GPU work is attributed to the right section."""

    def __init__(self, device: torch.device):
        self.device = device
        self.totals: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

    def get(self, name: str) -> float:
        return self.totals.get(name, 0.0)

    def wall(self) -> float:
        return time.perf_counter() - self._start


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_run_dir(cfg, name: str, overwrite: bool = False) -> Path:
    run_dir = Path(cfg.paths.runs_dir) / name
    if run_dir.exists() and (run_dir / "metrics.jsonl").exists() and not overwrite:
        raise FileExistsError(f"{run_dir} already has metrics; delete it or pick another --name")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return run_dir


def linear_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from ~0 to the base lr over warmup_steps optimizer steps, then constant (paper App B)."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))


def disable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
