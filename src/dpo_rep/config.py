"""YAML config loading with layered files and dotted CLI overrides.

Usage in scripts:
    cfg, args = parse_args()
    # python scripts/04_train_dpo.py --config configs/base.yaml --config configs/smoke.yaml \
    #     --set dpo.beta=0.5 --name dpo_beta0.5
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

import yaml


class Cfg(dict):
    """Dict with attribute access for nested configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, dotted: str) -> None:
    """Apply a single `a.b.c=value` override; value is parsed as YAML (so 1e-6, true, [1,2] work)."""
    if "=" not in dotted:
        raise ValueError(f"Override must look like key.path=value, got {dotted!r}")
    path, raw = dotted.split("=", 1)
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section {k!r} in override {dotted!r}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key {path!r} (typo?)")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:  # YAML 1.1 reads "5e-5" (no dot) as a string
            value = float(value)
        except ValueError:
            pass
    node[keys[-1]] = value


def load_config(paths: list[str], overrides: list[str] | None = None) -> Cfg:
    cfg: dict = {}
    for p in paths:
        with open(p) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for o in overrides or []:
        apply_override(cfg, o)
    return Cfg.wrap(cfg)


def parse_args(description: str = "", extra: list[tuple[str, dict]] | None = None) -> tuple[Cfg, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", action="append", default=None,
                        help="YAML config file(s), later files override earlier ones. Default: configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted config override, e.g. dpo.beta=0.5")
    parser.add_argument("--name", default=None, help="Run name (directory under paths.runs_dir)")
    for flag, kwargs in extra or []:
        parser.add_argument(flag, **kwargs)
    args = parser.parse_args()
    cfg = load_config(args.config or ["configs/base.yaml"], args.set)
    return cfg, args
