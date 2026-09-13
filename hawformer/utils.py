"""Utilities: seeding, logging, device selection, config loading."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

LOGGER_NAME = "hawformer"


def get_logger(name: str = LOGGER_NAME, logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_backends(tf32: bool = True) -> None:
    """Settings that change speed, not results, and are worth stating once.

    PyTorch has defaulted fp32 matmuls to full IEEE precision since 1.12, which
    leaves the tensor cores idle on any Ampere-or-later card.  TF32 keeps the
    fp32 exponent and rounds the mantissa to 10 bits: for a model trained under
    Huber loss on z-scored targets that is far below the noise floor of the
    data, and it is what every published throughput figure on these cards
    assumes.  expandable_segments keeps the caching allocator from fragmenting
    into a shape where a 5 GB working set no longer fits in 8 GB of VRAM --
    which, under WSL, does not raise OOM but silently spills to host memory
    over PCIe and costs two orders of magnitude.
    """
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:                                        # noqa: BLE001
            pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config, following an optional `inherit:` chain."""
    path = str(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(path), parent)
        base = load_config(parent_path)
        cfg = deep_update(base, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class Normalizer:
    """Z-score normaliser fitted on the training split only."""

    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, x: np.ndarray, mask: Optional[np.ndarray] = None) -> "Normalizer":
        v = x[mask] if mask is not None else x
        v = v[np.isfinite(v)]
        std = float(v.std())
        return cls(mean=float(v.mean()), std=std if std > 1e-6 else 1.0)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Normalizer":
        return cls(**d)
