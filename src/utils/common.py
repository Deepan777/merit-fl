"""Reproducibility, logging, and I/O utilities for MERIT-FL."""
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed: int):
    """Deterministic seeding across random, numpy, torch (CPU+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(base: str = "results", tag: str = "") -> str:
    """New directory results/<timestamp>[-tag][-n]; never reuses one. Creation
    is atomic, so parallel workers starting in the same second cannot collide."""
    name = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    i = 0
    while True:
        run = os.path.join(base, name if i == 0 else f"{name}-{i}")
        try:
            os.makedirs(run)
            return run
        except FileExistsError:
            i += 1


def get_logger(run_dir: str) -> logging.Logger:
    """One logger per run, writing to that run's run.log and the console."""
    logger = logging.getLogger(f"merit_fl.{os.path.abspath(run_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    handlers = [logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")]
    if not os.environ.get("MERIT_FL_QUIET"):          # tuning workers: file only
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def close_logger(logger: logging.Logger):
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def _clean(obj):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(obj, path):
    """Never-overwrite JSON saver (an existing file is backed up first)."""
    if os.path.exists(path):
        os.rename(path, path + "." + time.strftime("%H%M%S") + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_clean(obj), f, sort_keys=False)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict:
    """Code version, packages and hardware of a run (env.json)."""
    pkgs = {}
    for p in ("numpy", "pandas", "scikit-learn", "PyYAML", "torch", "scipy",
              "transformers", "accelerate", "bitsandbytes"):          # the last three: hf slow path
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    cuda = torch.cuda.is_available()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        # the experiment registry is rewritten by the runner itself while runs execute
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                               ".", ":(exclude)experiments/registry.csv",
                               ":(exclude)experiments/v2_registry.csv")),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
    }
