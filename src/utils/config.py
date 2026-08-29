"""Config loading utility.

Every experiment (architecture choice, hyperparameters, paths) is described
by a YAML file under configs/. This module is the single place that reads
those files into a plain dict, so every script (split, vocab, features,
train, evaluate, predict) loads config the same way.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "paths": {
        "raw_images_dir": "data/raw/images",
        "captions_file": "data/raw/captions.txt",
        "processed_dir": "data/processed",
        "features_dir": "data/features",
        "models_dir": "models",
    },
    "split": {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "vocab": {
        "min_freq": 5,
        "max_len": 35,
    },
    "encoder": {
        "type": "resnet50",
        "feature_dim": 2048,
        "freeze": True,
    },
    "decoder": {
        "type": "lstm",
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "training": {
        "batch_size": 32,
        "epochs": 30,
        "lr": 1e-3,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
        "lr_scheduler_patience": 2,
        "lr_scheduler_factor": 0.5,
        "early_stop_patience": 5,
        "num_workers": 2,
        "device": "auto",  # "auto" | "cpu" | "cuda"
    },
    "inference": {
        "decoding": "greedy",  # "greedy" | "beam"
        "beam_width": 3,
    },
    "run_name": "base_resnet_lstm",
}


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins), without mutating inputs."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and merge it on top of DEFAULT_CONFIG.

    Any key not specified in the YAML file falls back to DEFAULT_CONFIG,
    so config files only need to declare what they change from baseline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_update(DEFAULT_CONFIG, user_config)
    config["_config_path"] = str(path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Persist a resolved config dict to YAML (e.g. alongside a checkpoint)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)