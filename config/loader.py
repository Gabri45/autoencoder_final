"""
YAML configuration loader with validation and dot-notation overrides.

Relevant YAML sections: project, data, features, training, monitoring, evaluation.
Outputs: resolved dict used by train.py and evaluate.py; optionally saved as
config_resolved.yaml by TrainingMonitor.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("project", "data", "features", "training", "monitoring", "evaluation")

DEFAULTS: dict[str, Any] = {
    "project": {"output_dir": "./outputs", "seed": 42},
    "data": {
        "trajectory_glob": "iter_*/ratchet_*/md_noPBC.xtc",
        "stride": 1,
        "recompute_features": False,
        "topology_glob": "init_conf.gro",
    },
    "features": {
        "encoder": "rna_g4_enriched",
        "params": {"hbond_cutoff": 0.35},
        "cache_prefix": "features_mixed",
    },
    "training": {
        "mode": "ensemble",
        "n_ensemble": 5,
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.001,
        "val_split": 0.3,
        "device": "cuda",
        "early_stopping": {"patience": 20},
        "loss": {
            "beta_max": 1.0,
            "warmup_epochs": 50,
            "gamma": 1.0,
            "margin": 1.0,
            "free_bits": 0.5,
        },
        "model": {"latent_dim": 2, "n_classes": 5},
    },
    "monitoring": {
        "log_every_epochs": 1,
        "plot_every_epochs": 10,
        "save_csv": True,
        "flush_stdout": True,
    },
    "evaluation": {
        "mode": "auto",
        "plots": {
            "overview_4panel": True,
            "uncertainty": True,
            "hamming": True,
            "trajectory_grid_5x5": True,
            "landscape_grid_5x10": True,
            "confidence": True,
            "confidence_grid": True,
            "folded_only_4panel": True,
            "landscape_comparison": True,
        },
        "by_target": {
            "enabled": False,
            "folding_report": None,
            "data_dir": None,
            "trajectory_glob": None,
            "white_style": True,
            "output_subdir": None,
            "suffix": "_reference_ensemble_classic",
        },
        "new_dataset": {
            "enabled": False,
            "data_dir": None,
            "conf_dir": None,
            "folding_report": None,
            "trajectory_glob": "iter_*/ratchet_*/md_noPBC.xtc",
            "cache_prefix": "features_new",
            "output_subdir": "new_dataset",
            "reference_latent": None,
            "recompute_features": False,
            "per_target": None,
            "plot_latent_overview": True,
            "plot_landscape_grid": False,
            "plot_uncertainty": False,
            "landscape_grid_ncols": 5,
            "white_style": False,
            "suffix": "_NEW",
            "batch_size": 2048,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set a nested key using dot notation, e.g. training.epochs."""
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    raw = value
    if isinstance(value, str):
        lower = value.lower()
        if lower == "true":
            raw = True
        elif lower == "false":
            raw = False
        elif lower == "null" or lower == "none":
            raw = None
        else:
            try:
                if "." in value:
                    raw = float(value)
                else:
                    raw = int(value)
            except ValueError:
                pass
    current[keys[-1]] = raw


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """
    Apply CLI overrides of the form key=value or key.subkey=value.

    Args:
        config: Base configuration dict.
        overrides: List like ["training.epochs=10", "evaluation.plots.hamming=false"].

    Returns:
        Updated configuration dict.
    """
    if not overrides:
        return config
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key, _, value = item.partition("=")
        _set_nested(config, key.strip(), value.strip())
    return config


def _resolve_paths(config: dict, config_path: Path) -> dict:
    """Expand user paths and make output_dir absolute relative to config file."""
    config_dir = config_path.parent.resolve()
    project_root = config_dir.parent

    out = config["project"].get("output_dir", "./outputs")
    if not os.path.isabs(out):
        config["project"]["output_dir"] = str((project_root / out).resolve())

    for key in ("data_dir", "conf_dir"):
        if key in config.get("data", {}):
            val = config["data"][key]
            if val and not os.path.isabs(val):
                config["data"][key] = str((project_root / val).resolve())

    eval_cfg = config.get("evaluation", {})
    for section in ("by_target", "new_dataset"):
        sec = eval_cfg.get(section, {})
        for path_key in ("folding_report", "data_dir", "conf_dir", "reference_latent"):
            if path_key in sec and sec[path_key] and not os.path.isabs(sec[path_key]):
                sec[path_key] = str((project_root / sec[path_key]).resolve())

    return config


def _validate(config: dict) -> None:
    """Minimal schema validation."""
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    mode = config["training"].get("mode", "ensemble")
    if mode not in ("single", "ensemble"):
        raise ValueError(f"training.mode must be 'single' or 'ensemble', got {mode!r}")

    if not config["data"].get("data_dir"):
        raise ValueError("data.data_dir is required")


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """
    Load YAML config, merge defaults, apply overrides, resolve paths.

    Args:
        config_path: Path to YAML file.
        overrides: Optional CLI override strings.

    Returns:
        Fully resolved configuration dictionary.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    config = copy.deepcopy(DEFAULTS)
    _deep_merge(config, user_cfg)
    apply_overrides(config, overrides)
    _resolve_paths(config, path)
    _validate(config)
    return config


def save_config(config: dict, path: str) -> None:
    """Write resolved config to YAML (used by TrainingMonitor)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
