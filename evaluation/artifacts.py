"""
Artifact loading for single and ensemble evaluation modes.

Auto-detects mode from output_dir:
  ensemble_model_0.pt → ensemble
  vae_model.pt → single

YAML: evaluation.mode (auto | single | ensemble), project.output_dir
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

from model.vae import SemiSupervisedVAE


def detect_mode(output_dir: str, config_mode: str = "auto") -> str:
    """Detect evaluation mode from artifacts or config override."""
    if config_mode in ("single", "ensemble"):
        return config_mode

    if glob.glob(os.path.join(output_dir, "ensemble_model_*.pt")):
        return "ensemble"
    if os.path.exists(os.path.join(output_dir, "vae_model.pt")):
        return "single"
    raise FileNotFoundError(
        f"No vae_model.pt or ensemble_model_*.pt found in {output_dir}"
    )


def load_features(output_dir: str, cache_prefix: str = "features_mixed"):
    """Load cached features, labels, and indices."""
    dist_path = os.path.join(output_dir, f"{cache_prefix}.npy")
    if not os.path.exists(dist_path):
        print(f"ERROR: {dist_path} not found.")
        sys.exit(1)

    features = np.load(dist_path, mmap_mode="r")
    traj_labels = np.load(os.path.join(output_dir, "traj_labels.npy"))
    indices_path = os.path.join(output_dir, "traj_indices.npy")
    traj_indices = np.load(indices_path) if os.path.exists(indices_path) else None

    meta_path = os.path.join(output_dir, f"{cache_prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_continuous = meta["n_continuous"]
        n_binary = meta["n_binary"]
    else:
        n_continuous = 240
        n_binary = 0

    return features, traj_labels, traj_indices, n_continuous, n_binary


def load_single_artifacts(output_dir: str, device: str = "cpu"):
    """Load single-model checkpoint and scaler."""
    model_path = os.path.join(output_dir, "vae_model.pt")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_continuous = checkpoint.get("n_continuous", 240)
    n_binary = checkpoint.get("n_binary", 0)
    latent_dim = checkpoint.get("latent_dim", 2)

    model = SemiSupervisedVAE(
        n_continuous=n_continuous, n_binary=n_binary, latent_dim=latent_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    features, traj_labels, traj_indices, n_continuous, n_binary = load_features(output_dir)

    return {
        "mode": "single",
        "model": model,
        "scaler": scaler,
        "features": features,
        "traj_labels": traj_labels,
        "traj_indices": traj_indices,
        "n_continuous": n_continuous,
        "n_binary": n_binary,
        "output_dir": output_dir,
    }


def load_ensemble_artifacts(output_dir: str, device: str = "cpu"):
    """Load ensemble models and scaler."""
    scaler_path = os.path.join(output_dir, "ensemble_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model_files = sorted(glob.glob(os.path.join(output_dir, "ensemble_model_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No ensemble models in {output_dir}")

    models = []
    for mf in model_files:
        checkpoint = torch.load(mf, map_location=device, weights_only=False)
        model = SemiSupervisedVAE(
            n_continuous=checkpoint.get("n_continuous", 240),
            n_binary=checkpoint.get("n_binary", 0),
            latent_dim=checkpoint["latent_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.n_continuous = checkpoint.get("n_continuous", 240)
        model.n_binary = checkpoint.get("n_binary", 0)
        model.latent_dim = checkpoint["latent_dim"]
        model.to(device)
        model.eval()
        models.append(model)

    features, traj_labels, traj_indices, n_continuous, n_binary = load_features(output_dir)

    return {
        "mode": "ensemble",
        "models": models,
        "scaler": scaler,
        "features": features,
        "traj_labels": traj_labels,
        "traj_indices": traj_indices,
        "n_continuous": n_continuous,
        "n_binary": n_binary,
        "output_dir": output_dir,
    }


def load_artifacts(config: dict):
    """Load artifacts according to config mode (auto-detect or explicit)."""
    output_dir = config["project"]["output_dir"]
    eval_mode = config.get("evaluation", {}).get("mode", "auto")
    device_name = config.get("training", {}).get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = device_name if device_name != "cuda" else "cuda:0"

    mode = detect_mode(output_dir, eval_mode)
    if mode == "ensemble":
        artifacts = load_ensemble_artifacts(output_dir, device)
    else:
        artifacts = load_single_artifacts(output_dir, device)
        artifacts["model"] = artifacts["model"].to(device)

    artifacts["config"] = config
    artifacts["conf_dir"] = config["data"]["conf_dir"]
    artifacts["data_dir"] = config["data"]["data_dir"]
    artifacts["trajectory_glob"] = config["data"].get("trajectory_glob", "iter_*/ratchet_*/md_noPBC.xtc")
    return artifacts
