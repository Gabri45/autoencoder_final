"""
Feature computation pipeline with disk cache.

Loads or computes features via the registered encoder and caches:
  {prefix}.npy, {prefix}_meta.json, traj_labels.npy, traj_indices.npy

YAML: features.encoder, features.cache_prefix, data.recompute_features
"""

from __future__ import annotations

import json
import os

import numpy as np

from features.registry import get_encoder
from utils.trajectories import load_merged_trajectories


def load_or_compute_features(config: dict) -> tuple:
    """
    Load cached features or compute from trajectories.

    Args:
        config: Full resolved configuration dict.

    Returns:
        features, traj_labels, traj_indices, n_continuous, n_binary
        (features is memory-mapped when loaded from cache)
    """
    output_dir = config["project"]["output_dir"]
    data_cfg = config["data"]
    feat_cfg = config["features"]

    prefix = feat_cfg.get("cache_prefix", "features_mixed")
    dist_path = os.path.join(output_dir, f"{prefix}.npy")
    metadata_path = os.path.join(output_dir, f"{prefix}_meta.json")
    labels_path = os.path.join(output_dir, "traj_labels.npy")
    indices_path = os.path.join(output_dir, "traj_indices.npy")

    recompute = data_cfg.get("recompute_features", False)

    if (
        not recompute
        and os.path.exists(dist_path)
        and os.path.exists(labels_path)
        and os.path.exists(indices_path)
        and os.path.exists(metadata_path)
    ):
        print("Loading cached features...")
        features = np.load(dist_path, mmap_mode="r")
        traj_labels = np.load(labels_path)
        traj_indices = np.load(indices_path)
        with open(metadata_path) as f:
            meta = json.load(f)
        print(
            f"  Shape: {features.shape}, Labels: {len(traj_labels)}, "
            f"Indices: {len(traj_indices)}"
        )
        return (
            features,
            traj_labels,
            traj_indices,
            meta["n_continuous"],
            meta["n_binary"],
        )

    print("Computing features from trajectories...")
    os.makedirs(output_dir, exist_ok=True)

    traj, traj_labels, traj_indices = load_merged_trajectories(
        data_dir=data_cfg["data_dir"],
        conf_dir=data_cfg["conf_dir"],
        stride=data_cfg.get("stride", 1),
        trajectory_glob=data_cfg.get("trajectory_glob", "iter_*/ratchet_*/md_noPBC.xtc"),
        topology_glob=data_cfg.get("topology_glob", "init_conf.gro"),
    )
    if traj is None:
        return None, None, None, None, None

    encoder = get_encoder(feat_cfg["encoder"], **feat_cfg.get("params", {}))
    features, meta = encoder.compute(traj)
    n_continuous = meta.n_continuous
    n_binary = meta.n_binary

    np.save(dist_path, features)
    np.save(labels_path, traj_labels)
    np.save(indices_path, traj_indices)
    with open(metadata_path, "w") as f:
        json.dump({"n_continuous": n_continuous, "n_binary": n_binary}, f)
    print(f"Saved features to {dist_path}")

    features = np.load(dist_path, mmap_mode="r")
    return features, traj_labels, traj_indices, n_continuous, n_binary
