#!/usr/bin/env python3
"""
Training entry point for autoencoder_final.

Usage:
    python train.py --config configs/train_ensemble.yaml
    python train.py --config configs/train_single.yaml --override training.epochs=10
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure package root is on path when run as script
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.loader import load_config
from features import rna_g4  # noqa: F401 — register encoder
from features.pipeline import load_or_compute_features
from training.ensemble import EnsembleTrainer
from training.monitor import TrainingMonitor
from training.trainer import SingleModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE (single or ensemble)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config values, e.g. training.epochs=10",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.override)

    print("=" * 70)
    print(f"VAE TRAINING — mode={config['training']['mode']}")
    print("=" * 70)

    os.makedirs(config["project"]["output_dir"], exist_ok=True)
    monitor = TrainingMonitor(config)

    print("\n[1/2] Loading features...")
    features, traj_labels, traj_indices, n_continuous, n_binary = load_or_compute_features(config)
    if features is None:
        print("ERROR: Could not load features. Exiting.")
        sys.exit(1)

    print(f"Features: {features.shape} (continuous={n_continuous}, binary={n_binary})")

    mode = config["training"]["mode"]
    print(f"\n[2/2] Training ({mode})...")

    if mode == "single":
        trainer = SingleModelTrainer(config, monitor)
        checkpoints = [trainer.train(features, traj_labels, n_continuous, n_binary)]
    elif mode == "ensemble":
        trainer = EnsembleTrainer(config, monitor)
        checkpoints = trainer.train(features, traj_labels, n_continuous, n_binary)
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    summary = monitor.finalize(mode=mode, checkpoint_paths=checkpoints)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Output: {config['project']['output_dir']}")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Summary: {summary}")
    print("=" * 70)


if __name__ == "__main__":
    main()
