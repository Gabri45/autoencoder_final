#!/usr/bin/env python3
"""
Evaluation entry point for autoencoder_final.

Auto-detects single vs ensemble from artifacts in output_dir.
Dispatches plots enabled in evaluation.plots YAML section.

Usage:
    python evaluate.py --config configs/default_rna.yaml
    python evaluate.py --config configs/default_rna.yaml --override evaluation.plots.uncertainty=false
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from config.loader import load_config
from evaluation.artifacts import load_artifacts
from evaluation.by_target import run_by_target
from evaluation.encoder import encode_single, predict_ensemble
from evaluation.new_dataset import run_new_dataset
from evaluation.plots import run_plots
from evaluation.plots.references import (
    project_reference_pdbs_ensemble,
    project_reference_pdbs_single,
)
from features import rna_g4  # noqa: F401 — register encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE (single or ensemble)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config values, e.g. evaluation.plots.hamming=false",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    eval_cfg = config.get("evaluation", {})

    print("=" * 70)
    print("VAE EVALUATION")
    print("=" * 70)

    print("\n[1/3] Loading artifacts...")
    artifacts = load_artifacts(config)
    mode = artifacts["mode"]
    output_dir = artifacts["output_dir"]
    print(f"  Mode: {mode}")

    print("\n[2/3] Encoding latent space...")
    if mode == "single":
        latent_coords, y_pred, confidence, x_bin = encode_single(
            artifacts["model"], artifacts["scaler"], artifacts["features"],
            artifacts["n_continuous"], artifacts["n_binary"],
        )
        np.save(os.path.join(output_dir, "latent_coordinates.npy"), latent_coords)
        ref_coords, ref_labels, ref_binary = project_reference_pdbs_single(
            artifacts["model"], artifacts["scaler"], artifacts["conf_dir"],
            artifacts["n_continuous"], artifacts["n_binary"],
        )
        latent_uncertainty = None
    else:
        latent_coords, std_mu, latent_uncertainty, y_pred, confidence, x_bin = predict_ensemble(
            artifacts["models"], artifacts["scaler"], artifacts["features"],
        )
        np.save(os.path.join(output_dir, "ensemble_mean_latent.npy"), latent_coords)
        np.save(os.path.join(output_dir, "ensemble_uncertainty.npy"), {
            "std_mu": std_mu,
            "latent_uncertainty": latent_uncertainty,
            "y_pred": y_pred,
            "confidence": confidence,
        })
        ref_coords, ref_labels, ref_binary = project_reference_pdbs_ensemble(
            artifacts["models"], artifacts["scaler"], artifacts["conf_dir"],
        )

    np.save(os.path.join(output_dir, "predicted_classes.npy"), y_pred)
    print(f"  Encoded {latent_coords.shape[0]} frames → {latent_coords.shape}")

    ctx = {
        "mode": mode,
        "latent_coords": latent_coords,
        "traj_labels": artifacts["traj_labels"],
        "traj_indices": artifacts["traj_indices"],
        "y_pred": y_pred,
        "confidence": confidence,
        "latent_uncertainty": latent_uncertainty,
        "x_bin": x_bin,
        "ref_coords": ref_coords,
        "ref_labels": ref_labels,
        "ref_binary": ref_binary,
        "output_dir": output_dir,
        "data_dir": artifacts["data_dir"],
        "conf_dir": artifacts["conf_dir"],
        "trajectory_glob": artifacts["trajectory_glob"],
    }

    print("\n[3/3] Generating plots...")
    run_plots(ctx, eval_cfg.get("plots", {}))

    if eval_cfg.get("by_target", {}).get("enabled"):
        run_by_target(ctx, eval_cfg["by_target"])

    if eval_cfg.get("new_dataset", {}).get("enabled"):
        run_new_dataset(config, artifacts)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
