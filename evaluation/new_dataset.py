"""
Evaluate a new dataset by projecting it into a pre-trained ensemble latent space.

YAML: evaluation.new_dataset.*

Modes:
  - Quick (no folding_report): latent overview plot only
  - Full (with folding_report): per-target landscapes + grids + area metrics
"""

from __future__ import annotations

import copy
import os

import numpy as np

from evaluation.encoder import predict_ensemble
from evaluation.plots.references import project_reference_pdbs_ensemble
from evaluation.plots.uncertainty import plot_new_dataset_uncertainty
from evaluation.plots.target import (
    plot_new_latent_overview,
    plot_new_dataset_landscape_grid,
    plot_target_landscape,
    plot_target_landscape_clean,
    plot_target_trajectories_grid,
    compute_global_limits_from_coords,
)
from features.pipeline import load_or_compute_features
from utils.folding_report import map_names_to_indices, parse_folding_by_target
from utils.free_energy import calculate_explored_area, compute_free_energy
from utils.trajectories import find_trajectory_files


def run_new_dataset(config: dict, artifacts: dict) -> None:
    """Project new trajectories into the trained ensemble latent space."""
    print("\n" + "=" * 70)
    print("NEW DATASET EVALUATION")
    print("=" * 70)

    cfg = config.get("evaluation", {}).get("new_dataset", {})
    train_output_dir = artifacts["output_dir"]
    models = artifacts.get("models")
    scaler = artifacts.get("scaler")

    if not models or scaler is None:
        print("ERROR: new_dataset requires ensemble checkpoints in output_dir.")
        return

    data_dir = cfg.get("data_dir")
    if not data_dir:
        print("ERROR: evaluation.new_dataset.data_dir is required.")
        return

    folding_report = cfg.get("folding_report")
    per_target = cfg.get("per_target")
    if per_target is None:
        per_target = bool(folding_report and os.path.exists(folding_report))

    plot_latent_overview = cfg.get("plot_latent_overview", True)
    plot_landscape_grid = cfg.get("plot_landscape_grid", False)
    plot_uncertainty = cfg.get("plot_uncertainty", False)
    landscape_grid_ncols = cfg.get("landscape_grid_ncols", 5)

    output_dir = os.path.join(train_output_dir, cfg.get("output_subdir", "new_dataset"))
    os.makedirs(output_dir, exist_ok=True)

    conf_dir = cfg.get("conf_dir") or config["data"]["conf_dir"]
    trajectory_glob = cfg.get("trajectory_glob") or config["data"].get(
        "trajectory_glob", "iter_*/ratchet_*/md_noPBC.xtc"
    )
    topology_glob = cfg.get("topology_glob") or config["data"].get("topology_glob", "init_conf.gro")
    white_style = cfg.get("white_style", False)
    suffix = cfg.get("suffix", "_NEW")
    batch_size = cfg.get("batch_size", 2048)

    new_config = copy.deepcopy(config)
    new_config["data"]["data_dir"] = data_dir
    new_config["data"]["trajectory_glob"] = trajectory_glob
    new_config["data"]["recompute_features"] = cfg.get("recompute_features", False)
    new_config["features"]["cache_prefix"] = cfg.get("cache_prefix", "features_new")
    new_config["project"]["output_dir"] = output_dir

    print("\n[1/3] Loading/computing features for new dataset...")
    features, _, traj_indices, _, _ = load_or_compute_features(new_config)
    if features is None:
        print("ERROR: Could not load features for new dataset.")
        return

    print("\n[2/3] Projecting into latent space...")
    latent_coords, std_mu, latent_uncertainty, y_pred, confidence, _ = predict_ensemble(
        models, scaler, features, batch_size=batch_size,
    )
    np.save(os.path.join(output_dir, "ensemble_mean_latent_new.npy"), latent_coords)
    np.save(os.path.join(output_dir, "traj_indices_new.npy"), traj_indices)
    np.save(os.path.join(output_dir, "ensemble_uncertainty_new.npy"), {
        "std_mu": std_mu,
        "latent_uncertainty": latent_uncertainty,
        "y_pred": y_pred,
        "confidence": confidence,
    })

    ref_coords, ref_labels, _ = project_reference_pdbs_ensemble(models, scaler, conf_dir)

    print("\n[3/3] Generating plots...")
    ref_latent_path = cfg.get("reference_latent") or os.path.join(
        train_output_dir, "ensemble_mean_latent.npy"
    )
    old_latent = np.load(ref_latent_path) if os.path.exists(ref_latent_path) else None

    if old_latent is not None:
        global_limits = compute_global_limits_from_coords(old_latent)
        x_range, y_range = global_limits
        fe_old, xc_old, yc_old = compute_free_energy(old_latent, xlim=x_range, ylim=y_range)
        global_landscape = (fe_old, xc_old, yc_old)
        print(f"  Training background: {ref_latent_path}")
    else:
        print(f"  Warning: {ref_latent_path} not found. Using new dataset for axes.")
        global_limits = compute_global_limits_from_coords(latent_coords)
        x_range, y_range = global_limits
        fe_g, xc_g, yc_g = compute_free_energy(latent_coords, xlim=x_range, ylim=y_range)
        global_landscape = (fe_g, xc_g, yc_g)
        old_latent = None

    if plot_latent_overview:
        plot_new_latent_overview(
            latent_coords, traj_indices, ref_coords, ref_labels, output_dir,
            global_landscape=global_landscape, global_limits=global_limits,
            old_latent=old_latent,
        )

    if plot_uncertainty:
        plot_new_dataset_uncertainty(
            latent_coords, latent_uncertainty, confidence,
            ref_coords, ref_labels, output_dir,
            global_landscape=global_landscape, global_limits=global_limits,
            suffix=suffix,
        )

    if plot_landscape_grid:
        merged_files = find_trajectory_files(data_dir, trajectory_glob)
        if merged_files:
            plot_new_dataset_landscape_grid(
                latent_coords, traj_indices, output_dir, merged_files,
                ref_coords=ref_coords, ref_labels=ref_labels,
                global_limits=global_limits, conf_dir=conf_dir,
                topology_glob=topology_glob, n_cols=landscape_grid_ncols,
                suffix=suffix,
            )
        else:
            print("  Warning: no trajectory files found for landscape grid.")

    if not per_target:
        if not folding_report:
            print("  No folding_report → skipping per-target plots (quick mode).")
        print(f"\nNew dataset evaluation complete. Output: {output_dir}")
        return

    if not folding_report or not os.path.exists(folding_report):
        print("ERROR: per_target=true but folding_report not found.")
        return

    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    if not merged_files:
        merged_files = find_trajectory_files(data_dir, "iter_*/ratchet_*/merged_trajectory.xtc")

    folding_data = parse_folding_by_target(folding_report)
    processed = map_names_to_indices(folding_data, merged_files)

    results = {}
    for target_name, info in processed.items():
        target_mask = np.isin(traj_indices, info["folded_idx"] + info["misfolded_idx"])
        target_coords = latent_coords[target_mask]

        target_landscape = None
        if len(target_coords) > 0:
            fe_t, xc_t, yc_t = compute_free_energy(target_coords, xlim=x_range, ylim=y_range)
            target_landscape = (fe_t, xc_t, yc_t)

        plot_target_landscape(
            latent_coords, traj_indices, target_name, info, output_dir,
            ref_coords, ref_labels, global_landscape, global_limits,
            white_style=white_style, suffix=suffix,
        )
        plot_target_landscape_clean(
            latent_coords, traj_indices, target_name, info, output_dir,
            ref_coords, ref_labels, global_limits,
            white_style=white_style, suffix=suffix,
        )
        plot_target_trajectories_grid(
            latent_coords, traj_indices, target_name, info, output_dir, merged_files,
            ref_coords, ref_labels, global_landscape=global_landscape,
            target_landscape=target_landscape, global_limits=global_limits,
            white_style=white_style, suffix=suffix, title_tag=" (NEW DATA)",
        )

        folded_coords = latent_coords[np.isin(traj_indices, info["folded_idx"])]
        misfolded_coords = latent_coords[np.isin(traj_indices, info["misfolded_idx"])]
        results[target_name] = {
            "Folded Area": calculate_explored_area(folded_coords, x_range, y_range),
            "Misfolded Area": calculate_explored_area(misfolded_coords, x_range, y_range),
            "Total Area": calculate_explored_area(target_coords, x_range, y_range) if len(target_coords) > 0 else 0.0,
        }

    _save_area_table(results, os.path.join(output_dir, "exploration_areas_NEW.txt"))
    print(f"\nNew dataset evaluation complete. Output: {output_dir}")


def _save_area_table(results: dict, path: str) -> None:
    header = f"{'Target':<15} | {'Folded Area':<15} | {'Misfolded Area':<15} | {'Total Target Area':<15}"
    print(f"\n--- Exploration Areas (Visited Latent Space) ---")
    print(header)
    print("-" * 68)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 68 + "\n")
        for target_name, res in results.items():
            line = (
                f"{target_name:<15} | {res['Folded Area']:<15.4f} | "
                f"{res['Misfolded Area']:<15.4f} | {res['Total Area']:<15.4f}"
            )
            print(line)
            f.write(line + "\n")
    print(f"Saved {path}")
