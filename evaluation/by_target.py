"""
Per-target evaluation on the training dataset latent coordinates.

Uses pre-computed latent_coords from the standard evaluation step.
YAML: evaluation.by_target.*
"""

from __future__ import annotations

import os

import numpy as np

from evaluation.plots.target import (
    plot_target_landscape,
    plot_target_trajectories_grid,
    compute_global_limits_from_coords,
)
from utils.folding_report import map_names_to_indices, parse_folding_by_target
from utils.free_energy import calculate_explored_area, compute_free_energy
from utils.trajectories import find_trajectory_files


def run_by_target(ctx: dict, cfg: dict) -> None:
    """
    Generate per-target landscapes and exploration metrics.

    Requires ensemble_mean_latent.npy or latent_coords already in ctx.
    """
    print("\n" + "=" * 70)
    print("BY-TARGET EVALUATION")
    print("=" * 70)

    latent_coords = ctx["latent_coords"]
    traj_indices = ctx["traj_indices"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")

    data_dir = cfg.get("data_dir") or ctx["data_dir"]
    trajectory_glob = cfg.get("trajectory_glob") or ctx["trajectory_glob"]
    folding_report = cfg.get("folding_report")
    if not folding_report:
        print("ERROR: evaluation.by_target.folding_report is required.")
        return

    output_dir = ctx["output_dir"]
    if cfg.get("output_subdir"):
        output_dir = os.path.join(output_dir, cfg["output_subdir"])
    os.makedirs(output_dir, exist_ok=True)

    white_style = cfg.get("white_style", True)
    suffix = cfg.get("suffix", "_reference_ensemble_classic")

    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    if not merged_files:
        alt_glob = "iter_*/ratchet_*/merged_trajectory.xtc"
        merged_files = find_trajectory_files(data_dir, alt_glob)

    folding_data = parse_folding_by_target(folding_report)
    processed = map_names_to_indices(folding_data, merged_files)

    x_range, y_range = compute_global_limits_from_coords(latent_coords)
    global_limits = (x_range, y_range)
    fe_glob, xc_g, yc_g = compute_free_energy(latent_coords, xlim=x_range, ylim=y_range)
    global_landscape = (fe_glob, xc_g, yc_g)

    results = []
    for target_name, info in processed.items():
        print(f"\nTarget: {target_name} ({len(info['folded_idx'])} folded, {len(info['misfolded_idx'])} misfolded)")

        plot_target_landscape(
            latent_coords, traj_indices, target_name, info, output_dir,
            ref_coords, ref_labels, global_landscape, global_limits,
            white_style=white_style, suffix=suffix,
        )
        plot_target_trajectories_grid(
            latent_coords, traj_indices, target_name, info, output_dir, merged_files,
            ref_coords, ref_labels, global_landscape=global_landscape,
            global_limits=global_limits, white_style=white_style, suffix=suffix,
        )

        folded_mask = np.isin(traj_indices, info["folded_idx"])
        misfolded_mask = np.isin(traj_indices, info["misfolded_idx"])
        total_mask = folded_mask | misfolded_mask

        results.append({
            "target": target_name,
            "folded": calculate_explored_area(latent_coords[folded_mask], x_range, y_range),
            "misfolded": calculate_explored_area(latent_coords[misfolded_mask], x_range, y_range),
            "total": calculate_explored_area(latent_coords[total_mask], x_range, y_range),
        })

    _save_area_table(results, os.path.join(output_dir, "exploration_areas_by_target.txt"))
    print(f"\nBy-target evaluation complete. Output: {output_dir}")


def _save_area_table(results: list, path: str) -> None:
    header = f"{'Target':<15} | {'Folded Area':<15} | {'Misfolded Area':<15} | {'Total Target Area':<15}"
    print(f"\n--- Exploration Areas (Visited Latent Space) ---")
    print(header)
    print("-" * 68)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("-" * 68 + "\n")
        for res in results:
            line = (
                f"{res['target']:<15} | {res['folded']:<15.4f} | "
                f"{res['misfolded']:<15.4f} | {res['total']:<15.4f}"
            )
            print(line)
            f.write(line + "\n")
    print(f"Saved {path}")
