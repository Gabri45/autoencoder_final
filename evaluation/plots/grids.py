"""
Trajectory grid and landscape plots.

YAML: evaluation.plots.trajectory_grid_5x5, landscape_grid_5x10, landscape_comparison
Outputs: *_trajectories_grid.png, *_landscape_grid_5x10.png, *_landscape_comparison_2x2.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from evaluation.plots.references import add_reference_points
from utils.free_energy import compute_free_energy
from utils.qscore import check_if_unfolded
from utils.trajectories import find_trajectory_files, trajectory_title


def plot_trajectory_grid_5x5(ctx):
    """5x5 grid of individual trajectories colored by time."""
    latent_coords = ctx["latent_coords"]
    traj_indices = ctx["traj_indices"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]
    data_dir = ctx["data_dir"]
    conf_dir = ctx["conf_dir"]
    trajectory_glob = ctx.get("trajectory_glob")
    mode = ctx.get("mode", "single")

    if traj_indices is None:
        return

    unique_indices = np.unique(traj_indices)
    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    n_plots = min(len(unique_indices), 25)

    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle("Individual Trajectories in Latent Space (Color = Time)", fontsize=24, fontweight="bold", y=0.98)

    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    axes_flat = axes.flatten()

    for i in range(25):
        ax = axes_flat[i]
        if i < n_plots:
            traj_idx = unique_indices[i]
            mask = traj_indices == traj_idx
            points = latent_coords[mask]

            is_unfolded, status_text = check_if_unfolded(traj_idx, merged_files, conf_dir) if merged_files else (False, "")
            if len(points) > 0:
                time_progress = np.linspace(0, 1, len(points))
                ax.scatter(points[:, 0], points[:, 1], c=time_progress, cmap="viridis", s=2, alpha=0.6, rasterized=True)

            title = trajectory_title(traj_idx, merged_files) if merged_files else f"Traj {traj_idx}"
            if is_unfolded:
                title += f"\n{status_text}"
                ax.set_title(title, fontsize=10, fontweight="bold", color="red")
            else:
                ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            add_reference_points(ax, ref_coords, ref_labels)
        else:
            ax.axis("off")

    plt.setp(axes, xlim=(x_min - margin_x, x_max + margin_x), ylim=(y_min - margin_y, y_max + margin_y))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    fname = "vae_ensemble_trajectories_grid.png" if mode == "ensemble" else "vae_latent_space_grid.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_landscape_grid_5x10(ctx):
    """5x10 grid with free energy background and trajectory overlays (ensemble)."""
    latent_coords = ctx["latent_coords"]
    traj_indices = ctx["traj_indices"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]
    data_dir = ctx["data_dir"]
    conf_dir = ctx["conf_dir"]
    trajectory_glob = ctx.get("trajectory_glob")

    if traj_indices is None:
        return

    unique_indices = np.unique(traj_indices)
    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    n_trajs = len(unique_indices)

    fe, xc, yc = compute_free_energy(latent_coords)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))

    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    n_grids = int(np.ceil(n_trajs / 50.0))

    for grid_idx in range(n_grids):
        start_idx = grid_idx * 50
        end_idx = min(start_idx + 50, n_trajs)
        n_plots = end_idx - start_idx

        fig, axes = plt.subplots(5, 10, figsize=(25, 15), sharex=True, sharey=True)
        part_str = f" (Part {grid_idx+1}/{n_grids})" if n_grids > 1 else ""
        fig.suptitle(f"Trajectory Overlay on Free Energy Landscape (5x10 Grid){part_str}",
                     fontsize=24, fontweight="bold", y=0.98)

        for i in range(50):
            ax = axes.flatten()[i]
            if i < n_plots:
                ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r", alpha=0.5)
                traj_idx = unique_indices[start_idx + i]
                mask = traj_indices == traj_idx
                points = latent_coords[mask]

                if len(points) > 1:
                    pts = points.reshape(-1, 1, 2)
                    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    time_progress = np.linspace(0, 1, len(points))
                    lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 1))
                    lc.set_array(time_progress)
                    lc.set_linewidth(1.5)
                    lc.set_alpha(0.8)
                    ax.add_collection(lc)

                add_reference_points(ax, ref_coords, ref_labels)
                is_unfolded, status_text = check_if_unfolded(traj_idx, merged_files, conf_dir) if merged_files else (False, "")
                title = trajectory_title(traj_idx, merged_files) if merged_files else f"Traj {traj_idx}"
                if is_unfolded:
                    title += f"\n{status_text}"
                    ax.set_title(title, fontsize=8, fontweight="bold", color="red")
                else:
                    ax.set_title(title, fontsize=8, fontweight="bold")
            else:
                ax.axis("off")

        plt.setp(axes, xlim=(x_min - margin_x, x_max + margin_x), ylim=(y_min - margin_y, y_max + margin_y))
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = f"vae_ensemble_landscape_grid_5x10_part{grid_idx+1}.png" if n_grids > 1 else "vae_ensemble_landscape_grid_5x10.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def plot_landscape_comparison(ctx, white_theme=False):
    """2x2 comparison of folded vs unfolded pathways on free energy landscape."""
    latent_coords = ctx["latent_coords"]
    traj_indices = ctx["traj_indices"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]
    data_dir = ctx["data_dir"]
    conf_dir = ctx["conf_dir"]
    trajectory_glob = ctx.get("trajectory_glob")

    if traj_indices is None:
        return

    unique_indices = np.unique(traj_indices)
    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    if not merged_files:
        return

    folded_indices, unfolded_indices = [], []
    for traj_idx in unique_indices:
        is_unfolded, _ = check_if_unfolded(traj_idx, merged_files, conf_dir)
        if not is_unfolded and len(folded_indices) < 2:
            folded_indices.append(traj_idx)
        elif is_unfolded and len(unfolded_indices) < 2:
            unfolded_indices.append(traj_idx)
        if len(folded_indices) >= 2 and len(unfolded_indices) >= 2:
            break

    selected_trajs = folded_indices[:2] + unfolded_indices[:2]
    if not selected_trajs:
        return

    fe, xc, yc = compute_free_energy(latent_coords)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))

    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    text_color = "white" if white_theme else "black"

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    if white_theme:
        fig.patch.set_alpha(0.0)
    fig.suptitle("Folding vs. Misfolding Pathways on Free Energy Landscape",
                 fontsize=22, fontweight="bold", y=0.98, color=text_color)

    for i in range(4):
        ax = axes.flatten()[i]
        if white_theme:
            ax.patch.set_alpha(0.0)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")

        if i < len(selected_trajs):
            ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r", alpha=0.5)
            traj_idx = selected_trajs[i]
            mask = traj_indices == traj_idx
            points = latent_coords[mask]

            if len(points) > 1:
                pts = points.reshape(-1, 1, 2)
                segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
                time_progress = np.linspace(0, 1, len(points))
                lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 1))
                lc.set_array(time_progress)
                lc.set_linewidth(2.0)
                lc.set_alpha(0.8)
                ax.add_collection(lc)

            add_reference_points(ax, ref_coords, ref_labels)
            is_unfolded, _ = check_if_unfolded(traj_idx, merged_files, conf_dir)
            title = trajectory_title(traj_idx, merged_files)
            outcome = "[UNFOLDED]" if is_unfolded else "[FOLDED]"
            color = "white" if white_theme else ("red" if is_unfolded else "green")
            ax.set_title(f"{outcome} {title}", fontsize=12, fontweight="bold", color=color)
        else:
            ax.axis("off")

    plt.setp(axes, xlim=(x_min - margin_x, x_max + margin_x), ylim=(y_min - margin_y, y_max + margin_y))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    suffix = "_white" if white_theme else ""
    out_path = os.path.join(output_dir, f"vae_ensemble_landscape_comparison_2x2{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=white_theme)
    plt.close()
    print(f"Saved {out_path}")


def plot_landscape_comparison_both(ctx):
    """Generate both normal and white-theme comparison plots."""
    plot_landscape_comparison(ctx, white_theme=False)
    plot_landscape_comparison(ctx, white_theme=True)
