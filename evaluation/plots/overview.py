"""
4-panel latent space overview plots (single and ensemble).

YAML: evaluation.plots.overview_4panel, evaluation.plots.folded_only_4panel
Outputs: vae_latent_space.png / vae_ensemble_latent_space.png / vae_ensemble_folded_latent_space.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

from evaluation.plots.references import add_reference_points
from utils.free_energy import compute_free_energy
from utils.qscore import check_if_unfolded
from utils.trajectories import find_trajectory_files


def plot_latent_space_single(ctx):
    """4-panel overview for single model."""
    latent_coords = ctx["latent_coords"]
    traj_labels = ctx["traj_labels"]
    traj_indices = ctx["traj_indices"]
    y_pred = ctx["y_pred"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Semi-Supervised VAE Latent Space (2D)", fontsize=14, fontweight="bold", y=0.98)

    z1, z2 = latent_coords[:, 0], latent_coords[:, 1]

    ax = axes[0, 0]
    unique_labels = np.unique(traj_labels)
    scatter = ax.scatter(z1, z2, c=traj_labels, cmap="tab10", s=3, alpha=0.5, rasterized=True)
    ax.set_title("True Labels (States)")
    plt.colorbar(scatter, ax=ax, label="State ID", ticks=unique_labels)

    ax = axes[0, 1]
    scatter = ax.scatter(z1, z2, c=y_pred, cmap="tab10", s=3, alpha=0.5, rasterized=True)
    ax.set_title("Predicted Class")
    plt.colorbar(scatter, ax=ax, label="Class Index")

    ax = axes[1, 0]
    if traj_indices is not None:
        scatter = ax.scatter(z1, z2, c=traj_indices, cmap="nipy_spectral", s=3, alpha=0.5, rasterized=True)
        ax.set_title("Trajectory Index")
        plt.colorbar(scatter, ax=ax, label="Trajectory ID")
    else:
        ax.text(0.5, 0.5, "Trajectory Indices Not Available", ha="center", va="center")

    ax = axes[1, 1]
    free_energy, x_centers, y_centers = compute_free_energy(latent_coords)
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
    fe_plot = np.clip(free_energy.T, 0, np.nanpercentile(free_energy.T, 95))
    contour = ax.contourf(X_grid, Y_grid, fe_plot, levels=30, cmap="RdYlBu_r")
    ax.contour(X_grid, Y_grid, fe_plot, levels=15, colors="black", linewidths=0.3, alpha=0.5)
    ax.set_title("Free Energy Landscape")
    plt.colorbar(contour, ax=ax, label="F / kT")

    if ref_coords is not None:
        for ax in axes.flat:
            ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="white", s=100, zorder=20)
            ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="red", marker="*", s=80,
                       edgecolors="black", linewidths=0.5, zorder=21)
            for coord, label in zip(ref_coords, ref_labels):
                ax.text(coord[0], coord[1] + 0.2, label, fontsize=12, fontweight="bold",
                        ha="center", va="bottom", zorder=22,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_latent_space.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_ensemble_overview(ctx):
    """4-panel overview for ensemble mean latent."""
    latent_coords = ctx["latent_coords"]
    traj_labels = ctx["traj_labels"]
    traj_indices = ctx["traj_indices"]
    y_pred = ctx["y_pred"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Ensemble VAE Latent Space (Mean of Models)", fontsize=16, fontweight="bold")

    z1, z2 = latent_coords[:, 0], latent_coords[:, 1]

    ax = axes[0, 0]
    scatter = ax.scatter(z1, z2, c=traj_labels, cmap="tab10", s=2, alpha=0.5, rasterized=True)
    ax.set_title("True Labels (States)")
    plt.colorbar(scatter, ax=ax, label="State ID")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[0, 1]
    scatter = ax.scatter(z1, z2, c=y_pred, cmap="tab10", s=2, alpha=0.5, rasterized=True)
    ax.set_title("Predicted Class (Ensemble Mean)")
    plt.colorbar(scatter, ax=ax, label="Class Index")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[1, 0]
    if traj_indices is not None:
        scatter = ax.scatter(z1, z2, c=traj_indices, cmap="nipy_spectral", s=2, alpha=0.5, rasterized=True)
        ax.set_title("Trajectory Index")
        plt.colorbar(scatter, ax=ax, label="Traj ID")
    else:
        ax.text(0.5, 0.5, "No Trajectory Indices", ha="center")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[1, 1]
    fe, xc, yc = compute_free_energy(latent_coords)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))
    contour = ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r")
    ax.set_title("Free Energy Landscape")
    plt.colorbar(contour, ax=ax, label="F / kT")
    add_reference_points(ax, ref_coords, ref_labels)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_ensemble_latent_space.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_folded_only_overview(ctx):
    """4-panel overview restricted to folded trajectories (ensemble)."""
    latent_coords = ctx["latent_coords"]
    traj_labels = ctx["traj_labels"]
    traj_indices = ctx["traj_indices"]
    y_pred = ctx["y_pred"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]
    data_dir = ctx["data_dir"]
    conf_dir = ctx["conf_dir"]
    trajectory_glob = ctx.get("trajectory_glob", "iter_*/ratchet_*/md_noPBC.xtc")

    if traj_indices is None:
        print("Skipping folded-only plot: no traj_indices")
        return

    merged_files = find_trajectory_files(data_dir, trajectory_glob)
    unique_indices = np.unique(traj_indices)

    folded_trajs = []
    for traj_idx in unique_indices:
        is_unfolded, _ = check_if_unfolded(traj_idx, merged_files, conf_dir)
        if not is_unfolded:
            folded_trajs.append(traj_idx)

    if not folded_trajs:
        print("No folded trajectories found.")
        return

    mask = np.isin(traj_indices, folded_trajs)
    f_coords = latent_coords[mask]
    f_labels = traj_labels[mask]
    f_indices = traj_indices[mask]
    f_pred = y_pred[mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Ensemble VAE Latent Space (FOLDED ONLY: Q > 0.85)", fontsize=16, fontweight="bold")

    z1, z2 = f_coords[:, 0], f_coords[:, 1]

    ax = axes[0, 0]
    scatter = ax.scatter(z1, z2, c=f_labels, cmap="tab10", s=2, alpha=0.5, rasterized=True)
    ax.set_title("True Labels - Folded Only")
    plt.colorbar(scatter, ax=ax, label="State ID")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[0, 1]
    scatter = ax.scatter(z1, z2, c=f_pred, cmap="tab10", s=2, alpha=0.5, rasterized=True)
    ax.set_title("Predicted Class - Folded Only")
    plt.colorbar(scatter, ax=ax, label="Class Index")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[1, 0]
    scatter = ax.scatter(z1, z2, c=f_indices, cmap="nipy_spectral", s=2, alpha=0.5, rasterized=True)
    ax.set_title("Trajectory Index - Folded Only")
    plt.colorbar(scatter, ax=ax, label="Traj ID")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[1, 1]
    fe, xc, yc = compute_free_energy(f_coords)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))
    contour = ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r")
    ax.set_title("Free Energy - Folded Only")
    plt.colorbar(contour, ax=ax, label="F / kT")
    add_reference_points(ax, ref_coords, ref_labels)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_ensemble_folded_latent_space.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
