"""
Uncertainty and confidence plots.

YAML: evaluation.plots.uncertainty, confidence, confidence_grid
Outputs: vae_ensemble_uncertainty.png, vae_latent_confidence.png, vae_latent_confidence_grid.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

from evaluation.plots.references import add_reference_points
from utils.qscore import check_if_unfolded
from utils.trajectories import find_trajectory_files


def plot_ensemble_uncertainty(ctx):
    """Latent std and classification uncertainty (ensemble only)."""
    latent_coords = ctx["latent_coords"]
    latent_uncertainty = ctx["latent_uncertainty"]
    confidence = ctx["confidence"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Ensemble Uncertainty Estimation", fontsize=16, fontweight="bold")

    z1, z2 = latent_coords[:, 0], latent_coords[:, 1]

    ax = axes[0]
    scatter = ax.scatter(z1, z2, c=latent_uncertainty, cmap="inferno", s=2, alpha=0.6, rasterized=True)
    ax.set_title("Latent Space Position Spread (Std Dev)")
    plt.colorbar(scatter, ax=ax, label="Std Dev (Euclidean)")
    add_reference_points(ax, ref_coords, ref_labels)

    ax = axes[1]
    class_uncertainty = 1.0 - confidence
    scatter = ax.scatter(z1, z2, c=class_uncertainty, cmap="inferno", s=2, alpha=0.6, rasterized=True)
    ax.set_title("Classification Uncertainty (1 - Max Prob)")
    plt.colorbar(scatter, ax=ax, label="Uncertainty")
    add_reference_points(ax, ref_coords, ref_labels)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_ensemble_uncertainty.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_confidence(ctx):
    """Single-model confidence plot with opacity."""
    latent_coords = ctx["latent_coords"]
    y_pred = ctx["y_pred"]
    confidence = ctx["confidence"]
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    output_dir = ctx["output_dir"]

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_classes = np.unique(y_pred)
    cmap = plt.get_cmap("tab10")

    for state in unique_classes:
        mask = y_pred == state
        z_state = latent_coords[mask]
        conf_state = confidence[mask]
        color = cmap(state % 10)
        rgba_colors = np.zeros((len(z_state), 4))
        rgba_colors[:, 0:3] = color[0:3]
        rgba_colors[:, 3] = conf_state
        ax.scatter(z_state[:, 0], z_state[:, 1], c=rgba_colors, s=20, label=f"State {state}", rasterized=True)

    ax.set_title("Latent Space - Classification Confidence (Opacity)", fontsize=16)
    ax.legend()

    if ref_coords is not None:
        ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="white", s=100, zorder=20)
        ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="red", marker="*", s=80,
                   edgecolors="black", linewidths=0.5, zorder=21)
        if ref_labels:
            for coord, label in zip(ref_coords, ref_labels):
                ax.text(coord[0], coord[1], label, fontsize=10, fontweight="bold",
                        ha="center", va="bottom", zorder=22,
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_latent_confidence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_confidence_grid(ctx):
    """5x5 grid of trajectories colored by confidence."""
    latent_coords = ctx["latent_coords"]
    traj_indices = ctx["traj_indices"]
    y_pred = ctx["y_pred"]
    confidence = ctx["confidence"]
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
    n_plots = min(len(unique_indices), 25)

    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle("Individual Trajectories Colored by Confidence", fontsize=24, fontweight="bold", y=0.98)

    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    cmap = plt.get_cmap("tab10")
    axes_flat = axes.flatten()

    for i in range(25):
        ax = axes_flat[i]
        if i < n_plots:
            traj_idx = unique_indices[i]
            mask = traj_indices == traj_idx
            points = latent_coords[mask]
            traj_preds = y_pred[mask]
            traj_conf = confidence[mask]

            if len(points) > 0:
                rgba_colors = np.zeros((len(points), 4))
                for j, pred_class in enumerate(traj_preds):
                    rgba_colors[j, 0:3] = cmap(pred_class % 10)[0:3]
                rgba_colors[:, 3] = traj_conf
                ax.scatter(points[:, 0], points[:, 1], c=rgba_colors, s=3, rasterized=True)

            is_unfolded, status_text = check_if_unfolded(traj_idx, merged_files, conf_dir) if merged_files else (False, "")
            title = f"Traj {traj_idx}"
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
    out_path = os.path.join(output_dir, "vae_latent_confidence_grid.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_new_dataset_uncertainty(
    latent_coords,
    latent_uncertainty,
    confidence,
    ref_coords,
    ref_labels,
    output_dir,
    global_landscape=None,
    global_limits=None,
    suffix="",
):
    """
    Ensemble uncertainty for a new dataset projected into the trained latent space.

    Left: positional spread across ensemble models.
    Right: classification uncertainty (1 - max class probability).
    """
    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range = [latent_coords[:, 0].min(), latent_coords[:, 0].max()]
        y_range = [latent_coords[:, 1].min(), latent_coords[:, 1].max()]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"New Dataset: Ensemble Uncertainty{suffix}",
        fontsize=16, fontweight="bold",
    )

    z1, z2 = latent_coords[:, 0], latent_coords[:, 1]
    class_uncertainty = 1.0 - confidence

    for ax, values, title, cbar_label in (
        (axes[0], latent_uncertainty, "Latent Space Position Spread (Std Dev)", "Std Dev (Euclidean)"),
        (axes[1], class_uncertainty, "Classification Uncertainty (1 - Max Prob)", "Uncertainty"),
    ):
        if global_landscape:
            fe_g, xc_g, yc_g = global_landscape
            Xg, Yg = np.meshgrid(xc_g, yc_g)
            fe_plot = np.clip(fe_g.T, 0, np.nanpercentile(fe_g.T, 95))
            ax.contour(Xg, Yg, fe_plot, levels=20, cmap="Greys", alpha=0.35, linewidths=0.6)

        scatter = ax.scatter(z1, z2, c=values, cmap="inferno", s=3, alpha=0.7, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("Latent Z1")
        ax.set_ylabel("Latent Z2")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        plt.colorbar(scatter, ax=ax, label=cbar_label)
        add_reference_points(ax, ref_coords, ref_labels)

    plt.tight_layout()
    base = f"vae_new_dataset_uncertainty{suffix}"
    out_path = os.path.join(output_dir, f"{base}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
