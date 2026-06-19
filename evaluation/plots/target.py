"""
Target-specific free energy landscapes and trajectory grids.

Used by evaluation.by_target and evaluation.new_dataset.
YAML: evaluation.by_target.*, evaluation.new_dataset.*
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.collections import LineCollection

from evaluation.plots.references import add_reference_points
from utils.free_energy import compute_free_energy
from utils.qscore import check_if_unfolded
from utils.trajectories import trajectory_title


def _apply_white_style(fig, ax, cbar=None, white_style=False):
    if not white_style:
        return
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    if cbar:
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cbar.outline.set_edgecolor("white")
        cbar.set_label(cbar.ax.get_ylabel(), color="white")


def _add_refs(ax, ref_coords, ref_labels, white_style=False, force_standard=False):
    if ref_coords is None:
        return
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="white", s=100, zorder=20)
    ax.scatter(
        ref_coords[:, 0], ref_coords[:, 1], c="red", marker="*", s=80,
        edgecolors="black", linewidths=0.5, zorder=21,
    )
    if ref_labels:
        t_color = "black" if force_standard else ("white" if white_style else "black")
        s_color = "white" if force_standard else ("black" if white_style else "white")
        for coord, label in zip(ref_coords, ref_labels):
            ax.text(
                coord[0], coord[1], label, fontsize=12, fontweight="bold",
                color=t_color, ha="center", va="bottom", zorder=22,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground=s_color)],
            )


def plot_target_landscape(
    latent_coords, traj_indices, target_name, target_data, output_dir,
    ref_coords=None, ref_labels=None, global_landscape=None, global_limits=None,
    white_style=False, suffix="",
):
    """Target-specific FE landscape with optional global background."""
    folded_idx = target_data["folded_idx"]
    misfolded_idx = target_data["misfolded_idx"]
    all_idx = folded_idx + misfolded_idx
    if not all_idx:
        return

    target_coords = latent_coords[np.isin(traj_indices, all_idx)]
    if len(target_coords) == 0:
        return

    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range, y_range = compute_global_limits_from_coords(latent_coords)

    fe, xc, yc = compute_free_energy(target_coords, xlim=x_range, ylim=y_range)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))

    fig, ax = plt.subplots(figsize=(10, 8))
    if global_landscape:
        fe_g, xc_g, yc_g = global_landscape
        Xg_g, Yg_g = np.meshgrid(xc_g, yc_g)
        ax.contour(Xg_g, Yg_g, fe_g.T, levels=20, cmap="Greys", alpha=0.3, linewidths=0.8)

    contour = ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r")
    cbar = plt.colorbar(contour, ax=ax, label="F / kT (Target-Specific)")
    _add_refs(ax, ref_coords, ref_labels, white_style=white_style)
    ax.set_title(f"Target Latent Space: {target_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Latent Z1")
    ax.set_ylabel("Latent Z2")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.2)
    _apply_white_style(fig, ax, cbar, white_style)
    plt.tight_layout()

    fname = f"vae_target_landscape_{target_name}{suffix}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, transparent=white_style)
    print(f"Saved {out_path}")
    plt.close()


def plot_target_landscape_clean(
    latent_coords, traj_indices, target_name, target_data, output_dir,
    ref_coords=None, ref_labels=None, global_limits=None, white_style=False, suffix="",
):
    """Target FE landscape only, no global background."""
    folded_idx = target_data["folded_idx"]
    misfolded_idx = target_data["misfolded_idx"]
    all_idx = folded_idx + misfolded_idx
    if not all_idx:
        return

    target_coords = latent_coords[np.isin(traj_indices, all_idx)]
    if len(target_coords) == 0:
        return

    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range, y_range = compute_global_limits_from_coords(target_coords)

    fe, xc, yc = compute_free_energy(target_coords, xlim=x_range, ylim=y_range)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r")
    cbar = plt.colorbar(contour, ax=ax, label="F / kT (Target-Specific Only)")
    _add_refs(ax, ref_coords, ref_labels, white_style=white_style)
    ax.set_title(f"Target Energy Landscape: {target_name}\n(No Background)", fontsize=16, fontweight="bold")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    _apply_white_style(fig, ax, cbar, white_style)
    plt.tight_layout()

    fname = f"vae_target_landscape_clean_{target_name}{suffix}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, transparent=white_style)
    print(f"Saved {out_path}")
    plt.close()


def plot_target_trajectories_grid(
    latent_coords, traj_indices, target_name, target_data, output_dir, merged_files,
    ref_coords=None, ref_labels=None, global_landscape=None, target_landscape=None,
    global_limits=None, white_style=False, suffix="", title_tag="",
):
    """5x10 grid: folded first, misfolded on a new row."""
    folded_idx = target_data["folded_idx"]
    misfolded_idx = target_data["misfolded_idx"]
    n_folded, n_misfolded = len(folded_idx), len(misfolded_idx)
    if n_folded == 0 and n_misfolded == 0:
        return

    row_length = 10
    folded_rows = int(np.ceil(n_folded / row_length))
    misfolded_start = folded_rows * row_length
    total_slots = misfolded_start + n_misfolded
    n_grids = int(np.ceil(total_slots / 50.0))

    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range, y_range = compute_global_limits_from_coords(latent_coords)

    fe_target_clipped = None
    Xt_target = Yt_target = None
    if target_landscape:
        fe_t, xc_t, yc_t = target_landscape
        Xt_target, Yt_target = np.meshgrid(xc_t, yc_t)
        fe_target_clipped = np.clip(fe_t.T, 0, np.nanpercentile(fe_t.T, 95))

    fe_g_plot = None
    Xg = Yg = None
    if global_landscape:
        fe_g, xc_g, yc_g = global_landscape
        Xg, Yg = np.meshgrid(xc_g, yc_g)
        fe_g_plot = fe_g.T

    fe_bg_clipped = None
    Xt_bg = Yt_bg = None
    if global_landscape and target_landscape is None:
        fe_bg, xc_b, yc_b = global_landscape
        Xt_bg, Yt_bg = np.meshgrid(xc_b, yc_b)
        fe_bg_clipped = np.clip(fe_bg.T, 0, np.nanpercentile(fe_bg.T, 95))

    for grid_idx in range(n_grids):
        start_slot = grid_idx * 50
        fig, axes = plt.subplots(5, 10, figsize=(25, 15), sharex=True, sharey=True)
        part = f" (Part {grid_idx + 1}/{n_grids})" if n_grids > 1 else ""
        fig.suptitle(
            f"Target {target_name}: Trajectory Overlays{title_tag}{part}",
            fontsize=24, fontweight="bold", y=0.98,
        )

        for i, ax in enumerate(axes.flatten()):
            slot_idx = start_slot + i
            if slot_idx >= total_slots:
                ax.axis("off")
                continue

            traj_idx, status = None, ""
            if slot_idx < n_folded:
                traj_idx, status = folded_idx[slot_idx], "FOLDED"
            elif slot_idx >= misfolded_start:
                misf_i = slot_idx - misfolded_start
                if misf_i < n_misfolded:
                    traj_idx, status = misfolded_idx[misf_i], "MISFOLDED"

            if traj_idx is None:
                ax.axis("off")
                continue

            if fe_g_plot is not None:
                ax.contour(Xg, Yg, fe_g_plot, levels=15, cmap="Greys", alpha=0.2, linewidths=0.5)
            if fe_target_clipped is not None:
                ax.contourf(Xt_target, Yt_target, fe_target_clipped, levels=20, cmap="RdYlBu_r", alpha=0.7)
            elif fe_bg_clipped is not None:
                ax.contourf(Xt_bg, Yt_bg, fe_bg_clipped, levels=30, cmap="RdYlBu_r", alpha=0.5)

            points = latent_coords[traj_indices == traj_idx]
            if len(points) > 0:
                pts = points.reshape(-1, 1, 2)
                segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 1))
                lc.set_array(np.linspace(0, 1, len(points)))
                lc.set_linewidth(1.5)
                lc.set_alpha(0.8)
                ax.add_collection(lc)

            _add_refs(ax, ref_coords, ref_labels, white_style=white_style, force_standard=not white_style)
            title = trajectory_title(traj_idx, merged_files)
            t_color = "red" if status == "MISFOLDED" else ("white" if white_style else "black")
            ax.set_title(f"{title}\n{status}", fontsize=9, fontweight="bold", color=t_color)
            ax.grid(False)
            _apply_white_style(fig, ax, white_style=white_style)

        if white_style and fig._suptitle:
            fig.suptitle(fig._suptitle.get_text(), color="white", fontsize=24, fontweight="bold", y=0.98)
        plt.setp(axes, xlim=x_range, ylim=y_range)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        base = f"vae_target_trajectories_grid_{target_name}{suffix}"
        fname = f"{base}_part{grid_idx + 1}.png" if n_grids > 1 else f"{base}.png"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=white_style)
        print(f"Saved {out_path}")
        plt.close()


def plot_new_dataset_landscape_grid(
    latent_coords,
    traj_indices,
    output_dir,
    merged_files,
    ref_coords=None,
    ref_labels=None,
    global_limits=None,
    conf_dir=None,
    topology_glob="init_conf.gro",
    n_cols=5,
    suffix="",
):
    """
    Grid of per-trajectory overlays on the new dataset free energy landscape.

    Each subplot shows the same FE background (computed from all new-data frames)
    with one trajectory traced through latent space (color = time).
    """
    if traj_indices is None or len(merged_files) == 0:
        return

    unique_indices = np.unique(traj_indices)
    n_trajs = len(unique_indices)
    if n_trajs == 0:
        return

    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range, y_range = compute_global_limits_from_coords(latent_coords)

    fe, xc, yc = compute_free_energy(latent_coords, xlim=x_range, ylim=y_range)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))

    slots_per_page = 50
    n_grids = int(np.ceil(n_trajs / slots_per_page))

    for grid_idx in range(n_grids):
        start_idx = grid_idx * slots_per_page
        end_idx = min(start_idx + slots_per_page, n_trajs)
        n_plots = end_idx - start_idx

        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), sharex=True, sharey=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes)

        part = f" (Part {grid_idx + 1}/{n_grids})" if n_grids > 1 else ""
        fig.suptitle(
            f"New Dataset: Trajectories on Free Energy Landscape{suffix}{part}",
            fontsize=20, fontweight="bold", y=0.98,
        )

        for i in range(n_rows * n_cols):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            if i >= n_plots:
                ax.axis("off")
                continue

            ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r", alpha=0.5)

            traj_idx = unique_indices[start_idx + i]
            points = latent_coords[traj_indices == traj_idx]
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
            is_unfolded, status_text = (
                check_if_unfolded(traj_idx, merged_files, conf_dir, topology_glob)
                if conf_dir else (False, "")
            )
            title = trajectory_title(traj_idx, merged_files)
            if is_unfolded:
                title += f"\n{status_text}"
                ax.set_title(title, fontsize=8, fontweight="bold", color="red")
            else:
                ax.set_title(title, fontsize=8, fontweight="bold")
            ax.grid(False)

        plt.setp(axes, xlim=x_range, ylim=y_range)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        base = f"vae_new_dataset_landscape_grid{suffix}"
        fname = f"{base}_part{grid_idx + 1}.png" if n_grids > 1 else f"{base}.png"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def compute_global_limits_from_coords(latent_coords, margin_frac=0.05):
    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    return [x_min - mx, x_max + mx], [y_min - my, y_max + my]


def plot_new_latent_overview(
    latent_coords, traj_indices, ref_coords, ref_labels, output_dir,
    global_landscape=None, global_limits=None, old_latent=None,
):
    """
    Quick latent-space overview for a new dataset (no folding report needed).

    Left: new trajectories colored by trajectory index on training FE background.
    Right: free energy of the new data alone.
    """
    if global_limits:
        x_range, y_range = global_limits
    else:
        x_range, y_range = compute_global_limits_from_coords(latent_coords)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("New Dataset in Trained Latent Space", fontsize=14, fontweight="bold")

    # Panel 1: scatter on training background
    ax = axes[0]
    if global_landscape:
        fe_g, xc_g, yc_g = global_landscape
        Xg, Yg = np.meshgrid(xc_g, yc_g)
        fe_plot = np.clip(fe_g.T, 0, np.nanpercentile(fe_g.T, 95))
        ax.contour(Xg, Yg, fe_plot, levels=20, cmap="Greys", alpha=0.4, linewidths=0.6)
    if old_latent is not None:
        ax.scatter(
            old_latent[:, 0], old_latent[:, 1], s=1, alpha=0.08,
            c="gray", rasterized=True, label="training",
        )
    if traj_indices is not None:
        scatter = ax.scatter(
            latent_coords[:, 0], latent_coords[:, 1], c=traj_indices,
            cmap="nipy_spectral", s=2, alpha=0.5, rasterized=True, label="new",
        )
        plt.colorbar(scatter, ax=ax, label="Trajectory ID")
    else:
        ax.scatter(latent_coords[:, 0], latent_coords[:, 1], s=2, alpha=0.5, c="blue", rasterized=True)
    _add_refs(ax, ref_coords, ref_labels)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("Latent Z1")
    ax.set_ylabel("Latent Z2")
    ax.set_title("New trajectories (training FE background)")
    ax.legend(loc="upper right", markerscale=3)

    # Panel 2: FE of new data
    ax = axes[1]
    fe, xc, yc = compute_free_energy(latent_coords, xlim=x_range, ylim=y_range)
    Xg, Yg = np.meshgrid(xc, yc)
    fe_plot = np.clip(fe.T, 0, np.nanpercentile(fe.T, 95))
    contour = ax.contourf(Xg, Yg, fe_plot, levels=30, cmap="RdYlBu_r")
    plt.colorbar(contour, ax=ax, label="F / kT")
    _add_refs(ax, ref_coords, ref_labels)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("Latent Z1")
    ax.set_ylabel("Latent Z2")
    ax.set_title("New dataset free energy")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vae_new_dataset_latent_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
