"""
Hamming distance contrastive validation plots.

YAML: evaluation.plots.hamming
Outputs: vae_contrastive_hamming_distances.png / vae_ensemble_contrastive_hamming_distances.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as patheffects


def plot_hamming(ctx):
    """Plot latent space colored by Hamming distance to reference binary features."""
    latent_coords = ctx["latent_coords"]
    x_bin = ctx.get("x_bin")
    ref_coords = ctx.get("ref_coords")
    ref_labels = ctx.get("ref_labels")
    ref_binary = ctx.get("ref_binary")
    output_dir = ctx["output_dir"]
    mode = ctx.get("mode", "single")

    if x_bin is None or ref_binary is None or len(ref_binary) == 0:
        print("Skipping Hamming distance plots (no binary features).")
        return

    title_suffix = " (Ensemble Mean)" if mode == "ensemble" else ""
    n_refs = len(ref_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Contrastive Validation: Hamming Distance to Reference States{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for i in range(4):
        ax = axes.flatten()[i]
        if i < n_refs:
            hamming_dist = np.sum(np.abs(x_bin - ref_binary[i]), axis=1)
            scatter = ax.scatter(
                latent_coords[:, 0], latent_coords[:, 1], c=hamming_dist,
                cmap="coolwarm", s=2, alpha=0.5, rasterized=True,
            )
            ax.set_title(f"Distance to Reference {ref_labels[i]}")
            plt.colorbar(scatter, ax=ax, label="Hamming Distance")
            ax.scatter(ref_coords[i, 0], ref_coords[i, 1], c="red", marker="*", s=200,
                       edgecolors="black", linewidths=1.5, zorder=10)
            ax.text(ref_coords[i, 0], ref_coords[i, 1] + 0.1, ref_labels[i],
                    fontsize=12, fontweight="bold", ha="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")], zorder=11)
        else:
            ax.axis("off")

    plt.tight_layout()
    fname = "vae_ensemble_contrastive_hamming_distances.png" if mode == "ensemble" else "vae_contrastive_hamming_distances.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
