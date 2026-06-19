"""
Reference PDB projection for evaluation plots.

Projects conf_dir/1.pdb ... 4.pdb into latent space using the trained model(s).
"""

from __future__ import annotations

import os

import mdtraj as md
import numpy as np
import torch

from features.rna_g4 import compute_features_enriched


def project_reference_pdbs_single(model, scaler, conf_dir, n_continuous, n_binary, device=None):
    """
    Load reference PDBs and encode with a single model.

    Returns:
        ref_coords, ref_labels, ref_binary
    """
    if device is None:
        device = next(model.parameters()).device

    ref_coords, ref_labels, ref_binary = [], [], []
    print(f"\nProjecting reference configurations from {conf_dir}...")

    for i in range(1, 5):
        pdb_path = os.path.join(conf_dir, f"{i}.pdb")
        if not os.path.exists(pdb_path):
            print(f"  Warning: {pdb_path} not found.")
            continue

        print(f"  Processing {i}.pdb...")
        try:
            traj = md.load(pdb_path)
            dists, n_cont_tmp, n_bin_tmp = compute_features_enriched(traj)
            if dists is None:
                continue

            if n_continuous > 0:
                x_cont = dists[:, :n_continuous]
                x_cont_scaled = scaler.transform(x_cont).astype(np.float32)
            else:
                x_cont_scaled = np.zeros((dists.shape[0], 0), dtype=np.float32)

            if n_binary > 0:
                x_bin = dists[:, n_continuous:]
                dists_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)
            else:
                x_bin = None
                dists_scaled = x_cont_scaled

            model = model.to(device)
            tensor_x = torch.tensor(dists_scaled, device=device)

            with torch.no_grad():
                mu, _ = model.encode(tensor_x)

            ref_coords.append(mu.cpu().numpy()[0])
            ref_labels.append(str(i))
            if n_binary > 0:
                ref_binary.append(x_bin[0])
        except Exception as e:
            print(f"    Error processing {i}.pdb: {e}")

    if not ref_coords:
        return None, None, None

    ref_binary_arr = np.array(ref_binary) if ref_binary else None
    return np.array(ref_coords), ref_labels, ref_binary_arr


def project_reference_pdbs_ensemble(models, scaler, conf_dir):
    """Project references using ensemble mean latent position."""
    ref_coords, ref_labels, ref_binary = [], [], []
    print(f"\nProjecting reference configurations from {conf_dir}...")

    if not os.path.exists(conf_dir):
        print(f"Warning: Conf directory {conf_dir} not found.")
        return None, None, None

    n_continuous = models[0].n_continuous
    n_binary = models[0].n_binary
    device = next(models[0].parameters()).device

    for i in range(1, 10):
        pdb_path = os.path.join(conf_dir, f"{i}.pdb")
        if not os.path.exists(pdb_path):
            if i <= 4:
                print(f"  Warning: {pdb_path} not found.")
            continue

        print(f"  Processing {i}.pdb...")
        try:
            traj = md.load(pdb_path)
            dists, _, _ = compute_features_enriched(traj)
            if dists is None:
                continue

            if n_continuous > 0:
                x_cont = dists[:, :n_continuous]
                x_cont_scaled = scaler.transform(x_cont).astype(np.float32)
            else:
                x_cont_scaled = np.zeros((dists.shape[0], 0), dtype=np.float32)

            if n_binary > 0:
                x_bin = dists[:, n_continuous:]
                dists_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)
            else:
                x_bin = None
                dists_scaled = x_cont_scaled

            tensor_x = torch.tensor(dists_scaled, device=device)
            mus = []
            with torch.no_grad():
                for model in models:
                    mu, _ = model.encode(tensor_x)
                    mus.append(mu.cpu().numpy())

            mus = np.stack(mus)
            mean_mu = np.mean(mus, axis=0)
            ref_coords.append(mean_mu[0])
            ref_labels.append(str(i))
            if n_binary > 0:
                ref_binary.append(x_bin[0])
        except Exception as e:
            print(f"    Error processing {i}.pdb: {e}")

    if not ref_coords:
        return None, None, None

    ref_binary_arr = np.array(ref_binary) if ref_binary else None
    return np.array(ref_coords), ref_labels, ref_binary_arr


def add_reference_points(ax, ref_coords, ref_labels):
    """Add reference star markers and labels to a matplotlib axis."""
    import matplotlib.patheffects as patheffects

    if ref_coords is None:
        return
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], c="white", s=100, zorder=20)
    ax.scatter(
        ref_coords[:, 0], ref_coords[:, 1], c="red", marker="*", s=80,
        edgecolors="black", linewidths=0.5, zorder=21,
    )
    if ref_labels:
        for coord, label in zip(ref_coords, ref_labels):
            ax.text(
                coord[0], coord[1], label, fontsize=12, fontweight="bold",
                color="black", ha="center", va="bottom", zorder=22,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
            )
