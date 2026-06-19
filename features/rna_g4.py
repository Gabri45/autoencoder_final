"""
RNA G-quadruplex enriched feature encoder.

Ports compute_features_enriched from autoencoder_contrastive_classifier/train_vae.py:
G4 H-bond distances (all guanine pairs) + sin/cos chi angles + binary H-bond cutoff.

YAML: features.encoder=rna_g4_enriched, features.params.hbond_cutoff
"""

from __future__ import annotations

import itertools

import mdtraj as md
import numpy as np

from features.base import FeatureEncoder, FeatureMeta
from features.registry import register_encoder


def compute_g4_distances_full(traj):
    """
    Compute pairwise H-bond distances for ALL Guanine residues.

    For every pair of unique Guanine residues (Gi, Gj) where i != j,
    computes two distances:
    1. Gi N1 ... Gj O6
    2. Gi N2 ... Gj N7

    Returns:
        distances: np.ndarray of shape (n_frames, n_pairs * 2)
    """
    topology = traj.topology

    guanines = [r for r in topology.residues if r.name in ['DG', 'G', 'GUA', 'DGU']]
    guanines.sort(key=lambda x: x.index)

    pairs = []

    for g1, g2 in itertools.permutations(guanines, 2):
        try:
            n1 = next((a for a in g1.atoms if a.name == 'N1'), None)
            o6 = next((a for a in g2.atoms if a.name == 'O6'), None)

            if n1 and o6:
                pairs.append((n1.index, o6.index))

            n2 = next((a for a in g1.atoms if a.name == 'N2'), None)
            n7 = next((a for a in g2.atoms if a.name == 'N7'), None)

            if n2 and n7:
                pairs.append((n2.index, n7.index))

        except Exception as e:
            print(f"Error processing pair {g1}-{g2}: {e}")
            continue

    if not pairs:
        return None

    distances = md.compute_distances(traj, pairs)
    return distances


def compute_chi_angles(traj):
    """
    Compute glycosidic chi angles for Guanine residues.
    Returns:
        angles: np.ndarray of shape (n_frames, n_guanines)
    """
    topology = traj.topology
    guanines = [r for r in topology.residues if r.name in ['DG', 'G', 'GUA', 'DGU']]
    guanines.sort(key=lambda x: x.index)

    dihedrals = []

    for g in guanines:
        try:
            o4p = next((a for a in g.atoms if a.name in ["O4'", "O4*"]), None)
            c1p = next((a for a in g.atoms if a.name in ["C1'", "C1*"]), None)
            n9 = next((a for a in g.atoms if a.name == 'N9'), None)
            c4 = next((a for a in g.atoms if a.name == 'C4'), None)

            if o4p and c1p and n9 and c4:
                dihedrals.append([o4p.index, c1p.index, n9.index, c4.index])
        except Exception:
            continue

    if not dihedrals:
        return None

    angles = md.compute_dihedrals(traj, dihedrals)
    return angles


def compute_features_enriched(traj, hbond_cutoff: float = 0.35):
    """
    Compute both H-bond distances and dihedral angles (sin/cos).
    Concatenates them into a single feature array.
    """
    distances = compute_g4_distances_full(traj)
    if distances is None:
        return None

    chi_angles = compute_chi_angles(traj)
    if chi_angles is None:
        return distances

    sin_chi = np.sin(chi_angles)
    cos_chi = np.cos(chi_angles)

    binary_features = (distances < hbond_cutoff).astype(np.float32)

    features = np.concatenate([distances, sin_chi, cos_chi, binary_features], axis=1)

    n_binary = binary_features.shape[1]
    n_continuous = features.shape[1] - n_binary

    return features, n_continuous, n_binary


@register_encoder("rna_g4_enriched")
class RNAG4EnrichedEncoder(FeatureEncoder):
    """G4 distance + chi sin/cos + binary H-bond features for RNA G-quadruplex."""

    name = "rna_g4_enriched"

    def __init__(self, hbond_cutoff: float = 0.35):
        self.hbond_cutoff = hbond_cutoff

    def compute(self, traj) -> tuple[np.ndarray, FeatureMeta]:
        result = compute_features_enriched(traj, hbond_cutoff=self.hbond_cutoff)
        if result is None:
            raise ValueError("Could not compute G4 enriched features from trajectory")
        features, n_continuous, n_binary = result
        return features, FeatureMeta(n_continuous=n_continuous, n_binary=n_binary)

    def describe(self) -> dict:
        info = super().describe()
        info["hbond_cutoff"] = self.hbond_cutoff
        return info
