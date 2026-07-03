"""
Build atom-pair signatures from trajectories, references, and filters.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from features.transforms import sigmoid_squared

RNA_BACKBONE_ATOM_NAMES = frozenset({
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "C1*",
})

SELECTION_ALIASES = {
    "backbone": "__rna_backbone__",
    "heavy": "not element H",
}


def resolve_selection(selection: str) -> str:
    return SELECTION_ALIASES.get(selection, selection)


def select_atom_indices(traj: md.Trajectory, selection: str) -> np.ndarray:
    """Return atom indices for an MDTraj selection string or alias."""
    selection = resolve_selection(selection)
    if selection == "__rna_backbone__":
        indices = [a.index for a in traj.topology.atoms if a.name in RNA_BACKBONE_ATOM_NAMES]
        if not indices:
            raise ValueError("RNA backbone selection matched no atoms")
        return np.asarray(indices, dtype=np.int32)
    indices = traj.topology.select(selection)
    if len(indices) == 0:
        raise ValueError(f"Atom selection matched no atoms: {selection!r}")
    return np.asarray(indices, dtype=np.int32)


def residue_indices_for_atoms(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    """Residue index per selected atom."""
    return np.array([traj.topology.atom(int(i)).residue.index for i in atom_indices], dtype=np.int32)


def build_sequential_pair_signature(
    atom_indices: np.ndarray,
    residue_indices: np.ndarray,
    min_residues_apart: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """
    All atom pairs with sufficient sequence separation.

    Returns:
        pairs: (M, 2) int array
        labels: human-readable labels
    """
    pairs = []
    labels = []
    for i in range(len(atom_indices)):
        for j in range(i + 1, len(atom_indices)):
            if residue_indices[j] - residue_indices[i] >= min_residues_apart:
                ai, aj = int(atom_indices[i]), int(atom_indices[j])
                pairs.append([ai, aj])
                labels.append(f"pair_{ai}_{aj}")
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), []
    return np.asarray(pairs, dtype=np.int32), labels


def build_cmap_signature(
    ref_traj: md.Trajectory,
    selection: str = "not element H",
    min_residues_apart: int = 3,
    signature_mode: str = "top_k",
    top_k: int = 500,
    threshold: float = 0.6,
    distance_scale_angstrom: float = 7.5,
    external_pairs: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a CMAP signature from a reference structure.

    signature_mode: top_k | threshold | all | external
    """
    if signature_mode == "external":
        if external_pairs is None or len(external_pairs) == 0:
            raise ValueError("external_pairs required for signature_mode=external")
        pairs = np.asarray(external_pairs, dtype=np.int32)
        labels = [f"cmap_{a}_{b}" for a, b in pairs]
        return pairs, labels

    atom_indices = select_atom_indices(ref_traj, selection)
    res_indices = residue_indices_for_atoms(ref_traj, atom_indices)
    pairs, labels = build_sequential_pair_signature(atom_indices, res_indices, min_residues_apart)

    if len(pairs) == 0:
        return pairs, labels

    if signature_mode == "all":
        return pairs, [f"cmap_{i}" for i in range(len(pairs))]

    distances = md.compute_distances(ref_traj, pairs)[0]
    values = sigmoid_squared(distances, distance_scale_angstrom)

    if signature_mode == "top_k":
        k = min(top_k, len(pairs))
        order = np.argsort(-values)[:k]
        pairs = pairs[order]
        labels = [f"cmap_top_{i}" for i in range(len(pairs))]
    elif signature_mode == "threshold":
        mask = values >= threshold
        pairs = pairs[mask]
        labels = [f"cmap_thr_{i}" for i in range(len(pairs))]
    else:
        raise ValueError(f"Unknown signature_mode {signature_mode!r}")

    return pairs, labels


def filter_pairs_top_k(
    ref_traj: md.Trajectory,
    pairs: np.ndarray,
    top_k: int,
    distance_scale_angstrom: float = 7.5,
) -> tuple[np.ndarray, list[str]]:
    """Keep top-K pairs by sigmoid_squared on reference."""
    if len(pairs) == 0:
        return pairs, []
    distances = md.compute_distances(ref_traj, pairs)[0]
    values = sigmoid_squared(distances, distance_scale_angstrom)
    k = min(top_k, len(pairs))
    order = np.argsort(-values)[:k]
    pairs = pairs[order]
    return pairs, [f"top_{i}" for i in range(len(pairs))]


def filter_pairs_threshold(
    ref_traj: md.Trajectory,
    pairs: np.ndarray,
    threshold: float,
    distance_scale_angstrom: float = 7.5,
) -> tuple[np.ndarray, list[str]]:
    if len(pairs) == 0:
        return pairs, []
    distances = md.compute_distances(ref_traj, pairs)[0]
    values = sigmoid_squared(distances, distance_scale_angstrom)
    mask = values >= threshold
    pairs = pairs[mask]
    return pairs, [f"thr_{i}" for i in range(len(pairs))]


def build_spatial_pair_signature(
    ref_traj: md.Trajectory,
    pairs: np.ndarray,
    spatial_cutoff_nm: float = 0.6,
) -> tuple[np.ndarray, list[str]]:
    """Filter pair list to those within spatial_cutoff_nm on the reference."""
    if len(pairs) == 0:
        return pairs, []
    distances = md.compute_distances(ref_traj, pairs)[0]
    mask = distances <= spatial_cutoff_nm
    filtered = pairs[mask]
    labels = [f"spatial_{a}_{b}" for a, b in filtered]
    return filtered, labels


def load_reference_trajectory(reference_pdb: str | None, traj: md.Trajectory) -> md.Trajectory:
    """Load reference PDB or use first frame of trajectory."""
    if reference_pdb:
        return md.load(reference_pdb)
    return traj[0]
