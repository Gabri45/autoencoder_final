"""
Distance and dihedral transforms for feature blocks.

Maps raw geometric quantities to network inputs (continuous features).
"""

from __future__ import annotations

import numpy as np


def sigmoid_squared(distances_nm: np.ndarray, distance_scale_angstrom: float = 7.5) -> np.ndarray:
    """
    GROMACS/ratchet-style soft contact map (vectorized).

    Args:
        distances_nm: Distances in nm, any shape.
        distance_scale_angstrom: Characteristic contact scale in Angstrom.

    Returns:
        Mapped values in approximately [0, 1].
    """
    scale_nm = distance_scale_angstrom / 10.0
    x = distances_nm / scale_nm
    x = x * x
    cond0 = x <= 151.29
    cond1 = np.abs(x - 56.25) < 1e-5
    x = np.where(cond1, 0.0, x)
    mapped = np.where(cond1, 0.6, (1.0 - (x / 56.25) ** 3) / (1.0 - (x / 56.25) ** 5))
    return np.where(cond0, mapped, 0.0).astype(np.float32)


def raw_distance(distances_nm: np.ndarray) -> np.ndarray:
    """Identity transform — distances in nm."""
    return distances_nm.astype(np.float32)


def sin_cos_dihedral(angles_rad: np.ndarray) -> np.ndarray:
    """Stack sin and cos along the last axis (n_frames, n_angles) -> (n_frames, 2*n_angles)."""
    return np.concatenate(
        [np.sin(angles_rad), np.cos(angles_rad)],
        axis=-1,
    ).astype(np.float32)


TRANSFORMS = {
    "raw_distance": raw_distance,
    "sigmoid_squared": sigmoid_squared,
    "sin_cos": sin_cos_dihedral,
}


def apply_distance_transform(
    distances_nm: np.ndarray,
    name: str,
    distance_scale_angstrom: float = 7.5,
) -> np.ndarray:
    """Apply a named distance transform."""
    if name == "sigmoid_squared":
        return sigmoid_squared(distances_nm, distance_scale_angstrom)
    if name == "raw_distance":
        return raw_distance(distances_nm)
    raise ValueError(f"Unknown distance transform {name!r}. Available: {list(TRANSFORMS)}")
