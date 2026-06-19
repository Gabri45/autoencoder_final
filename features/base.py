"""
Abstract base for pluggable feature encoders.

Each encoder converts an mdtraj trajectory into a feature matrix and metadata
describing continuous vs binary feature counts. Register new encoders via
features.registry.register_encoder and select them in YAML via features.encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureMeta:
    """Metadata returned alongside computed features."""

    n_continuous: int
    n_binary: int
    feature_names: list[str] | None = None


class FeatureEncoder(ABC):
    """Contract for any molecular feature encoding strategy."""

    name: str = "base"

    @abstractmethod
    def compute(self, traj) -> tuple[np.ndarray, FeatureMeta]:
        """
        Compute feature matrix from trajectory.

        Args:
            traj: mdtraj.Trajectory object.

        Returns:
            features: Array of shape (n_frames, n_features).
            meta: FeatureMeta with n_continuous and n_binary counts.
        """

    def describe(self) -> dict:
        """Human-readable description for logs and config snapshots."""
        return {"name": self.name, "class": self.__class__.__name__}
