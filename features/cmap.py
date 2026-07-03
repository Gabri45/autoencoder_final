"""
Standalone CMAP feature encoder.

YAML: features.encoder=cmap, features.params.reference_pdb, top_k, selection, ...
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from features.base import FeatureEncoder, FeatureMeta
from features.blocks import CmapBlock, compute_blocks
from features.manifest import SignatureManifest
from features.registry import register_encoder
from features.signatures.providers import load_reference_trajectory


@register_encoder("cmap")
class CmapEncoder(FeatureEncoder):
    """Soft contact map features from a fixed signature (top-K or threshold)."""

    name = "cmap"

    def __init__(
        self,
        reference_pdb: str | None = None,
        selection: str = "backbone",
        min_residues_apart: int = 3,
        signature_mode: str = "top_k",
        top_k: int = 500,
        threshold: float = 0.6,
        distance_scale_angstrom: float = 7.5,
        binary_cutoff: float | None = 0.6,
    ):
        self.reference_pdb = reference_pdb
        self.block_cfg = {
            "type": "cmap",
            "selection": selection,
            "min_residues_apart": min_residues_apart,
            "signature_mode": signature_mode,
            "top_k": top_k,
            "threshold": threshold,
            "distance_scale_angstrom": distance_scale_angstrom,
            "binary_cutoff": binary_cutoff,
        }
        self._blocks = None
        self._manifest: SignatureManifest | None = None

    def _ensure_blocks(self, traj: md.Trajectory):
        if self._blocks is not None:
            return
        ref = load_reference_trajectory(self.reference_pdb, traj)
        block = CmapBlock(ref, **{k: v for k, v in self.block_cfg.items() if k != "type"})
        self._blocks = [block]
        self._manifest = SignatureManifest(
            pairs=block.pairs.tolist(),
            labels=block.labels,
            encoder=self.name,
            params=self.describe(),
        )

    def compute(self, traj) -> tuple[np.ndarray, FeatureMeta]:
        self._ensure_blocks(traj)
        features, n_cont, n_bin, labels = compute_blocks(traj, self._blocks)
        if features.shape[1] == 0:
            raise ValueError("CMAP encoder produced zero features; check reference/selection/top_k")
        meta = FeatureMeta(n_continuous=n_cont, n_binary=n_bin, feature_names=labels)
        return features.astype(np.float32), meta

    def get_manifest(self) -> SignatureManifest | None:
        return self._manifest

    def describe(self) -> dict:
        info = super().describe()
        info.update(self.block_cfg)
        info["reference_pdb"] = self.reference_pdb
        return info
