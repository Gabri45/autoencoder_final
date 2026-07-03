"""
Composite feature encoder — YAML-driven blocks for any molecular system.

YAML:
  features:
    encoder: composite
    params:
      reference_pdb: /path/to/reference.pdb   # optional; uses traj frame 0 if omitted
      preset: rna_general                     # optional shorthand
      blocks: [...]

Presets:
  rna_g4_enriched — equivalent to legacy G4 encoder (g4 H-bonds + guanine chi + binary)
  rna_general     — WC distances + all chi + CMAP top-K + binaries
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from features.base import FeatureEncoder, FeatureMeta
from features.blocks import build_block_from_config, compute_blocks
from features.manifest import SignatureManifest
from features.registry import register_encoder
from features.signatures.providers import load_reference_trajectory

PRESETS: dict[str, list[dict]] = {
    "rna_g4_enriched": [
        {
            "type": "distance_pairs",
            "template": "g4_hoogsteen",
            "transform": "raw_distance",
            "binary_cutoff_nm": 0.35,
        },
        {
            "type": "dihedral",
            "template": "guanine_chi",
        },
    ],
    "rna_general": [
        {
            "type": "distance_pairs",
            "template": "watson_crick",
            "transform": "raw_distance",
            "binary_cutoff_nm": 0.35,
            "spatial_filter_nm": 0.45,
        },
        {
            "type": "dihedral",
            "template": "all_nucleotide_chi",
        },
        {
            "type": "cmap",
            "selection": "backbone",
            "signature_mode": "top_k",
            "top_k": 500,
            "binary_cutoff": 0.6,
        },
    ],
}


@register_encoder("composite")
class CompositeEncoder(FeatureEncoder):
    """Declarative multi-block encoder for generalized molecular features."""

    name = "composite"

    def __init__(
        self,
        blocks: list[dict] | None = None,
        preset: str | None = None,
        reference_pdb: str | None = None,
    ):
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset {preset!r}. Available: {list(PRESETS)}")
            self.block_configs = [dict(b) for b in PRESETS[preset]]
            self.preset = preset
        elif blocks:
            self.block_configs = blocks
            self.preset = None
        else:
            raise ValueError("composite encoder requires 'blocks' or 'preset' in params")

        self.reference_pdb = reference_pdb
        self._blocks = None
        self._manifest: SignatureManifest | None = None

    def _ensure_blocks(self, traj: md.Trajectory):
        if self._blocks is not None:
            return
        ref = load_reference_trajectory(self.reference_pdb, traj)
        self._blocks = [build_block_from_config(cfg, ref) for cfg in self.block_configs]

        all_pairs, all_dihedrals, all_labels = [], [], []
        block_summaries = []
        for cfg, block in zip(self.block_configs, self._blocks):
            summary = {"type": cfg["type"], **{k: v for k, v in cfg.items() if k != "type"}}
            if hasattr(block, "pairs"):
                all_pairs.extend(block.pairs.tolist())
                summary["n_pairs"] = len(block.pairs)
            if hasattr(block, "dihedrals"):
                all_dihedrals.extend(block.dihedrals.tolist())
                summary["n_dihedrals"] = len(block.dihedrals)
            if hasattr(block, "labels"):
                all_labels.extend(block.labels)
                summary["n_features"] = len(block.labels)
            block_summaries.append(summary)

        self._manifest = SignatureManifest(
            pairs=all_pairs,
            dihedrals=all_dihedrals,
            labels=all_labels,
            blocks=block_summaries,
            encoder=self.name,
            params=self.describe(),
        )

    def compute(self, traj) -> tuple:
        self._ensure_blocks(traj)
        features, n_cont, n_bin, labels = compute_blocks(traj, self._blocks)
        if features.shape[1] == 0:
            raise ValueError("Composite encoder produced zero features; check block configuration")
        meta = FeatureMeta(n_continuous=n_cont, n_binary=n_bin, feature_names=labels)
        return features.astype(np.float32), meta

    def get_manifest(self) -> SignatureManifest | None:
        return self._manifest

    def describe(self) -> dict:
        info = super().describe()
        info["preset"] = self.preset
        info["reference_pdb"] = self.reference_pdb
        info["blocks"] = self.block_configs
        return info
