"""
Composable feature blocks for the generalized encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np

from features.signatures.providers import (
    build_cmap_signature,
    build_spatial_pair_signature,
    filter_pairs_top_k,
    load_reference_trajectory,
)
from features.signatures.templates import build_dihedrals_from_template, build_pairs_from_template
from features.transforms import apply_distance_transform, sin_cos_dihedral


@dataclass
class BlockOutput:
    continuous: np.ndarray
    binary: np.ndarray
    labels: list[str]


class DistancePairsBlock:
    def __init__(
        self,
        ref_traj: md.Trajectory,
        template: str = "watson_crick",
        transform: str = "raw_distance",
        distance_scale_angstrom: float = 7.5,
        binary_cutoff_nm: float | None = None,
        spatial_filter_nm: float | None = None,
        top_k: int | None = None,
    ):
        self.transform = transform
        self.distance_scale_angstrom = distance_scale_angstrom
        self.binary_cutoff_nm = binary_cutoff_nm
        pairs, labels = build_pairs_from_template(ref_traj, template)
        if spatial_filter_nm is not None and len(pairs) > 0:
            pairs, labels = build_spatial_pair_signature(ref_traj, pairs, spatial_filter_nm)
        if top_k is not None and len(pairs) > 0:
            pairs, labels = filter_pairs_top_k(
                ref_traj, pairs, top_k, distance_scale_angstrom
            )
        self.pairs = pairs
        self.labels = [f"dist_{l}" for l in labels]

    def compute(self, traj: md.Trajectory) -> BlockOutput:
        if len(self.pairs) == 0:
            n = traj.n_frames
            return BlockOutput(
                continuous=np.zeros((n, 0), dtype=np.float32),
                binary=np.zeros((n, 0), dtype=np.float32),
                labels=[],
            )
        distances = md.compute_distances(traj, self.pairs).astype(np.float32)
        continuous = apply_distance_transform(
            distances, self.transform, self.distance_scale_angstrom
        )
        binary = np.zeros((traj.n_frames, 0), dtype=np.float32)
        if self.binary_cutoff_nm is not None:
            binary = (distances < self.binary_cutoff_nm).astype(np.float32)
        return BlockOutput(continuous=continuous, binary=binary, labels=self.labels)


class DihedralBlock:
    def __init__(self, ref_traj: md.Trajectory, template: str = "all_nucleotide_chi"):
        dihedrals, labels = build_dihedrals_from_template(ref_traj, template)
        self.dihedrals = dihedrals
        self.labels = [f"dihedral_{l}" for l in labels]

    def compute(self, traj: md.Trajectory) -> BlockOutput:
        if len(self.dihedrals) == 0:
            n = traj.n_frames
            return BlockOutput(
                continuous=np.zeros((n, 0), dtype=np.float32),
                binary=np.zeros((n, 0), dtype=np.float32),
                labels=[],
            )
        angles = md.compute_dihedrals(traj, self.dihedrals).astype(np.float32)
        continuous = sin_cos_dihedral(angles)
        feat_labels = [f"{l}_sin" for l in self.labels] + [f"{l}_cos" for l in self.labels]
        return BlockOutput(
            continuous=continuous,
            binary=np.zeros((traj.n_frames, 0), dtype=np.float32),
            labels=feat_labels,
        )


class CmapBlock:
    def __init__(
        self,
        ref_traj: md.Trajectory,
        selection: str = "not element H",
        min_residues_apart: int = 3,
        signature_mode: str = "top_k",
        top_k: int = 500,
        threshold: float = 0.6,
        distance_scale_angstrom: float = 7.5,
        binary_cutoff: float | None = 0.6,
    ):
        self.distance_scale_angstrom = distance_scale_angstrom
        self.binary_cutoff = binary_cutoff
        pairs, labels = build_cmap_signature(
            ref_traj,
            selection=selection,
            min_residues_apart=min_residues_apart,
            signature_mode=signature_mode,
            top_k=top_k,
            threshold=threshold,
            distance_scale_angstrom=distance_scale_angstrom,
        )
        self.pairs = pairs
        self.labels = [f"cmap_{l}" for l in labels]

    def compute(self, traj: md.Trajectory) -> BlockOutput:
        if len(self.pairs) == 0:
            n = traj.n_frames
            return BlockOutput(
                continuous=np.zeros((n, 0), dtype=np.float32),
                binary=np.zeros((n, 0), dtype=np.float32),
                labels=[],
            )
        distances = md.compute_distances(traj, self.pairs).astype(np.float32)
        continuous = apply_distance_transform(
            distances, "sigmoid_squared", self.distance_scale_angstrom
        )
        binary = np.zeros((traj.n_frames, 0), dtype=np.float32)
        if self.binary_cutoff is not None:
            binary = (continuous > self.binary_cutoff).astype(np.float32)
        return BlockOutput(continuous=continuous, binary=binary, labels=self.labels)


def build_block_from_config(block_cfg: dict, ref_traj: md.Trajectory):
    """Instantiate a block from a YAML block dict."""
    btype = block_cfg["type"]
    if btype == "distance_pairs":
        return DistancePairsBlock(
            ref_traj,
            template=block_cfg.get("template", "watson_crick"),
            transform=block_cfg.get("transform", "raw_distance"),
            distance_scale_angstrom=block_cfg.get("distance_scale_angstrom", 7.5),
            binary_cutoff_nm=block_cfg.get("binary_cutoff_nm"),
            spatial_filter_nm=block_cfg.get("spatial_filter_nm"),
            top_k=block_cfg.get("top_k"),
        )
    if btype == "dihedral":
        return DihedralBlock(ref_traj, template=block_cfg.get("template", "all_nucleotide_chi"))
    if btype == "cmap":
        return CmapBlock(
            ref_traj,
            selection=block_cfg.get("selection", "not element H"),
            min_residues_apart=block_cfg.get("min_residues_apart", 3),
            signature_mode=block_cfg.get("signature_mode", "top_k"),
            top_k=block_cfg.get("top_k", 500),
            threshold=block_cfg.get("threshold", 0.6),
            distance_scale_angstrom=block_cfg.get("distance_scale_angstrom", 7.5),
            binary_cutoff=block_cfg.get("binary_cutoff"),
        )
    raise ValueError(f"Unknown block type {btype!r}")


def compute_blocks(traj: md.Trajectory, blocks: list) -> tuple[np.ndarray, int, int, list[str]]:
    """Run all blocks and concatenate into feature matrix."""
    cont_parts, bin_parts, all_labels = [], [], []
    for block in blocks:
        out = block.compute(traj)
        if out.continuous.shape[1] > 0:
            cont_parts.append(out.continuous)
        if out.binary.shape[1] > 0:
            bin_parts.append(out.binary)
        all_labels.extend(out.labels)

    n_frames = traj.n_frames
    continuous = np.concatenate(cont_parts, axis=1) if cont_parts else np.zeros((n_frames, 0), dtype=np.float32)
    binary = np.concatenate(bin_parts, axis=1) if bin_parts else np.zeros((n_frames, 0), dtype=np.float32)
    features = np.concatenate([continuous, binary], axis=1) if binary.shape[1] > 0 else continuous
    return features, continuous.shape[1], binary.shape[1], all_labels
