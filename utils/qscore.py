"""
Best-Hummer-Eaton Q-score and unfolding checks.

Used for trajectory labeling (training) and grid plot annotations (evaluation).
"""

from __future__ import annotations

import glob
import os
from itertools import combinations

import mdtraj as md
import numpy as np

Q_THRESHOLD = 0.85


def best_hummer_q(traj, native):
    """Compute Q using Best-Hummer-Eaton method."""
    BETA_CONST = 50
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45

    heavy = native.topology.select('not element H')
    heavy_pairs = np.array(
        [(i, j) for (i, j) in combinations(heavy, 2)
         if abs(native.topology.atom(i).residue.index -
                native.topology.atom(j).residue.index) > 3]
    )

    if len(heavy_pairs) == 0:
        return np.zeros(traj.n_frames)

    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]

    if len(native_contacts) == 0:
        return np.zeros(traj.n_frames)

    r = md.compute_distances(traj, native_contacts)
    r0 = md.compute_distances(native[0], native_contacts)
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)

    return q


def check_if_unfolded(traj_idx, merged_files, conf_dir: str, topology_glob: str = "init_conf.gro"):
    """
    Check if the specific trajectory is unfolded (never reached Q > 0.85).
    Returns (is_unfolded, status_text)
    """
    if traj_idx >= len(merged_files):
        return False, "IndexError"

    xtc_path = merged_files[traj_idx]
    traj_dir = os.path.dirname(xtc_path)

    top_files = glob.glob(os.path.join(traj_dir, topology_glob))
    if not top_files:
        hopfield_gros = glob.glob(os.path.join(traj_dir, "hopfield_*", topology_glob))
        if not hopfield_gros:
            return False, "NoTop"
        top_file = sorted(hopfield_gros)[0]
    else:
        top_file = sorted(top_files)[0]

    try:
        traj = md.load(xtc_path, top=top_file)
        last_frame = traj[-1]
    except Exception:
        return False, "LoadErr"

    max_q = 0
    for k in ["1", "2", "3", "4"]:
        ref_path = os.path.join(conf_dir, f"{k}.pdb")
        if os.path.exists(ref_path):
            ref = md.load(ref_path)
            q = best_hummer_q(last_frame, ref)[0]
            if q > max_q:
                max_q = q

    if max_q < Q_THRESHOLD:
        return True, f"UNFOLDED (Q={max_q:.2f})"
    return False, f"Folded (Q={max_q:.2f})"
