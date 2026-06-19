"""
Trajectory loading and Q-based state labeling.

Ports load_merged_trajectories from train_vae.py with configurable paths and globs.
YAML: data.data_dir, data.conf_dir, data.trajectory_glob, data.topology_glob, data.stride
"""

from __future__ import annotations

import glob
import os

import mdtraj as md
import numpy as np

from utils.qscore import best_hummer_q


def load_merged_trajectories(
    data_dir: str,
    conf_dir: str,
    stride: int = 1,
    trajectory_glob: str = "iter_*/ratchet_*/md_noPBC.xtc",
    topology_glob: str = "init_conf.gro",
    csv_path=None,
):
    """
    Load all merged trajectories and concatenate them.
    Assign labels based on the fraction of native contacts Q.

    Returns:
        combined: mdtraj.Trajectory or None
        traj_labels: np.ndarray of per-frame labels
        traj_indices: np.ndarray of trajectory index per frame
    """
    refs = []
    for k in ["1", "2", "3", "4"]:
        ref_path = os.path.join(conf_dir, f"{k}.pdb")
        if os.path.exists(ref_path):
            refs.append(md.load(ref_path))

    if not refs:
        print(f"ERROR: No reference PDBs found in {conf_dir}")
        return None, None, None

    merged_files = sorted(glob.glob(os.path.join(data_dir, trajectory_glob)))

    if not merged_files:
        print("ERROR: No merged trajectories found!")
        return None, None, None

    print(f"Found {len(merged_files)} merged trajectories")

    all_trajs = []
    traj_labels = []
    traj_indices = []

    for i, merged_xtc in enumerate(merged_files):
        traj_dir = os.path.dirname(merged_xtc)
        parts = merged_xtc.split(os.sep)
        iter_name = parts[-3]
        ratchet_name = parts[-2]
        traj_name = f"{iter_name}/{ratchet_name}"

        top_files = glob.glob(os.path.join(traj_dir, topology_glob))
        if not top_files:
            print(f"  Skipping {traj_name}: no topology found")
            continue

        top_file = sorted(top_files)[0]

        try:
            traj = md.load(merged_xtc, top=top_file, stride=stride)

            q_vals = np.array([best_hummer_q(traj, ref) for ref in refs])
            max_q = np.max(q_vals, axis=0)
            argmax_q = np.argmax(q_vals, axis=0)

            labels_final = []
            for f in range(traj.n_frames):
                if f * stride < 800:
                    labels_final.append(0)
                elif max_q[f] > 0.85:
                    labels_final.append(argmax_q[f] + 1)
                else:
                    labels_final.append(-1)

            traj_labels.extend(labels_final)
            traj_indices.extend([i] * len(labels_final))

            if i == 0:
                print(f"    DEBUG: Labels for first trajectory ({len(labels_final)} frames):")
                prev_label = None
                start_f = 0
                for f_idx, label in enumerate(labels_final):
                    if label != prev_label:
                        if prev_label is not None:
                            print(f"      Frame {start_f:4d} - {f_idx-1:4d}: Label {prev_label}")
                        prev_label = label
                        start_f = f_idx
                print(f"      Frame {start_f:4d} - {len(labels_final)-1:4d}: Label {prev_label}")

            all_trajs.append(traj)
            print(f"  Loaded {traj_name}: {traj.n_frames} frames")

        except Exception as e:
            print(f"  Error loading {traj_name}: {e}")

    if not all_trajs:
        return None, None, None

    combined = md.join(all_trajs)
    print(f"\nTotal: {combined.n_frames} frames")

    return combined, np.array(traj_labels), np.array(traj_indices)


def find_trajectory_files(data_dir: str, trajectory_glob: str) -> list[str]:
    """Return sorted list of trajectory file paths matching glob."""
    return sorted(glob.glob(os.path.join(data_dir, trajectory_glob)))


def trajectory_title(traj_idx: int, merged_files: list[str]) -> str:
    """Format iter/ratchet title from trajectory file path."""
    if traj_idx >= len(merged_files):
        return f"Traj {traj_idx}"
    fpath = merged_files[traj_idx]
    parts = fpath.split(os.sep)
    iter_part = next((p for p in reversed(parts) if p.startswith("iter_")), "?")
    ratchet_part = next((p for p in reversed(parts) if p.startswith("ratchet_")), "?")
    return f"{iter_part.replace('iter_', 'iter ')} - {ratchet_part.replace('ratchet_', 'ratchet ')}"
