"""
Parser for folding_by_target.txt reports.

YAML: evaluation.by_target.folding_report, evaluation.new_dataset.folding_report
"""

from __future__ import annotations

import os

TARGETS = ["143D", "1KF1", "2HY9", "2JPZ", "NEVER_FOLDED"]


def parse_folding_by_target(filepath: str) -> dict:
    """
    Parse folding report into {target: {folded: [...], misfolded: [...]}}.

    Trajectory names use format iter_X/ratchet_Y.
    """
    data = {t: {"folded": [], "misfolded": []} for t in TARGETS}

    if not os.path.exists(filepath):
        print(f"Warning: folding report not found: {filepath}")
        return data

    current_target = None
    current_state = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TARGET: "):
                current_target = line.split("TARGET: ")[1].strip()
            elif line.startswith("NEVER FOLDED"):
                current_target = "NEVER_FOLDED"
                current_state = "folded"
            elif "MISFOLDED (" in line:
                current_state = "misfolded"
            elif "FOLDED (" in line:
                current_state = "folded"
            elif line.startswith("- iter_"):
                traj_name = line.split("- ")[1].strip()
                if current_target and current_state:
                    data[current_target][current_state].append(traj_name)

    return data


def map_names_to_indices(folding_data: dict, merged_files: list[str]) -> dict:
    """Map iter_X/ratchet_Y names to trajectory indices."""
    name_to_idx = {}
    for i, fpath in enumerate(merged_files):
        parts = fpath.split(os.sep)
        iter_p = next((p for p in reversed(parts) if p.startswith("iter_")), "?")
        ratchet_p = next((p for p in reversed(parts) if p.startswith("ratchet_")), "?")
        name_to_idx[f"{iter_p}/{ratchet_p}"] = i

    processed = {}
    for target, info in folding_data.items():
        processed[target] = {
            "folded_idx": [name_to_idx[n] for n in info["folded"] if n in name_to_idx],
            "misfolded_idx": [name_to_idx[n] for n in info["misfolded"] if n in name_to_idx],
        }
    return processed
