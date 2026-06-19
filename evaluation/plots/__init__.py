"""
Plot registry — dispatch evaluation plots from YAML flags.

Register new plots by adding entries to PLOT_REGISTRY.
YAML: evaluation.plots.*
"""

from __future__ import annotations

from typing import Callable

from evaluation.plots import grids, hamming, overview, uncertainty

PlotFn = Callable[[dict], None]

PLOT_REGISTRY: dict[str, dict] = {
    "overview_4panel": {
        "single": overview.plot_latent_space_single,
        "ensemble": overview.plot_ensemble_overview,
    },
    "uncertainty": {
        "ensemble": uncertainty.plot_ensemble_uncertainty,
    },
    "hamming": {
        "single": hamming.plot_hamming,
        "ensemble": hamming.plot_hamming,
    },
    "trajectory_grid_5x5": {
        "single": grids.plot_trajectory_grid_5x5,
        "ensemble": grids.plot_trajectory_grid_5x5,
    },
    "landscape_grid_5x10": {
        "ensemble": grids.plot_landscape_grid_5x10,
    },
    "confidence": {
        "single": uncertainty.plot_confidence,
    },
    "confidence_grid": {
        "single": uncertainty.plot_confidence_grid,
    },
    "folded_only_4panel": {
        "ensemble": overview.plot_folded_only_overview,
    },
    "landscape_comparison": {
        "ensemble": grids.plot_landscape_comparison_both,
    },
}


def run_plots(ctx: dict, plot_config: dict) -> None:
    """
    Run all enabled plots for the current evaluation mode.

    Args:
        ctx: Evaluation context (latent_coords, labels, refs, paths, mode, ...).
        plot_config: evaluation.plots section from YAML.
    """
    mode = ctx.get("mode", "single")

    for plot_key, enabled in plot_config.items():
        if not enabled:
            continue
        if plot_key not in PLOT_REGISTRY:
            print(f"Warning: unknown plot key {plot_key!r}, skipping.")
            continue

        handlers = PLOT_REGISTRY[plot_key]
        fn = handlers.get(mode)
        if fn is None:
            print(f"Skipping {plot_key}: not available for mode={mode}")
            continue

        print(f"\nGenerating plot: {plot_key}...")
        fn(ctx)
