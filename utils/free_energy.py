"""
Free energy landscape computation on 2D latent coordinates.

F = -kT * ln(P) with Gaussian smoothing of the 2D histogram.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_free_energy(latent_coords, bins=100, sigma=2.0, kT=1.0, xlim=None, ylim=None):
    """
    Compute free energy landscape F = -kT * ln(P) on 2D latent space.

    Args:
        latent_coords: (N, 2) array of latent coordinates.
        bins: Number of histogram bins per axis.
        sigma: Gaussian smoothing sigma.
        kT: Thermal energy scale.
        xlim, ylim: Optional fixed ranges for histogram.

    Returns:
        free_energy, x_centers, y_centers
    """
    range_ = None
    if xlim is not None and ylim is not None:
        range_ = [list(xlim), list(ylim)]

    hist, xedges, yedges = np.histogram2d(
        latent_coords[:, 0], latent_coords[:, 1], bins=bins, range=range_
    )

    hist_smooth = gaussian_filter(hist.astype(float), sigma=sigma)
    hist_smooth[hist_smooth == 0] = np.nan

    prob = hist_smooth / np.nansum(hist_smooth)
    free_energy = -kT * np.log(prob)
    free_energy -= np.nanmin(free_energy)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return free_energy, x_centers, y_centers


def calculate_explored_area(coords, x_range, y_range, bins=100) -> float:
    """Total latent-space area covered by at least one frame."""
    if len(coords) == 0:
        return 0.0
    hist, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=bins, range=[x_range, y_range]
    )
    cell_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
    return float(np.count_nonzero(hist > 0) * cell_area)


def compute_global_limits(latent_coords, margin_frac=0.05):
    """Return (x_range, y_range) with margin around latent coords."""
    x_min, x_max = latent_coords[:, 0].min(), latent_coords[:, 0].max()
    y_min, y_max = latent_coords[:, 1].min(), latent_coords[:, 1].max()
    margin_x = (x_max - x_min) * margin_frac
    margin_y = (y_max - y_min) * margin_frac
    return [x_min - margin_x, x_max + margin_x], [y_min - margin_y, y_max + margin_y]
