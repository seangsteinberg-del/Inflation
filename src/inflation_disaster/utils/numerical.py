"""Numerical utilities: interpolation, differentiation, integration."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_spline_second_derivative(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit cubic spline to (x, y) and return second derivative.

    Parameters
    ----------
    x : array
        Knot points (e.g., strike prices). Must be sorted ascending.
    y : array
        Function values (e.g., call prices).
    x_eval : array, optional
        Points at which to evaluate. Defaults to x.

    Returns
    -------
    x_eval : array
        Evaluation points.
    d2y : array
        Second derivative at evaluation points.
    """
    cs = CubicSpline(x, y, bc_type="natural")
    if x_eval is None:
        x_eval = x
    d2y = cs(x_eval, nu=2)
    return x_eval, d2y


def cubic_spline_first_derivative(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit cubic spline to (x, y) and return first derivative."""
    cs = CubicSpline(x, y, bc_type="natural")
    if x_eval is None:
        x_eval = x
    dy = cs(x_eval, nu=1)
    return x_eval, dy


def integrate_bins(
    grid: np.ndarray,
    density: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Integrate a density over discrete bins using trapezoidal rule.

    Parameters
    ----------
    grid : array
        Fine grid points where density is evaluated.
    density : array
        Density values on the grid.
    bin_edges : array
        Bin edges (length n_bins + 1).

    Returns
    -------
    probabilities : array
        Integrated probability in each bin (length n_bins).
    """
    n_bins = len(bin_edges) - 1
    probabilities = np.zeros(n_bins)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (grid >= lower) & (grid <= upper)
        if mask.sum() >= 2:
            probabilities[i] = np.trapz(density[mask], grid[mask])
        elif mask.sum() == 1:
            # Single point: approximate with neighboring grid spacing
            idx = np.where(mask)[0][0]
            if idx > 0:
                dx = grid[idx] - grid[idx - 1]
            elif idx < len(grid) - 1:
                dx = grid[idx + 1] - grid[idx]
            else:
                dx = 0.0
            probabilities[i] = density[idx] * dx

    # Ensure non-negative and normalize
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum()
    if total > 0:
        probabilities /= total

    return probabilities


def make_fine_grid(
    lower: float = -0.03,
    upper: float = 0.08,
    n_points: int = 1000,
) -> np.ndarray:
    """Create a fine grid for density evaluation.

    Default range covers -3% to 8% annual inflation, which spans
    well beyond the traded strike range of -2% to 6%.
    """
    return np.linspace(lower, upper, n_points)
