"""Pareto distribution fitting for consumption disasters during inflation disasters.

Fits the distribution of inverse consumption drops z = 1/(1 - beta*d) during
joint inflation-output disasters, following Gabaix (2012) and Barro & Liao (2021).

F(z) = 1 - (z_0/z)^alpha, for z >= z_0 > 1

Paper estimates:
  High-inflation disasters: alpha=5.45, z_0=1.03
  Deflation disasters: alpha=15.18, z_0=1.06
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pareto as pareto_dist

from inflation_disaster.data.schemas import ParetoFit

log = logging.getLogger("inflation_disaster.models.pareto")


def fit_pareto(
    z_values: np.ndarray,
    z_min: float | None = None,
) -> tuple[float, float]:
    """Fit Pareto distribution to consumption drop data via MLE.

    Parameters
    ----------
    z_values : array
        Inverse consumption drops z = 1/(1+g) where g is GDP growth during disaster.
        z > 1 means GDP fell.
    z_min : float, optional
        Minimum z (location parameter). If None, estimated from data.

    Returns
    -------
    alpha : float
        Tail exponent.
    z_0 : float
        Location parameter (minimum z).
    """
    z = z_values[z_values > 1.0]  # only disaster episodes where GDP fell

    if len(z) < 3:
        log.warning(f"Too few disaster observations ({len(z)}), using defaults")
        return 6.0, 1.03

    if z_min is None:
        z_0 = z.min()
    else:
        z_0 = z_min

    # MLE for Pareto: alpha_hat = n / sum(log(z/z_0))
    n = len(z)
    log_sum = np.sum(np.log(z / z_0))
    if log_sum < 1e-12:
        log.warning("All disaster z-values nearly identical, using default alpha=6.0")
        return 6.0, z_0
    alpha = n / log_sum

    log.info(f"Pareto fit: alpha={alpha:.2f}, z_0={z_0:.3f}, n={n}")
    return alpha, z_0


def fit_pareto_separate(
    high_disasters, low_disasters,
) -> tuple[ParetoFit, ParetoFit]:
    """Fit Pareto distributions separately for high-inflation and deflation disasters.

    Parameters
    ----------
    high_disasters : DataFrame
        From identify_inflation_disasters(), with columns: z, has_output_disaster.
    low_disasters : DataFrame
        Same structure.

    Returns
    -------
    high_fit : ParetoFit
    low_fit : ParetoFit
    """
    # High-inflation disasters
    high_joint = high_disasters[high_disasters["has_output_disaster"]]
    if len(high_joint) > 0:
        z_high = high_joint["z"].values
        alpha_h, z0_h = fit_pareto(z_high)
    else:
        alpha_h, z0_h = 5.45, 1.03  # paper defaults

    p_tilde_h = (
        len(high_joint) / len(high_disasters)
        if len(high_disasters) > 0
        else 0.356
    )

    high_fit = ParetoFit(
        disaster_type="high",
        alpha=alpha_h,
        z_0=z0_h,
        n_observations=len(high_joint),
        p_tilde=p_tilde_h,
        risk_adjustment_factor=0.0,  # computed below
    )

    # Deflation disasters
    low_joint = low_disasters[low_disasters["has_output_disaster"]]
    if len(low_joint) > 0:
        z_low = low_joint["z"].values
        alpha_l, z0_l = fit_pareto(z_low)
    else:
        alpha_l, z0_l = 15.18, 1.06  # paper defaults

    p_tilde_l = (
        len(low_joint) / len(low_disasters)
        if len(low_disasters) > 0
        else 0.085
    )

    low_fit = ParetoFit(
        disaster_type="low",
        alpha=alpha_l,
        z_0=z0_l,
        n_observations=len(low_joint),
        p_tilde=p_tilde_l,
        risk_adjustment_factor=0.0,
    )

    return high_fit, low_fit
