"""Epstein-Zin risk adjustment for inflation disaster probabilities.

Computes the Q-to-P ratio (risk adjustment factor) using Epstein-Zin preferences
and the Pareto distribution of consumption drops during inflation disasters.

Following Gabaix (2012) and Barro & Liao (2021), the key insight is that
inflation disasters that coincide with output disasters have elevated marginal
utility, so risk-neutral probabilities overstate the physical probability.

The adjustment factor = P/Q = 1 / [(m_tilde - 1) * p_tilde + 1]

where:
  m_tilde = E[z^gamma | joint disaster] / E[1] -- ratio of marginal utility
            during a joint inflation+output disaster vs. normal times
  p_tilde = P(output disaster | inflation disaster) -- conditional probability
  gamma = RRA (relative risk aversion)

Paper results:
  High inflation: P/Q = 0.66
  Deflation: P/Q = 0.96
"""

from __future__ import annotations

import logging

import numpy as np

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import ParetoFit

log = logging.getLogger("inflation_disaster.models.epstein_zin")


def expected_marginal_utility_ratio(
    alpha: float,
    z_0: float,
    gamma: float,
) -> float:
    """Compute E[z^gamma] for Pareto-distributed z.

    For z ~ Pareto(alpha, z_0):
        E[z^gamma] = alpha * z_0^gamma / (alpha - gamma)  if alpha > gamma
        = infinity                                          if alpha <= gamma

    The ratio m_tilde = E[z^gamma] accounts for the elevated marginal utility
    during consumption disasters.

    Parameters
    ----------
    alpha : float
        Pareto tail exponent.
    z_0 : float
        Minimum z (location parameter).
    gamma : float
        Risk aversion (RRA).

    Returns
    -------
    m_tilde : float
        Expected marginal utility ratio.
    """
    if alpha <= gamma:
        log.warning(
            f"alpha={alpha} <= gamma={gamma}, infinite marginal utility. "
            "Using truncated estimate."
        )
        # Truncate at a large but finite z
        z_max = z_0 * 10
        n_points = 10000
        z_grid = np.linspace(z_0 + 1e-12, z_max, n_points)
        pdf = alpha * z_0**alpha / z_grid ** (alpha + 1)
        # Normalize for truncation: P(z <= z_max) = 1 - (z_0/z_max)^alpha
        cdf_at_max = 1.0 - (z_0 / z_max) ** alpha
        pdf_truncated = pdf / max(cdf_at_max, 1e-12)
        integrand = z_grid**gamma * pdf_truncated
        return float(np.trapz(integrand, z_grid))

    m_tilde = alpha * z_0**gamma / (alpha - gamma)
    return m_tilde


def risk_adjustment_factor(
    pareto_fit: ParetoFit,
    gamma: float | None = None,
) -> float:
    """Compute the P/Q risk adjustment factor.

    P/Q = 1 / [(m_tilde - 1) * p_tilde + 1]

    This is equation (2) from the paper, rearranged.

    Parameters
    ----------
    pareto_fit : ParetoFit
        Fitted Pareto parameters and conditional disaster probability.
    gamma : float, optional
        Risk aversion coefficient. Default from config (3.0).

    Returns
    -------
    factor : float
        P/Q ratio. Always in (0, 1] since m_tilde >= 1.
    """
    if gamma is None:
        gamma = settings.rra

    m_tilde = expected_marginal_utility_ratio(
        pareto_fit.alpha, pareto_fit.z_0, gamma
    )
    p_tilde = pareto_fit.p_tilde

    # Q = [(m_tilde - 1) * p_tilde + 1] * P
    # => P/Q = 1 / [(m_tilde - 1) * p_tilde + 1]
    denominator = (m_tilde - 1.0) * p_tilde + 1.0
    factor = 1.0 / denominator

    log.info(
        f"Risk adjustment ({pareto_fit.disaster_type}): "
        f"m_tilde={m_tilde:.3f}, p_tilde={p_tilde:.3f}, P/Q={factor:.3f}"
    )

    return factor


def compute_risk_adjustments(
    high_fit: ParetoFit,
    low_fit: ParetoFit,
    gamma: float | None = None,
) -> tuple[float, float]:
    """Compute risk adjustment factors for both tails.

    Returns
    -------
    adj_high : float
        P/Q factor for high-inflation disasters.
    adj_low : float
        P/Q factor for deflation disasters.
    """
    adj_high = risk_adjustment_factor(high_fit, gamma)
    adj_low = risk_adjustment_factor(low_fit, gamma)

    # Update the ParetoFit objects
    high_fit.risk_adjustment_factor = adj_high
    low_fit.risk_adjustment_factor = adj_low

    return adj_high, adj_low


def default_risk_adjustments() -> tuple[float, float]:
    """Return the paper's calibrated risk adjustment factors.

    Use these when not re-estimating from historical data.
    """
    return settings.risk_adj_high, settings.risk_adj_low
