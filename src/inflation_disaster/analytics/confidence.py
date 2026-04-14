"""Confidence bands for disaster probabilities via bootstrap and delta method.

Reproduces Figure 7 of the paper: 90% confidence bands around the
5y5y high-inflation disaster probability, accounting for estimation
uncertainty in both the horizon (Markov chain) and risk (Pareto) adjustments.
"""

from __future__ import annotations

import logging

import numpy as np

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams, ParetoFit
from inflation_disaster.models.markov_chain import simulate_forward_distribution
from inflation_disaster.models.epstein_zin import risk_adjustment_factor
from inflation_disaster.adjustments.horizon_adj import extract_disaster_probabilities

log = logging.getLogger("inflation_disaster.analytics.confidence")


def bootstrap_risk_adjustment(
    z_values_high: np.ndarray,
    z_values_low: np.ndarray,
    p_tilde_high: float,
    p_tilde_low: float,
    n_bootstrap: int = 1000,
    gamma: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for P/Q risk adjustment factors.

    Resamples the historical disaster z-values and re-estimates the Pareto
    distribution and P/Q ratio for each bootstrap sample.

    Parameters
    ----------
    z_values_high : array
        Inverse consumption drops during high-inflation disasters.
    z_values_low : array
        Inverse consumption drops during deflation disasters.
    p_tilde_high, p_tilde_low : float
        Conditional probabilities of output disaster given inflation disaster.
    n_bootstrap : int
    gamma : float
        Risk aversion.
    seed : int

    Returns
    -------
    adj_high_samples : array of shape (n_bootstrap,)
    adj_low_samples : array of shape (n_bootstrap,)
    """
    from inflation_disaster.models.pareto import fit_pareto

    rng = np.random.RandomState(seed)
    adj_high_samples = np.zeros(n_bootstrap)
    adj_low_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample high-inflation disasters
        if len(z_values_high) > 0:
            idx_h = rng.choice(len(z_values_high), size=len(z_values_high), replace=True)
            z_h = z_values_high[idx_h]
            alpha_h, z0_h = fit_pareto(z_h)
            fit_h = ParetoFit(
                disaster_type="high", alpha=alpha_h, z_0=z0_h,
                n_observations=len(z_h), p_tilde=p_tilde_high,
                risk_adjustment_factor=0.0,
            )
            adj_high_samples[b] = risk_adjustment_factor(fit_h, gamma)
        else:
            adj_high_samples[b] = settings.risk_adj_high

        # Resample deflation disasters
        if len(z_values_low) > 0:
            idx_l = rng.choice(len(z_values_low), size=len(z_values_low), replace=True)
            z_l = z_values_low[idx_l]
            alpha_l, z0_l = fit_pareto(z_l)
            fit_l = ParetoFit(
                disaster_type="low", alpha=alpha_l, z_0=z0_l,
                n_observations=len(z_l), p_tilde=p_tilde_low,
                risk_adjustment_factor=0.0,
            )
            adj_low_samples[b] = risk_adjustment_factor(fit_l, gamma)
        else:
            adj_low_samples[b] = settings.risk_adj_low

    return adj_high_samples, adj_low_samples


def compute_confidence_bands(
    q_prob_5y5y_high: float,
    q_prob_5y5y_low: float,
    adj_high_samples: np.ndarray,
    adj_low_samples: np.ndarray,
    confidence: float = 0.90,
) -> dict:
    """Compute confidence bands for physical disaster probabilities.

    Parameters
    ----------
    q_prob_5y5y_high : float
        Q-measure 5y5y high-inflation disaster probability for this date.
    q_prob_5y5y_low : float
        Q-measure 5y5y deflation disaster probability.
    adj_high_samples : array
        Bootstrap samples of P/Q ratio for high inflation.
    adj_low_samples : array
        Bootstrap samples of P/Q ratio for deflation.
    confidence : float
        Confidence level (default 0.90 for 90% band).

    Returns
    -------
    dict with keys: p_high_lower, p_high_upper, p_high_median,
                    p_low_lower, p_low_upper, p_low_median
    """
    alpha = (1 - confidence) / 2

    p_high_samples = q_prob_5y5y_high * adj_high_samples
    p_low_samples = q_prob_5y5y_low * adj_low_samples

    return {
        "p_high_lower": float(np.percentile(p_high_samples, 100 * alpha)),
        "p_high_upper": float(np.percentile(p_high_samples, 100 * (1 - alpha))),
        "p_high_median": float(np.median(p_high_samples)),
        "p_low_lower": float(np.percentile(p_low_samples, 100 * alpha)),
        "p_low_upper": float(np.percentile(p_low_samples, 100 * (1 - alpha))),
        "p_low_median": float(np.median(p_low_samples)),
    }
