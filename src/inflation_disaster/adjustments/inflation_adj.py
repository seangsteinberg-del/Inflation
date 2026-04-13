"""First adjustment: Inflation (N -> Q probability conversion).

Conventional option-implied probabilities (N-probabilities) do not account for
the fact that nominal payoffs have different real values across inflation states.
A $1 payoff in a high-inflation state is worth less in real terms.

The adjustment: q(pi) = n(pi) * exp((pi - pi^e) * H)

where pi^e is expected inflation (breakeven from swaps) and H is the horizon.

For high inflation: Q > N (understated by conventional methods)
For deflation: Q < N (overstated by conventional methods)

Paper median adjustment factors:
  5y high inflation: 1.09
  10y high inflation: 1.24
  5y deflation: 0.84
  10y deflation: 0.69
"""

from __future__ import annotations

import numpy as np

from inflation_disaster.config import settings


def inflation_adjustment_factor_continuous(
    inflation_rate: float,
    expected_inflation: float,
    horizon: int,
) -> float:
    """Point adjustment factor for a specific inflation rate.

    factor = exp((pi - pi^e) * H)
    """
    return np.exp((inflation_rate - expected_inflation) * horizon)


def adjust_bin_probabilities(
    n_probs: np.ndarray,
    expected_inflation: float,
    horizon: int,
    bin_midpoints: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Convert N-measure bin probabilities to Q-measure.

    Parameters
    ----------
    n_probs : array of shape (8,)
        N-measure bin probabilities.
    expected_inflation : float
        Expected inflation (from swap rate), in decimal (e.g. 0.02).
    horizon : int
        Maturity in years (5 or 10).
    bin_midpoints : array, optional
        Midpoint inflation rate for each bin. Default from config.

    Returns
    -------
    q_probs : array of shape (8,)
        Q-measure bin probabilities (renormalized to sum to 1).
    aggregate_adj_factor : float
        Ratio of Q to N total disaster probability (for reporting).
    """
    if bin_midpoints is None:
        bin_midpoints = settings.bin_midpoints / 100.0  # percent to decimal

    # Apply adjustment factor to each bin
    factors = np.array([
        inflation_adjustment_factor_continuous(pi, expected_inflation, horizon)
        for pi in bin_midpoints
    ])

    q_unnormalized = n_probs * factors

    # Renormalize to sum to 1
    total = q_unnormalized.sum()
    q_probs = q_unnormalized / total if total > 0 else q_unnormalized

    return q_probs, float(total / n_probs.sum()) if n_probs.sum() > 0 else 1.0


def compute_inflation_adj_factor(
    n_prob_disaster: float,
    q_prob_disaster: float,
) -> float:
    """Compute the inflation adjustment factor Q/N for a disaster tail."""
    if n_prob_disaster > 1e-10:
        return q_prob_disaster / n_prob_disaster
    return 1.0
