"""Second adjustment: Horizon (spot -> 5y5y forward probability).

Traded options give probabilities for cumulative inflation over 5y or 10y from now.
Policymakers care about the 5y5y forward: inflation between years 5 and 10.

The horizon adjustment uses the estimated Markov chain model to decompose
the 10y distribution into 5y + 5y5y components.

Key insight: if inflation is sluggish, a 10y disaster probability understates
the 5y5y forward probability for high inflation (because current high inflation
contributes to the 10y cumulative but may mean-revert before year 5).

Paper median adjustment factors (Q_5y5y / Q_10y):
  US high inflation: 0.38
  EZ high inflation: 0.93
  US deflation: 1.41
  EZ deflation: 2.26
"""

from __future__ import annotations

import numpy as np

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams
from inflation_disaster.models.markov_chain import simulate_forward_distribution


def compute_forward_distribution(
    params: MarkovParams,
    initial_state: np.ndarray,
    T: int = 5,
    H: int = 5,
    n_paths: int | None = None,
) -> np.ndarray:
    """Compute the 5y5y forward inflation distribution.

    Uses the Markov chain to simulate:
    1. State distribution at year T
    2. Distribution of average inflation over [T, T+H]

    Parameters
    ----------
    params : MarkovParams
        Estimated Markov chain parameters.
    initial_state : array of shape (8,)
        Current inflation state distribution.
    T : int
        Forward start (years).
    H : int
        Forward horizon (years).
    n_paths : int
        Monte Carlo paths.

    Returns
    -------
    forward_probs : array of shape (8,)
        Probability of average inflation in [T, T+H] falling in each bin.
    """
    return simulate_forward_distribution(
        params, initial_state, T=T, H=H, n_paths=n_paths,
    )


def extract_disaster_probabilities(
    bin_probs: np.ndarray,
    threshold: float = 2.0,
    target: float = 2.0,
) -> tuple[float, float]:
    """Extract high-inflation and deflation disaster probabilities from bins.

    Parameters
    ----------
    bin_probs : array of shape (8,)
        Bin probabilities.
    threshold : float
        Disaster threshold in percentage points (d).
    target : float
        Inflation target in percent.

    Returns
    -------
    prob_high : float
        Prob[avg inflation > target + threshold]
    prob_low : float
        Prob[avg inflation < target - threshold]
    """
    # Bins: {<=-1, (-1,0], (0,1], (1,2], (2,3], (3,4], (4,5], >5}
    # Disaster if avg inflation > target + threshold
    # For d=2%: high inflation > 4%, deflation < 0%
    # For d=3%: high inflation > 5%, deflation < -1%

    high_cutoff = target + threshold  # e.g., 4%
    low_cutoff = target - threshold   # e.g., 0%

    bin_edges = list(settings.bin_edges)  # [-inf, -1, 0, 1, 2, 3, 4, 5, inf]

    prob_high = 0.0
    prob_low = 0.0

    for i in range(len(bin_probs)):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # High inflation: bin is fully above cutoff
        if lower >= high_cutoff:
            prob_high += bin_probs[i]
        # Partial overlap for the bin straddling the cutoff
        elif upper > high_cutoff and lower < high_cutoff:
            # Linear interpolation within bin
            if np.isfinite(upper) and np.isfinite(lower):
                frac = (upper - high_cutoff) / (upper - lower)
            else:
                frac = 1.0  # infinite bin straddling cutoff: assume all above
            prob_high += bin_probs[i] * frac

        # Deflation: bin is fully below cutoff
        if upper <= low_cutoff:
            prob_low += bin_probs[i]
        elif lower < low_cutoff and upper > low_cutoff:
            if np.isfinite(upper) and np.isfinite(lower):
                frac = (low_cutoff - lower) / (upper - lower)
            else:
                frac = 1.0  # infinite bin straddling cutoff: assume all below
            prob_low += bin_probs[i] * frac

    return float(prob_high), float(prob_low)


def compute_horizon_adj_factor(
    q_10y_disaster: float,
    q_5y5y_disaster: float,
) -> float:
    """Compute the horizon adjustment factor Q_5y5y / Q_10y."""
    if q_10y_disaster > 1e-10:
        return q_5y5y_disaster / q_10y_disaster
    return 1.0
