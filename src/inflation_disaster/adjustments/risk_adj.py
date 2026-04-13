"""Third adjustment: Risk (Q -> P physical probability conversion).

Risk-neutral probabilities overstate the physical probability of inflation
disasters because marginal utility is elevated during disasters (investors
pay a premium for insurance against these states).

P = Q * (P/Q ratio)

The P/Q ratio is estimated from historical data on joint inflation-output disasters
using Epstein-Zin preferences. It differs for high inflation vs deflation because:
- High inflation more often coincides with deep recessions (P/Q = 0.66)
- Deflation is less systematically associated with output drops (P/Q = 0.96)

Paper estimates: high=0.66, deflation=0.96, pooled=0.82
"""

from __future__ import annotations

from inflation_disaster.config import settings
from inflation_disaster.models.epstein_zin import default_risk_adjustments


def apply_risk_adjustment(
    q_prob_high: float,
    q_prob_low: float,
    adj_high: float | None = None,
    adj_low: float | None = None,
) -> tuple[float, float]:
    """Convert Q-measure disaster probabilities to P-measure.

    Parameters
    ----------
    q_prob_high : float
        Q-measure probability of high-inflation disaster.
    q_prob_low : float
        Q-measure probability of deflation disaster.
    adj_high : float, optional
        P/Q ratio for high inflation. Default: 0.66 from paper.
    adj_low : float, optional
        P/Q ratio for deflation. Default: 0.96 from paper.

    Returns
    -------
    p_prob_high : float
        Physical probability of high-inflation disaster.
    p_prob_low : float
        Physical probability of deflation disaster.
    """
    if adj_high is None or adj_low is None:
        default_high, default_low = default_risk_adjustments()
        if adj_high is None:
            adj_high = default_high
        if adj_low is None:
            adj_low = default_low

    p_prob_high = q_prob_high * adj_high
    p_prob_low = q_prob_low * adj_low

    return float(p_prob_high), float(p_prob_low)
