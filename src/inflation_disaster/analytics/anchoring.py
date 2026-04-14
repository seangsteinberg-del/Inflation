"""Conditional anchoring analysis (reproduces Figure 5 of the paper).

Computes counterfactual disaster probabilities under different
initial inflation conditions to measure how well expectations are anchored.

If expectations are fully anchored, the 5y5y disaster probability should
be insensitive to current inflation. If not, higher current inflation
leads to higher 5y5y disaster probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams
from inflation_disaster.adjustments.horizon_adj import (
    compute_forward_distribution,
    extract_disaster_probabilities,
)
from inflation_disaster.adjustments.risk_adj import apply_risk_adjustment


def compute_conditional_probabilities(
    markov_params: MarkovParams,
    initial_states: dict[str, np.ndarray],
    threshold: float = 2.0,
    risk_adj_high: float | None = None,
    risk_adj_low: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute disaster probabilities under different initial conditions.

    Parameters
    ----------
    markov_params : MarkovParams
        Estimated parameters for a given date.
    initial_states : dict
        Keys are labels (e.g., "Baseline", "(2,3]", "(3,4]"),
        values are 8-element state vectors.
    threshold : float
        Disaster threshold d.
    risk_adj_high, risk_adj_low : float, optional
        P/Q factors.

    Returns
    -------
    dict mapping labels to (prob_high_p, prob_low_p) tuples.
    """
    results = {}
    target = settings.target_inflation

    for label, state in initial_states.items():
        fwd_dist = compute_forward_distribution(markov_params, state, T=5, H=5)
        q_high, q_low = extract_disaster_probabilities(fwd_dist, threshold, target)
        p_high, p_low = apply_risk_adjustment(q_high, q_low, risk_adj_high, risk_adj_low)
        results[label] = (p_high, p_low)

    return results


def make_counterfactual_states() -> dict[str, np.ndarray]:
    """Create standard counterfactual initial state vectors.

    Returns states for:
    - "Baseline": current inflation (caller should override)
    - "(2,3]": inflation at 2-3% (well-anchored)
    - "(3,4]": inflation at 3-4% (mildly elevated)
    - "(4,5]": inflation at 4-5% (elevated)
    """
    return {
        "(2,3]": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float),
        "(3,4]": np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=float),
        "(4,5]": np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float),
    }


def anchoring_time_series(
    monthly_params: list[tuple],
    dates: list,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute conditional anchoring analysis over the full sample.

    Parameters
    ----------
    monthly_params : list of (date, MarkovParams, initial_state) tuples
    dates : list of dates
    threshold : float

    Returns
    -------
    DataFrame with columns: date, label, prob_high_p, prob_low_p
    """
    counterfactual_states = make_counterfactual_states()
    records = []

    for dt, params, baseline_state in monthly_params:
        # Add baseline
        all_states = {"Baseline": baseline_state}
        all_states.update(counterfactual_states)

        results = compute_conditional_probabilities(params, all_states, threshold)

        for label, (p_high, p_low) in results.items():
            records.append({
                "date": dt,
                "label": label,
                "prob_high_p_5y5y": p_high,
                "prob_low_p_5y5y": p_low,
            })

    return pd.DataFrame(records)
