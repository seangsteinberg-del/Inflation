"""Full pipeline orchestrating all three adjustments.

Input: raw Bloomberg option prices for a single date
Output: disaster probabilities under N, Q, and P measures

Pipeline steps:
1. Clean option surface
2. Calibrate SABR model
3. Extract N-density via Breeden-Litzenberger
4. Apply inflation adjustment (N -> Q)
5. Compute bin probabilities
6. Estimate Markov chain via GMM (or use pre-estimated params)
7. Compute 5y5y forward distribution (horizon adjustment)
8. Apply risk adjustment (Q -> P)
9. Package results
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Literal

import numpy as np

from inflation_disaster.adjustments.horizon_adj import (
    compute_forward_distribution,
    compute_horizon_adj_factor,
    extract_disaster_probabilities,
)
from inflation_disaster.adjustments.inflation_adj import (
    adjust_bin_probabilities,
    compute_inflation_adj_factor,
)
from inflation_disaster.adjustments.risk_adj import apply_risk_adjustment
from inflation_disaster.config import settings
from inflation_disaster.data.schemas import (
    CleanedSurface,
    DisasterProbability,
    MarkovParams,
)
from inflation_disaster.models.breeden_litzenberger import (
    extract_full_distributions,
)
from inflation_disaster.models.markov_chain import (
    simulate_cumulative_distribution,
)
from inflation_disaster.models.sabr import (
    calibrate_sabr,
    sabr_to_prices,
)
from inflation_disaster.utils.numerical import make_fine_grid

log = logging.getLogger("inflation_disaster.adjustments.pipeline")


class DisasterProbabilityPipeline:
    """End-to-end pipeline for computing inflation disaster probabilities.

    Parameters
    ----------
    region : "US" or "EZ"
    risk_adj_high : float, optional
        P/Q factor for high inflation disasters. Default from paper (0.66).
    risk_adj_low : float, optional
        P/Q factor for deflation disasters. Default from paper (0.96).
    """

    def __init__(
        self,
        region: Literal["US", "EZ"],
        risk_adj_high: float | None = None,
        risk_adj_low: float | None = None,
    ):
        self.region = region
        self.risk_adj_high = risk_adj_high or settings.risk_adj_high
        self.risk_adj_low = risk_adj_low or settings.risk_adj_low

    def process_single_date(
        self,
        surface_5y: CleanedSurface,
        surface_10y: CleanedSurface,
        markov_params: MarkovParams,
        current_inflation_state: np.ndarray,
        threshold: float = 2.0,
    ) -> DisasterProbability:
        """Run full pipeline for one date.

        Parameters
        ----------
        surface_5y : CleanedSurface
            5-year option surface (cleaned + SABR-smoothed).
        surface_10y : CleanedSurface
            10-year option surface.
        markov_params : MarkovParams
            Pre-estimated Markov chain parameters for this date.
        current_inflation_state : array of shape (8,)
            Current state distribution (e.g., [0,0,0,1,0,0,0,0] if inflation is 1-2%).
        threshold : float
            Disaster threshold d in percent (2.0 or 3.0).

        Returns
        -------
        DisasterProbability with all measures and adjustment factors.
        """
        dt = surface_5y.date
        expected_infl_5y = surface_5y.swap_rate
        expected_infl_10y = surface_10y.swap_rate

        # --- Step 1: Extract N and Q distributions ---
        fine_grid = make_fine_grid()

        # 5-year
        if len(surface_5y.fine_call_prices) > 0:
            fine_5y = surface_5y.fine_strikes
            prices_5y = surface_5y.fine_call_prices
        else:
            sabr_params_5y = calibrate_sabr(
                forward=1.0 + expected_infl_5y,
                strikes=surface_5y.strikes / 100.0 + 1.0,
                market_vols=surface_5y.implied_vols,
                T=5,
            )
            fine_5y = fine_grid
            prices_5y = sabr_to_prices(
                forward=1.0 + expected_infl_5y,
                fine_strikes=fine_5y + 1.0,
                T=5,
                params=sabr_params_5y,
                discount_factor=np.exp(-surface_5y.real_rate * 5),
            )

        n_dist_5y, q_dist_5y = extract_full_distributions(
            fine_5y, prices_5y,
            nominal_rate=expected_infl_5y + surface_5y.real_rate,
            real_rate=surface_5y.real_rate,
            expected_inflation=expected_infl_5y,
            maturity=5,
            date=dt,
            region=self.region,
        )

        # 10-year (same process)
        if len(surface_10y.fine_call_prices) > 0:
            fine_10y = surface_10y.fine_strikes
            prices_10y = surface_10y.fine_call_prices
        else:
            sabr_params_10y = calibrate_sabr(
                forward=1.0 + expected_infl_10y,
                strikes=surface_10y.strikes / 100.0 + 1.0,
                market_vols=surface_10y.implied_vols,
                T=10,
            )
            fine_10y = fine_grid
            prices_10y = sabr_to_prices(
                forward=1.0 + expected_infl_10y,
                fine_strikes=fine_10y + 1.0,
                T=10,
                params=sabr_params_10y,
                discount_factor=np.exp(-surface_10y.real_rate * 10),
            )

        n_dist_10y, q_dist_10y = extract_full_distributions(
            fine_10y, prices_10y,
            nominal_rate=expected_infl_10y + surface_10y.real_rate,
            real_rate=surface_10y.real_rate,
            expected_inflation=expected_infl_10y,
            maturity=10,
            date=dt,
            region=self.region,
        )

        # --- Step 2: Extract disaster probabilities at each stage ---
        target = settings.target_inflation

        # N-measure disaster probs
        prob_high_n_5y, prob_low_n_5y = extract_disaster_probabilities(
            n_dist_5y.bin_probabilities, threshold, target
        )
        prob_high_n_10y, prob_low_n_10y = extract_disaster_probabilities(
            n_dist_10y.bin_probabilities, threshold, target
        )

        # Q-measure disaster probs
        prob_high_q_5y, prob_low_q_5y = extract_disaster_probabilities(
            q_dist_5y.bin_probabilities, threshold, target
        )
        prob_high_q_10y, prob_low_q_10y = extract_disaster_probabilities(
            q_dist_10y.bin_probabilities, threshold, target
        )

        # --- Step 3: Horizon adjustment (5y5y forward) ---
        forward_dist = compute_forward_distribution(
            markov_params, current_inflation_state, T=5, H=5,
        )
        prob_high_q_5y5y, prob_low_q_5y5y = extract_disaster_probabilities(
            forward_dist, threshold, target
        )

        # --- Step 4: Risk adjustment (Q -> P) ---
        prob_high_p_5y5y, prob_low_p_5y5y = apply_risk_adjustment(
            prob_high_q_5y5y, prob_low_q_5y5y,
            self.risk_adj_high, self.risk_adj_low,
        )

        # --- Step 5: Compute adjustment factors ---
        infl_adj_5y = compute_inflation_adj_factor(prob_high_n_5y, prob_high_q_5y)
        infl_adj_10y = compute_inflation_adj_factor(prob_high_n_10y, prob_high_q_10y)
        horizon_adj_high = compute_horizon_adj_factor(prob_high_q_10y, prob_high_q_5y5y)
        horizon_adj_low = compute_horizon_adj_factor(prob_low_q_10y, prob_low_q_5y5y)

        return DisasterProbability(
            date=dt,
            region=self.region,
            threshold=threshold,
            prob_high_n_5y=prob_high_n_5y,
            prob_high_q_5y=prob_high_q_5y,
            prob_high_n_10y=prob_high_n_10y,
            prob_high_q_10y=prob_high_q_10y,
            prob_high_q_5y5y=prob_high_q_5y5y,
            prob_high_p_5y5y=prob_high_p_5y5y,
            prob_low_n_5y=prob_low_n_5y,
            prob_low_q_5y=prob_low_q_5y,
            prob_low_n_10y=prob_low_n_10y,
            prob_low_q_10y=prob_low_q_10y,
            prob_low_q_5y5y=prob_low_q_5y5y,
            prob_low_p_5y5y=prob_low_p_5y5y,
            inflation_adj_5y=infl_adj_5y,
            inflation_adj_10y=infl_adj_10y,
            horizon_adj_high=horizon_adj_high,
            horizon_adj_low=horizon_adj_low,
            risk_adj_high=self.risk_adj_high,
            risk_adj_low=self.risk_adj_low,
        )
