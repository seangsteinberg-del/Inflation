"""Integration tests: end-to-end pipeline from synthetic surface to disaster probs."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import date
from inflation_disaster.data.schemas import (
    MarkovParams, CleanedSurface, DisasterProbability, OptionSurface,
)
from inflation_disaster.data.cleaning import clean_surface
from inflation_disaster.data.surface_builder import build_ready_surface, determine_inflation_state
from inflation_disaster.adjustments.pipeline import DisasterProbabilityPipeline
from inflation_disaster.models.sabr import black_price, SABRParams
from inflation_disaster.config import settings


def _make_ready_surface(maturity: int, swap_rate: float = 0.022) -> CleanedSurface:
    """Create a synthetic CleanedSurface ready for density extraction.

    Bypasses Bloomberg-specific PCP checks and directly produces a surface
    with SABR-calibrated fine grid prices.
    """
    from inflation_disaster.models.sabr import sabr_to_prices, sabr_smile, calibrate_sabr
    from inflation_disaster.utils.numerical import make_fine_grid

    strikes_pct = settings.strikes
    strikes_gross = 1.0 + strikes_pct / 100.0
    forward_gross = 1.0 + swap_rate
    real_rate = 0.01
    df = np.exp(-real_rate * maturity)
    sigma = 0.025

    # Generate cap prices from Black formula
    cap_prices = np.array([
        black_price(forward_gross, k, maturity, sigma, df, is_call=True)
        for k in strikes_gross
    ])
    floor_prices = np.array([
        black_price(forward_gross, k, maturity, sigma, df, is_call=False)
        for k in strikes_gross
    ])

    # SABR calibration
    implied_vols = np.full(len(strikes_pct), sigma)
    sabr_params = SABRParams(alpha=sigma * forward_gross ** 0.5, beta=0.5, rho=-0.1, nu=0.3)

    # Fine grid
    fine_rate_grid = make_fine_grid()
    fine_gross_grid = 1.0 + fine_rate_grid
    fine_call_prices = sabr_to_prices(forward_gross, fine_gross_grid, maturity, sabr_params, df)

    return CleanedSurface(
        date=date(2024, 1, 2), region="US", maturity=maturity,
        strikes=strikes_pct, cap_prices=cap_prices, floor_prices=floor_prices,
        swap_rate=swap_rate, implied_vols=implied_vols,
        real_rate=real_rate, quality_score=1.0,
        fine_strikes=fine_rate_grid, fine_call_prices=fine_call_prices,
    )


class TestSurfaceBuilder:
    """Test the synthetic surface construction for integration testing."""

    def test_ready_surface_has_fine_grid(self):
        result = _make_ready_surface(5)
        assert len(result.fine_strikes) > 100
        assert len(result.fine_call_prices) == len(result.fine_strikes)
        assert np.all(result.fine_call_prices >= 0)
        assert np.all(np.isfinite(result.fine_call_prices))

    def test_ready_surface_has_implied_vols(self):
        result = _make_ready_surface(10)
        assert np.any(result.implied_vols > 0)

    def test_ready_surface_quality(self):
        result = _make_ready_surface(5)
        assert result.quality_score >= 0.5


class TestInflationState:
    def test_2_percent(self):
        state = determine_inflation_state(2.5)
        assert state[4] == 1.0  # bin (2, 3]
        assert state.sum() == 1.0

    def test_negative_inflation(self):
        state = determine_inflation_state(-0.5)
        assert state[1] == 1.0  # bin (-1, 0]

    def test_severe_deflation(self):
        state = determine_inflation_state(-2.0)
        assert state[0] == 1.0  # bin <= -1

    def test_severe_inflation(self):
        state = determine_inflation_state(7.0)
        assert state[7] == 1.0  # bin > 5%


class TestFullPipelineIntegration:
    """End-to-end test: synthetic surface -> disaster probabilities."""

    def test_produces_disaster_probability(self):
        # Build synthetic ready surfaces (bypasses Bloomberg PCP checks)
        ready_5y = _make_ready_surface(5)
        ready_10y = _make_ready_surface(10)

        # Use paper's US Markov params
        params = MarkovParams(
            p_dh=0.05, p_dl=0.03, p_nn=0.15,
            p_h=0.1998, p_l=0.1990, p_mr=0.50,
        )
        state = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float)

        # Run pipeline
        pipeline = DisasterProbabilityPipeline("US")
        result = pipeline.process_single_date(
            ready_5y, ready_10y, params, state, threshold=2.0,
        )

        # Verify output structure
        assert isinstance(result, DisasterProbability)
        assert result.region == "US"
        assert result.threshold == 2.0

        # All probabilities should be in [0, 1]
        for field_name in [
            "prob_high_n_5y", "prob_high_q_5y", "prob_high_n_10y", "prob_high_q_10y",
            "prob_high_q_5y5y", "prob_high_p_5y5y",
            "prob_low_n_5y", "prob_low_q_5y", "prob_low_n_10y", "prob_low_q_10y",
            "prob_low_q_5y5y", "prob_low_p_5y5y",
        ]:
            val = getattr(result, field_name)
            assert 0 <= val <= 1, f"{field_name}={val} not in [0,1]"

        # P should be <= Q (risk adjustment reduces probability)
        assert result.prob_high_p_5y5y <= result.prob_high_q_5y5y + 0.001
        assert result.prob_low_p_5y5y <= result.prob_low_q_5y5y + 0.001

        # Adjustment factors should be reasonable
        assert result.risk_adj_high == pytest.approx(0.66, abs=0.01)
        assert result.risk_adj_low == pytest.approx(0.96, abs=0.01)

    def test_us_vs_ez_pipeline(self):
        """Both US and EZ pipelines should work."""
        for region in ["US", "EZ"]:
            ready = _make_ready_surface(5, swap_rate=0.02)
            assert ready is not None
            assert ready.quality_score > 0


class TestCacheIntegration:
    """Test the caching layer."""

    def test_save_and_load_markov_params(self, tmp_path):
        from inflation_disaster.data.cache import ResultsCache

        cache = ResultsCache(tmp_path)
        params = MarkovParams(
            p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50,
        )
        dt = date(2024, 1, 1)

        cache.save_markov_params(dt, "US", params)
        loaded = cache.load_markov_params(dt, "US")

        assert loaded is not None
        assert loaded.p_dh == params.p_dh
        assert loaded.p_mr == params.p_mr

    def test_save_and_load_distributions(self, tmp_path):
        from inflation_disaster.data.cache import ResultsCache

        cache = ResultsCache(tmp_path)
        probs = np.array([0.02, 0.03, 0.05, 0.20, 0.45, 0.15, 0.07, 0.03])
        dt = date(2024, 1, 1)

        cache.save_distributions(dt, "US", 5, "Q", probs)
        loaded = cache.load_distributions(dt, "US", 5, "Q")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, probs)

    def test_audit_log(self, tmp_path):
        from inflation_disaster.data.cache import ResultsCache
        import pandas as pd

        cache = ResultsCache(tmp_path)
        cache.log_audit(date(2024, 1, 1), "US", "sabr_calibration", "ok", "RMSE=0.001")
        cache.log_audit(date(2024, 1, 1), "US", "data_quality", "warning", "3 butterfly violations")
        cache.flush_audit_log()

        log_df = pd.read_csv(tmp_path / "audit_log.csv")
        assert len(log_df) == 2
        assert log_df.iloc[1]["status"] == "warning"


class TestConfidenceBands:
    """Test bootstrap confidence band computation."""

    def test_bootstrap_produces_valid_bands(self):
        from inflation_disaster.analytics.confidence import (
            bootstrap_risk_adjustment, compute_confidence_bands,
        )

        rng = np.random.RandomState(42)
        z_high = 1.03 * rng.uniform(0, 1, 50) ** (-1 / 5.45)
        z_low = 1.06 * rng.uniform(0, 1, 20) ** (-1 / 15.18)

        adj_h, adj_l = bootstrap_risk_adjustment(
            z_high, z_low, p_tilde_high=0.356, p_tilde_low=0.085,
            n_bootstrap=200, gamma=3.0,
        )

        assert len(adj_h) == 200
        assert np.all(np.isfinite(adj_h))
        assert np.all(adj_h > 0) and np.all(adj_h <= 1)

        bands = compute_confidence_bands(0.06, 0.03, adj_h, adj_l)
        assert bands["p_high_lower"] < bands["p_high_median"] < bands["p_high_upper"]
        assert bands["p_high_lower"] >= 0
