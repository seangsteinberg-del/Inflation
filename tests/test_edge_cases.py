"""Edge case tests: crash prevention, NaN handling, extreme inputs.

These tests verify the code doesn't crash on degenerate inputs that
can arise from bad Bloomberg data, empty markets, or unusual parameter values.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import MarkovParams, SABRParams, InflationDistribution
from inflation_disaster.models.markov_chain import (
    build_transition_matrix, simulate_cumulative_distribution,
    simulate_forward_distribution, stationary_distribution,
)
from inflation_disaster.models.sabr import (
    sabr_implied_vol, calibrate_sabr, black_price, sabr_to_prices,
)
from inflation_disaster.models.breeden_litzenberger import (
    extract_n_distribution, n_to_q_adjustment, density_to_bin_probabilities,
)
from inflation_disaster.models.pareto import fit_pareto
from inflation_disaster.models.epstein_zin import expected_marginal_utility_ratio
from inflation_disaster.adjustments.inflation_adj import (
    inflation_adjustment_factor_continuous, adjust_bin_probabilities,
)
from inflation_disaster.adjustments.horizon_adj import extract_disaster_probabilities
from inflation_disaster.adjustments.risk_adj import apply_risk_adjustment
from inflation_disaster.utils.numerical import integrate_bins, make_fine_grid


class TestSABREdgeCases:
    """SABR model edge cases."""

    def test_rho_near_positive_one(self):
        """rho = 0.99 should not produce NaN."""
        p = SABRParams(alpha=0.03, beta=0.5, rho=0.99, nu=0.4)
        vol = sabr_implied_vol(1.02, 1.04, 10, p)
        assert np.isfinite(vol) and vol > 0

    def test_rho_near_negative_one(self):
        """rho = -0.99 should not produce NaN."""
        p = SABRParams(alpha=0.03, beta=0.5, rho=-0.99, nu=0.4)
        vol = sabr_implied_vol(1.02, 1.04, 10, p)
        assert np.isfinite(vol) and vol > 0

    def test_very_small_alpha(self):
        p = SABRParams(alpha=1e-5, beta=0.5, rho=0.0, nu=0.3)
        vol = sabr_implied_vol(1.02, 1.04, 10, p)
        assert np.isfinite(vol)

    def test_very_large_nu(self):
        p = SABRParams(alpha=0.03, beta=0.5, rho=0.0, nu=2.9)
        vol = sabr_implied_vol(1.02, 1.04, 10, p)
        assert np.isfinite(vol)

    def test_strike_equals_forward(self):
        p = SABRParams(alpha=0.03, beta=0.5, rho=-0.2, nu=0.4)
        vol = sabr_implied_vol(1.02, 1.02, 10, p)
        assert np.isfinite(vol) and vol > 0

    def test_black_price_zero_strike(self):
        """Strike = 0 should return intrinsic value."""
        price = black_price(1.02, 0.0, 10, 0.03, 0.9, is_call=True)
        assert np.isfinite(price)
        assert price == pytest.approx(1.02 * 0.9, abs=0.01)

    def test_black_price_negative_strike(self):
        """Negative strike (deflation floor) should not crash."""
        price = black_price(1.02, -0.05, 10, 0.03, 0.9, is_call=True)
        assert np.isfinite(price)

    def test_black_price_zero_vol(self):
        """Zero vol should return intrinsic."""
        price = black_price(1.05, 1.02, 10, 0.0, 0.9, is_call=True)
        assert price == pytest.approx(0.9 * (1.05 - 1.02), abs=1e-8)

    def test_calibration_all_zero_vols(self):
        """Should not crash or return NaN with all-zero vols."""
        result = calibrate_sabr(1.02, np.linspace(0.96, 1.08, 13), np.zeros(13), 5)
        assert result.alpha > 0 and np.isfinite(result.alpha)

    def test_calibration_single_valid_vol(self):
        """Single valid vol point should use defaults."""
        vols = np.zeros(13)
        vols[6] = 0.03  # only ATM
        result = calibrate_sabr(1.02, np.linspace(0.96, 1.08, 13), vols, 5)
        assert result.alpha > 0


class TestMarkovEdgeCases:
    """Markov chain edge cases."""

    def test_all_mass_in_disaster(self):
        """Initial state all in high-inflation disaster."""
        params = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        state = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])
        dist = simulate_cumulative_distribution(params, state, 5, n_paths=50000)
        assert abs(dist.sum() - 1.0) < 0.01
        assert np.all(np.isfinite(dist))

    def test_all_mass_in_deflation(self):
        """Initial state all in deflation disaster."""
        params = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        state = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
        dist = simulate_cumulative_distribution(params, state, 5, n_paths=50000)
        assert abs(dist.sum() - 1.0) < 0.01

    def test_uniform_initial_state(self):
        """Uniform initial distribution."""
        params = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        state = np.ones(8) / 8
        dist = simulate_cumulative_distribution(params, state, 5, n_paths=50000)
        assert abs(dist.sum() - 1.0) < 0.01

    def test_very_small_params(self):
        """Tiny disaster probabilities should work."""
        params = MarkovParams(p_dh=1e-6, p_dl=1e-6, p_nn=0.01, p_h=0.19, p_l=0.19, p_mr=0.50)
        P = build_transition_matrix(params)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_stationary_always_nonneg(self):
        """Stationary distribution should never have negative entries."""
        for _ in range(20):
            rng = np.random.RandomState(42 + _)
            try:
                params = MarkovParams(
                    p_dh=rng.uniform(0.001, 0.1),
                    p_dl=rng.uniform(0.001, 0.1),
                    p_nn=rng.uniform(0.01, 0.2),
                    p_h=rng.uniform(0.05, 0.19),
                    p_l=rng.uniform(0.05, 0.19),
                    p_mr=rng.uniform(0.2, 0.6),
                )
            except ValueError:
                continue  # infeasible params
            P = build_transition_matrix(params)
            pi = stationary_distribution(P)
            assert np.all(pi >= 0), f"Negative stationary dist with params: {params}"

    def test_forward_distribution_long_horizon(self):
        """T=10, H=10 should still work."""
        params = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        state = np.array([0, 0, 0, 0, 1, 0, 0, 0.0])
        dist = simulate_forward_distribution(params, state, T=10, H=10, n_paths=50000)
        assert abs(dist.sum() - 1.0) < 0.01


class TestDensityEdgeCases:
    """Density extraction edge cases."""

    def test_n_density_from_constant_prices(self):
        """Constant call prices (degenerate) should produce zero density."""
        grid = np.linspace(-0.03, 0.08, 100)
        prices = np.ones_like(grid) * 0.5
        _, density = extract_n_distribution(grid, prices, 0.03, 5)
        # Second derivative of constant is zero -> density is zero
        assert np.all(density == 0)

    def test_q_adjustment_at_target(self):
        """At expected inflation, Q should approximately equal N."""
        grid = np.linspace(-0.03, 0.08, 500)
        # Simple Gaussian density centered at 2%
        n_density = np.exp(-0.5 * ((grid - 0.02) / 0.01) ** 2)
        n_density /= np.trapezoid(n_density, grid)

        q_density = n_to_q_adjustment(grid, n_density, 0.02, 5)
        # At the center, N and Q should be similar (small horizon, at expected)
        center = np.argmin(np.abs(grid - 0.02))
        assert abs(q_density[center] / n_density[center] - 1.0) < 0.2

    def test_bin_probabilities_sum_to_one(self):
        """Bin probabilities must sum to ~1."""
        grid = make_fine_grid()
        density = np.exp(-0.5 * ((grid - 0.02) / 0.01) ** 2)
        density /= np.trapezoid(density, grid)
        probs = density_to_bin_probabilities(grid, density)
        assert abs(probs.sum() - 1.0) < 0.05


class TestParetoEdgeCases:
    def test_empty_array(self):
        """Empty z array should return defaults."""
        alpha, z0 = fit_pareto(np.array([]))
        assert alpha > 0 and z0 > 0

    def test_all_below_one(self):
        """All z <= 1 (no actual disasters) should return defaults."""
        alpha, z0 = fit_pareto(np.array([0.9, 0.95, 0.99]))
        assert alpha > 0

    def test_single_value(self):
        """Single observation should return defaults."""
        alpha, z0 = fit_pareto(np.array([1.05]))
        assert alpha > 0

    def test_identical_values(self):
        """Identical z values should not crash."""
        alpha, z0 = fit_pareto(np.array([1.05, 1.05, 1.05, 1.05]))
        assert np.isfinite(alpha) and alpha > 0


class TestEpsteinZinEdgeCases:
    def test_alpha_near_gamma(self):
        """alpha barely above gamma should give large but finite m_tilde."""
        m = expected_marginal_utility_ratio(3.01, 1.03, 3.0)
        assert np.isfinite(m) and m > 1.0

    def test_alpha_equals_gamma(self):
        """alpha = gamma triggers truncated numerical integration."""
        m = expected_marginal_utility_ratio(3.0, 1.03, 3.0)
        assert np.isfinite(m) and m > 0

    def test_alpha_below_gamma(self):
        """alpha < gamma triggers truncated numerical integration."""
        m = expected_marginal_utility_ratio(2.5, 1.03, 3.0)
        assert np.isfinite(m) and m > 0


class TestAdjustmentEdgeCases:
    def test_zero_probabilities(self):
        """Zero input probabilities should not crash."""
        p_h, p_l = apply_risk_adjustment(0.0, 0.0)
        assert p_h == 0.0 and p_l == 0.0

    def test_one_probabilities(self):
        """Probability of 1.0 should produce valid output."""
        p_h, p_l = apply_risk_adjustment(1.0, 1.0)
        assert 0.0 <= p_h <= 1.0
        assert 0.0 <= p_l <= 1.0

    def test_inflation_adj_very_long_horizon(self):
        """30-year horizon should still produce finite adjustment."""
        factor = inflation_adjustment_factor_continuous(0.06, 0.02, 30)
        assert np.isfinite(factor)
        assert factor > 1.0  # high inflation -> upward adjustment

    def test_disaster_probs_all_mass_in_center(self):
        """All mass in center bins -> zero disaster probability."""
        probs = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0.0])
        high, low = extract_disaster_probabilities(probs, 2.0, 2.0)
        assert high == 0.0
        assert low == 0.0

    def test_disaster_probs_all_mass_in_tails(self):
        """All mass in extreme bins."""
        probs = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
        high, low = extract_disaster_probabilities(probs, 2.0, 2.0)
        assert high > 0.4
        assert low > 0.4


class TestNumericalEdgeCases:
    def test_fine_grid_covers_extremes(self):
        """Fine grid should cover well beyond strike range."""
        grid = make_fine_grid()
        assert grid[0] <= -0.04
        assert grid[-1] >= 0.10
        assert len(grid) >= 1000

    def test_integrate_bins_empty_density(self):
        """Zero density should produce zero probabilities."""
        grid = np.linspace(-0.05, 0.12, 100)
        density = np.zeros_like(grid)
        edges = np.array([-1e10, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 1e10])
        probs = integrate_bins(grid, density, edges)
        assert np.allclose(probs, 0.0) or probs.sum() == 0.0

    def test_integrate_bins_delta_density(self):
        """Density concentrated at one point."""
        grid = np.linspace(-0.05, 0.12, 1500)
        density = np.zeros_like(grid)
        # Put all mass near 2.5% (center of bin 4: (0.02, 0.03])
        center = np.argmin(np.abs(grid - 0.025))
        density[center] = 1000.0  # tall narrow spike
        density[center - 1] = 500.0
        density[center + 1] = 500.0
        edges = np.array([-1e10, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 1e10])
        probs = integrate_bins(grid, density, edges)
        # Most mass should be in the bin containing 2.5%
        assert probs.sum() > 0
        assert probs[4] > 0.5  # bin (0.02, 0.03] should have most mass


class TestAnalyticsModules:
    """Test the analytics modules exist and work."""

    def test_results_to_dataframe(self):
        from inflation_disaster.analytics.disaster_probs import results_to_dataframe
        from inflation_disaster.data.schemas import DisasterProbability
        from datetime import date

        r = DisasterProbability(
            date=date(2024, 1, 1), region="US", threshold=2.0,
            prob_high_n_5y=0.06, prob_high_q_5y=0.067, prob_high_n_10y=0.089,
            prob_high_q_10y=0.11, prob_high_q_5y5y=0.052, prob_high_p_5y5y=0.035,
            prob_low_n_5y=0.027, prob_low_q_5y=0.023, prob_low_n_10y=0.021,
            prob_low_q_10y=0.015, prob_low_q_5y5y=0.025, prob_low_p_5y5y=0.024,
            inflation_adj_5y=1.12, inflation_adj_10y=1.23, horizon_adj_high=0.41,
            horizon_adj_low=1.31, risk_adj_high=0.66, risk_adj_low=0.96,
        )
        df = results_to_dataframe([r])
        assert len(df) == 1
        assert df.iloc[0]["prob_high_p_5y5y"] == 0.035

    def test_anchoring_counterfactuals(self):
        from inflation_disaster.analytics.anchoring import (
            compute_conditional_probabilities, make_counterfactual_states,
        )
        params = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        states = make_counterfactual_states()
        results = compute_conditional_probabilities(params, states)
        assert len(results) == 3
        for label, (p_h, p_l) in results.items():
            assert 0 <= p_h <= 1
            assert 0 <= p_l <= 1
