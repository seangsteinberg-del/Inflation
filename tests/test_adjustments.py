"""Tests for the three adjustments (inflation, horizon, risk)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import MarkovParams, ParetoFit
from inflation_disaster.adjustments.inflation_adj import (
    inflation_adjustment_factor_continuous,
    adjust_bin_probabilities,
)
from inflation_disaster.adjustments.horizon_adj import (
    extract_disaster_probabilities,
    compute_horizon_adj_factor,
)
from inflation_disaster.adjustments.risk_adj import apply_risk_adjustment
from inflation_disaster.models.epstein_zin import (
    expected_marginal_utility_ratio,
    risk_adjustment_factor,
    default_risk_adjustments,
)
from inflation_disaster.models.pareto import fit_pareto


class TestInflationAdjustment:
    """Test the N -> Q adjustment."""

    def test_no_adjustment_at_expected_inflation(self):
        """Factor should be 1.0 when inflation = expected."""
        factor = inflation_adjustment_factor_continuous(0.02, 0.02, 10)
        assert abs(factor - 1.0) < 1e-10

    def test_high_inflation_upward(self):
        """High inflation should have adjustment > 1 (N understates Q)."""
        factor = inflation_adjustment_factor_continuous(0.04, 0.02, 10)
        assert factor > 1.0

    def test_low_inflation_downward(self):
        """Low inflation should have adjustment < 1 (N overstates Q)."""
        factor = inflation_adjustment_factor_continuous(0.00, 0.02, 10)
        assert factor < 1.0

    def test_longer_horizon_larger_adjustment(self):
        """10y adjustment should be larger than 5y."""
        factor_5 = inflation_adjustment_factor_continuous(0.04, 0.02, 5)
        factor_10 = inflation_adjustment_factor_continuous(0.04, 0.02, 10)
        assert factor_10 > factor_5

    def test_bin_probabilities_sum_to_one(self):
        """Q-probabilities should sum to 1 after adjustment."""
        n_probs = np.array([0.01, 0.02, 0.05, 0.20, 0.45, 0.20, 0.05, 0.02])
        q_probs, _ = adjust_bin_probabilities(n_probs, 0.02, 10)
        assert abs(q_probs.sum() - 1.0) < 1e-10

    def test_paper_adjustment_magnitude(self):
        """Adjustment factor should be in the right ballpark per Table 1."""
        # Paper: 10y high inflation adj ~ 1.24
        factor_10y = inflation_adjustment_factor_continuous(0.04, 0.02, 10)
        # exp(0.02 * 10) = exp(0.2) ≈ 1.22
        assert 1.1 < factor_10y < 1.4


class TestHorizonAdjustment:
    def test_disaster_probs_nonneg(self):
        """Disaster probabilities must be non-negative."""
        probs = np.array([0.02, 0.03, 0.05, 0.20, 0.45, 0.15, 0.07, 0.03])
        high, low = extract_disaster_probabilities(probs, threshold=2.0, target=2.0)
        assert high >= 0
        assert low >= 0

    def test_disaster_probs_bounded(self):
        """Disaster probabilities must not exceed 1."""
        probs = np.array([0.02, 0.03, 0.05, 0.20, 0.45, 0.15, 0.07, 0.03])
        high, low = extract_disaster_probabilities(probs, threshold=2.0, target=2.0)
        assert high <= 1.0
        assert low <= 1.0

    def test_high_threshold_captures_top_bins(self):
        """With d=2%, high disaster is >4%, which is bins 6 (4-5%) and 7 (>5%)."""
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7])
        high, low = extract_disaster_probabilities(probs, threshold=2.0, target=2.0)
        assert abs(high - 1.0) < 0.01  # all mass is in disaster bins

    def test_deflation_captures_bottom_bins(self):
        """With d=2%, deflation is <0%, which is bins 0 (<=-1%) and 1 (-1%,0%)."""
        probs = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        high, low = extract_disaster_probabilities(probs, threshold=2.0, target=2.0)
        assert abs(low - 1.0) < 0.01

    def test_horizon_adj_factor(self):
        assert compute_horizon_adj_factor(0.10, 0.04) == pytest.approx(0.4, abs=0.01)


class TestRiskAdjustment:
    def test_paper_high_inflation_value(self):
        """P/Q for high inflation should be ~0.66."""
        high_fit = ParetoFit(
            disaster_type="high", alpha=5.45, z_0=1.03,
            n_observations=50, p_tilde=0.356, risk_adjustment_factor=0.0,
        )
        factor = risk_adjustment_factor(high_fit, gamma=3.0)
        assert abs(factor - 0.66) < 0.01

    def test_paper_deflation_value(self):
        """P/Q for deflation should be ~0.96."""
        low_fit = ParetoFit(
            disaster_type="low", alpha=15.18, z_0=1.06,
            n_observations=20, p_tilde=0.085, risk_adjustment_factor=0.0,
        )
        factor = risk_adjustment_factor(low_fit, gamma=3.0)
        assert abs(factor - 0.96) < 0.01

    def test_risk_adj_reduces_probability(self):
        """Physical probability should be <= risk-neutral probability."""
        p_high, p_low = apply_risk_adjustment(0.10, 0.05)
        assert p_high <= 0.10
        assert p_low <= 0.05

    def test_defaults_match_paper(self):
        """Default adjustment factors should match paper."""
        adj_h, adj_l = default_risk_adjustments()
        assert adj_h == 0.66
        assert adj_l == 0.96

    def test_marginal_utility_ratio_correct(self):
        """E[z^gamma] = alpha * z_0^gamma / (alpha - gamma)."""
        alpha, z_0, gamma = 5.45, 1.03, 3.0
        result = expected_marginal_utility_ratio(alpha, z_0, gamma)
        expected = alpha * z_0**gamma / (alpha - gamma)
        assert abs(result - expected) < 1e-10


class TestParetoFit:
    def test_known_pareto_data(self):
        """Fit should recover known alpha from Pareto-distributed samples."""
        rng = np.random.RandomState(42)
        true_alpha = 6.0
        z_0 = 1.03
        # Generate Pareto samples: z = z_0 * u^(-1/alpha) where u ~ Uniform(0,1)
        u = rng.uniform(0.0, 1.0, size=1000)
        z_samples = z_0 * u ** (-1 / true_alpha)

        fitted_alpha, fitted_z0 = fit_pareto(z_samples, z_min=z_0)
        assert abs(fitted_alpha - true_alpha) < 0.5  # should be close
        assert fitted_z0 == z_0

    def test_handles_identical_values(self):
        """Should not crash if all z are the same."""
        z = np.array([1.05, 1.05, 1.05, 1.05])
        alpha, z_0 = fit_pareto(z)
        assert alpha > 0  # should return default
