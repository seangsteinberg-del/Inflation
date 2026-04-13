"""Tests for the 8-state Markov chain inflation model."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import MarkovParams
from inflation_disaster.models.markov_chain import (
    build_transition_matrix,
    validate_transition_matrix,
    stationary_distribution,
    simulate_cumulative_distribution,
    simulate_forward_distribution,
    marginal_one_year_distribution,
    N_STATES,
)


@pytest.fixture
def us_params():
    """Paper's US parameter estimates."""
    return MarkovParams(
        p_dh=0.05, p_dl=0.03, p_nn=0.15,
        p_h=0.1998, p_l=0.1990, p_mr=0.50,
    )


@pytest.fixture
def center_state():
    """Initial state concentrated in 2-3% bin."""
    return np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)


class TestTransitionMatrix:
    """Test the 8x8 transition matrix construction."""

    def test_valid_stochastic_matrix(self, us_params):
        P = build_transition_matrix(us_params)
        assert validate_transition_matrix(P)

    def test_rows_sum_to_one(self, us_params):
        P = build_transition_matrix(us_params)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(N_STATES), atol=1e-10)

    def test_all_nonnegative(self, us_params):
        P = build_transition_matrix(us_params)
        assert np.all(P >= 0), f"Negative entries: {P[P < 0]}"

    def test_disaster_exit_structure(self, us_params):
        """Row 0 exits to states 1-5, row 7 exits to states 2-6."""
        P = build_transition_matrix(us_params)

        # Row 0: deflation disaster
        assert P[0, 6] == 0.0
        assert P[0, 7] == 0.0
        assert P[0, 1] > 0  # exits to normal states
        assert abs(P[0, 0] - (1 - 5 * us_params.p_l)) < 1e-10

        # Row 7: high inflation disaster
        assert P[7, 0] == 0.0
        assert P[7, 1] == 0.0
        assert P[7, 2] > 0  # exits to normal states
        assert abs(P[7, 7] - (1 - 5 * us_params.p_h)) < 1e-10

    def test_symmetry_rows_3_4(self, us_params):
        """Rows 3 and 4 (at-target) should have identical structure."""
        P = build_transition_matrix(us_params)
        # Both have: p_dl at col 0, p_nn neighbors, p_n diagonal, p_dh at col 7
        assert P[3, 0] == P[4, 0]  # same p_dl
        assert P[3, 7] == P[4, 7]  # same p_dh
        assert P[3, 3] == P[4, 4]  # same diagonal
        assert P[3, 2] == P[4, 5]  # same local movement

    def test_rejects_infeasible_params(self):
        """Should reject params that make rows negative."""
        with pytest.raises(ValueError, match="Infeasible"):
            MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.25, p_l=0.19, p_mr=0.50)


class TestStationaryDistribution:
    def test_is_valid_distribution(self, us_params):
        P = build_transition_matrix(us_params)
        pi = stationary_distribution(P)
        assert np.all(pi >= 0)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_is_fixed_point(self, us_params):
        P = build_transition_matrix(us_params)
        pi = stationary_distribution(P)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)

    def test_mode_near_target(self, us_params):
        """Stationary distribution should peak near 2% target (bins 3,4)."""
        P = build_transition_matrix(us_params)
        pi = stationary_distribution(P)
        assert pi[3] + pi[4] > 0.5, "Most mass should be near target"


class TestSimulation:
    def test_cumulative_sums_to_one(self, us_params, center_state):
        dist = simulate_cumulative_distribution(
            us_params, center_state, horizon=5, n_paths=50000
        )
        assert abs(dist.sum() - 1.0) < 0.01

    def test_forward_sums_to_one(self, us_params, center_state):
        dist = simulate_forward_distribution(
            us_params, center_state, T=5, H=5, n_paths=50000
        )
        assert abs(dist.sum() - 1.0) < 0.01

    def test_different_seeds(self, us_params, center_state):
        """Spot and forward should use different random sequences."""
        dist_spot = simulate_cumulative_distribution(
            us_params, center_state, horizon=5, n_paths=100000
        )
        dist_fwd = simulate_forward_distribution(
            us_params, center_state, T=5, H=5, n_paths=100000
        )
        # They should differ (different initial conditions and seeds)
        assert not np.allclose(dist_spot, dist_fwd, atol=0.005)

    def test_longer_horizon_more_dispersed(self, us_params, center_state):
        """10y distribution should be more dispersed than 5y."""
        dist_5 = simulate_cumulative_distribution(
            us_params, center_state, horizon=5, n_paths=100000
        )
        dist_10 = simulate_cumulative_distribution(
            us_params, center_state, horizon=10, n_paths=100000, seed=99
        )
        # The peak probability should be lower for 10y (more spread)
        assert dist_10.max() < dist_5.max()

    def test_marginal_distribution(self, us_params, center_state):
        """One-year marginal should be valid distribution."""
        marg = marginal_one_year_distribution(us_params, center_state, year=5)
        assert np.all(marg >= -1e-10)
        assert abs(marg.sum() - 1.0) < 1e-10
