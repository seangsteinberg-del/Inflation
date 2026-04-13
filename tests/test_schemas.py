"""Tests for Pydantic data models."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import (
    MarkovParams,
    SABRParams,
    InflationDistribution,
)


class TestMarkovParams:
    def test_valid_params(self):
        p = MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)
        assert p.p_dh == 0.05

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            MarkovParams(p_dh=-0.01, p_dl=0.03, p_nn=0.15, p_h=0.19, p_l=0.19, p_mr=0.50)

    def test_rejects_infeasible_middle_row(self):
        """1 - 2*p_nn - p_dl - p_dh must be >= 0."""
        with pytest.raises(ValueError, match="Infeasible middle"):
            MarkovParams(p_dh=0.3, p_dl=0.3, p_nn=0.3, p_h=0.1, p_l=0.1, p_mr=0.2)

    def test_rejects_infeasible_boundary_row(self):
        """1 - p_dl - p_nn - p_mr must be >= 0 for row 1."""
        with pytest.raises(ValueError, match="Infeasible row 1"):
            MarkovParams(p_dh=0.01, p_dl=0.3, p_nn=0.3, p_h=0.1, p_l=0.1, p_mr=0.5)

    def test_rejects_infeasible_disaster_exit(self):
        """1 - 5*p_h must be >= 0."""
        with pytest.raises(ValueError, match="Infeasible row 7"):
            MarkovParams(p_dh=0.05, p_dl=0.03, p_nn=0.15, p_h=0.25, p_l=0.19, p_mr=0.50)


class TestSABRParams:
    def test_valid(self):
        p = SABRParams(alpha=0.03, beta=0.5, rho=-0.2, nu=0.4)
        assert p.alpha == 0.03

    def test_rejects_bad_beta(self):
        with pytest.raises(ValueError, match="beta"):
            SABRParams(alpha=0.03, beta=1.5, rho=-0.2, nu=0.4)

    def test_rejects_bad_rho(self):
        with pytest.raises(ValueError, match="rho"):
            SABRParams(alpha=0.03, beta=0.5, rho=1.0, nu=0.4)

    def test_rejects_zero_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            SABRParams(alpha=0.0, beta=0.5, rho=0.0, nu=0.4)


class TestInflationDistribution:
    def test_valid(self):
        probs = np.array([0.02, 0.03, 0.05, 0.20, 0.45, 0.15, 0.07, 0.03])
        d = InflationDistribution(
            date="2024-01-01", region="US", horizon=5, measure="N",
            bin_probabilities=probs,
        )
        assert d.measure == "N"

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="8 bin"):
            InflationDistribution(
                date="2024-01-01", region="US", horizon=5, measure="N",
                bin_probabilities=np.array([0.5, 0.5]),
            )

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="Negative"):
            InflationDistribution(
                date="2024-01-01", region="US", horizon=5, measure="N",
                bin_probabilities=np.array([0.5, 0.5, -0.1, 0.0, 0.0, 0.0, 0.1, 0.0]),
            )

    def test_accepts_5y5y_horizon(self):
        probs = np.array([0.02, 0.03, 0.05, 0.20, 0.45, 0.15, 0.07, 0.03])
        d = InflationDistribution(
            date="2024-01-01", region="US", horizon="5y5y", measure="Q",
            bin_probabilities=probs,
        )
        assert d.horizon == "5y5y"
