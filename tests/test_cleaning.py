"""Tests for data cleaning functions."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.cleaning import (
    check_cap_monotonicity,
    check_floor_monotonicity,
    check_butterfly,
)


class TestMonotonicity:
    def test_monotone_caps_pass(self):
        """Decreasing cap prices should have no violations."""
        strikes = np.array([1.0, 2.0, 3.0, 4.0])
        prices = np.array([0.10, 0.08, 0.05, 0.02])
        violations = check_cap_monotonicity(prices, strikes)
        assert violations.sum() == 0

    def test_non_monotone_cap_detected(self):
        strikes = np.array([1.0, 2.0, 3.0, 4.0])
        prices = np.array([0.10, 0.12, 0.05, 0.02])  # violation at index 1
        violations = check_cap_monotonicity(prices, strikes)
        assert violations[1] == True

    def test_monotone_floors_pass(self):
        """Increasing floor prices should have no violations."""
        strikes = np.array([1.0, 2.0, 3.0, 4.0])
        prices = np.array([0.02, 0.05, 0.08, 0.10])
        violations = check_floor_monotonicity(prices, strikes)
        assert violations.sum() == 0

    def test_non_monotone_floor_detected(self):
        strikes = np.array([1.0, 2.0, 3.0, 4.0])
        prices = np.array([0.02, 0.05, 0.03, 0.10])  # violation at index 2
        violations = check_floor_monotonicity(prices, strikes)
        assert violations[2] == True


class TestButterfly:
    def test_convex_prices_pass(self):
        """Convex call prices should have no butterfly violations."""
        strikes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Quadratic (convex) call prices
        prices = np.array([0.16, 0.09, 0.04, 0.01, 0.0])
        violations = check_butterfly(prices, strikes)
        assert violations.sum() == 0

    def test_concave_prices_detected(self):
        """Concave prices should trigger butterfly violation."""
        strikes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Concave: c(2) is too high relative to neighbors
        prices = np.array([0.04, 0.09, 0.16, 0.09, 0.04])
        violations = check_butterfly(prices, strikes)
        # Interior points 1,2,3 should have violations
        assert violations.sum() > 0
