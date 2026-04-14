"""Tests for YOY caplet stripping and forward distribution extraction."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.models.yoy import (
    strip_caplets,
    strip_floorlets,
    _implied_vol_bisection,
    extract_average_forward_yoy,
)
from inflation_disaster.models.sabr import black_price


class TestCapletStripping:
    def test_first_maturity_equals_cap(self):
        """First maturity caplet should equal the full cap price."""
        cap_prices = {
            5: np.array([0.10, 0.08, 0.05]),
            7: np.array([0.15, 0.12, 0.08]),
            10: np.array([0.20, 0.16, 0.10]),
        }
        dfs = {5: 0.90, 7: 0.85, 10: 0.78}
        strikes = np.array([1.0, 2.0, 3.0])

        caplets = strip_caplets(cap_prices, dfs, strikes)
        np.testing.assert_array_equal(caplets[5], cap_prices[5])

    def test_caplets_nonnegative(self):
        """Stripped caplets should be non-negative."""
        cap_prices = {
            5: np.array([0.10, 0.08, 0.05]),
            7: np.array([0.15, 0.12, 0.08]),
            10: np.array([0.20, 0.16, 0.10]),
        }
        dfs = {5: 0.90, 7: 0.85, 10: 0.78}
        strikes = np.array([1.0, 2.0, 3.0])

        caplets = strip_caplets(cap_prices, dfs, strikes)
        for year, prices in caplets.items():
            assert np.all(prices >= 0), f"Negative caplet at year {year}"

    def test_floorlet_stripping(self):
        """Floorlet stripping should mirror caplet stripping."""
        floor_prices = {
            5: np.array([0.02, 0.05, 0.08]),
            10: np.array([0.04, 0.08, 0.12]),
        }
        dfs = {5: 0.90, 10: 0.78}
        strikes = np.array([1.0, 2.0, 3.0])

        floorlets = strip_floorlets(floor_prices, dfs, strikes)
        assert 5 in floorlets and 10 in floorlets
        assert np.all(floorlets[10] >= 0)


class TestImpliedVol:
    def test_recovers_known_vol(self):
        """Should recover the vol used to generate the price."""
        F, K, T, sigma, df = 1.02, 1.04, 1.0, 0.03, 0.98
        price = black_price(F, K, T, sigma, df, is_call=True)
        recovered = _implied_vol_bisection(F, K, T, price, df)
        assert abs(recovered - sigma) < 1e-4

    def test_handles_deep_otm(self):
        """Deep OTM option should give a finite implied vol."""
        F, K, T, df = 1.02, 1.10, 1.0, 0.98
        price = 1e-6  # very small price
        vol = _implied_vol_bisection(F, K, T, price, df)
        assert np.isfinite(vol) and vol > 0
