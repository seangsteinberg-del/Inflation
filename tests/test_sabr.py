"""Tests for SABR model calibration."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import SABRParams
from inflation_disaster.models.sabr import (
    sabr_implied_vol,
    _sabr_atm_vol,
    sabr_smile,
    calibrate_sabr,
    black_price,
    sabr_to_prices,
)


class TestSABRImpliedVol:
    """Test the Hagan (2002) SABR implied vol formula."""

    def test_atm_consistency(self):
        """General formula must equal ATM formula when F = K."""
        params = SABRParams(alpha=0.03, beta=0.5, rho=-0.2, nu=0.4)
        F = 1.02
        T = 10.0

        general = sabr_implied_vol(F, F, T, params)
        atm = _sabr_atm_vol(F, T, params.alpha, params.beta, params.rho, params.nu)

        assert abs(general - atm) < 1e-10, f"ATM mismatch: general={general}, atm={atm}"

    def test_positive_vol(self):
        """Implied vol must always be positive."""
        params = SABRParams(alpha=0.02, beta=0.5, rho=0.0, nu=0.3)
        for F in [0.98, 1.0, 1.02, 1.05]:
            for K in [0.96, 0.98, 1.0, 1.02, 1.04, 1.06]:
                for T in [1, 5, 10]:
                    vol = sabr_implied_vol(F, K, T, params)
                    assert vol > 0, f"Non-positive vol at F={F}, K={K}, T={T}: {vol}"

    def test_rho_zero_symmetry(self):
        """With rho=0, smile should be approximately symmetric around ATM."""
        params = SABRParams(alpha=0.03, beta=0.5, rho=0.0, nu=0.3)
        F = 1.02
        T = 5.0

        vol_down = sabr_implied_vol(F, F - 0.02, T, params)
        vol_up = sabr_implied_vol(F, F + 0.02, T, params)

        # Not exactly symmetric due to beta < 1, but should be close
        assert abs(vol_down - vol_up) / vol_down < 0.3

    def test_negative_rho_skew(self):
        """Negative rho should produce downward-skewed smile (higher vol at lower strikes)."""
        params = SABRParams(alpha=0.03, beta=0.5, rho=-0.5, nu=0.4)
        F = 1.02
        T = 10.0

        vol_low = sabr_implied_vol(F, F - 0.03, T, params)
        vol_high = sabr_implied_vol(F, F + 0.03, T, params)

        assert vol_low > vol_high, "Negative rho should produce downward skew"


class TestSABRCalibration:
    """Test SABR calibration to market data."""

    def test_recovery_from_sabr_smile(self):
        """Calibration should recover params from SABR-generated smile."""
        true_params = SABRParams(alpha=0.03, beta=0.5, rho=-0.15, nu=0.35)
        F = 1.02
        T = 10.0
        strikes = np.linspace(0.96, 1.08, 13)

        # Generate "market" vols from the true params
        market_vols = sabr_smile(F, strikes, T, true_params)

        # Calibrate
        fitted = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)

        # Should recover approximately
        assert abs(fitted.alpha - true_params.alpha) < 0.005
        assert abs(fitted.rho - true_params.rho) < 0.1
        assert abs(fitted.nu - true_params.nu) < 0.1

    def test_handles_few_points(self):
        """Should not crash with only 2 valid vol points."""
        F = 1.02
        T = 5.0
        strikes = np.array([0.98, 1.02])
        vols = np.array([0.03, 0.025])

        result = calibrate_sabr(F, strikes, vols, T)
        assert result.alpha > 0


class TestBlackPrice:
    """Test Black-76 pricing formula."""

    def test_atm_call_positive(self):
        """ATM call has positive value."""
        price = black_price(1.02, 1.02, 10.0, 0.03, 0.95, is_call=True)
        assert price > 0

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call should be close to discounted intrinsic."""
        F, K, df = 1.06, 0.96, 0.90
        price = black_price(F, K, 10.0, 0.03, df, is_call=True)
        intrinsic = df * (F - K)
        assert price >= intrinsic * 0.99

    def test_put_call_parity(self):
        """call - put = discount * (F - K)."""
        F, K, T, sigma, df = 1.02, 1.04, 10.0, 0.03, 0.90
        call = black_price(F, K, T, sigma, df, is_call=True)
        put = black_price(F, K, T, sigma, df, is_call=False)
        assert abs((call - put) - df * (F - K)) < 1e-10
