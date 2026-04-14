"""Shared test fixtures for the inflation disaster test suite."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.data.schemas import MarkovParams, SABRParams, OptionSurface, CleanedSurface
from inflation_disaster.config import settings


@pytest.fixture
def us_markov_params():
    """Paper's US Markov chain parameter estimates."""
    return MarkovParams(
        p_dh=0.05, p_dl=0.03, p_nn=0.15,
        p_h=0.1998, p_l=0.1990, p_mr=0.50,
    )


@pytest.fixture
def ez_markov_params():
    """Paper's EZ Markov chain parameter estimates."""
    return MarkovParams(
        p_dh=0.04, p_dl=0.05, p_nn=0.12,
        p_h=0.0617, p_l=0.1999, p_mr=0.47,
    )


@pytest.fixture
def center_state():
    """Initial state concentrated in 2-3% bin."""
    return np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)


@pytest.fixture
def sabr_params():
    """Typical SABR parameters for inflation options."""
    return SABRParams(alpha=0.03, beta=0.5, rho=-0.15, nu=0.35)


@pytest.fixture
def gaussian_density():
    """A Gaussian test density centered at 2% with 1% std."""
    from inflation_disaster.utils.numerical import make_fine_grid
    grid = make_fine_grid()
    density = np.exp(-0.5 * ((grid - 0.02) / 0.01) ** 2)
    total = np.trapezoid(density, grid)
    density /= total
    return grid, density


@pytest.fixture
def sample_option_surface():
    """A synthetic option surface with realistic structure."""
    from datetime import date
    from inflation_disaster.models.sabr import sabr_to_prices, black_price

    strikes_pct = settings.strikes  # -2 to 6 in 0.5 steps
    strikes_gross = 1.0 + strikes_pct / 100.0
    forward = 1.022  # 2.2% expected inflation
    T = 5
    df = 0.85  # 5y discount factor

    flat_vol = 0.025
    cap_prices = np.array([
        black_price(forward, k, T, 0.025, df, is_call=True)
        for k in strikes_gross
    ])
    floor_prices = np.array([
        black_price(forward, k, T, 0.025, df, is_call=False)
        for k in strikes_gross
    ])

    return OptionSurface(
        date=date(2024, 1, 2),
        region="US",
        maturity=5,
        strikes=strikes_pct,
        cap_prices=cap_prices,
        floor_prices=floor_prices,
        swap_rate=0.022,
    )
