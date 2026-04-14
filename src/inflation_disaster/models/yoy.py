"""Year-on-year caplet/floorlet stripping and forward distribution extraction.

Converts YOY inflation cap/floor prices into forward-starting annual inflation
distributions for years 5-9. These are used as the third set of moments
for the GMM estimator (matching the average forward YOY distribution).

The paper averages the five annual distributions (years 5-9) because they
are approximately equal, indicating inflation has reached its ergodic state
by year 5. This averaging also makes the approach robust to data noise in
the less-liquid YOY market.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import InflationDistribution, SABRParams
from inflation_disaster.models.sabr import calibrate_sabr, sabr_to_prices, black_price
from inflation_disaster.utils.numerical import make_fine_grid

log = logging.getLogger("inflation_disaster.models.yoy")

# NumPy compat
_trapz = getattr(np, "trapezoid", None) or np.trapz


def strip_caplets(
    yoy_cap_prices: dict[int, np.ndarray],
    discount_factors: dict[int, float],
    strikes: np.ndarray,
) -> dict[int, np.ndarray]:
    """Extract individual year caplet prices from YOY cap term structure.

    A YOY cap of maturity T is a portfolio of T caplets, one for each year.
    caplet_t = yoy_cap_T(k) - yoy_cap_{T-1}(k), adjusted for discounting.

    Parameters
    ----------
    yoy_cap_prices : dict
        Keys are maturities (years), values are arrays of cap prices by strike.
    discount_factors : dict
        Keys are maturities, values are cumulative discount factors.
    strikes : array
        Strike prices (same for all maturities).

    Returns
    -------
    caplets : dict
        Keys are years (e.g., 5, 6, 7, 8, 9), values are caplet prices by strike.
    """
    sorted_mats = sorted(yoy_cap_prices.keys())
    caplets = {}

    for i, mat in enumerate(sorted_mats):
        if i == 0:
            # First maturity: caplet = entire cap price (no prior maturity to subtract)
            caplets[mat] = yoy_cap_prices[mat].copy()
        else:
            prev_mat = sorted_mats[i - 1]
            # Caplet for year mat = cap(mat) - cap(mat-1)
            # Market cap prices are already present-valued, so no DF adjustment needed.
            caplets[mat] = yoy_cap_prices[mat] - yoy_cap_prices[prev_mat]
            # Ensure non-negative
            caplets[mat] = np.maximum(caplets[mat], 0.0)

    return caplets


def strip_floorlets(
    yoy_floor_prices: dict[int, np.ndarray],
    discount_factors: dict[int, float],
    strikes: np.ndarray,
) -> dict[int, np.ndarray]:
    """Extract individual year floorlet prices from YOY floor term structure."""
    sorted_mats = sorted(yoy_floor_prices.keys())
    floorlets = {}

    for i, mat in enumerate(sorted_mats):
        if i == 0:
            floorlets[mat] = yoy_floor_prices[mat].copy()
        else:
            prev_mat = sorted_mats[i - 1]
            # Market floor prices are already present-valued, no DF adjustment needed.
            floorlets[mat] = yoy_floor_prices[mat] - yoy_floor_prices[prev_mat]
            floorlets[mat] = np.maximum(floorlets[mat], 0.0)

    return floorlets


def extract_yoy_distribution(
    caplet_prices: np.ndarray,
    strikes: np.ndarray,
    forward_rate: float,
    discount_factor: float,
    year: int,
    region: str = "US",
    date=None,
) -> InflationDistribution:
    """Extract Q-distribution from a single year's caplet prices.

    Uses SABR smoothing + Breeden-Litzenberger, same as for ZC options
    but applied to the individual year caplet.

    Parameters
    ----------
    caplet_prices : array
        Caplet prices at each strike for this specific year.
    strikes : array
        Strike prices in percent.
    forward_rate : float
        Forward inflation rate for this year (decimal, e.g. 0.02).
    discount_factor : float
        Discount factor to this year.
    year : int
        Which year (e.g. 5, 6, 7, 8, 9).
    region : str
    date : date, optional

    Returns
    -------
    InflationDistribution (Q-measure) for this year.
    """
    from datetime import date as date_type

    if date is None:
        date = date_type.today()

    # Convert strikes to gross inflation for SABR
    gross_strikes = 1.0 + strikes / 100.0
    gross_forward = 1.0 + forward_rate

    # Compute implied vols from caplet prices
    implied_vols = np.zeros_like(strikes, dtype=float)
    for i, (k, price) in enumerate(zip(gross_strikes, caplet_prices)):
        if price > 1e-8 and k > 0:
            # Invert Black formula to get implied vol (bisection)
            vol = _implied_vol_bisection(gross_forward, k, 1.0, price, discount_factor)
            implied_vols[i] = vol

    # SABR calibration
    valid = implied_vols > 1e-6
    if valid.sum() >= 3:
        sabr_params = calibrate_sabr(
            gross_forward, gross_strikes[valid], implied_vols[valid], T=1.0
        )
    else:
        sabr_params = SABRParams(alpha=0.01, beta=0.5, rho=0.0, nu=0.3)

    # Reconstruct prices on fine grid
    fine_rate_grid = make_fine_grid()
    fine_gross_grid = 1.0 + fine_rate_grid
    fine_prices = sabr_to_prices(
        gross_forward, fine_gross_grid, T=1.0, params=sabr_params,
        discount_factor=discount_factor,
    )

    # Extract density via second derivative
    cs = CubicSpline(fine_rate_grid, fine_prices, bc_type="natural")
    d2_prices = cs(fine_rate_grid, nu=2)

    # Q-density (for 1-year horizon, inflation and N->Q adjustments are small)
    q_density = np.maximum(d2_prices / discount_factor, 0.0)
    total = _trapz(q_density, fine_rate_grid)
    if total > 0:
        q_density /= total

    # Bin probabilities
    bin_edges = np.array(settings.bin_edges) / 100.0
    n_bins = len(bin_edges) - 1
    probs = np.zeros(n_bins)
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            mask = fine_rate_grid <= upper
        else:
            mask = (fine_rate_grid > lower) & (fine_rate_grid <= upper)
        if mask.sum() >= 2:
            probs[i] = _trapz(q_density[mask], fine_rate_grid[mask])
    total_p = probs.sum()
    if total_p > 0:
        probs /= total_p

    return InflationDistribution(
        date=date, region=region, horizon=year, measure="Q",
        bin_probabilities=probs,
    )


def extract_average_forward_yoy(
    caplet_prices_by_year: dict[int, np.ndarray],
    strikes: np.ndarray,
    forward_rates: dict[int, float],
    discount_factors: dict[int, float],
    years: list[int] | None = None,
    region: str = "US",
    date=None,
) -> InflationDistribution:
    """Extract the average forward YOY distribution over years 5-9.

    This is the third moment set used in the GMM estimator.

    Parameters
    ----------
    caplet_prices_by_year : dict
        Year -> caplet prices array.
    strikes : array
        Strike prices in percent.
    forward_rates : dict
        Year -> forward inflation rate (decimal).
    discount_factors : dict
        Year -> discount factor.
    years : list, optional
        Years to average over. Default: [5, 6, 7, 8, 9].

    Returns
    -------
    InflationDistribution with horizon="yoy_avg", averaged across years.
    """
    from datetime import date as date_type

    if years is None:
        years = [5, 6, 7, 8, 9]
    if date is None:
        date = date_type.today()

    avg_probs = np.zeros(settings.n_bins)
    n_valid = 0

    for year in years:
        if year not in caplet_prices_by_year:
            continue
        dist = extract_yoy_distribution(
            caplet_prices_by_year[year], strikes,
            forward_rates.get(year, 0.02),
            discount_factors.get(year, 0.95),
            year=year, region=region, date=date,
        )
        avg_probs += dist.bin_probabilities
        n_valid += 1

    if n_valid > 0:
        avg_probs /= n_valid

    # Renormalize
    total = avg_probs.sum()
    if total > 0:
        avg_probs /= total

    return InflationDistribution(
        date=date, region=region, horizon="yoy_avg", measure="Q",
        bin_probabilities=avg_probs,
    )


def _implied_vol_bisection(
    forward: float,
    strike: float,
    T: float,
    market_price: float,
    discount_factor: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Invert Black-76 formula to get implied vol via bisection."""
    lo, hi = 1e-6, 3.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        model_price = black_price(forward, strike, T, mid, discount_factor, is_call=True)
        if model_price > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return (lo + hi) / 2.0
