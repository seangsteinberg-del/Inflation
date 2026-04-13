"""Breeden-Litzenberger density extraction from option prices.

Extracts risk-neutral probability distributions from inflation option prices:
- N-probabilities (nominal risk-neutral): eq. (8) -- N(k) = 1 + e^i * a'(k)
- Q-probabilities (real risk-neutral): eq. (10) -- Q(k) = e^r * k * a''(k)

The inflation adjustment (N -> Q) is the paper's first key contribution.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import InflationDistribution
from inflation_disaster.utils.numerical import integrate_bins, make_fine_grid

log = logging.getLogger("inflation_disaster.models.breeden_litzenberger")


def extract_n_distribution(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    nominal_rate: float,
    maturity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract N-probability density from option prices via eq. (8).

    N(k) = 1 + e^{i*T} * a'(k)

    where a'(k) is the first derivative of cap price w.r.t. strike,
    and i is the nominal interest rate.

    The density n(k) = N'(k) = e^{i*T} * a''(k).

    Parameters
    ----------
    strikes : array
        Strike prices on a fine grid (annual inflation rates, e.g. 0.02 for 2%).
    call_prices : array
        Call (cap) prices on the same grid.
    nominal_rate : float
        Annualized nominal interest rate.
    maturity : int
        Maturity in years.

    Returns
    -------
    grid : array
        Strike grid.
    n_density : array
        N-probability density.
    """
    # Fit cubic spline to call prices
    cs = CubicSpline(strikes, call_prices, bc_type="natural")

    # N-density: n(k) = e^{i*T} * a''(k)
    discount_inv = np.exp(nominal_rate * maturity)
    d2_prices = cs(strikes, nu=2)

    n_density = discount_inv * d2_prices

    # Ensure non-negative
    n_density = np.maximum(n_density, 0.0)

    return strikes, n_density


def extract_q_distribution(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    real_rate: float,
    maturity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Q-probability density directly from option prices via eq. (10).

    Q(k) = e^{r*T} * k * a''(k)

    This is the real (inflation-adjusted) risk-neutral density.
    The direct route avoids going through N first.

    WARNING: This function assumes strikes are in annual decimal rates and
    computes gross strikes as (1+rate)^T. The second derivative a''(k) is
    w.r.t. the annual rate, which requires a Jacobian correction that is
    NOT applied here. For production use, prefer extract_n_distribution()
    followed by n_to_q_adjustment() instead.

    Parameters
    ----------
    strikes : array
        Strike prices (annual inflation rates).
    call_prices : array
        Call (cap) prices.
    real_rate : float
        Annualized real interest rate.
    maturity : int
        Maturity in years.

    Returns
    -------
    grid : array
        Strike grid.
    q_density : array
        Q-probability density.
    """
    cs = CubicSpline(strikes, call_prices, bc_type="natural")
    d2_prices = cs(strikes, nu=2)

    # Q(k) = e^{r*T} * k * a''(k) where k is the gross inflation rate
    # If strikes are annual decimal rates (e.g. 0.02 for 2%), convert to gross:
    # gross = (1 + annual_rate) for single-year options, or (1 + rate)^T for T-year
    # For the paper's ZC options (paying on cumulative inflation), k = (1 + rate)^T
    gross_strikes = (1.0 + strikes) ** maturity

    real_discount_inv = np.exp(real_rate * maturity)
    q_density = real_discount_inv * gross_strikes * d2_prices

    q_density = np.maximum(q_density, 0.0)

    return strikes, q_density


def n_to_q_adjustment(
    strikes: np.ndarray,
    n_density: np.ndarray,
    expected_inflation: float,
    maturity: int,
) -> np.ndarray:
    """Convert N-density to Q-density via equation (4).

    q(pi) = n(pi) * exp((pi - pi^e) * T)

    where pi^e = i - r is expected inflation (breakeven).

    This is the paper's first adjustment: nominal payoffs must be converted
    to real payoffs. High inflation states are worth less in real terms
    (so N understates their probability), low inflation states worth more.

    Parameters
    ----------
    strikes : array
        Inflation rates.
    n_density : array
        N-probability density.
    expected_inflation : float
        Expected inflation rate (e.g., from swap rate).
    maturity : int
        Maturity in years.

    Returns
    -------
    q_density : array
        Q-probability density (renormalized to integrate to 1).
    """
    # Adjustment factor: exp((pi - pi^e) * T) for each inflation rate
    adjustment = np.exp((strikes - expected_inflation) * maturity)

    q_density = n_density * adjustment

    # Renormalize
    total = np.trapezoid(q_density, strikes)
    if total > 0:
        q_density /= total

    return q_density


def density_to_bin_probabilities(
    grid: np.ndarray,
    density: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate density over the 8 inflation bins.

    Bins: {<=-1%, (-1,0], (0,1], (1,2], (2,3], (3,4], (4,5], >5%}

    Parameters
    ----------
    grid : array
        Fine grid of inflation rates (in decimal, e.g., 0.02 for 2%).
    density : array
        Probability density on the grid.
    bin_edges : array, optional
        Bin edges in decimal. Default from config.

    Returns
    -------
    probabilities : array of length 8
    """
    if bin_edges is None:
        bin_edges = np.array(settings.bin_edges) / 100.0  # convert percent to decimal

    return integrate_bins(grid, density, bin_edges)


def compute_disaster_probabilities_from_density(
    grid: np.ndarray,
    density: np.ndarray,
    target: float = 0.02,
    threshold: float = 0.02,
) -> tuple[float, float]:
    """Compute disaster probabilities directly from continuous density.

    Parameters
    ----------
    grid : array
        Inflation rate grid.
    density : array
        Probability density.
    target : float
        Inflation target (default 2% = 0.02).
    threshold : float
        Disaster threshold d (default 2pp = 0.02).

    Returns
    -------
    prob_high : float
        Prob[pi > target + threshold]
    prob_low : float
        Prob[pi < target - threshold]
    """
    high_mask = grid > (target + threshold)
    low_mask = grid < (target - threshold)

    prob_high = np.trapezoid(density[high_mask], grid[high_mask]) if high_mask.sum() > 1 else 0.0
    prob_low = np.trapezoid(density[low_mask], grid[low_mask]) if low_mask.sum() > 1 else 0.0

    return float(prob_high), float(prob_low)


def extract_full_distributions(
    fine_strikes: np.ndarray,
    call_prices: np.ndarray,
    nominal_rate: float,
    real_rate: float,
    expected_inflation: float,
    maturity: int,
    date=None,
    region: str = "US",
) -> tuple[InflationDistribution, InflationDistribution]:
    """Full extraction pipeline: option prices -> N and Q bin distributions.

    Parameters
    ----------
    fine_strikes : array
        Dense strike grid (decimal inflation rates).
    call_prices : array
        Call prices on the fine grid (from SABR reconstruction).
    nominal_rate, real_rate : float
        Interest rates.
    expected_inflation : float
        Breakeven inflation from swaps.
    maturity : int
        Years.
    date : date, optional
    region : str

    Returns
    -------
    n_dist : InflationDistribution (N-measure)
    q_dist : InflationDistribution (Q-measure)
    """
    from datetime import date as date_type

    if date is None:
        date = date_type.today()

    # Extract N-density
    _, n_density = extract_n_distribution(
        fine_strikes, call_prices, nominal_rate, maturity
    )

    # Convert to Q-density
    q_density = n_to_q_adjustment(
        fine_strikes, n_density, expected_inflation, maturity
    )

    # Bin edges in decimal
    bin_edges = np.array(settings.bin_edges) / 100.0

    # Integrate over bins
    n_probs = density_to_bin_probabilities(fine_strikes, n_density, bin_edges)
    q_probs = density_to_bin_probabilities(fine_strikes, q_density, bin_edges)

    n_dist = InflationDistribution(
        date=date,
        region=region,
        horizon=maturity,
        measure="N",
        bin_probabilities=n_probs,
    )

    q_dist = InflationDistribution(
        date=date,
        region=region,
        horizon=maturity,
        measure="Q",
        bin_probabilities=q_probs,
    )

    return n_dist, q_dist
