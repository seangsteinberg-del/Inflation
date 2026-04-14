"""Surface builder: wires cleaning -> SABR calibration -> fine grid prices.

This module bridges the gap between raw/cleaned option surfaces and the
density extraction pipeline. It takes a CleanedSurface with zero implied_vols
and populates the SABR-smoothed vols and fine-grid call prices.
"""

from __future__ import annotations

import logging

import numpy as np

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import CleanedSurface, OptionSurface
from inflation_disaster.data.cleaning import clean_surface
from inflation_disaster.models.sabr import calibrate_sabr, sabr_smile, sabr_to_prices, black_price
from inflation_disaster.utils.numerical import make_fine_grid

log = logging.getLogger("inflation_disaster.data.surface_builder")


def build_ready_surface(
    surface: OptionSurface,
    discount_factor: float | None = None,
) -> CleanedSurface | None:
    """Full pipeline: raw surface -> cleaned -> SABR-calibrated -> fine grid prices.

    This is the single function that takes raw Bloomberg data and produces
    a surface ready for Breeden-Litzenberger density extraction.

    Parameters
    ----------
    surface : OptionSurface
        Raw option surface from Bloomberg.
    discount_factor : float, optional
        Nominal discount factor. If None, estimated.

    Returns
    -------
    CleanedSurface with populated implied_vols, fine_strikes, fine_call_prices.
    None if surface fails quality checks.
    """
    # Step 1: Clean
    cleaned, diagnostics = clean_surface(surface, discount_factor)
    if cleaned is None:
        return None

    # Step 2: Compute implied vols from cleaned cap prices
    # Use Black formula inversion via bisection
    forward_gross = 1.0 + surface.swap_rate
    T = surface.maturity
    df = np.exp(-cleaned.real_rate * T) if cleaned.real_rate != 0 else (discount_factor or 0.95)

    strikes_gross = cleaned.strikes / 100.0 + 1.0
    implied_vols = np.zeros(len(cleaned.strikes))

    for i, (k, price) in enumerate(zip(strikes_gross, cleaned.cap_prices)):
        if price > 1e-8 and k > 0:
            vol = _implied_vol_bisection(forward_gross, k, T, price, df)
            implied_vols[i] = vol

    # Step 3: SABR calibration
    valid = implied_vols > 1e-6
    if valid.sum() >= 3:
        sabr_params = calibrate_sabr(
            forward_gross, strikes_gross[valid], implied_vols[valid], T,
            beta=settings.sabr_beta,
        )
    else:
        log.warning(f"Only {valid.sum()} valid vols for SABR calibration")
        from inflation_disaster.data.schemas import SABRParams
        med_vol = np.median(implied_vols[valid]) if valid.sum() > 0 else 0.01
        sabr_params = SABRParams(
            alpha=med_vol * forward_gross ** (1 - settings.sabr_beta),
            beta=settings.sabr_beta, rho=0.0, nu=0.3,
        )

    # Step 4: Reconstruct on fine grid
    fine_rate_grid = make_fine_grid()
    fine_gross_grid = 1.0 + fine_rate_grid
    fine_call_prices = sabr_to_prices(
        forward_gross, fine_gross_grid, T, sabr_params, df,
    )

    # Step 5: SABR smile on original strike grid (for diagnostics)
    sabr_vols_on_grid = sabr_smile(forward_gross, strikes_gross, T, sabr_params)

    # Build the final CleanedSurface
    return CleanedSurface(
        date=cleaned.date,
        region=cleaned.region,
        maturity=cleaned.maturity,
        strikes=cleaned.strikes,
        cap_prices=cleaned.cap_prices,
        floor_prices=cleaned.floor_prices,
        swap_rate=cleaned.swap_rate,
        implied_vols=sabr_vols_on_grid,
        real_rate=cleaned.real_rate,
        quality_score=cleaned.quality_score,
        fine_strikes=fine_rate_grid,
        fine_call_prices=fine_call_prices,
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
    """Invert Black-76 to get implied vol via bisection."""
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


def determine_inflation_state(
    current_annual_inflation: float,
) -> np.ndarray:
    """Convert a current annual inflation rate to an 8-state probability vector.

    Parameters
    ----------
    current_annual_inflation : float
        Current YoY inflation in percent (e.g. 3.2 for 3.2%).

    Returns
    -------
    state : array of shape (8,)
        One-hot vector indicating which bin current inflation falls in.
    """
    edges = list(settings.bin_edges)
    state = np.zeros(8)
    for i in range(8):
        if edges[i] <= current_annual_inflation < edges[i + 1]:
            state[i] = 1.0
            return state
    # Fallback: if above all edges, put in last bin
    if current_annual_inflation >= edges[-1]:
        state[7] = 1.0
    else:
        state[0] = 1.0
    return state
