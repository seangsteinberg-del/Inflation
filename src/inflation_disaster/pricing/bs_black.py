"""Standard Black-Scholes (1973) lognormal option pricing and IV inversion.

Paper Appendix A.2, verbatim:
  "We then transform the data and calculate Black and Scholes (1973) implied
   volatilities. This nonlinear transformation makes it easier to adjust for
   data inaccuracies and errors."

The ZC inflation cap payoff at maturity T:
    max((1 + y_realized)^T - (1 + K)^T, 0)

is priced under lognormal dynamics on the CPI ratio F = (1+y_swap)^T with
strike K_eff = (1+K)^T:

    cap = DF * [ F * N(d1) - K_eff * N(d2) ]
    d1  = [ln(F/K_eff) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
    d2  = d1 - sigma * sqrt(T)

This is the paper's "Black-Scholes implied volatility" convention:
lognormal in the CPI ratio, T as time-to-maturity, DF the nominal discount
factor.

The floor payoff max((1+K)^T - (1+y_realized)^T, 0) prices via put-call parity:
    floor = cap - DF * (F - K_eff)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_cap_price(
    swap_rate: float, strike: float, maturity: float,
    sigma: float, discount_factor: float,
) -> float:
    """Black-Scholes ZC cap price (lognormal on CPI ratio)."""
    F = (1.0 + swap_rate) ** maturity
    K = (1.0 + strike) ** maturity
    if sigma <= 0 or maturity <= 0:
        return discount_factor * max(F - K, 0.0)
    sqrt_T = np.sqrt(maturity)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_floor_price(
    swap_rate: float, strike: float, maturity: float,
    sigma: float, discount_factor: float,
) -> float:
    """Black-Scholes ZC floor price (lognormal on CPI ratio)."""
    F = (1.0 + swap_rate) ** maturity
    K = (1.0 + strike) ** maturity
    if sigma <= 0 or maturity <= 0:
        return discount_factor * max(K - F, 0.0)
    sqrt_T = np.sqrt(maturity)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bs_implied_vol(
    price: float, swap_rate: float, strike: float, maturity: float,
    discount_factor: float, is_cap: bool = True,
    vol_lo: float = 1e-6, vol_hi: float = 5.0,
) -> float | None:
    """Invert Black-Scholes for lognormal implied vol.

    Returns None if price is below intrinsic or Brent fails to converge.
    """
    F = (1.0 + swap_rate) ** maturity
    K = (1.0 + strike) ** maturity
    intrinsic = discount_factor * (max(F - K, 0.0) if is_cap else max(K - F, 0.0))
    if price < intrinsic - 1e-10:
        return None
    if price <= intrinsic + 1e-12:
        return 0.0

    pricer = bs_cap_price if is_cap else bs_floor_price

    def f(sigma):
        return pricer(swap_rate, strike, maturity, sigma, discount_factor) - price

    try:
        return float(brentq(f, vol_lo, vol_hi, xtol=1e-8, maxiter=200))
    except ValueError:
        return None


def invert_surface_to_vols(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    swap_rate: float,
    maturity: int,
    discount_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert an entire cap/floor price grid to BS lognormal implied vols.

    Returns (cap_ivols, floor_ivols) as 1-D arrays aligned with `strikes`.
    NaN where inversion fails (price below intrinsic or no convergence).
    """
    strikes = np.asarray(strikes, dtype=float)
    cap_ivols = np.full(len(strikes), np.nan)
    floor_ivols = np.full(len(strikes), np.nan)
    for i, K in enumerate(strikes):
        if i < len(cap_prices) and cap_prices[i] > 0:
            v = bs_implied_vol(cap_prices[i], swap_rate, K, maturity,
                               discount_factor, is_cap=True)
            if v is not None:
                cap_ivols[i] = v
        if i < len(floor_prices) and floor_prices[i] > 0:
            v = bs_implied_vol(floor_prices[i], swap_rate, K, maturity,
                               discount_factor, is_cap=False)
            if v is not None:
                floor_ivols[i] = v
    return cap_ivols, floor_ivols
