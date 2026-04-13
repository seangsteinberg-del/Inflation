"""SABR stochastic volatility model for inflation options.

Implements the Hagan et al. (2002) closed-form approximation for implied volatility
under the SABR model. Used to smooth the inflation option volatility smile before
extracting risk-neutral densities via Breeden-Litzenberger.

The SABR model dynamics are:
    dF = alpha * F^beta * dW_1
    dalpha = nu * alpha * dW_2
    dW_1 * dW_2 = rho * dt

For inflation options, beta is typically fixed at 0.5.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from inflation_disaster.data.schemas import SABRParams

log = logging.getLogger("inflation_disaster.models.sabr")


def sabr_implied_vol(
    forward: float,
    strike: float,
    T: float,
    params: SABRParams,
) -> float:
    """Hagan et al. (2002) SABR implied volatility approximation.

    Parameters
    ----------
    forward : float
        Forward inflation rate (e.g., 1.02 for 2% expected inflation).
    strike : float
        Strike price (gross inflation rate).
    T : float
        Time to expiry in years.
    params : SABRParams
        SABR model parameters.

    Returns
    -------
    sigma : float
        Black implied volatility.
    """
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu

    # Handle ATM case separately to avoid division by zero
    if abs(forward - strike) < 1e-10:
        return _sabr_atm_vol(forward, T, alpha, beta, rho, nu)

    # Ensure positive values
    if forward <= 0 or strike <= 0:
        return alpha  # fallback

    FK = forward * strike
    FK_beta = FK ** ((1 - beta) / 2)
    log_FK = np.log(forward / strike)

    # z and x(z) from Hagan et al.
    z = (nu / alpha) * FK_beta * log_FK
    sqrt_term = np.sqrt(max(1 - 2 * rho * z + z**2, 0.0))
    arg = (sqrt_term + z - rho) / (1 - rho)
    if arg <= 0:
        return _sabr_atm_vol(forward, T, alpha, beta, rho, nu)
    x_z = np.log(arg)

    if abs(x_z) < 1e-10:
        zeta = 1.0
    else:
        zeta = z / x_z

    # Series expansion terms
    term1 = (1 - beta) ** 2 / 24 * alpha**2 / FK ** (1 - beta)
    term2 = rho * beta * nu * alpha / (4 * FK_beta)
    term3 = (2 - 3 * rho**2) * nu**2 / 24

    # Denominator correction -- single polynomial per Hagan (2002)
    denom = (
        1.0
        + (1 - beta) ** 2 / 24 * log_FK**2
        + (1 - beta) ** 4 / 1920 * log_FK**4
    )

    sigma = (
        alpha
        / (FK_beta * denom)
        * zeta
        * (1 + (term1 + term2 + term3) * T)
    )

    return max(sigma, 1e-6)


def _sabr_atm_vol(
    forward: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """SABR ATM volatility (F = K)."""
    F_beta = forward ** (1 - beta)

    term1 = (1 - beta) ** 2 / 24 * alpha**2 / forward ** (2 * (1 - beta))
    term2 = rho * beta * nu * alpha / (4 * forward ** (1 - beta))
    term3 = (2 - 3 * rho**2) * nu**2 / 24

    sigma = alpha / forward ** (1 - beta) * (1 + (term1 + term2 + term3) * T)
    return max(sigma, 1e-6)


def sabr_smile(
    forward: float,
    strikes: np.ndarray,
    T: float,
    params: SABRParams,
) -> np.ndarray:
    """Compute SABR implied volatility across a range of strikes."""
    return np.array([sabr_implied_vol(forward, k, T, params) for k in strikes])


def calibrate_sabr(
    forward: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    beta: float = 0.5,
    weights: np.ndarray | None = None,
) -> SABRParams:
    """Calibrate SABR parameters to market implied volatilities.

    Parameters
    ----------
    forward : float
        Forward rate.
    strikes : array
        Strike prices.
    market_vols : array
        Market implied volatilities at each strike.
    T : float
        Time to expiry.
    beta : float
        CEV exponent (fixed, default 0.5 for inflation).
    weights : array, optional
        Weights for each strike in the objective. Default: equal weights
        with ATM strikes weighted 2x.

    Returns
    -------
    SABRParams : calibrated parameters.
    """
    if weights is None:
        weights = np.ones(len(strikes))
        # Up-weight ATM strikes
        atm_idx = np.argmin(np.abs(strikes - forward))
        weights[max(0, atm_idx - 1) : atm_idx + 2] *= 2.0
        weights /= weights.sum()

    # Filter out zero or NaN vols
    valid = (market_vols > 1e-6) & np.isfinite(market_vols)
    if valid.sum() < 3:
        log.warning("Fewer than 3 valid vol points, using defaults")
        positive_vols = market_vols[market_vols > 0]
        atm_vol = np.nanmedian(positive_vols) if len(positive_vols) > 0 else 0.01
        if np.isnan(atm_vol):
            atm_vol = 0.01
        return SABRParams(
            alpha=atm_vol * forward ** (1 - beta),
            beta=beta,
            rho=0.0,
            nu=0.3,
        )

    strikes_v = strikes[valid]
    vols_v = market_vols[valid]
    weights_v = weights[valid]
    weights_v /= weights_v.sum()

    def objective(x):
        alpha, rho, nu = x
        params = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        model_vols = sabr_smile(forward, strikes_v, T, params)
        squared_errors = (model_vols - vols_v) ** 2
        return np.sum(weights_v * squared_errors)

    # Initial guess: alpha from ATM vol, rho=0, nu=0.3
    atm_vol = np.interp(forward, strikes_v, vols_v)
    alpha_init = atm_vol * forward ** (1 - beta)

    # Multiple random starts
    best_result = None
    best_cost = np.inf

    starts = [
        (alpha_init, 0.0, 0.3),
        (alpha_init * 0.8, -0.2, 0.4),
        (alpha_init * 1.2, 0.2, 0.2),
        (alpha_init, -0.3, 0.5),
        (alpha_init, 0.3, 0.1),
    ]

    bounds = [
        (1e-6, 2.0),  # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 3.0),  # nu
    ]

    for x0 in starts:
        try:
            result = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        log.warning("SABR calibration failed, using initial guess")
        return SABRParams(alpha=alpha_init, beta=beta, rho=0.0, nu=0.3)

    alpha_opt, rho_opt, nu_opt = best_result.x

    params = SABRParams(alpha=alpha_opt, beta=beta, rho=rho_opt, nu=nu_opt)
    log.debug(
        f"SABR calibrated: alpha={alpha_opt:.4f}, rho={rho_opt:.4f}, "
        f"nu={nu_opt:.4f}, RMSE={np.sqrt(best_cost / len(vols_v)):.6f}"
    )

    return params


def black_price(
    forward: float,
    strike: float,
    T: float,
    sigma: float,
    discount_factor: float,
    is_call: bool = True,
) -> float:
    """Black (1976) option pricing formula.

    Used to convert implied vols back to option prices on a fine strike grid.
    """
    from scipy.stats import norm

    if sigma <= 0 or T <= 0 or strike <= 0 or forward <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return intrinsic * discount_factor

    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def sabr_to_prices(
    forward: float,
    fine_strikes: np.ndarray,
    T: float,
    params: SABRParams,
    discount_factor: float,
) -> np.ndarray:
    """Convert SABR-calibrated smile to call prices on a fine grid.

    Returns array of call (cap) prices for each fine strike.
    """
    vols = sabr_smile(forward, fine_strikes, T, params)
    prices = np.array([
        black_price(forward, k, T, vol, discount_factor, is_call=True)
        for k, vol in zip(fine_strikes, vols)
    ])
    return prices
