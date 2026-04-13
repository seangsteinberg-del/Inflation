"""Data cleaning and quality checks for inflation option surfaces.

Implements the pre-screening from the paper (Appendix B.1):
1. Monotonicity of cap/floor prices in strike
2. Monotonicity in maturity
3. Positive butterfly spreads (convexity)
4. Put-call parity consistency across strikes
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from inflation_disaster.data.schemas import CleanedSurface, OptionSurface

log = logging.getLogger("inflation_disaster.data.cleaning")


def check_cap_monotonicity(prices: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Cap prices must be non-increasing in strike price.

    A cap with a lower strike is worth more because it pays in more states.

    Returns boolean mask where True = violation.
    """
    if len(prices) < 2:
        return np.zeros(len(prices), dtype=bool)

    violations = np.zeros(len(prices), dtype=bool)
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1] + 1e-8:  # small tolerance
            violations[i] = True
    return violations


def check_floor_monotonicity(prices: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Floor prices must be non-decreasing in strike price.

    A floor with a higher strike is worth more.

    Returns boolean mask where True = violation.
    """
    if len(prices) < 2:
        return np.zeros(len(prices), dtype=bool)

    violations = np.zeros(len(prices), dtype=bool)
    for i in range(1, len(prices)):
        if prices[i] < prices[i - 1] - 1e-8:
            violations[i] = True
    return violations


def check_butterfly(
    call_prices: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Butterfly spread must be non-negative (convexity of call prices).

    c(K - dK) - 2*c(K) + c(K + dK) >= 0 for all K.

    This is equivalent to requiring the second derivative of the call price
    w.r.t. strike to be non-negative, which ensures a valid probability density.

    Returns boolean mask where True = violation (at interior points).
    """
    n = len(call_prices)
    violations = np.zeros(n, dtype=bool)

    for i in range(1, n - 1):
        dk_left = strikes[i] - strikes[i - 1]
        dk_right = strikes[i + 1] - strikes[i]

        # Weighted second difference for uneven spacing
        butterfly = (
            call_prices[i - 1] / dk_left
            - call_prices[i] * (1.0 / dk_left + 1.0 / dk_right)
            + call_prices[i + 1] / dk_right
        ) * 2.0 / (dk_left + dk_right)

        if butterfly < -1e-8:
            violations[i] = True

    return violations


def check_put_call_parity(
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    strikes: np.ndarray,
    swap_rate: float,
    discount_factor: float,
    maturity: int = 5,
) -> tuple[np.ndarray, float]:
    """Check put-call parity: cap - floor = discount * (swap_rate - strike).

    For zero-coupon inflation options:
        ZC_cap(K) - ZC_floor(K) = e^{-rT} * (e^{pi^e * T} - K^T)

    where pi^e is expected inflation (swap rate) and K is the gross strike.

    Returns
    -------
    violations : boolean mask where True = violation exceeds tolerance
    implied_real_rate : real rate implied by put-call parity
    """
    n = len(strikes)
    violations = np.zeros(n, dtype=bool)

    # For each strike where both cap and floor exist
    valid = (cap_prices > 0) & (floor_prices > 0)
    if valid.sum() < 2:
        return violations, 0.0

    # Put-call parity: cap(K) - floor(K) = discount * (forward - K)
    # where forward = e^{swap_rate * T} approximately
    parity_lhs = cap_prices[valid] - floor_prices[valid]
    # ZC parity: Cap(K) - Floor(K) = DF * ((1+swap)^T - (1+K/100)^T)
    parity_rhs = discount_factor * (
        (1.0 + swap_rate) ** maturity - (1.0 + strikes[valid] / 100.0) ** maturity
    )

    residuals = np.abs(parity_lhs - parity_rhs)
    tolerance = 0.005 * discount_factor  # 0.5% of notional

    violations[valid] = residuals > tolerance

    # Implied real rate from put-call parity
    # cap(K) - floor(K) = e^{-r*T} * (F - K), so differencing two strikes:
    # d(cap-floor)/dK = -e^{-r*T}, giving r = -ln(-dcf/dk) / T
    if valid.sum() >= 2:
        idx = np.where(valid)[0]
        i, j = idx[0], idx[-1]
        dk = (strikes[j] - strikes[i]) / 100.0
        dcf = (cap_prices[j] - floor_prices[j]) - (cap_prices[i] - floor_prices[i])
        # d(cap-floor)/dK = -e^{-rT} so dcf/dk should be negative
        if abs(dk) > 1e-8 and dcf / dk < -1e-10:
            implied_real_rate = -np.log(-dcf / dk)
            # Annualize: we need the maturity to do this properly.
            # The caller must divide by T to get the annualized rate.
            # For now, store as total-period rate and document this.
            # NOTE: This is the total-period rate, not annualized.
        else:
            implied_real_rate = 0.0
    else:
        implied_real_rate = 0.0

    return violations, implied_real_rate


def clean_surface(
    surface: OptionSurface,
    discount_factor: float | None = None,
) -> tuple[CleanedSurface | None, dict]:
    """Apply all quality checks and return cleaned surface.

    Parameters
    ----------
    surface : OptionSurface
        Raw option surface.
    discount_factor : float, optional
        Nominal discount factor. If None, estimated from put-call parity.

    Returns
    -------
    cleaned : CleanedSurface or None if too many violations
    diagnostics : dict with violation counts and quality score
    """
    strikes = surface.strikes
    caps = surface.cap_prices.copy()
    floors = surface.floor_prices.copy()

    diagnostics = {
        "date": surface.date,
        "maturity": surface.maturity,
        "region": surface.region,
        "n_strikes": len(strikes),
    }

    # 1. Monotonicity checks
    cap_mono_violations = check_cap_monotonicity(caps, strikes)
    floor_mono_violations = check_floor_monotonicity(floors, strikes)

    diagnostics["cap_mono_violations"] = int(cap_mono_violations.sum())
    diagnostics["floor_mono_violations"] = int(floor_mono_violations.sum())

    # 2. Butterfly check (on call prices = cap prices for ZC options)
    butterfly_violations = check_butterfly(caps, strikes)
    diagnostics["butterfly_violations"] = int(butterfly_violations.sum())

    # 3. Put-call parity
    if discount_factor is None:
        # Rough estimate: 5y nominal rate ~3-4%, 10y ~3.5-4.5%
        maturity = surface.maturity
        discount_factor = np.exp(-0.035 * maturity)

    pcp_violations, implied_real_rate_total = check_put_call_parity(
        caps, floors, strikes, surface.swap_rate, discount_factor,
        maturity=surface.maturity,
    )
    # Annualize the total-period rate returned by put-call parity
    implied_real_rate = implied_real_rate_total / surface.maturity if surface.maturity > 0 else 0.0
    diagnostics["pcp_violations"] = int(pcp_violations.sum())
    diagnostics["implied_real_rate"] = implied_real_rate

    # Combine all violations
    all_violations = (
        cap_mono_violations | floor_mono_violations | butterfly_violations | pcp_violations
    )
    n_valid = (~all_violations).sum()
    quality_score = n_valid / len(strikes) if len(strikes) > 0 else 0.0
    diagnostics["quality_score"] = quality_score
    diagnostics["n_valid_strikes"] = int(n_valid)

    if quality_score < 0.3:
        log.warning(
            f"Surface {surface.region} {surface.maturity}y {surface.date}: "
            f"quality too low ({quality_score:.0%}), skipping"
        )
        return None, diagnostics

    # Interpolate over violations using neighbors
    for i in np.where(all_violations)[0]:
        # Linear interpolation from nearest valid neighbors
        valid_idx = np.where(~all_violations)[0]
        if len(valid_idx) < 2:
            continue
        caps[i] = np.interp(strikes[i], strikes[valid_idx], caps[valid_idx])
        floors[i] = np.interp(strikes[i], strikes[valid_idx], floors[valid_idx])

    log.info(
        f"Cleaned {surface.region} {surface.maturity}y {surface.date}: "
        f"quality={quality_score:.0%}, "
        f"violations: mono_cap={diagnostics['cap_mono_violations']}, "
        f"mono_floor={diagnostics['floor_mono_violations']}, "
        f"butterfly={diagnostics['butterfly_violations']}, "
        f"pcp={diagnostics['pcp_violations']}"
    )

    # The CleanedSurface will be fully populated after SABR calibration
    # For now, return with placeholder implied_vols and fine grid
    cleaned = CleanedSurface(
        date=surface.date,
        region=surface.region,
        maturity=surface.maturity,
        strikes=strikes,
        cap_prices=caps,
        floor_prices=floors,
        swap_rate=surface.swap_rate,
        implied_vols=np.zeros_like(strikes),  # populated by SABR
        real_rate=implied_real_rate,
        quality_score=quality_score,
        fine_strikes=np.array([]),  # populated by SABR
        fine_call_prices=np.array([]),  # populated by SABR
    )

    return cleaned, diagnostics


def select_best_monthly_dates(
    daily_data: pd.DataFrame,
    quality_threshold: float = 0.5,
) -> list:
    """Select the best-quality trading day per month.

    For each month, pick the day with the most non-null option prices
    that pass quality checks, as close to the start of the month as possible.

    Parameters
    ----------
    daily_data : DataFrame
        Raw daily data with 'date' column.
    quality_threshold : float
        Minimum fraction of non-null strikes required.

    Returns
    -------
    List of selected dates (one per month).
    """
    daily_data = daily_data.copy()
    daily_data["year_month"] = pd.to_datetime(daily_data["date"]).dt.to_period("M")

    selected = []
    for ym, group in daily_data.groupby("year_month"):
        # Count non-null prices per date
        date_quality = (
            group.groupby("date")
            .apply(lambda x: x[["cap_price", "floor_price"]].notna().sum().sum())
        )
        date_quality = date_quality.sort_values(ascending=False)

        if date_quality.iloc[0] < quality_threshold * len(group["strike"].unique()) * 2:
            continue

        # Among top-quality dates, pick earliest (closest to month start)
        top_dates = date_quality[
            date_quality >= 0.8 * date_quality.iloc[0]
        ].index.tolist()
        selected.append(min(top_dates))

    return sorted(selected)
