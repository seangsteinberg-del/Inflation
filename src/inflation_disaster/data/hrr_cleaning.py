"""Pre-cleaning screens for inflation cap/floor data — HRR Internet Appendix A.1.

Paper (Appendix A.1, verbatim):

  "We only use data if it passes the following requirements:
   (1) cap and floor premia are monotonic in the strike price,
   (2) cap and floor premia increase monotonically with maturity,
   (3) butterfly spreads, which represent one way of constructing nominal
       Arrow-Debreu security payoffs, have positive prices, and
   (4) the put-call parity implied real rates are consistent across
       strike prices."

Each screen returns a diagnostic that flags which strikes/maturities pass
and which fail. The pipeline applies the screens to the raw paste data
before handing off to SABR smoothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("inflation_disaster.data.hrr_cleaning")


@dataclass
class ScreenResult:
    passed: bool
    failures: list[str]
    details: dict


def screen_1_monotone_in_strike(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    tolerance: float = 1e-6,
) -> ScreenResult:
    """Screen 1: cap premia MONOTONE DECREASING in K, floor premia MONOTONE INCREASING in K.

    Paper text: "cap and floor premia are monotonic in the strike price".
    Cap premium falls as strike rises (OTM -> further OTM = cheaper).
    Floor premium rises as strike rises (OTM -> closer to ATM = dearer).
    """
    strikes = np.asarray(strikes)
    order = np.argsort(strikes)
    k = strikes[order]
    c = np.asarray(cap_prices)[order]
    f = np.asarray(floor_prices)[order]

    failures = []
    cap_diffs = np.diff(c)
    floor_diffs = np.diff(f)

    cap_ok = np.all(cap_diffs <= tolerance)
    floor_ok = np.all(floor_diffs >= -tolerance)

    if not cap_ok:
        bad_idx = np.where(cap_diffs > tolerance)[0]
        failures.append(
            f"cap non-monotone at strike pairs: {[(k[i]*100, k[i+1]*100) for i in bad_idx]}"
        )
    if not floor_ok:
        bad_idx = np.where(floor_diffs < -tolerance)[0]
        failures.append(
            f"floor non-monotone at strike pairs: {[(k[i]*100, k[i+1]*100) for i in bad_idx]}"
        )

    return ScreenResult(
        passed=cap_ok and floor_ok,
        failures=failures,
        details={"cap_diffs": cap_diffs, "floor_diffs": floor_diffs},
    )


def screen_2_monotone_in_maturity(
    surfaces_by_T: dict,
    tolerance: float = 1e-6,
) -> ScreenResult:
    """Screen 2: cap and floor premia MONOTONE INCREASING in maturity.

    Paper text: "cap and floor premia increase monotonically with maturity".
    Longer maturity -> more uncertainty -> higher option premium.

    Parameters
    ----------
    surfaces_by_T : dict {T: (strikes, cap_prices, floor_prices)}
    """
    failures = []
    tenors = sorted(surfaces_by_T.keys())

    # Check at each common strike
    # Find strikes common to all tenors
    common_strikes = None
    for T in tenors:
        s = np.asarray(surfaces_by_T[T][0])
        if common_strikes is None:
            common_strikes = set(s.tolist())
        else:
            common_strikes &= set(s.tolist())

    common_strikes = sorted(common_strikes) if common_strikes else []

    for K in common_strikes:
        caps_at_K = []
        floors_at_K = []
        for T in tenors:
            strikes, caps, floors = surfaces_by_T[T]
            idx = int(np.where(np.asarray(strikes) == K)[0][0])
            caps_at_K.append(caps[idx])
            floors_at_K.append(floors[idx])
        cap_diffs = np.diff(caps_at_K)
        floor_diffs = np.diff(floors_at_K)
        if np.any(cap_diffs < -tolerance):
            failures.append(f"cap non-monotone in T at K={K*100:.2f}%: {caps_at_K}")
        if np.any(floor_diffs < -tolerance):
            failures.append(f"floor non-monotone in T at K={K*100:.2f}%: {floors_at_K}")

    return ScreenResult(
        passed=len(failures) == 0,
        failures=failures,
        details={"common_strikes": common_strikes, "tenors": tenors},
    )


def screen_3_positive_butterflies(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    tolerance: float = 1e-8,
) -> ScreenResult:
    """Screen 3: butterfly spreads non-negative (Arrow-Debreu prices >= 0).

    Paper text: "butterfly spreads, which represent one way of constructing
    nominal Arrow-Debreu security payoffs, have positive prices."

    For three equi-spaced strikes K_{i-1}, K_i, K_{i+1}:
        butterfly = cap(K_{i-1}) - 2*cap(K_i) + cap(K_{i+1}) >= 0
    """
    strikes = np.asarray(strikes)
    order = np.argsort(strikes)
    k = strikes[order]
    c = np.asarray(cap_prices)[order]

    failures = []
    butterflies = []
    for i in range(1, len(k) - 1):
        # Only evaluate if strikes are roughly equispaced (within 10% tolerance)
        dk_left = k[i] - k[i - 1]
        dk_right = k[i + 1] - k[i]
        if abs(dk_left - dk_right) > 0.1 * max(dk_left, dk_right):
            continue
        bfly = c[i - 1] - 2.0 * c[i] + c[i + 1]
        butterflies.append((k[i], bfly))
        if bfly < -tolerance:
            failures.append(
                f"butterfly negative at K={k[i]*100:.2f}%: {bfly:.6f}"
            )

    return ScreenResult(
        passed=len(failures) == 0,
        failures=failures,
        details={"butterflies": butterflies},
    )


def screen_4_real_rate_consistent(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    swap_rate: float,
    maturity: int,
    max_dispersion_bp: float = 30.0,
) -> ScreenResult:
    """Screen 4: put-call parity implied real rates consistent across K.

    Paper text: "the put-call parity implied real rates are consistent
    across strike prices."

    Computes the real rate implied at each strike. Flags if the standard
    deviation across strikes exceeds `max_dispersion_bp` basis points.
    """
    from inflation_disaster.pricing.put_call_parity import extract_real_rate

    try:
        res = extract_real_rate(strikes, cap_prices, floor_prices,
                                swap_rate, maturity)
    except ValueError as e:
        return ScreenResult(passed=False, failures=[str(e)], details={})

    dispersion_bp = res.cross_strike_dispersion * 10_000
    passed = dispersion_bp <= max_dispersion_bp
    failures = []
    if not passed:
        failures.append(
            f"real-rate dispersion {dispersion_bp:.1f} bp > {max_dispersion_bp} bp threshold"
        )
    return ScreenResult(
        passed=passed,
        failures=failures,
        details={"dispersion_bp": dispersion_bp,
                 "per_strike_real_rate": res.per_strike_real_rate,
                 "median_real_rate": res.real_rate},
    )


def apply_all_screens(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    swap_rate: float,
    maturity: int,
    all_maturity_surfaces: dict | None = None,
    max_real_rate_dispersion_bp: float = 30.0,
) -> dict:
    """Run all 4 screens on a single (region, T) surface and return a summary.

    Parameters
    ----------
    all_maturity_surfaces : optional dict for screen 2; if None, screen 2 is skipped.
    """
    results = {
        "screen_1_monotone_K": screen_1_monotone_in_strike(strikes, cap_prices, floor_prices),
        "screen_3_butterfly": screen_3_positive_butterflies(strikes, cap_prices),
        "screen_4_real_rate": screen_4_real_rate_consistent(
            strikes, cap_prices, floor_prices, swap_rate, maturity,
            max_dispersion_bp=max_real_rate_dispersion_bp,
        ),
    }
    if all_maturity_surfaces is not None:
        results["screen_2_monotone_T"] = screen_2_monotone_in_maturity(all_maturity_surfaces)
    return results
