"""End-to-end ZC cap/floor paste -> Q-density -> 5y5y forward -> P-measure.

Full HRR 2024 replication path, Bloomberg-only, from the SWIL ZC Volatility
paste:

  1. Parse cap + floor SLN vol matrices (SWIL tab 5).
  2. Price each (K, T) to a ZC cap price via shifted-lognormal Black-76
     on the CPI ratio. Combined cap+floor grid via put-call parity.
  3. Apply Breeden-Litzenberger (eq. 10) to get Q-density of average
     annualized inflation at 5Y and 10Y.
  4. Integrate Q-density over 8 bins to get bin distributions.
  5. Pull live YOY cap/floor data + swap rates + nominal rates from
     Bloomberg via pull_live_prices().
  6. Extract forward 1Y annual distribution (years 6-10) from YOY caplet
     stripping.
  7. Calibrate 8-state Markov chain to {cum_5y, cum_10y, fwd_avg_6_10,
     swap-rate means}.
  8. Simulate 5y5y forward distribution, extract Q-probabilities.
  9. Apply paper's P/Q risk adjustment (0.66 high, 0.96 low; HRR Table 1).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import logging
from datetime import date

import numpy as np
from scipy.interpolate import CubicSpline

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("run_zc_paste")


# ---------------------------------------------------------------------------
# Paper's Q -> P risk-adjustment multipliers (HRR 2024 Table 1, p. 33).
# Derived from Barro/JST disaster calibration with Epstein-Zin RRA=3,
# Pareto tail sizes xi_h=5.45, xi_l=15.18.
# ---------------------------------------------------------------------------
RISK_ADJ_HIGH = 0.66   # P(>4%) / Q(>4%)
RISK_ADJ_LOW = 0.96    # P(<0%) / Q(<0%)


BIN_EDGES = np.array([-np.inf, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, np.inf])
BIN_LABELS = ["<=-1", "(-1,0]", "(0,1]", "(1,2]", "(2,3]", "(3,4]", "(4,5]", ">5%"]


def q_density_from_zc_surface(
    surface, nominal_rate: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Breeden-Litzenberger on a ZC cap surface (HRR eq. 10).

    Returns (fine_grid_annual, q_density, {cdf, bin_dist, prob_queries}).
    """
    strikes_dec = np.asarray(surface.strikes) / 100.0
    cap_prices = np.asarray(surface.cap_prices)
    T = surface.maturity
    swap = surface.swap_rate
    real_rate = nominal_rate - swap  # breakeven

    cs = CubicSpline(strikes_dec, cap_prices, bc_type="natural", extrapolate=False)
    fine_k = np.linspace(strikes_dec[0], strikes_dec[-1], 4000)
    fine_cap = cs(fine_k)
    ok = ~np.isnan(fine_cap)
    fine_k, fine_cap = fine_k[ok], fine_cap[ok]

    # 2nd derivative w.r.t. annual strike
    d1_ = np.gradient(fine_cap, fine_k)
    d2_ = np.gradient(d1_, fine_k)

    # Q-density (annual-rate space) with Jacobian cumulative->annual
    jacobian = T * (1.0 + fine_k) ** (T - 1)
    q_density = np.exp(real_rate * T) * d2_ * jacobian
    q_density = np.maximum(q_density, 0.0)

    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total = trap(q_density, fine_k)
    if total > 0:
        q_density /= total

    # CDF
    cdf = np.concatenate([[0.0], np.cumsum(
        (q_density[:-1] + q_density[1:]) / 2 * np.diff(fine_k)
    )])
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    # 8-bin integration
    bin_dist = np.zeros(8)
    for i in range(8):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        mask = (fine_k > lo) & (fine_k <= hi)
        if mask.sum() > 1:
            bin_dist[i] = trap(q_density[mask], fine_k[mask])
    # Extrapolate tails beyond fine grid range using edge mass
    left_missing = max(0.0, 1.0 - bin_dist.sum())
    # Put any missing mass in the outermost bins proportionally
    if left_missing > 0:
        # Check if more mass lies below or above the grid
        below = strikes_dec[0] > BIN_EDGES[1]  # grid doesn't reach -1%
        above = strikes_dec[-1] < BIN_EDGES[-2]  # grid doesn't reach 5%
        if below:
            bin_dist[0] += left_missing / 2
        if above:
            bin_dist[-1] += left_missing / 2
        if not (below or above):
            bin_dist /= bin_dist.sum()  # renormalize noise
    bin_dist /= bin_dist.sum()

    def p_above(thr):
        idx = np.searchsorted(fine_k, thr)
        return float(1.0 - cdf[idx]) if idx < len(cdf) else 0.0

    def p_below(thr):
        idx = np.searchsorted(fine_k, thr)
        return float(cdf[idx - 1]) if idx > 0 else 0.0

    extras = {
        "cdf": cdf,
        "bin_dist": bin_dist,
        "p_below_0": p_below(0.0),
        "p_below_1": p_below(0.01),
        "p_above_3": p_above(0.03),
        "p_above_4": p_above(0.04),
        "p_above_5": p_above(0.05),
        "swap_rate": swap,
        "real_rate": real_rate,
        "nominal_rate": nominal_rate,
    }
    return fine_k, q_density, extras


def main():
    from inflation_disaster.data.zc_paste import (
        build_combined_surface,
        SAMPLE_CAP_PASTE_US_2026_04_21,
        SAMPLE_FLOOR_PASTE_US_2026_04_21,
    )
    from inflation_disaster.data.surface_builder import determine_inflation_state
    from inflation_disaster.models.markov_chain import (
        build_transition_matrix, simulate_forward_distribution,
    )

    # Import live Bloomberg fetch + YOY extraction + Markov calibration
    from run_live import (
        pull_live_prices, extract_forward_annual_distribution,
        calibrate_markov_from_market, _mc_forward_probs,
    )
    from inflation_disaster.adjustments.horizon_adj import extract_disaster_probabilities

    print("=" * 70)
    print("ZC-PASTE PIPELINE (HRR 2024, Bloomberg-only)")
    print(f"Date: {date.today()}")
    print("=" * 70)

    # --- Step 0: pull live Bloomberg data ---
    print("\nSTEP 0: Pulling live Bloomberg (swaps, nominals, CPI, YOY caps/floors)")
    data = pull_live_prices()

    swap_rates = {
        1: 0.031275, 2: 0.028049, 3: 0.026683,
        5: 0.025610, 7: 0.02501, 10: 0.024680,
        12: 0.02458, 15: 0.02436, 20: 0.02408, 30: 0.02373,
    }
    # Overwrite with live values where available
    for (region, mat), sr in data["swap_rates"].items():
        if region == "US":
            swap_rates[mat] = sr
    nominal_rates = {
        mat: rate for (region, mat), rate in data["nominal_rates"].items()
        if region == "US"
    }
    print(f"  US swap rates: {[f'{m}Y={swap_rates.get(m,0):.3%}' for m in [1,2,3,5,7,10]]}")
    print(f"  US nominal:    {[f'{m}Y={r:.3%}' for m, r in nominal_rates.items()]}")
    print(f"  US CPI YoY:    {data['cpi_us']:.1f}%")

    # --- Step 1: ZC surfaces from paste + B-L at 5Y and 10Y ---
    print("\nSTEP 1: ZC cap+floor surfaces from paste -> Breeden-Litzenberger")
    surfaces = build_combined_surface(
        cap_paste=SAMPLE_CAP_PASTE_US_2026_04_21,
        floor_paste=SAMPLE_FLOOR_PASTE_US_2026_04_21,
        region="US",
        valuation_date=date.today(),
        swap_rates=swap_rates,
        nominal_rates=nominal_rates,
        value_type="vol",
        input_units="percent",
        shift=0.0,
        target_tenors=[5, 10],
    )

    zc_results = {}
    for T in [5, 10]:
        r_nom = nominal_rates.get(T, 0.04)
        _, _, ext = q_density_from_zc_surface(surfaces[T], r_nom)
        zc_results[T] = ext
        print(f"\n  {T}Y  swap={ext['swap_rate']:.3%}  real={ext['real_rate']:.3%}  "
              f"nominal={ext['nominal_rate']:.3%}")
        print(f"       bins: {' '.join(f'{p:5.1%}' for p in ext['bin_dist'])}")
        print(f"       labels:{' '.join(f'{b:>5s}' for b in BIN_LABELS)}")
        print(f"       Q(<0%)={ext['p_below_0']:.2%}  "
              f"Q(<1%)={ext['p_below_1']:.2%}  "
              f"Q(>3%)={ext['p_above_3']:.2%}  "
              f"Q(>4%)={ext['p_above_4']:.2%}  "
              f"Q(>5%)={ext['p_above_5']:.2%}")

    # --- Step 2: Forward 1Y annual distribution from YOY caplet stripping ---
    print("\nSTEP 2: Forward 1Y annual distribution (yr 6-10) from YOY caplets")
    fwd_dist = extract_forward_annual_distribution(
        data["us_caps"], data["us_floors"], data["swap_rates"], data["nominal_rates"],
        region="US",
    )
    if fwd_dist is not None:
        print(f"  Forward yr 6-10: {' '.join(f'{p:5.1%}' for p in fwd_dist)}")
    else:
        print("  (no forward YOY data available)")

    # --- Step 3: Markov chain calibrated to ZC cumulatives + YOY forward ---
    print("\nSTEP 3: Markov calibration (ZC 5Y + ZC 10Y + YOY forward)")
    cpi = data["cpi_us"]
    initial_state = determine_inflation_state(cpi)
    print(f"  CPI YoY: {cpi:.1f}% -> initial bin {np.argmax(initial_state)}")

    # The existing calibrate_markov_from_market fits `init @ P^T` to the
    # target. We feed it our ZC-derived cumulative bins at 5Y and 10Y as
    # the target distributions. This approximates the cumulative with the
    # marginal (OK if chain mixes fast; HRR's actual method is GMM with
    # path-averaged moments, which we could swap in later for more fidelity).
    annual_dists_as_cumul = {
        5: zc_results[5]["bin_dist"],
        10: zc_results[10]["bin_dist"],
    }
    markov_params = calibrate_markov_from_market(
        initial_state, "US",
        annual_dists_as_cumul, fwd_dist,
        data["swap_rates"], data["nominal_rates"],
    )
    print(f"  p_dh={markov_params.p_dh:.4f}  p_dl={markov_params.p_dl:.4f}  "
          f"p_nn={markov_params.p_nn:.4f}")
    print(f"  p_h ={markov_params.p_h:.4f}  p_l ={markov_params.p_l:.4f}  "
          f"p_mr={markov_params.p_mr:.4f}")

    # --- Step 4: 5Y5Y forward Q-probabilities ---
    print("\nSTEP 4: 5Y5Y forward Q-probs (5 seeds x 500K MC)")
    q_5y5y_high, q_5y5y_low = _mc_forward_probs(
        markov_params, initial_state, n_seeds=5, n_paths=500_000,
    )
    print(f"  Q(>4%, 5y5y) = {q_5y5y_high:.3%}")
    print(f"  Q(<0%, 5y5y) = {q_5y5y_low:.3%}")

    # --- Step 5: Risk adjustment Q -> P (paper Table 1) ---
    print("\nSTEP 5: Risk adjustment Q->P  (HRR Table 1: 0.66 high, 0.96 low)")
    p_5y5y_high = q_5y5y_high * RISK_ADJ_HIGH
    p_5y5y_low = q_5y5y_low * RISK_ADJ_LOW
    print(f"  P(>4%, 5y5y) = {p_5y5y_high:.3%}")
    print(f"  P(<0%, 5y5y) = {p_5y5y_low:.3%}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: US ({date.today()})")
    print("=" * 70)
    print(f"{'Measure':<28s} {'Q':>8s} {'P':>8s}")
    print("-" * 48)

    def row(label, q, p=None):
        pstr = f"{p:>7.2%}" if p is not None else f"{'--':>7s}"
        print(f"  {label:<26s} {q:>7.2%} {pstr}")

    row("Q(avg < 0%, 5y)", zc_results[5]["p_below_0"],
        zc_results[5]["p_below_0"] * RISK_ADJ_LOW)
    row("Q(avg > 4%, 5y)", zc_results[5]["p_above_4"],
        zc_results[5]["p_above_4"] * RISK_ADJ_HIGH)
    row("Q(avg < 0%, 10y)", zc_results[10]["p_below_0"],
        zc_results[10]["p_below_0"] * RISK_ADJ_LOW)
    row("Q(avg > 4%, 10y)", zc_results[10]["p_above_4"],
        zc_results[10]["p_above_4"] * RISK_ADJ_HIGH)
    row("Q(avg > 4%, 5y5y)", q_5y5y_high, p_5y5y_high)
    row("Q(avg < 0%, 5y5y)", q_5y5y_low, p_5y5y_low)
    print("=" * 70)


if __name__ == "__main__":
    main()
