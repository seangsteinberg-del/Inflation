"""Forward-starting 1Y YOY caplet/floorlet stripping via Rubinstein (1991).

Paper Appendix A.2, verbatim:
  "For the year-on-year data, we first extract individual caplet and
   floorlet prices from the market prices of caps and floors. We then
   use the Rubinstein (1991) transformation to price forward starting
   options based on their specific option tenor, which is the time
   between reset dates. We discount using the real interest rate which
   is extracted from the put-call parity relationship of the zero coupon
   options (Birru and Figlewski, 2012). For the individual caplet and
   floorlet prices we then follow the same SABR implied volatility
   smoothing procedure with the same constraint that smoothing cannot
   introduce arbitrage opportunities."

Rubinstein (1991), "Pay Now, Choose Later" — the key result:
  For a forward-starting option on an asset whose distribution between
  time t1 and t1+dt depends only on the ratio S_{t1+dt}/S_{t1}, the price
  at t=0 today satisfies:
      Price_forward_starting(K) = DF(0 -> t1) * E_Q[ Black_Scholes(S_{t1}, K, dt, sigma_{t1,t1+dt}) ]

  For a STRIKE expressed as a percentage of S_{t1} (proportional strike):
      V(k, t1, t1+dt) = S_0 * DF_real(t1) * BS(1, k, dt, sigma)

  That is: a forward-starting ATM proportional option on inflation YOY
  is priced as a spot ATM option discounted by the REAL rate from now
  to t1 (because the inflation payoff grows with the CPI).

YOY inflation options in the market:
  A "YOY cap" at strike K, maturity T years, pays SUM of annual caplets:
      V_cap(K, T) = sum_{j=1}^{T} V_caplet(K, j-1, j)
  where V_caplet(K, t1, t1+1) is the forward-starting 1Y YOY caplet
  which pays max((CPI_{t1+1}/CPI_{t1}) - (1+K), 0).

Stripping procedure:
  Given market prices of YOY caps at maturities T1 < T2 < ... < Tn,
  individual forward caplets are recovered by differencing:
      V_caplet(K, t1+1, t2) * (t2 - t1) = V_cap(K, t2) - V_cap(K, t1)

  For consecutive maturities (Tj, Tj+1) with dt=1, V_caplet(K, Tj, Tj+1) is
  a single 1-year forward-starting option. For non-consecutive maturities
  (e.g., US post-2018 data with gaps), this gives the SUM of caplets over
  the missing years.

  The paper averages the 5 annual forward distributions over years t+6
  to t+10 (Sec. 4.3.1: "we take the average of these five annual
  distributions"), so we need their pooled distribution, not per-year.

Pool the pooled prices into per-year-equivalent quantities:
  V_cap(K, 10Y) - V_cap(K, 5Y) = SUM caplet prices for years 6-10
  Average over 5 years: avg_caplet_price = (V_cap_10Y - V_cap_5Y) / 5

  Then each avg caplet is a single-year ATM-ish option with implied
  distribution (under Rubinstein) priced at dt=1 year starting from
  the beginning of year 6. The effective "forward CPI ratio forward"
  is the 5y5y forward rate of ZC inflation:
      F_{5,10} = [(1+y_{0,10})^10 / (1+y_{0,5})^5]^(1/5) = forward annual
  (approximately — exact paper convention may differ slightly).

Build the density:
  1. Invert avg_caplet_price at each strike K into BS lognormal IV
     using Rubinstein: sigma such that
        DF_real(0->5) * BS_call(F_{5,10}, 1+K, dt=1, sigma) = avg_caplet_price
     where DF_real is from put-call parity (paper's explicit choice).
  2. SABR-smooth the IV smile per Appendix A.2.
  3. Re-price on dense strike grid, apply B-L (eq. 10 but for 1Y horizon):
        q_1y(k) = e^{r_real * 1} * k * a''(k)
  4. Discretize into the 8 paper bins.

Note: for dt=1 the "gross cumulative inflation" and "annual inflation rate"
coincide, so B-L is directly in annual-rate space.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from inflation_disaster.pricing.bs_black import bs_implied_vol
from inflation_disaster.models.sabr import calibrate_sabr, sabr_smile
from inflation_disaster.data.schemas import SABRParams

log = logging.getLogger("inflation_disaster.analytics.hrr_forward_yoy")


_BIN_EDGES_ANNUAL = np.array([-np.inf, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, np.inf])
_BIN_LABELS = ("<=-1", "(-1,0]", "(0,1]", "(1,2]", "(2,3]", "(3,4]", "(4,5]", ">5")


@dataclass
class ForwardYOYResult:
    """Output of YOY forward-starting distribution extraction."""
    region: str
    year_range: tuple          # (start, end) inclusive, e.g. (6, 10)
    forward_rate: float         # the 5y5y forward annual rate, decimal
    discount_factor_real: float # DF_real(0 -> start), decimal
    strikes_annual: np.ndarray  # strikes used
    avg_caplet_prices: np.ndarray  # avg per-year caplet prices (decimal)
    implied_vols: np.ndarray    # BS lognormal IV at each strike
    sabr_params: SABRParams
    fine_strikes: np.ndarray    # dense strike grid
    fine_caplet_prices: np.ndarray  # SABR-reconstructed caplet prices on fine grid
    q_density_annual: np.ndarray  # pdf of annual inflation
    bin_dist: np.ndarray        # 8-bin distribution


def _rubinstein_caplet_price(
    forward_ratio: float,
    strike_ratio: float,
    vol: float,
    dt: float,
    discount_factor_real: float,
    is_cap: bool = True,
) -> float:
    """Rubinstein (1991) forward-starting caplet/floorlet pricer.

    Forward-starting 1-year inflation caplet paying max(CPI_ratio - 1 - K, 0)
    where CPI_ratio is (CPI_{t+1}/CPI_t) ~ lognormal with log-vol `vol` and
    forward = forward_ratio = (1 + forward_inflation_rate).

    Price is discounted by REAL rate (not nominal) per paper text.
    """
    F = forward_ratio
    K = strike_ratio
    if vol <= 0 or dt <= 0:
        return discount_factor_real * max(F - K if is_cap else K - F, 0.0)
    sqrt_T = np.sqrt(dt)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * dt) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    if is_cap:
        return discount_factor_real * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount_factor_real * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _rubinstein_invert_vol(
    price: float,
    forward_ratio: float,
    strike_ratio: float,
    dt: float,
    discount_factor_real: float,
    is_cap: bool = True,
    vol_lo: float = 1e-4, vol_hi: float = 3.0,
) -> float | None:
    """Invert a forward-starting caplet price back to BS lognormal IV."""
    from scipy.optimize import brentq
    F = forward_ratio
    K = strike_ratio
    intrinsic = discount_factor_real * (max(F - K, 0.0) if is_cap else max(K - F, 0.0))
    if price < intrinsic - 1e-12:
        return None
    if price <= intrinsic + 1e-12:
        return 0.0

    def fn(sigma):
        return _rubinstein_caplet_price(F, K, sigma, dt, discount_factor_real,
                                        is_cap) - price
    try:
        return float(brentq(fn, vol_lo, vol_hi, xtol=1e-8, maxiter=200))
    except ValueError:
        return None


def extract_forward_yoy_distribution(
    yoy_caps: dict,                 # {(strike_pct, mat_yr): price_bp_decimal}
    yoy_floors: dict,
    swap_rates: dict,               # {(region, mat): decimal}
    nominal_rates: dict,            # {(region, mat): decimal}
    real_rate_5y: float,            # from put-call parity on ZC 5Y
    region: str,
    year_range: tuple = (6, 10),
    beta: float = 0.5,
) -> ForwardYOYResult | None:
    """Strip forward YOY caplets/floorlets for years 6-10 and build
    the average 1-year Q distribution per paper Sec. 4.3.1.

    Parameters
    ----------
    yoy_caps, yoy_floors : dicts from `pull_live_prices()` — prices in bps.
    swap_rates, nominal_rates : dicts from `pull_live_prices()`.
    real_rate_5y : real rate to year 5 (paper uses put-call parity on the 5Y
                   ZC options).
    region : 'US' or 'EZ'.
    year_range : (start, end) inclusive, paper's is (6, 10).
    beta : SABR beta (fixed at 0.5 per paper convention).
    """
    start_yr, end_yr = year_range
    # Forward-starting annuity: T to T+H years. We use the SUM of YOY caplets
    # at maturity T+H minus at maturity T (they cover years 1..T+H and 1..T
    # respectively). So the spread covers years T+1 .. T+H.
    # With T=5, H=5: covers years 6..10 = 5 caplets.

    swap_T = swap_rates.get((region, start_yr - 1), None) or swap_rates.get((region, 5))
    swap_TH = swap_rates.get((region, end_yr), None) or swap_rates.get((region, 10))
    nom_T = nominal_rates.get((region, start_yr - 1), None) or nominal_rates.get((region, 5))
    if swap_T is None or swap_TH is None:
        log.warning(f"Missing swap rates for year_range={year_range}")
        return None

    # Forward annualized inflation rate from T to T+H (paper convention)
    # (1 + y_{0,T+H})^{T+H} = (1 + y_{0,T})^T * (1 + f_{T,T+H})^H
    # => f = ((1+y_{0,T+H})^{T+H} / (1+y_{0,T})^T)^(1/H) - 1
    T = start_yr - 1
    H = end_yr - start_yr + 1
    forward_rate = ((1.0 + swap_TH) ** (T + H) / (1.0 + swap_T) ** T) ** (1.0 / H) - 1.0

    # Discount factor to start of forward period, REAL rate (paper text)
    df_real = float(np.exp(-real_rate_5y * T))

    # Find strikes that have both T-year and (T+H)-year YOY cap/floor data
    # Prices in yoy_caps/yoy_floors are decimals (bps/10000)
    caps_T  = {k: p for (k, m), p in yoy_caps.items() if m == T}
    caps_TH = {k: p for (k, m), p in yoy_caps.items() if m == T + H}
    floors_T  = {k: p for (k, m), p in yoy_floors.items() if m == T}
    floors_TH = {k: p for (k, m), p in yoy_floors.items() if m == T + H}

    # Per-year forward-starting caplet at strike K:
    #   avg_caplet_price(K) = [YoY_cap(T+H, K) - YoY_cap(T, K)] / H
    # YoY prices from pull_live_prices come in bps, so divide by 10000 for decimal.
    avg_cap = {}
    for K in sorted(set(caps_T) & set(caps_TH)):
        p = (caps_TH[K] - caps_T[K]) / H / 10000.0
        if p > 0:
            avg_cap[float(K)] = p
    avg_floor = {}
    for K in sorted(set(floors_T) & set(floors_TH)):
        p = (floors_TH[K] - floors_T[K]) / H / 10000.0
        if p > 0:
            avg_floor[float(K)] = p

    if not avg_cap and not avg_floor:
        log.warning("No overlapping YOY strikes with both T and T+H maturities")
        return None

    # Rubinstein: convert each per-year caplet/floorlet price to BS IV
    # using forward_ratio = 1 + forward_rate, strike_ratio = 1 + K, dt = 1
    F_ratio = 1.0 + forward_rate
    strikes_used = sorted(set(avg_cap.keys()) | set(avg_floor.keys()))
    ivols = []
    price_array = []
    for K in strikes_used:
        K_ratio = 1.0 + K / 100.0  # K is in percent
        iv = None
        if K in avg_cap:
            iv = _rubinstein_invert_vol(
                avg_cap[K], F_ratio, K_ratio, dt=1.0,
                discount_factor_real=df_real, is_cap=True,
            )
            price_array.append(avg_cap[K])
        elif K in avg_floor:
            iv = _rubinstein_invert_vol(
                avg_floor[K], F_ratio, K_ratio, dt=1.0,
                discount_factor_real=df_real, is_cap=False,
            )
            price_array.append(avg_floor[K])
        ivols.append(iv if iv is not None else np.nan)

    strikes_used = np.array(strikes_used, dtype=float) / 100.0  # decimal
    ivols = np.array(ivols, dtype=float)
    strike_ratios = 1.0 + strikes_used

    valid = np.isfinite(ivols) & (ivols > 1e-6)
    if valid.sum() < 3:
        log.warning(f"Only {valid.sum()} valid IV points for forward YOY SABR")
        return None

    # SABR-smooth on ratios (paper Appendix A.2, same as ZC smoothing)
    sabr = calibrate_sabr(
        forward=F_ratio,
        strikes=strike_ratios[valid],
        market_vols=ivols[valid],
        T=1.0,
        beta=beta,
    )

    # Dense strike grid covering tails (paper's 8-bin range)
    fine_strikes = np.linspace(-0.05, 0.10, 151)
    fine_ratios = 1.0 + fine_strikes
    fine_vols = sabr_smile(F_ratio, fine_ratios, 1.0, sabr)

    # Price caplets on fine grid
    fine_caplet_prices = np.array([
        _rubinstein_caplet_price(F_ratio, K_r, v, 1.0, df_real, is_cap=True)
        for K_r, v in zip(fine_ratios, fine_vols)
    ])

    # Breeden-Litzenberger on annual inflation (dt=1, so annual = gross-1).
    # Paper eq. 10 derivation for the forward 1Y caplet:
    #    a_fwd(K) = DF_real(0->T_end) * integral((1 - K/f) q_fwd(f) df)
    #    a_fwd''(K) = DF_real(0->T_end) * q_fwd(K) / K
    # =>  q_fwd(K) = K * a_fwd''(K) * exp(r * T_end)
    #
    # We priced the averaged caplet with df_real = exp(-r * T_start) as the
    # representative discount (paper averages years 6..10; df at T_start=5
    # stands in for the average). To undo this discount in a'' and recover
    # q_fwd, the exponent must match: exp(r * T_start), NOT exp(r * 1).
    spline = CubicSpline(fine_ratios, fine_caplet_prices,
                         bc_type="natural", extrapolate=False)
    a_dd = spline(fine_ratios, nu=2)
    a_dd = np.maximum(a_dd, 0.0)
    q_density_ratio = np.exp(real_rate_5y * T) * fine_ratios * a_dd
    # Convert density-of-ratio to density-of-annual-rate: d(K_ratio)/d(K) = 1,
    # so they're numerically equal with K = K_ratio - 1
    q_density_annual = q_density_ratio.copy()
    q_density_annual = np.maximum(q_density_annual, 0.0)

    # Integrate into bins
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    bin_dist = np.zeros(8)
    for i in range(8):
        lo_e, hi_e = _BIN_EDGES_ANNUAL[i], _BIN_EDGES_ANNUAL[i + 1]
        mask = (fine_strikes > lo_e) & (fine_strikes <= hi_e)
        if mask.sum() > 1:
            bin_dist[i] = trap(q_density_annual[mask], fine_strikes[mask])
    total_mass = bin_dist.sum()
    # Allocate any remaining mass to the appropriate tail based on grid boundaries
    missing = max(0.0, 1.0 - total_mass)
    if missing > 0:
        # Tails outside strike grid: left below -5%, right above +10%
        # For YOY 1-year, both are vanishingly small — split 50/50 if any
        bin_dist[0] += missing / 2
        bin_dist[7] += missing / 2
    if bin_dist.sum() > 0:
        bin_dist = bin_dist / bin_dist.sum()

    return ForwardYOYResult(
        region=region,
        year_range=year_range,
        forward_rate=forward_rate,
        discount_factor_real=df_real,
        strikes_annual=strikes_used,
        avg_caplet_prices=np.array(price_array),
        implied_vols=ivols,
        sabr_params=sabr,
        fine_strikes=fine_strikes,
        fine_caplet_prices=fine_caplet_prices,
        q_density_annual=q_density_annual,
        bin_dist=bin_dist,
    )
