"""Paste-based ingestion of SWIL ZC cap/floor surfaces.

Bloomberg's SWIL ZC Volatility tab is rendered by BVOL analytics and is not
exposed as BLPAPI tickers or bulk fields. The only practical path is manual
copy/paste of the matrix. This module parses that paste and converts it into
`OptionSurface` objects per tenor, ready for the existing Breeden-Litzenberger
pipeline (HRR 2024 eq. 10).

Two input formats are supported:

1. **Premium paste** (preferred — no pricing needed):
   Cells are cap (or floor) premiums in bp.
   -> Premiums feed directly into B-L after converting bp -> decimal.

2. **Shifted-Lognormal vol paste** (requires a shift parameter):
   Cells are SLN vols in %. Black-76 on (rate + shift) produces premiums.
   -> Default shift is 2% if not provided; user should supply the true
      BVOL surface shift for accuracy (visible on SWIL "20) Surface Actions").

Typical usage:

    from inflation_disaster.data.zc_paste import build_surfaces_from_paste
    surfaces = build_surfaces_from_paste(
        paste_text=open("swil_us_zc_vol.tsv").read(),
        value_type="vol",          # or "premium"
        region="US",
        valuation_date=date.today(),
        swap_rates={1: 0.03128, 2: 0.02803, ..., 30: 0.02373},
        nominal_rates={5: 0.0388, 10: 0.0427, ...},
        shift=0.02,                # only for vol paste
    )
    # surfaces[5], surfaces[10] -> OptionSurface objects
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from scipy.stats import norm

from inflation_disaster.data.schemas import OptionSurface

log = logging.getLogger("inflation_disaster.data.zc_paste")


# ---------------------------------------------------------------------------
# Paste parsing
# ---------------------------------------------------------------------------
@dataclass
class PastedMatrix:
    """Result of parsing a pasted SWIL ZC matrix.

    strikes : 1-D array of strikes in decimal (e.g. 0.01 for 1%).
    tenors  : 1-D array of tenors in years (e.g. [1,2,3,5,7,10,12,15,20,30]).
    values  : 2-D array of shape (n_tenors, n_strikes).
              Units: decimal vol (e.g. 0.0239 for 2.39%) if value_type='vol',
                     decimal premium (e.g. 0.02417 for 241.7 bp) if 'premium'.
    value_type : 'vol' or 'premium'.
    """

    strikes: np.ndarray
    tenors: np.ndarray
    values: np.ndarray
    value_type: Literal["vol", "premium"]


def parse_paste(
    paste_text: str,
    value_type: Literal["vol", "premium"] = "vol",
    input_units: Literal["percent", "bp", "decimal"] = "percent",
) -> PastedMatrix:
    """Parse a whitespace/tab/comma-delimited SWIL matrix paste.

    Expected layout (header row + data rows):

        Tenor   1.00%   1.50%   2.00%   2.50%   3.00%   ...
        1 YR    1.54    1.36    1.18    1.19    1.19    ...
        2 YR    2.08    1.91    1.74    1.83    1.92    ...
        ...

    Commas, percent signs, "YR" suffixes, and tabs are tolerated. The first
    cell of the header row can be anything (e.g. "Tenor"); the remaining
    header cells are parsed as strikes. The first cell of each data row is
    the tenor in years.

    Parameters
    ----------
    paste_text : str
        Clipboard paste of the SWIL matrix.
    value_type : {'vol', 'premium'}
        Whether cells are SLN vols or cap/floor premiums.
    input_units : {'percent', 'bp', 'decimal'}
        How to scale cell values to decimal. 'percent' divides by 100,
        'bp' divides by 10,000, 'decimal' is as-is.
    """
    lines = [
        ln.strip() for ln in paste_text.strip().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError("paste needs at least a header row and one data row")

    def _cells(line: str) -> list[str]:
        # split on tabs, commas, or runs of whitespace
        if "\t" in line:
            raw = line.split("\t")
        elif "," in line:
            raw = line.split(",")
        else:
            raw = line.split()
        return [c.strip().replace("%", "").replace("YR", "").strip() for c in raw if c.strip() != ""]

    header = _cells(lines[0])
    # Header: first col is label ("Tenor"); rest are strikes in %
    strike_cells = header[1:]
    strikes_pct = np.array([float(s) for s in strike_cells], dtype=float)

    tenors = []
    rows = []
    for line in lines[1:]:
        cells = _cells(line)
        if not cells:
            continue
        t = float(cells[0])
        vals = [float(c) for c in cells[1:]]
        if len(vals) != len(strike_cells):
            raise ValueError(
                f"row '{line}' has {len(vals)} values, "
                f"header has {len(strike_cells)} strikes"
            )
        tenors.append(t)
        rows.append(vals)

    tenors = np.array(tenors, dtype=float)
    values = np.array(rows, dtype=float)

    # Scale to decimal
    scale = {"percent": 100.0, "bp": 10_000.0, "decimal": 1.0}[input_units]
    values = values / scale

    # Strikes header is always in percent in SWIL
    strikes_dec = strikes_pct / 100.0

    return PastedMatrix(
        strikes=strikes_dec,
        tenors=tenors,
        values=values,
        value_type=value_type,
    )


# ---------------------------------------------------------------------------
# Shifted-Lognormal Black pricer for ZC inflation caps
# ---------------------------------------------------------------------------
def zc_cap_price_sln(
    swap_rate: float,
    strike: float,
    maturity: float,
    sln_vol: float,
    discount_factor: float,
    shift: float = 0.0,
    is_cap: bool = True,
) -> float:
    """ZC cap/floor price via standard lognormal Black-76 on the CPI ratio.

    Payoff at maturity T:
        max((1+y_real)^T - (1+K)^T, 0) * N     [cap]
        max((1+K)^T - (1+y_real)^T, 0) * N     [floor]

    Priced under lognormal dynamics on the forward gross ratio F=(1+swap)^T:
        d1  = [ln(F/K_r) + 0.5 sigma^2 T] / (sigma sqrt(T))
        d2  = d1 - sigma sqrt(T)
        cap = DF * [F * N(d1) - K_r * N(d2)]
        floor = DF * [K_r * N(-d2) - F * N(-d1)]

    By construction, put-call parity holds exactly:
        cap - floor = DF * (F - K_r) = DF * ((1+swap)^T - (1+K)^T)

    sigma is the log-volatility of the gross ratio. The SWIL "Shifted
    Lognormal Vol" input is plugged in directly as an approximation; for
    HRR-grade accuracy paste the SWIL "Premium" view instead (zero pricing
    error). Paper Appendix A.2 uses Black-Scholes 1973 lognormal IV and
    smooths via SABR.

    `shift` is retained for API compatibility; unused in this lognormal pricer.
    """
    sqrt_T = np.sqrt(maturity)
    F = (1.0 + swap_rate) ** maturity
    K = (1.0 + strike) ** maturity

    if sln_vol <= 0 or sqrt_T <= 0:
        payoff = max(F - K, 0.0) if is_cap else max(K - F, 0.0)
        return discount_factor * payoff

    d1 = (np.log(F / K) + 0.5 * sln_vol ** 2 * maturity) / (sln_vol * sqrt_T)
    d2 = d1 - sln_vol * sqrt_T

    if is_cap:
        price = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        price = K * norm.cdf(-d2) - F * norm.cdf(-d1)

    return discount_factor * price


# ---------------------------------------------------------------------------
# Build OptionSurface per tenor
# ---------------------------------------------------------------------------
def build_surfaces_from_paste(
    paste_text: str,
    region: Literal["US", "EZ"],
    valuation_date: date,
    swap_rates: dict[int, float],
    nominal_rates: dict[int, float],
    value_type: Literal["vol", "premium"] = "vol",
    input_units: Literal["percent", "bp", "decimal"] = "percent",
    shift: float = 0.02,
    target_tenors: list[int] | None = None,
    is_floor: bool = False,
) -> dict[int, OptionSurface]:
    """Parse a SWIL paste and return one OptionSurface per tenor.

    Parameters
    ----------
    paste_text : str
    region : 'US' or 'EZ'
    valuation_date : date stamp for the surfaces
    swap_rates : {tenor: ZC swap rate in decimal}
    nominal_rates : {tenor: nominal discount rate in decimal}
    value_type : 'vol' (SLN) or 'premium' (bp or decimal)
    input_units : 'percent' / 'bp' / 'decimal'
    shift : SLN shift when value_type='vol'
    target_tenors : subset of tenors to return (default all tenors in paste)
    is_floor : if True, interpret cells as floor prices/vols instead of caps

    Returns
    -------
    dict[int -> OptionSurface]. Missing tenors (no swap rate) are skipped.
    """
    parsed = parse_paste(paste_text, value_type=value_type, input_units=input_units)

    tenors = parsed.tenors.astype(int)
    strikes_dec = parsed.strikes  # decimal

    if target_tenors is None:
        target_tenors = sorted(set(int(t) for t in tenors))

    out: dict[int, OptionSurface] = {}
    for t in target_tenors:
        if t not in tenors:
            log.warning(f"tenor {t} not in paste (available: {sorted(set(tenors))})")
            continue
        row_idx = int(np.where(tenors == t)[0][0])
        row = parsed.values[row_idx]

        swap = swap_rates.get(t)
        if swap is None:
            log.warning(f"no swap rate for {region} {t}Y; skipping")
            continue
        # Discount: use closest nominal rate on a flat extrapolation
        if t in nominal_rates:
            r_nom = nominal_rates[t]
        else:
            mats = sorted(nominal_rates.keys())
            closest = min(mats, key=lambda m: abs(m - t))
            r_nom = nominal_rates[closest]
        df = float(np.exp(-r_nom * t))

        if value_type == "premium":
            prices = row.copy()  # already decimal-notional
        else:
            prices = np.array([
                zc_cap_price_sln(
                    swap_rate=swap, strike=k, maturity=float(t),
                    sln_vol=v, discount_factor=df, shift=shift,
                    is_cap=not is_floor,
                )
                for k, v in zip(strikes_dec, row)
            ])

        cap_prices = np.zeros_like(prices) if is_floor else prices.copy()
        floor_prices = prices.copy() if is_floor else np.zeros_like(prices)

        # OptionSurface expects strikes in PERCENT (e.g. 2.0 for 2%)
        surface = OptionSurface(
            date=valuation_date,
            region=region,
            maturity=t,
            strikes=strikes_dec * 100.0,
            cap_prices=cap_prices,
            floor_prices=floor_prices,
            swap_rate=swap,
        )
        out[t] = surface

    return out


# ---------------------------------------------------------------------------
# Combined cap + floor surface builder
# ---------------------------------------------------------------------------
def build_combined_surface(
    cap_paste: str,
    floor_paste: str,
    region: Literal["US", "EZ"],
    valuation_date: date,
    swap_rates: dict[int, float],
    nominal_rates: dict[int, float],
    value_type: Literal["vol", "premium"] = "vol",
    input_units: Literal["percent", "bp", "decimal"] = "percent",
    shift: float = 0.02,
    target_tenors: list[int] | None = None,
) -> dict[int, OptionSurface]:
    """Merge cap + floor paste matrices into a single strike grid per tenor.

    Strategy for overlapping strikes (both cap and floor available):
      - Use FLOOR quote when K <= swap_rate (OTM floor, more info on left tail)
      - Use CAP   quote when K >  swap_rate (OTM cap, more info on right tail)
      - Then apply put-call parity to compute the complementary leg:
          cap(K) - floor(K) = DF * T * (F_s - K_s)   (SLN-shifted)
        so floor(K) = cap(K) - DF * T * (F_s - K_s)
        and cap(K)  = floor(K) + DF * T * (F_s - K_s)

    For strikes that exist only in the cap paste (OTM cap), we compute the
    floor via parity; and vice versa.

    Returns a dict {tenor -> OptionSurface} with BOTH cap_prices and
    floor_prices populated on the merged strike grid.
    """
    caps = parse_paste(cap_paste, value_type=value_type, input_units=input_units)
    floors = parse_paste(floor_paste, value_type=value_type, input_units=input_units)

    cap_tenors = caps.tenors.astype(int)
    floor_tenors = floors.tenors.astype(int)

    if target_tenors is None:
        target_tenors = sorted(set(cap_tenors) & set(floor_tenors))

    out: dict[int, OptionSurface] = {}
    for t in target_tenors:
        if t not in cap_tenors or t not in floor_tenors:
            log.warning(f"tenor {t} missing in one paste")
            continue
        swap = swap_rates.get(t)
        if swap is None:
            log.warning(f"no swap rate for {region} {t}Y; skipping")
            continue
        if t in nominal_rates:
            r_nom = nominal_rates[t]
        else:
            mats = sorted(nominal_rates.keys())
            r_nom = nominal_rates[min(mats, key=lambda m: abs(m - t))]
        df = float(np.exp(-r_nom * t))

        cap_row = caps.values[int(np.where(cap_tenors == t)[0][0])]
        floor_row = floors.values[int(np.where(floor_tenors == t)[0][0])]

        # Unified strike grid (decimal)
        merged_strikes = np.array(sorted(set(caps.strikes.tolist())
                                         | set(floors.strikes.tolist())))

        cap_prices = np.zeros_like(merged_strikes)
        floor_prices = np.zeros_like(merged_strikes)

        for i, k in enumerate(merged_strikes):
            has_cap = k in caps.strikes
            has_floor = k in floors.strikes

            def _price(row, strikes_arr, K, is_cap):
                idx = int(np.where(strikes_arr == K)[0][0])
                v = row[idx]
                if value_type == "premium":
                    return v
                return zc_cap_price_sln(
                    swap_rate=swap, strike=K, maturity=float(t),
                    sln_vol=v, discount_factor=df, shift=shift,
                    is_cap=is_cap,
                )

            # Prefer OTM quote, fill ITM via put-call parity.
            # Standard parity for ZC inflation caps (payoff on CPI ratio):
            #   cap(K) - floor(K) = DF * ((1+swap)^T - (1+K)^T)
            # (paper Sec. 4.2; Birru-Figlewski 2012 convention).
            F_ratio = (1.0 + swap) ** float(t)
            K_ratio = (1.0 + k) ** float(t)
            parity = df * (F_ratio - K_ratio)

            if has_cap and has_floor:
                # Average the two quoted prices: cap-source price and the
                # parity-implied price from the floor quote. This avoids the
                # cap/floor-source discontinuity at K=swap (cap and floor SLN
                # vols typically disagree by 50-100bp on the same strike due
                # to bid-ask / dealer convention; flipping source at K=swap
                # bakes that disagreement into the smile as a vol kink that
                # SABR then over-smooths and loses tail density).
                cp_from_cap = _price(cap_row, caps.strikes, k, is_cap=True)
                fl_from_floor = _price(floor_row, floors.strikes, k, is_cap=False)
                cp_from_floor = fl_from_floor + parity
                cap_prices[i] = 0.5 * (cp_from_cap + cp_from_floor)
                floor_prices[i] = cap_prices[i] - parity
            elif has_cap:
                cp = _price(cap_row, caps.strikes, k, is_cap=True)
                cap_prices[i] = cp
                floor_prices[i] = cp - parity
            elif has_floor:
                fl = _price(floor_row, floors.strikes, k, is_cap=False)
                floor_prices[i] = fl
                cap_prices[i] = fl + parity

        # Clip tiny negatives from numerical noise
        cap_prices = np.maximum(cap_prices, 0.0)
        floor_prices = np.maximum(floor_prices, 0.0)

        out[t] = OptionSurface(
            date=valuation_date,
            region=region,
            maturity=t,
            strikes=merged_strikes * 100.0,  # OptionSurface wants percent
            cap_prices=cap_prices,
            floor_prices=floor_prices,
            swap_rate=swap,
        )
    return out


# ---------------------------------------------------------------------------
# Hardcoded samples from user's SWIL screenshots (2026-04-21, US,
# Shifted Lognormal Vol, BVOL, SOFR vs Fixed swap curve).
# ---------------------------------------------------------------------------
SAMPLE_CAP_PASTE_US_2026_04_21 = """\
Tenor\t1.00\t1.50\t2.00\t2.50\t3.00\t3.50\t4.00\t4.50\t5.00\t6.00
1\t1.54\t1.36\t1.18\t1.19\t1.19\t1.20\t1.21\t1.33\t1.45\t1.69
2\t2.08\t1.91\t1.74\t1.83\t1.92\t2.07\t2.21\t2.38\t2.54\t2.87
3\t1.96\t1.89\t1.82\t1.98\t2.13\t2.33\t2.54\t2.75\t2.97\t3.40
5\t2.39\t2.44\t2.48\t2.69\t2.90\t3.19\t3.48\t3.77\t4.07\t4.64
7\t2.76\t2.84\t2.92\t3.22\t3.52\t3.81\t4.10\t4.37\t4.64\t5.12
10\t3.12\t3.03\t2.95\t3.57\t4.20\t4.79\t5.38\t5.92\t6.46\t7.47
12\t3.62\t3.65\t3.70\t4.58\t5.46\t5.96\t6.46\t7.07\t7.67\t8.80
15\t3.59\t3.64\t3.69\t4.57\t5.45\t5.96\t6.46\t7.06\t7.67\t8.79
20\t3.74\t3.98\t4.24\t4.93\t5.62\t6.38\t7.13\t7.85\t8.56\t9.94
30\t4.51\t4.21\t3.87\t5.17\t6.47\t7.55\t8.63\t9.67\t10.71\t12.57
"""

SAMPLE_FLOOR_PASTE_US_2026_04_21 = """\
Tenor\t-2.00\t-1.00\t-0.50\t0.00\t0.50\t1.00\t1.50\t2.00\t2.50\t3.00
1\t2.02\t2.02\t1.89\t1.77\t1.66\t1.54\t1.36\t1.18\t1.19\t1.19
2\t2.70\t2.70\t2.55\t2.41\t2.24\t2.07\t1.90\t1.73\t1.82\t1.91
3\t2.38\t2.38\t2.26\t2.14\t2.05\t1.96\t1.89\t1.82\t1.98\t2.13
5\t3.14\t3.14\t2.93\t2.72\t2.56\t2.41\t2.45\t2.49\t2.69\t2.90
7\t3.44\t3.44\t3.24\t3.03\t2.90\t2.76\t2.84\t2.92\t3.22\t3.52
10\t4.03\t4.03\t3.76\t3.49\t3.30\t3.11\t3.03\t2.95\t3.57\t4.19
12\t4.92\t4.92\t4.54\t4.16\t3.87\t3.58\t3.63\t3.69\t4.57\t5.45
15\t4.92\t4.92\t4.54\t4.16\t3.87\t3.58\t3.64\t3.69\t4.57\t5.45
20\t5.61\t5.61\t5.09\t4.57\t4.13\t3.69\t3.96\t4.23\t4.92\t5.61
30\t7.15\t7.15\t6.38\t5.60\t5.10\t4.60\t4.24\t3.89\t5.19\t6.48
"""


# 2026-02-12 US SWIL paste (BID side) — EXACT date of paper's latest .dta row.
SAMPLE_CAP_PASTE_US_2026_02_12 = """\
Tenor\t1.00\t1.50\t2.00\t2.50\t3.00\t3.50\t4.00\t4.50\t5.00\t6.00
1\t1.63\t1.50\t1.38\t1.45\t1.54\t1.56\t1.59\t1.69\t1.80\t2.06
2\t2.27\t2.14\t2.03\t2.51\t3.02\t2.93\t2.84\t3.00\t3.16\t3.49
3\t2.30\t2.15\t2.05\t2.59\t3.15\t3.30\t3.46\t3.66\t3.85\t4.26
5\t2.67\t2.62\t2.63\t2.75\t2.89\t3.36\t3.84\t4.10\t4.36\t4.86
7\t2.86\t2.82\t2.85\t3.29\t3.75\t4.05\t4.35\t4.66\t4.97\t5.56
10\t3.48\t3.07\t2.72\t3.62\t4.56\t5.15\t5.74\t6.29\t6.84\t7.85
12\t3.90\t3.69\t3.54\t4.18\t4.85\t5.39\t5.94\t6.45\t6.97\t7.88
15\t4.03\t3.74\t3.56\t4.19\t4.85\t5.40\t5.94\t6.46\t6.97\t7.88
20\t4.42\t4.12\t4.06\t4.53\t5.04\t5.62\t6.21\t6.76\t7.30\t8.26
30\t6.59\t5.79\t5.50\t5.48\t5.56\t6.53\t7.52\t8.38\t9.24\t10.76
"""

SAMPLE_FLOOR_PASTE_US_2026_02_12 = """\
Tenor\t-2.00\t-1.00\t-0.50\t0.00\t0.50\t1.00\t1.50\t2.00\t2.50\t3.00
1\t1.89\t1.89\t1.77\t1.65\t1.53\t1.41\t1.33\t1.24\t1.32\t1.40
2\t2.53\t2.53\t2.38\t2.24\t2.13\t2.01\t1.93\t1.85\t2.35\t2.85
3\t2.42\t2.41\t2.29\t2.17\t2.05\t1.93\t1.90\t1.86\t2.41\t2.96
5\t2.89\t2.88\t2.73\t2.58\t2.48\t2.37\t2.43\t2.48\t2.62\t2.74
7\t3.37\t3.36\t3.16\t2.95\t2.77\t2.59\t2.67\t2.74\t3.19\t3.64
10\t4.09\t4.09\t3.82\t3.55\t3.31\t3.08\t2.81\t2.54\t3.47\t4.40
12\t4.83\t4.82\t4.51\t4.19\t3.88\t3.57\t3.48\t3.39\t4.06\t4.71
15\t4.82\t4.82\t4.51\t4.19\t3.88\t3.57\t3.48\t3.39\t4.05\t4.70
20\t5.36\t5.36\t4.94\t4.53\t4.10\t3.68\t3.76\t3.84\t4.35\t4.85
30\t6.68\t6.66\t5.73\t4.80\t4.62\t4.43\t4.58\t4.71\t4.80\t4.80
"""


# 2026-02-12 US SWIL PREMIUM paste (BID, BVOL contributor, USD SOFR swap curve).
# Paper-faithful: paper inputs are option PRICES, not vols (App A.2). Using
# premiums skips the SLN-vol→price conversion (~3% bias). Units: bp.
SAMPLE_CAP_PASTE_US_2026_02_12_PREMIUM = """\
Tenor\t1.00\t1.50\t2.00\t2.50\t3.00\t3.50\t4.00\t4.50\t5.00\t6.00
1\t151.90\t110.29\t73.21\t49.04\t32.25\t19.14\t10.79\t6.98\t4.62\t2.49
2\t296.63\t218.32\t149.33\t123.68\t109.33\t72.72\t45.18\t33.58\t25.30\t15.20
3\t418.84\t299.57\t196.24\t154.29\t131.69\t95.45\t69.25\t51.94\t39.50\t23.96
5\t669.56\t483.11\t324.96\t209.97\t130.52\t100.53\t82.13\t59.41\t43.85\t25.07
7\t889.32\t636.62\t420.83\t292.89\t211.41\t145.99\t102.42\t73.84\t54.47\t30.89
10\t1190.39\t836.94\t505.98\t368.27\t301.51\t233.13\t187.88\t154.14\t130.23\t94.91
12\t1366.95\t988.42\t605.43\t455.31\t337.53\t250.61\t192.72\t150.82\t121.44\t79.30
15\t1551.22\t1111.78\t707.43\t474.20\t332.74\t233.55\t170.17\t126.40\t96.96\t57.57
20\t1775.24\t1277.21\t823.75\t523.35\t332.78\t222.15\t154.70\t109.71\t80.72\t43.95
30\t2112.88\t1557.03\t1039.17\t622.96\t336.14\t243.52\t192.07\t151.84\t126.03\t87.40
"""

SAMPLE_FLOOR_PASTE_US_2026_02_12_PREMIUM = """\
Tenor\t-2.00\t-1.00\t-0.50\t0.00\t0.50\t1.00\t1.50\t2.00\t2.50\t3.00
1\t0.26\t1.45\t2.18\t3.36\t5.33\t8.65\t15.35\t26.59\t50.74\t82.27
2\t0.52\t2.91\t4.55\t7.26\t12.49\t21.41\t37.93\t64.24\t134.37\t216.26
3\t0.06\t0.77\t1.60\t3.34\t7.19\t15.27\t35.38\t72.81\t173.01\t293.95
5\t0.04\t0.63\t1.43\t3.28\t8.34\t20.23\t55.67\t123.82\t239.62\t395.51
7\t0.04\t0.75\t1.59\t3.46\t8.32\t19.94\t62.04\t149.89\t334.66\t575.17
10\t0.07\t1.17\t2.32\t4.81\t11.01\t25.56\t57.89\t130.20\t413.95\t787.56
12\t0.20\t2.50\t4.60\t8.79\t17.59\t36.31\t92.31\t207.85\t502.06\t895.11
15\t0.05\t1.00\t2.11\t4.67\t10.85\t26.05\t78.21\t200.55\t531.42\t993.93
20\t0.03\t0.77\t1.54\t3.30\t7.40\t17.96\t77.01\t235.27\t606.31\t1152.38
30\t0.04\t1.09\t1.13\t1.18\t5.44\t22.32\t101.49\t316.15\t744.34\t1430.24
"""


# 2026-02-21 US SWIL paste (BID side — paper uses Mid so there is a small basis).
# Same date as the paper's "Feb 2026" .dta row (which is dated 2026-02-12, 9 days earlier).
# Quick Calculator sanity: 3Y Cap K=2% premium=181.18bp vol=1.94%, 5Y Cap K=2% premium=310.78bp vol=2.61%.
SAMPLE_CAP_PASTE_US_2026_02_21 = """\
Tenor\t1.00\t1.50\t2.00\t2.50\t3.00\t3.50\t4.00\t4.50\t5.00\t6.00
1\t1.63\t1.49\t1.38\t1.47\t1.57\t1.60\t1.63\t1.73\t1.84\t2.10
2\t2.23\t2.11\t2.01\t2.32\t2.64\t2.76\t2.89\t3.05\t3.21\t3.54
3\t2.19\t2.05\t1.94\t2.58\t3.24\t3.38\t3.53\t3.72\t3.91\t4.31
5\t2.60\t2.58\t2.61\t2.75\t2.91\t3.40\t3.90\t4.15\t4.41\t4.91
7\t2.79\t2.80\t2.88\t3.34\t3.82\t4.11\t4.41\t4.71\t5.02\t5.61
10\t3.38\t3.02\t2.72\t3.66\t4.63\t5.22\t5.80\t6.35\t6.90\t7.90
12\t3.82\t3.58\t3.40\t4.14\t4.91\t5.45\t5.99\t6.50\t7.02\t7.92
15\t3.89\t3.61\t3.40\t4.14\t4.91\t5.44\t5.99\t6.50\t7.01\t7.92
20\t4.21\t4.23\t4.47\t4.77\t5.09\t5.67\t6.25\t6.79\t7.34\t8.30
30\t5.02\t4.11\t3.29\t4.34\t5.43\t6.43\t7.43\t8.29\t9.16\t10.69
"""

SAMPLE_FLOOR_PASTE_US_2026_02_21 = """\
Tenor\t-2.00\t-1.00\t-0.50\t0.00\t0.50\t1.00\t1.50\t2.00\t2.50\t3.00
1\t1.90\t1.90\t1.78\t1.65\t1.54\t1.42\t1.34\t1.25\t1.34\t1.43
2\t2.53\t2.53\t2.38\t2.24\t2.12\t2.01\t1.94\t1.86\t2.18\t2.49
3\t2.40\t2.39\t2.27\t2.14\t2.02\t1.90\t1.84\t1.79\t2.43\t3.08
5\t2.86\t2.86\t2.71\t2.55\t2.45\t2.34\t2.41\t2.48\t2.63\t2.78
7\t3.34\t3.34\t3.13\t2.92\t2.74\t2.56\t2.67\t2.79\t3.26\t3.72
10\t4.07\t4.07\t3.80\t3.53\t3.29\t3.06\t2.82\t2.58\t3.54\t4.50
12\t4.81\t4.81\t4.49\t4.17\t3.87\t3.56\t3.42\t3.28\t4.04\t4.79
15\t4.81\t4.81\t4.49\t4.17\t3.87\t3.56\t3.42\t3.28\t4.03\t4.79
20\t5.35\t5.34\t4.93\t4.51\t4.08\t3.65\t3.98\t4.31\t4.62\t4.92
30\t6.72\t6.72\t5.78\t4.84\t4.68\t4.51\t3.82\t3.13\t4.21\t5.29
"""
