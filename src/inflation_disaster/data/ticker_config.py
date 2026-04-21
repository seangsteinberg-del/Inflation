"""Bloomberg ticker configuration for inflation options.

Discovered tickers (April 2026):
  US YOY Cap:   USISC{strike}{maturity} Curncy  (e.g. USISC35 = 3% strike, 5Y)
  US YOY Floor: USISF{strike}{maturity} Curncy  (e.g. USISF15 = 1% strike, 5Y)
  US ZC Cap:    USISCD{maturity} Curncy (strike 1.5%), USISCQ{maturity} (strike 4.5%)
  EU YOY Cap:   EUISC{strike}{maturity} Curncy  (e.g. EUISC25 = 2% strike, 5Y)
  EU YOY Floor: EUISF{strike}{maturity} Curncy  (e.g. EUISF15 = 1% strike, 5Y)
  EU ZC Cap:    EUISCC{maturity} Curncy (strike 4.5%)

Inflation swaps:
  US: USSWIT{maturity} Curncy
  EZ: EUSWI{maturity} Curncy

Forward:
  US 5y5y: FWISUS55 Index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from inflation_disaster.config import settings

log = logging.getLogger("inflation_disaster.data.ticker_config")


# Verified live Bloomberg tickers (April 2026)
TICKER_TEMPLATES = {
    "US": {
        "yoy_cap": "USISC{strike}{maturity} Curncy",
        "yoy_floor": "USISF{strike}{maturity} Curncy",
        "zc_cap_low": "USISCD{maturity} Curncy",   # strike 1.5%
        "zc_cap_high": "USISCQ{maturity} Curncy",   # strike 4.5%
        "swap": "USSWIT{maturity} Curncy",
        "forward_5y5y": "FWISUS55 Index",
        "nominal_5y": "USGG5YR Index",
        "nominal_10y": "USGG10YR Index",
        "cpi_yoy": "CPI YOY Index",
    },
    "EZ": {
        "yoy_cap": "EUISC{strike}{maturity} Curncy",
        "yoy_floor": "EUISF{strike}{maturity} Curncy",
        "zc_cap_high": "EUISCC{maturity} Curncy",  # strike 4.5%
        "swap": "EUSWI{maturity} Curncy",
        "nominal_5y": "GDBR5 Index",
        "nominal_10y": "GDBR10 Index",
    },
}

# Available strikes per region (verified with live data)
AVAILABLE_STRIKES = {
    "US": {
        "yoy_cap": [3, 4, 5, 6],         # 0-2% caps have no prices
        "yoy_floor": [1, 2],             # 0% and 3% floors sparse
    },
    "EZ": {
        "yoy_cap": [1, 2, 3, 4, 5],
        "yoy_floor": [1, 2],
    },
}

# Available maturities (verified)
AVAILABLE_MATURITIES = {
    "US": {
        "yoy_cap": [2, 3, 5, 7, 10],     # 1Y sparse
        "yoy_floor": [1, 2, 5, 7, 10],
    },
    "EZ": {
        "yoy_cap": [1, 2, 3, 5, 7, 10],
        "yoy_floor": [1, 2, 3, 5, 7, 10],
    },
}


# -----------------------------------------------------------------------------
# ZC cap full strike grid (SWIL ZC vol tab)
# -----------------------------------------------------------------------------
# Bloomberg's ZC inflation cap tickers embed the strike as a letter code in the
# stem (e.g. USISCD = 1.5%, USISCQ = 4.5%). The full letter->strike mapping
# is NOT publicly documented; only the two letters above are verified from
# live data. Populate the rest by running:
#
#     python -m inflation_disaster.data.bloomberg --discover-zc-strikes US
#
# which probes USISC{A..Z}5 and reports which tickers return PX_LAST.
# Fill in the discovered letters here (strike in %, value = single letter).
#
# Paper replication (HRR 2024) wants: -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,
# 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0  (17 strikes, 0.5% step).
ZC_CAP_STRIKE_LETTERS = {
    "US": {
        # strike_pct: letter   (None = not yet discovered — run discovery tool)
        1.5: "D",    # verified
        4.5: "Q",    # verified
        # -2.0: None, -1.5: None, -1.0: None, -0.5: None, 0.0: None,
        # 0.5: None, 1.0: None, 2.0: None, 2.5: None, 3.0: None,
        # 3.5: None, 4.0: None, 5.0: None, 5.5: None, 6.0: None,
    },
    "EZ": {
        # Only one verified so far (EUISCC = 4.5%).
        # EZ stem is EUISC{letter} — discovery needs the right family.
        4.5: "C",
    },
}

# Ticker stem patterns (letter replaces {letter})
ZC_CAP_TICKER_STEM = {
    "US": "USISC{letter}{maturity} Curncy",
    "EZ": "EUISC{letter}{maturity} Curncy",
}


def build_zc_cap_grid_tickers(
    region: Literal["US", "EZ"],
    maturities: list[int] | None = None,
    strikes: list[float] | None = None,
) -> dict[tuple[float, int], str]:
    """Build full ZC cap ticker grid from ZC_CAP_STRIKE_LETTERS.

    Returns dict mapping (strike_pct, maturity) -> Bloomberg ticker.
    Only strikes present in ZC_CAP_STRIKE_LETTERS[region] are included.
    """
    letter_map = ZC_CAP_STRIKE_LETTERS[region]
    stem = ZC_CAP_TICKER_STEM[region]
    if maturities is None:
        maturities = [5, 10]
    if strikes is None:
        strikes = sorted(letter_map.keys())

    out = {}
    for k in strikes:
        letter = letter_map.get(k)
        if letter is None:
            continue
        for m in maturities:
            out[(k, m)] = stem.format(letter=letter, maturity=m)
    return out


def build_yoy_cap_tickers(
    region: Literal["US", "EZ"],
    strikes: list[int] | None = None,
    maturities: list[int] | None = None,
) -> dict[tuple[int, int], str]:
    """Build ticker dict for YOY cap premiums.

    Returns dict mapping (strike, maturity) -> Bloomberg ticker.
    """
    template = TICKER_TEMPLATES[region]["yoy_cap"]
    if strikes is None:
        strikes = AVAILABLE_STRIKES[region]["yoy_cap"]
    if maturities is None:
        maturities = AVAILABLE_MATURITIES[region]["yoy_cap"]

    return {
        (k, m): template.format(strike=k, maturity=m)
        for k in strikes for m in maturities
    }


def build_yoy_floor_tickers(
    region: Literal["US", "EZ"],
    strikes: list[int] | None = None,
    maturities: list[int] | None = None,
) -> dict[tuple[int, int], str]:
    """Build ticker dict for YOY floor premiums."""
    template = TICKER_TEMPLATES[region]["yoy_floor"]
    if strikes is None:
        strikes = AVAILABLE_STRIKES[region]["yoy_floor"]
    if maturities is None:
        maturities = AVAILABLE_MATURITIES[region]["yoy_floor"]

    return {
        (k, m): template.format(strike=k, maturity=m)
        for k in strikes for m in maturities
    }


def build_swap_tickers(
    region: Literal["US", "EZ"],
    maturities: list[int] | None = None,
) -> dict[int, str]:
    """Build ticker dict for inflation swap rates."""
    template = TICKER_TEMPLATES[region]["swap"]
    if maturities is None:
        maturities = [1, 2, 3, 5, 7, 10]
    return {m: template.format(maturity=m) for m in maturities}


def strike_to_code(strike: float) -> str:
    """Convert strike price to Bloomberg ticker code.

    -2.0 -> 'N200', -0.5 -> 'N050', 0.0 -> '000', 1.5 -> '150', 4.0 -> '400'
    """
    if strike < 0:
        return f"N{abs(strike * 100):03.0f}"
    return f"{strike * 100:03.0f}"
