"""Bloomberg ticker configuration.

Separates ticker templates from code so they can be adjusted per terminal
without modifying source. Override by placing a ticker_overrides.json in
the project root or data/ directory.

Default ticker conventions follow common Bloomberg patterns for inflation
options, but THESE MAY NEED ADJUSTMENT for your specific terminal setup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from inflation_disaster.config import settings

log = logging.getLogger("inflation_disaster.data.ticker_config")

# Default ticker templates
_DEFAULTS = {
    "US": {
        "zc_cap": "USCP{maturity:02d}{strike_code} Curncy",
        "zc_floor": "USFP{maturity:02d}{strike_code} Curncy",
        "yoy_cap": "USCPYY{maturity:02d}{strike_code} Curncy",
        "yoy_floor": "USFPYY{maturity:02d}{strike_code} Curncy",
        "swap": "USSWIT{maturity} Curncy",
        "nominal_5y": "USGG5YR Index",
        "nominal_10y": "USGG10YR Index",
    },
    "EZ": {
        "zc_cap": "EUCP{maturity:02d}{strike_code} Curncy",
        "zc_floor": "EUFP{maturity:02d}{strike_code} Curncy",
        "yoy_cap": "EUCPYY{maturity:02d}{strike_code} Curncy",
        "yoy_floor": "EUFPYY{maturity:02d}{strike_code} Curncy",
        "swap": "EUSWI{maturity} Curncy",
        "nominal_5y": "GDBR5 Index",
        "nominal_10y": "GDBR10 Index",
    },
}

# Post-2021 US data has 1% strike spacing instead of 0.5%
POST_2021_US_STRIKE_STEP = 1.0
POST_2021_US_DATE = "2021-08-10"


def load_ticker_config(
    region: Literal["US", "EZ"],
    override_path: Path | None = None,
) -> dict:
    """Load ticker templates, with optional JSON overrides.

    Parameters
    ----------
    region : "US" or "EZ"
    override_path : Path, optional
        Path to a JSON file with ticker overrides.

    Returns
    -------
    dict of ticker templates.
    """
    config = _DEFAULTS[region].copy()

    # Check for overrides file
    search_paths = [
        override_path,
        settings.data_dir / "ticker_overrides.json",
        settings.project_root / "ticker_overrides.json",
    ]

    for path in search_paths:
        if path and path.exists():
            try:
                overrides = json.loads(path.read_text())
                if region in overrides:
                    config.update(overrides[region])
                    log.info(f"Loaded ticker overrides from {path}")
            except Exception as e:
                log.warning(f"Failed to load ticker overrides from {path}: {e}")
            break

    return config


def get_strikes_for_date(dt_str: str, region: str = "US") -> tuple:
    """Return appropriate strike grid for a given date.

    After August 2021, US inflation option data switched from 0.5% to 1%
    strike spacing.

    Parameters
    ----------
    dt_str : str
        Date string (YYYY-MM-DD).
    region : str

    Returns
    -------
    (strike_min, strike_max, strike_step) tuple
    """
    if region == "US" and dt_str >= POST_2021_US_DATE:
        return (-2.0, 6.0, POST_2021_US_STRIKE_STEP)
    return (settings.strike_min, settings.strike_max, settings.strike_step)


def strike_to_code(strike: float) -> str:
    """Convert strike price to Bloomberg ticker code.

    -2.0 -> 'N200', -0.5 -> 'N050', 0.0 -> '000', 1.5 -> '150', 4.0 -> '400'
    """
    if strike < 0:
        return f"N{abs(strike * 100):03.0f}"
    return f"{strike * 100:03.0f}"
