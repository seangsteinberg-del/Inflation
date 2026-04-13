"""Jorda-Schularick-Taylor Macrohistory Database loader and disaster identification.

Loads historical inflation and GDP data for 18 advanced economies (1875-2015)
to calibrate the risk adjustment factor (third adjustment).

Data source: https://www.macrohistory.net/database/
Output disaster dates from Barro (2006).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from inflation_disaster.config import settings

log = logging.getLogger("inflation_disaster.data.jst")


def load_jst_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the JST Macrohistory Database.

    Parameters
    ----------
    path : Path, optional
        Path to the JST Excel/CSV file. If None, looks in data/historical/.

    Returns
    -------
    DataFrame with columns: country, year, cpi, rgdppc (real GDP per capita),
        inflation (annual CPI inflation rate), rgdp_growth (real GDP growth).
    """
    if path is None:
        historical_dir = settings.historical_dir
        # Try common filenames
        for fname in [
            "JSTdatasetR6.xlsx",
            "JSTdatasetR5.xlsx",
            "JSTdatasetR4.xlsx",
            "jst_dataset.xlsx",
            "jst_dataset.csv",
        ]:
            candidate = historical_dir / fname
            if candidate.exists():
                path = candidate
                break

    if path is None or not path.exists():
        raise FileNotFoundError(
            f"JST dataset not found in {settings.historical_dir}. "
            "Download from https://www.macrohistory.net/database/ "
            "and place in data/historical/"
        )

    log.info(f"Loading JST dataset from {path}")

    if path.suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path)

    # Standardize column names (JST dataset uses varying conventions)
    col_map = {}
    for col in raw.columns:
        cl = col.lower().strip()
        if cl in ("country", "iso"):
            col_map[col] = "country"
        elif cl in ("year",):
            col_map[col] = "year"
        elif cl in ("cpi",):
            col_map[col] = "cpi"
        elif cl in ("rgdpmad", "rgdppc"):
            col_map[col] = "rgdppc"

    raw = raw.rename(columns=col_map)

    # Filter to our countries and date range
    target_countries = set(settings.jst_countries)
    df = raw[raw["country"].isin(target_countries)].copy()
    df = df[
        (df["year"] >= settings.jst_start_year) & (df["year"] <= settings.jst_end_year)
    ]

    # Compute inflation and GDP growth
    df = df.sort_values(["country", "year"])
    df["inflation"] = df.groupby("country")["cpi"].pct_change()
    df["rgdp_growth"] = df.groupby("country")["rgdppc"].pct_change()

    df = df.dropna(subset=["inflation", "rgdp_growth"])
    log.info(
        f"Loaded {len(df)} country-year observations "
        f"({df['country'].nunique()} countries)"
    )
    return df


def _butterworth_trend(
    series: np.ndarray,
    cutoff_years: int = 20,
    order: int = 2,
) -> np.ndarray:
    """Extract low-frequency trend using Butterworth highpass filter.

    The paper uses a bandpass filter that isolates fluctuations of frequency
    lower than 20 years to define the inflation target.
    """
    if len(series) < 2 * cutoff_years:
        return np.full_like(series, np.nanmean(series))

    # Butterworth lowpass to get trend
    # Nyquist frequency = 0.5 (annual data), cutoff at 1/cutoff_years
    nyquist = 0.5
    cutoff_freq = 1.0 / cutoff_years
    normalized_cutoff = cutoff_freq / nyquist

    # Clamp to valid range
    normalized_cutoff = min(normalized_cutoff, 0.99)

    b, a = butter(order, normalized_cutoff, btype="low")

    # Pad to handle edge effects
    pad_len = min(3 * max(len(a), len(b)), len(series) - 1)
    trend = filtfilt(b, a, series, padlen=pad_len)
    return trend


def identify_inflation_disasters(
    df: pd.DataFrame,
    method: str = "C2.T3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify inflation disaster episodes using peak/trough cycles.

    Parameters
    ----------
    df : DataFrame
        Output from load_jst_dataset() with columns: country, year, inflation, rgdp_growth.
    method : str
        Disaster identification method. Default "C2.T3" matches the paper's baseline:
        C2 = peak/trough cycles, T3 = Butterworth filter for inflation target.

    Returns
    -------
    high_disasters : DataFrame
        High-inflation disaster episodes with associated GDP drops.
    low_disasters : DataFrame
        Low-inflation (deflation) disaster episodes with associated GDP drops.

    Each has columns: country, start_year, end_year, peak_inflation,
        has_output_disaster, min_gdp_growth, avg_gdp_growth, z (inverse consumption drop).
    """
    high_disasters = []
    low_disasters = []

    for country, group in df.groupby("country"):
        group = group.sort_values("year").reset_index(drop=True)
        inflation = group["inflation"].values
        gdp_growth = group["rgdp_growth"].values
        years = group["year"].values

        if len(inflation) < 10:
            continue

        # Compute inflation target using Butterworth filter
        target = _butterworth_trend(inflation)

        # Identify peaks and troughs (5-year window)
        for i in range(2, len(inflation) - 2):
            window = inflation[i - 2 : i + 3]
            is_peak = inflation[i] == window.max() and inflation[i] > window[0]
            is_trough = inflation[i] == window.min() and inflation[i] < window[0]

            if not (is_peak or is_trough):
                continue

            # Check if this is a disaster: inflation deviates from target by > target level
            local_target = target[i]
            threshold = max(abs(local_target), 0.02)  # at least 2pp

            # Average inflation in 5-year window centered on peak/trough
            avg_inf = np.mean(inflation[max(0, i - 2) : min(len(inflation), i + 3)])

            if is_peak and (avg_inf - local_target) > threshold:
                # High-inflation disaster
                # Find the cycle boundaries
                cycle_start = max(0, i - 2)
                cycle_end = min(len(inflation) - 1, i + 2)
                cycle_gdp = gdp_growth[cycle_start : cycle_end + 1]

                # Check for output disaster (following Barro 2006: >10% cumulative drop)
                min_gdp = np.min(cycle_gdp) if len(cycle_gdp) > 0 else 0.0
                has_output_disaster = min_gdp < -0.10

                # Compute z = 1/(1+g) where g is the worst annual GDP growth
                z = 1.0 / (1.0 + min_gdp) if min_gdp < 0 else 1.0

                high_disasters.append({
                    "country": country,
                    "start_year": int(years[cycle_start]),
                    "end_year": int(years[cycle_end]),
                    "peak_inflation": float(inflation[i]),
                    "avg_inflation": float(avg_inf),
                    "has_output_disaster": has_output_disaster,
                    "min_gdp_growth": float(min_gdp),
                    "avg_gdp_growth": float(np.mean(cycle_gdp)),
                    "z": float(z),
                })

            elif is_trough and (local_target - avg_inf) > threshold:
                # Deflation disaster
                cycle_start = max(0, i - 2)
                cycle_end = min(len(inflation) - 1, i + 2)
                cycle_gdp = gdp_growth[cycle_start : cycle_end + 1]

                min_gdp = np.min(cycle_gdp) if len(cycle_gdp) > 0 else 0.0
                has_output_disaster = min_gdp < -0.10
                z = 1.0 / (1.0 + min_gdp) if min_gdp < 0 else 1.0

                low_disasters.append({
                    "country": country,
                    "start_year": int(years[cycle_start]),
                    "end_year": int(years[cycle_end]),
                    "peak_inflation": float(inflation[i]),
                    "avg_inflation": float(avg_inf),
                    "has_output_disaster": has_output_disaster,
                    "min_gdp_growth": float(min_gdp),
                    "avg_gdp_growth": float(np.mean(cycle_gdp)),
                    "z": float(z),
                })

    high_df = pd.DataFrame(high_disasters)
    low_df = pd.DataFrame(low_disasters)

    log.info(
        f"Identified {len(high_df)} high-inflation disasters, "
        f"{len(low_df)} deflation disasters "
        f"across {df['country'].nunique()} countries"
    )

    if not high_df.empty:
        n_joint_high = high_df["has_output_disaster"].sum()
        log.info(
            f"  High inflation: {n_joint_high}/{len(high_df)} "
            f"({n_joint_high / len(high_df):.1%}) overlap with output disasters"
        )
    if not low_df.empty:
        n_joint_low = low_df["has_output_disaster"].sum()
        log.info(
            f"  Deflation: {n_joint_low}/{len(low_df)} "
            f"({n_joint_low / len(low_df):.1%}) overlap with output disasters"
        )

    return high_df, low_df


def compute_disaster_statistics(
    high_disasters: pd.DataFrame,
    low_disasters: pd.DataFrame,
) -> dict:
    """Compute summary statistics matching Table 1 and 2 of the appendix.

    Returns dict with p_tilde (conditional probability of output disaster),
    and disaster counts.
    """
    stats = {}

    # Overall
    all_disasters = pd.concat([high_disasters, low_disasters])
    n_total = len(all_disasters)
    n_joint = all_disasters["has_output_disaster"].sum()
    stats["n_inflation_disasters"] = n_total
    stats["n_joint_disasters"] = int(n_joint)
    stats["p_tilde_pooled"] = n_joint / n_total if n_total > 0 else 0.0

    # High inflation
    n_high = len(high_disasters)
    n_joint_high = high_disasters["has_output_disaster"].sum() if n_high > 0 else 0
    stats["n_high_disasters"] = n_high
    stats["p_tilde_high"] = n_joint_high / n_high if n_high > 0 else 0.0

    # Deflation
    n_low = len(low_disasters)
    n_joint_low = low_disasters["has_output_disaster"].sum() if n_low > 0 else 0
    stats["n_low_disasters"] = n_low
    stats["p_tilde_low"] = n_joint_low / n_low if n_low > 0 else 0.0

    return stats
