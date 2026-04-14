"""Adjustment factor decomposition (reproduces Table 1 of the paper).

For each date, decomposes the total adjustment from naive N-probability
to final P-probability into the three factors:
  total = inflation_adj * horizon_adj * risk_adj
"""

from __future__ import annotations

import pandas as pd


def compute_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Add decomposition columns to the results DataFrame.

    Parameters
    ----------
    df : DataFrame from results_to_dataframe()

    Returns
    -------
    DataFrame with additional columns for total adjustment factors.
    """
    out = df.copy()

    # High inflation: total adjustment = P_5y5y / N_10y
    out["total_adj_high"] = out["prob_high_p_5y5y"] / out["prob_high_n_10y"].clip(lower=1e-10)

    # Deflation: total adjustment = P_5y5y / N_10y
    out["total_adj_low"] = out["prob_low_p_5y5y"] / out["prob_low_n_10y"].clip(lower=1e-10)

    # Verify decomposition: infl_adj * horizon_adj * risk_adj ~ total_adj
    out["decomp_check_high"] = (
        out["inflation_adj_10y"] * out["horizon_adj_high"] * out["risk_adj_high"]
    )
    out["decomp_check_low"] = (
        out["inflation_adj_10y"] * out["horizon_adj_low"] * out["risk_adj_low"]
    )

    return out


def table_1_panel_b(df: pd.DataFrame, region: str = "US") -> pd.DataFrame:
    """Reproduce Table 1 Panel B: High inflation adjustment factors.

    Returns median adjustment factors for the specified region.
    """
    subset = df[df["region"] == region]
    if subset.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "N to Q, 5y": [subset["inflation_adj_5y"].median()],
        "N to Q, 10y": [subset["inflation_adj_10y"].median()],
        "Q 10y to Q 5y5y": [subset["horizon_adj_high"].median()],
        "Q to P, 5y5y": [subset["risk_adj_high"].median()],
    }, index=[region])
