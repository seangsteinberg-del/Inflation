"""Final disaster probability time series construction.

Collects DisasterProbability results across all months and regions
into structured DataFrames for analysis and visualization.
"""

from __future__ import annotations

import pandas as pd

from inflation_disaster.data.schemas import DisasterProbability


def results_to_dataframe(
    results: list[DisasterProbability],
) -> pd.DataFrame:
    """Convert a list of DisasterProbability objects to a DataFrame.

    Parameters
    ----------
    results : list of DisasterProbability

    Returns
    -------
    DataFrame with one row per date/region/threshold, all probability
    measures and adjustment factors as columns.
    """
    records = []
    for r in results:
        records.append({
            "date": r.date,
            "region": r.region,
            "threshold": r.threshold,
            "prob_high_n_5y": r.prob_high_n_5y,
            "prob_high_q_5y": r.prob_high_q_5y,
            "prob_high_n_10y": r.prob_high_n_10y,
            "prob_high_q_10y": r.prob_high_q_10y,
            "prob_high_q_5y5y": r.prob_high_q_5y5y,
            "prob_high_p_5y5y": r.prob_high_p_5y5y,
            "prob_low_n_5y": r.prob_low_n_5y,
            "prob_low_q_5y": r.prob_low_q_5y,
            "prob_low_n_10y": r.prob_low_n_10y,
            "prob_low_q_10y": r.prob_low_q_10y,
            "prob_low_q_5y5y": r.prob_low_q_5y5y,
            "prob_low_p_5y5y": r.prob_low_p_5y5y,
            "inflation_adj_5y": r.inflation_adj_5y,
            "inflation_adj_10y": r.inflation_adj_10y,
            "horizon_adj_high": r.horizon_adj_high,
            "horizon_adj_low": r.horizon_adj_low,
            "risk_adj_high": r.risk_adj_high,
            "risk_adj_low": r.risk_adj_low,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["region", "date"]).reset_index(drop=True)
    return df


def summary_statistics(
    df: pd.DataFrame,
    period_start: str | None = None,
    period_end: str | None = None,
) -> pd.DataFrame:
    """Compute summary statistics matching Table 1 of the paper.

    Returns median probabilities and adjustment factors by region.
    """
    if period_start:
        df = df[df["date"] >= period_start]
    if period_end:
        df = df[df["date"] <= period_end]

    stats = df.groupby("region").agg({
        "prob_high_n_5y": "median",
        "prob_high_q_5y": "median",
        "prob_high_n_10y": "median",
        "prob_high_q_10y": "median",
        "prob_high_q_5y5y": "median",
        "prob_high_p_5y5y": "median",
        "prob_low_n_5y": "median",
        "prob_low_q_5y": "median",
        "prob_low_n_10y": "median",
        "prob_low_q_10y": "median",
        "prob_low_q_5y5y": "median",
        "prob_low_p_5y5y": "median",
        "inflation_adj_5y": "median",
        "inflation_adj_10y": "median",
        "horizon_adj_high": "median",
        "horizon_adj_low": "median",
    })

    return stats
