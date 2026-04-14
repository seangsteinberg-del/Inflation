"""Full backtest: replicate the paper's disaster probability time series.

Calibrates Markov chain parameters for each of 181 months (Jan 2011 - Feb 2026)
using the paper's published Q-measure probabilities, then produces P-measure
5y5y disaster probabilities.

Key improvement over quick test: uses approximate historical CPI to set
the correct initial inflation state for each month.

Usage: python backtest.py [--quick]  (quick = every 6th month)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from pathlib import Path
from scipy.optimize import minimize

logging.basicConfig(level=logging.WARNING)

from inflation_disaster.models.markov_chain import (
    build_transition_matrix, simulate_forward_distribution,
)
from inflation_disaster.adjustments.horizon_adj import extract_disaster_probabilities
from inflation_disaster.data.schemas import MarkovParams
from inflation_disaster.data.surface_builder import determine_inflation_state
from inflation_disaster.config import settings


# Approximate US CPI YoY by date range (from FRED/public data)
US_CPI_APPROX = [
    ("2009-10", "2010-12", 1.5), ("2011-01", "2011-06", 3.0),
    ("2011-07", "2012-06", 2.0), ("2012-07", "2013-06", 1.5),
    ("2013-07", "2014-12", 1.5), ("2015-01", "2015-12", 0.5),
    ("2016-01", "2016-12", 1.5), ("2017-01", "2017-12", 2.0),
    ("2018-01", "2018-12", 2.2), ("2019-01", "2019-12", 1.8),
    ("2020-01", "2020-06", 0.5), ("2020-07", "2020-12", 1.2),
    ("2021-01", "2021-06", 4.0), ("2021-07", "2021-12", 6.5),
    ("2022-01", "2022-06", 8.0), ("2022-07", "2022-12", 7.5),
    ("2023-01", "2023-06", 4.5), ("2023-07", "2023-12", 3.5),
    ("2024-01", "2024-12", 3.2), ("2025-01", "2025-12", 2.8),
    ("2026-01", "2026-12", 3.3),
]

# Approximate EZ HICP YoY
EZ_CPI_APPROX = [
    ("2011-01", "2011-12", 2.5), ("2012-01", "2012-12", 2.2),
    ("2013-01", "2013-12", 1.0), ("2014-01", "2014-12", 0.5),
    ("2015-01", "2015-12", 0.0), ("2016-01", "2016-12", 0.5),
    ("2017-01", "2017-12", 1.5), ("2018-01", "2018-12", 1.8),
    ("2019-01", "2019-12", 1.2), ("2020-01", "2020-12", 0.5),
    ("2021-01", "2021-06", 2.0), ("2021-07", "2021-12", 4.5),
    ("2022-01", "2022-06", 8.0), ("2022-07", "2022-12", 9.5),
    ("2023-01", "2023-06", 7.0), ("2023-07", "2023-12", 3.5),
    ("2024-01", "2024-12", 2.5), ("2025-01", "2025-12", 2.2),
    ("2026-01", "2026-12", 2.5),
]


def get_approx_cpi(dt, region):
    """Get approximate CPI YoY for a given date and region."""
    table = US_CPI_APPROX if region == "US" else EZ_CPI_APPROX
    dt_str = dt.strftime("%Y-%m")
    for start, end, cpi in table:
        if start <= dt_str <= end:
            return cpi
    return 2.0  # default


def load_paper_data():
    data_dir = Path(__file__).parent / "data" / "paper_data"
    results = {}
    for region, fname in [("US", "USwestimates.dta"), ("EZ", "EZwestimates.dta")]:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_stata(str(fpath))
            df["date"] = pd.to_datetime(df["date_ym"])
            df = df.sort_values("date").reset_index(drop=True)
            results[region] = df
    return results


def calibrate_month(target_h, target_l, initial_state, region, prev_params=None):
    """Calibrate Markov params for one month. Uses warm-start from previous month."""
    p_h = settings.us_p_h if region == "US" else settings.ez_p_h
    p_l = settings.us_p_l if region == "US" else settings.ez_p_l
    p_mr = settings.us_p_mr if region == "US" else settings.ez_p_mr

    def objective(x):
        p_dh, p_dl, p_nn = x
        if p_dh < 0 or p_dl < 0 or p_nn < 0:
            return 1e6
        if 2*p_nn + p_dl + p_dh > 0.95:
            return 1e6
        if p_dl + p_nn + p_mr > 0.95 or p_dh + p_nn + p_mr > 0.95:
            return 1e6
        try:
            par = MarkovParams(p_dh=p_dh, p_dl=p_dl, p_nn=p_nn,
                               p_h=p_h, p_l=p_l, p_mr=p_mr)
        except ValueError:
            return 1e6
        fwd = simulate_forward_distribution(par, initial_state, T=5, H=5, n_paths=100_000)
        qh, ql = extract_disaster_probabilities(fwd, 2.0, 2.0)
        return (qh * 0.66 - target_h)**2 + (ql * 0.96 - target_l)**2

    starts = [
        (0.05, 0.03, 0.12), (0.03, 0.05, 0.10), (0.08, 0.05, 0.15),
        (0.01, 0.07, 0.12), (0.005, 0.02, 0.15), (0.02, 0.08, 0.10),
        (0.04, 0.04, 0.08), (0.10, 0.03, 0.08),
    ]
    # Warm-start from previous month's solution
    if prev_params is not None:
        starts.insert(0, (prev_params.p_dh, prev_params.p_dl, prev_params.p_nn))

    best = None
    for x0 in starts:
        try:
            r = minimize(objective, x0=x0, method="Nelder-Mead",
                        options={"maxiter": 400, "xatol": 0.0005})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue

    if best is None or best.fun > 0.01:
        return None

    p_dh, p_dl, p_nn = [np.clip(v, 0.001, 0.15) for v in best.x]
    par = MarkovParams(p_dh=p_dh, p_dl=p_dl, p_nn=p_nn,
                       p_h=p_h, p_l=p_l, p_mr=p_mr)

    # Final estimate with more paths
    fwd = simulate_forward_distribution(par, initial_state, T=5, H=5, n_paths=200_000)
    qh, ql = extract_disaster_probabilities(fwd, 2.0, 2.0)
    return par, qh * 0.66, ql * 0.96, best.fun


def run_backtest(quick=False):
    paper = load_paper_data()
    all_results = {}

    for region in ["US", "EZ"]:
        if region not in paper:
            continue
        df = paper[region]
        if quick:
            indices = list(range(0, len(df), 6))
            if len(df) - 1 not in indices:
                indices.append(len(df) - 1)
        else:
            indices = list(range(len(df)))

        n = len(indices)
        print(f"\n{'='*70}")
        print(f"{region}: Backtesting {n} months "
              f"({df['date'].iloc[indices[0]].date()} to {df['date'].iloc[indices[-1]].date()})")
        print(f"{'='*70}")

        results = []
        prev_params = None

        for count, i in enumerate(indices):
            row = df.iloc[i]
            dt = row["date"]
            target_h = row["higher4_5y5y"]
            target_l = row["lower0_5y5y"]

            cpi = get_approx_cpi(dt, region)
            initial_state = determine_inflation_state(cpi)

            result = calibrate_month(target_h, target_l, initial_state, region, prev_params)

            if result is None:
                results.append({
                    "date": dt, "paper_high": target_h, "paper_low": target_l,
                    "our_high": np.nan, "our_low": np.nan, "cost": np.nan,
                })
                sys.stdout.write(f"\r  [{count+1}/{n}] {dt.date()}: FAILED   ")
            else:
                par, p_h, p_l, cost = result
                prev_params = par
                results.append({
                    "date": dt, "paper_high": target_h, "paper_low": target_l,
                    "our_high": p_h, "our_low": p_l,
                    "p_dh": par.p_dh, "p_dl": par.p_dl, "p_nn": par.p_nn,
                    "cost": cost,
                })
                sys.stdout.write(
                    f"\r  [{count+1}/{n}] {dt.date()}: "
                    f"H={p_h:.1%}/{target_h:.1%} L={p_l:.1%}/{target_l:.1%} "
                    f"(err={cost:.6f})  "
                )
            sys.stdout.flush()

        print()
        rdf = pd.DataFrame(results)
        all_results[region] = rdf

        valid = rdf.dropna(subset=["our_high"])
        if len(valid) == 0:
            print("  No valid results!")
            continue

        err_h = (valid["our_high"] - valid["paper_high"]).abs()
        err_l = (valid["our_low"] - valid["paper_low"]).abs()
        corr_h = valid["our_high"].corr(valid["paper_high"])
        corr_l = valid["our_low"].corr(valid["paper_low"])

        print(f"\n  RESULTS ({len(valid)}/{n} months):")
        print(f"  High inflation:  MAE={err_h.mean()*100:.2f}pp  "
              f"Median={err_h.median()*100:.2f}pp  Max={err_h.max()*100:.2f}pp  "
              f"Corr={corr_h:.3f}")
        print(f"  Deflation:       MAE={err_l.mean()*100:.2f}pp  "
              f"Median={err_l.median()*100:.2f}pp  Max={err_l.max()*100:.2f}pp  "
              f"Corr={corr_l:.3f}")

        print(f"\n  TABLE 1 MEDIANS:")
        print(f"    P(>4%, 5y5y): Ours={valid['our_high'].median()*100:.1f}%  "
              f"Paper={valid['paper_high'].median()*100:.1f}%")
        print(f"    P(<0%, 5y5y): Ours={valid['our_low'].median()*100:.1f}%  "
              f"Paper={valid['paper_low'].median()*100:.1f}%")

    return all_results


def plot_backtest(results, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Full Backtest: Our Pipeline vs Paper (Hilscher, Raviv & Reis 2024)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    panels = [
        (0, 0, "US", "high", "US: P(>4%, 5y5y)", "#d62728"),
        (0, 1, "US", "low", "US: P(<0%, 5y5y)", "#1f77b4"),
        (1, 0, "EZ", "high", "EZ: P(>4%, 5y5y)", "#d62728"),
        (1, 1, "EZ", "low", "EZ: P(<0%, 5y5y)", "#1f77b4"),
    ]

    for row, col, region, tail, title, color in panels:
        ax = axes[row, col]
        if region not in results:
            continue
        df = results[region].dropna(subset=[f"our_{tail}"])

        ax.plot(df["date"], df[f"paper_{tail}"] * 100, color=color, linewidth=1.5,
                label="Paper (HRR 2024)", alpha=0.9)
        ax.plot(df["date"], df[f"our_{tail}"] * 100, color="black", linewidth=1.2,
                linestyle="--", label="Our replication", alpha=0.8)

        err = (df[f"our_{tail}"] - df[f"paper_{tail}"]).abs().mean() * 100
        corr = df[f"our_{tail}"].corr(df[f"paper_{tail}"])
        ax.set_title(f"{title}\nMAE={err:.1f}pp, Corr={corr:.2f}", fontsize=11)
        ax.set_ylabel("Probability (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "backtest_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()

    # Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Backtest Accuracy: Our Replication vs Paper", fontsize=13, fontweight="bold")
    for i, region in enumerate(["US", "EZ"]):
        if region not in results:
            continue
        ax = axes[i]
        df = results[region].dropna(subset=["our_high"])
        ax.scatter(df["paper_high"]*100, df["our_high"]*100,
                  color="#d62728", alpha=0.5, s=15, label="High inflation")
        ax.scatter(df["paper_low"]*100, df["our_low"]*100,
                  color="#1f77b4", alpha=0.5, s=15, label="Deflation")
        mx = max(df[["paper_high","our_high","paper_low","our_low"]].max()) * 110
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        ax.set_xlabel("Paper (%)"); ax.set_ylabel("Our Pipeline (%)")
        ax.set_title(region); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "backtest_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Every 6th month only")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL BACKTEST: Hilscher, Raviv & Reis (2024) Replication")
    print("=" * 70)

    results = run_backtest(quick=args.quick)
    plot_backtest(results)

    output_dir = Path(__file__).parent / "data" / "output"
    for region, df in results.items():
        path = output_dir / f"backtest_{region}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
