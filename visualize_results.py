"""Visualize inflation disaster probability results vs paper.

Creates publication-quality charts comparing:
1. Paper's full time series of disaster probabilities
2. Our live Bloomberg-derived estimate (latest point)
3. Key adjustment factors and distributions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from pathlib import Path


def load_paper_data():
    """Load all paper estimates from .dta files."""
    data_dir = Path(__file__).parent / "data" / "paper_data"
    results = {}
    for region, fname in [("US", "USwestimates.dta"), ("EZ", "EZwestimates.dta")]:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_stata(str(fpath))
            df["date"] = pd.to_datetime(df["date_ym"])
            results[region] = df
    return results


def plot_disaster_probabilities(paper_data, our_results=None, output_path=None):
    """Create the main 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Inflation Disaster Probabilities: Paper vs Live Bloomberg Replication",
        fontsize=14, fontweight="bold", y=0.98,
    )

    panels = [
        (0, 0, "US", "higher4_5y5y", "US: P(avg inflation > 4%, 5y5y)", "#d62728"),
        (0, 1, "US", "lower0_5y5y", "US: P(avg inflation < 0%, 5y5y)", "#1f77b4"),
        (1, 0, "EZ", "higher4_5y5y", "EZ: P(avg inflation > 4%, 5y5y)", "#d62728"),
        (1, 1, "EZ", "lower0_5y5y", "EZ: P(avg inflation < 0%, 5y5y)", "#1f77b4"),
    ]

    for row, col, region, var, title, color in panels:
        ax = axes[row, col]
        if region in paper_data:
            df = paper_data[region]
            ax.plot(df["date"], df[var] * 100, color=color, linewidth=1.5,
                    label="Paper (HRR 2024)", alpha=0.9)
            ax.fill_between(df["date"], 0, df[var] * 100, color=color, alpha=0.15)

            # Mark our live estimate
            if our_results and region in our_results:
                r = our_results[region]
                our_val = r.get("p_5y5y_high" if "higher4" in var else "p_5y5y_low", 0)
                today = date.today()
                ax.scatter([today], [our_val * 100], color="black", s=100,
                          zorder=5, marker="D", label=f"Live Bloomberg ({today})")
                ax.annotate(f"{our_val*100:.1f}%",
                           (today, our_val * 100),
                           textcoords="offset points", xytext=(10, 10),
                           fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Probability (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.close()


def plot_q_measure_comparison(paper_data, output_path=None):
    """Plot Q-measure spot probabilities over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Measure Spot Disaster Probabilities (from ZC Options)",
                 fontsize=14, fontweight="bold", y=0.98)

    panels = [
        (0, 0, "US", "zc_higher4_5y", "US: Q(>4%, 5Y ZC)", "#d62728"),
        (0, 1, "US", "zc_higher4_10y", "US: Q(>4%, 10Y ZC)", "#ff7f0e"),
        (1, 0, "EZ", "zc_higher4_5y", "EZ: Q(>4%, 5Y ZC)", "#d62728"),
        (1, 1, "EZ", "zc_higher4_10y", "EZ: Q(>4%, 10Y ZC)", "#ff7f0e"),
    ]

    for row, col, region, var, title, color in panels:
        ax = axes[row, col]
        if region in paper_data:
            df = paper_data[region]
            ax.plot(df["date"], df[var] * 100, color=color, linewidth=1.5)
            ax.fill_between(df["date"], 0, df[var] * 100, color=color, alpha=0.15)

            # Also plot deflation
            defl_var = var.replace("higher4", "lower0")
            if defl_var in df.columns:
                ax.plot(df["date"], df[defl_var] * 100, color="#1f77b4",
                       linewidth=1.5, linestyle="--", label="Deflation Q(<0%)")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Probability (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.close()


def plot_summary_table(paper_data, our_results, output_path=None):
    """Create a visual summary comparing our results to paper."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Build comparison table
    rows = []
    for region in ["US", "EZ"]:
        if region in our_results and region in paper_data:
            df = paper_data[region]
            latest = df.iloc[-1]
            r = our_results[region]

            rows.extend([
                [f"{region} P(>4%, 5y5y)",
                 f"{r['p_5y5y_high']*100:.1f}%",
                 f"{latest['higher4_5y5y']*100:.1f}%",
                 f"{(r['p_5y5y_high']-latest['higher4_5y5y'])*100:+.1f}pp"],
                [f"{region} P(<0%, 5y5y)",
                 f"{r['p_5y5y_low']*100:.1f}%",
                 f"{latest['lower0_5y5y']*100:.1f}%",
                 f"{(r['p_5y5y_low']-latest['lower0_5y5y'])*100:+.1f}pp"],
            ])

    table = ax.table(
        cellText=rows,
        colLabels=["Measure", "Our Estimate", "Paper (Feb 2026)", "Difference"],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color the difference column
    for i, row in enumerate(rows):
        diff_val = float(row[3].replace("pp", "").replace("+", ""))
        if abs(diff_val) < 0.5:
            table[i+1, 3].set_facecolor("#c8e6c9")  # green
        elif abs(diff_val) < 1.5:
            table[i+1, 3].set_facecolor("#fff9c4")  # yellow
        else:
            table[i+1, 3].set_facecolor("#ffcdd2")  # red

    ax.set_title(
        f"Live Bloomberg Replication vs Paper — {date.today()}\n"
        "Hilscher, Raviv & Reis (2024)",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    paper = load_paper_data()
    print(f"Loaded paper data: US={len(paper.get('US', []))} obs, "
          f"EZ={len(paper.get('EZ', []))} obs")

    # Our latest results (from the run_live.py output)
    our_results = {
        "US": {
            "p_5y5y_high": 0.025, "p_5y5y_low": 0.043,
            "q_5y5y_high": 0.037, "q_5y5y_low": 0.042,
        },
        "EZ": {
            "p_5y5y_high": 0.053, "p_5y5y_low": 0.029,
            "q_5y5y_high": 0.080, "q_5y5y_low": 0.030,
        },
    }

    plot_disaster_probabilities(
        paper, our_results,
        output_path=output_dir / "disaster_probs_comparison.png",
    )

    plot_q_measure_comparison(
        paper,
        output_path=output_dir / "q_measure_spot_probs.png",
    )

    plot_summary_table(
        paper, our_results,
        output_path=output_dir / "results_summary_table.png",
    )

    print("\nAll visualizations saved to data/output/")
