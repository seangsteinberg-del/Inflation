"""Time series plots for disaster probabilities (reproducing Figures 3, 4)."""

from __future__ import annotations

from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inflation_disaster.visualization.style import COLORS, apply_paper_style


def plot_deflation_probability(
    df: pd.DataFrame,
    region: str = "US",
    save_path: str | None = None,
):
    """Plot Figure 3: 5y5y deflation disaster probabilities over time.

    Parameters
    ----------
    df : DataFrame
        Must have columns: date, prob_low_p_5y5y, and optionally
        prob_severe_low_p_5y5y (for d=3% threshold).
    region : str
    save_path : str, optional
    """
    apply_paper_style()
    fig, ax = plt.subplots()

    if df.empty or "prob_low_p_5y5y" not in df.columns:
        ax.set_title(f"No deflation data available — {region}")
        return fig, ax

    dates = pd.to_datetime(df["date"])

    ax.plot(dates, df["prob_low_p_5y5y"], color=COLORS["deflation"],
            linewidth=1.5, label="Deflation (<0%)")

    if "prob_severe_low_p_5y5y" in df.columns:
        ax.plot(dates, df["prob_severe_low_p_5y5y"], color=COLORS["serious_deflation"],
                linewidth=1.5, linestyle="--", label="Serious deflation (<-1%)")

    if "prob_low_p_5y" in df.columns:
        ax.plot(dates, df["prob_low_p_5y"], color=COLORS["5y_deflation"],
                linewidth=1.0, linestyle=":", label="5-year deflation")

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right")
    ax.set_title(f"Probabilities of a deflation disaster — {region}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_high_inflation_probability(
    df: pd.DataFrame,
    save_path: str | None = None,
):
    """Plot Figure 4a: 5y5y high-inflation disaster probabilities for US and EZ.

    Parameters
    ----------
    df : DataFrame
        Must have columns: date, region, prob_high_p_5y5y.
    """
    apply_paper_style()
    fig, ax = plt.subplots()

    for region in ["US", "EZ"]:
        subset = df[df["region"] == region]
        if subset.empty:
            continue
        dates = pd.to_datetime(subset["date"])
        style = "-" if region == "US" else "--"
        ax.plot(dates, subset["prob_high_p_5y5y"], color=COLORS[region],
                linewidth=1.5, linestyle=style, label=region)

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right")
    ax.set_title("Probability of a high-inflation disaster (>4%)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_density_snapshots(
    densities: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Risk-neutral densities, 10-year horizon",
    save_path: str | None = None,
):
    """Plot Figure 4c/d: Risk-neutral density at selected dates.

    Parameters
    ----------
    densities : dict
        Keys are date labels (e.g., "Mar 2020"), values are (grid, density) tuples.
    """
    apply_paper_style()
    fig, ax = plt.subplots()

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(densities)))
    for (label, (grid, density)), color in zip(densities.items(), colors):
        ax.plot(grid * 100, density, label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Inflation (%)")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, 6)
    ax.legend()
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_markov_parameters(
    params_df: pd.DataFrame,
    region: str = "US",
    save_path: str | None = None,
):
    """Plot Figure 6: Markov model parameter estimates over time.

    Parameters
    ----------
    params_df : DataFrame
        Must have columns: date, p_dl, p_dh, p_nn.
    """
    apply_paper_style()
    fig, ax = plt.subplots()

    dates = pd.to_datetime(params_df["date"])

    ax.plot(dates, params_df["p_dl"], label="$p_{dl}$", linewidth=1.2)
    ax.plot(dates, params_df["p_dh"], label="$p_{dh}$", linewidth=1.2, linestyle="--")
    ax.plot(dates, params_df["p_nn"], label="$p_{nn}$", linewidth=1.2, linestyle=":")

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    ax.set_title(f"Inflation dynamics: model parameter estimates — {region}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
