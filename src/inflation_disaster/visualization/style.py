"""Chart styling consistent with paper figures."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLORS = {
    "US": "#003366",
    "EZ": "#CC0000",
    "US_light": "#6699CC",
    "EZ_light": "#FF6666",
    "deflation": "#003366",
    "serious_deflation": "#003366",
    "high_inflation": "#CC0000",
    "severe_inflation": "#CC0000",
    "5y_deflation": "#669900",
}

LINE_STYLES = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
}


def apply_paper_style():
    """Set matplotlib style to match paper figures."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
