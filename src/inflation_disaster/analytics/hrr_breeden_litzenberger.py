"""Breeden-Litzenberger density extraction — HRR 2024 eq. (10), p. 20.

Paper formula, verbatim (eq. 10):

    q_G(k) = e^{rT} * k * a''(k)

where:
  - k is the **gross cumulative inflation strike**, i.e. k = (1 + annual_rate)^T
  - a(k) is the ZC cap price as a function of gross cumulative strike
  - a''(k) is the second derivative wrt k
  - r is the REAL interest rate (from put-call parity, Birru-Figlewski 2012)
  - T is the horizon in years
  - q_G(k) is the Q-measure **DENSITY** of gross cumulative inflation at k

Derivation (consistent with paper eq. 8-11, p. 20):
  Eq. 11: N(k) = 1 + e^{iT} a'(k) is the N-CDF of gross inflation
  -> n_G(k) = e^{iT} a''(k) is the N-DENSITY of gross inflation
  Eq. 5: q(pi) = n(pi) e^{pi - pi^e}  (measure change, log-inflation)
  Converting to gross-inflation via Jacobian:
       q_G(k) = n_G(k) * k * e^{-pi^e}
              = e^{iT} a''(k) * k * e^{-(i-r)T}
              = e^{rT} * k * a''(k)   -> paper eq. (10) exactly.

So eq. (10) is the DENSITY, not the CDF (paper's Q() notation is loose).

CDF of gross inflation:
  Q_G(K) = integral from 0 to K of q_G(k') dk'

Tail probabilities:
  P(avg annual > theta)  = integral from K* to inf of q_G(k) dk, K* = (1+theta)^T
  P(avg annual < theta)  = integral from 0 to K* of q_G(k) dk

This replaces the earlier annual-rate-space version which had an incorrect
Jacobian that systematically suppressed the tail at long horizons.

The function expects cap prices at several strikes. It:
  1. Converts strikes from annual-rate to gross cumulative.
  2. Fits a cubic spline in gross-strike space with natural BCs.
  3. Computes a''(k) on a dense grid.
  4. Builds Q(k) via eq. (10).
  5. Enforces monotonicity and [0,1] bounds (B-L arbitrage sanity).
  6. Returns both the CDF on a dense grid and tail probabilities at
     standard thresholds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

log = logging.getLogger("inflation_disaster.analytics.hrr_breeden_litzenberger")

# Default thresholds for disaster probabilities
_STANDARD_THRESHOLDS = (-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


@dataclass
class HRRDensityResult:
    """Output of HRR B-L extraction at a single (region, maturity)."""

    region: str
    maturity: int
    real_rate: float
    swap_rate: float

    # Dense grid in annual-rate space
    annual_grid: np.ndarray              # avg annualized inflation, decimal
    gross_grid: np.ndarray               # (1+annual_grid)^T, gross cumulative
    q_cdf: np.ndarray                    # Q-measure CDF on annual_grid
    q_density_annual: np.ndarray         # pdf in annual-rate space (derived from CDF)

    # 8-bin probability vector (paper convention)
    bin_dist: np.ndarray                 # shape (8,), bins {≤-1, ..., >5}
    bin_labels: tuple

    # Tail probabilities at standard thresholds (decimals)
    prob_above: dict[float, float]       # P(avg > θ) for θ in standard set
    prob_below: dict[float, float]       # P(avg < θ)


_BIN_EDGES_ANNUAL = np.array([-np.inf, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, np.inf])
_BIN_LABELS = ("<=-1", "(-1,0]", "(0,1]", "(1,2]", "(2,3]", "(3,4]", "(4,5]", ">5")


def hrr_extract_q(
    strikes_annual: np.ndarray,
    cap_prices: np.ndarray,
    real_rate: float,
    maturity: int,
    swap_rate: float,
    region: str = "US",
    dense_grid_points: int = 4000,
    dense_grid_min_annual: float | None = None,
    dense_grid_max_annual: float | None = None,
) -> HRRDensityResult:
    """Extract Q-CDF and disaster probabilities via HRR eq. (10).

    Parameters
    ----------
    strikes_annual : array of strikes in **annualized decimal rate** form.
                     E.g. [−0.02, −0.01, 0.0, 0.01, 0.02, 0.025, 0.03, ..., 0.06].
    cap_prices : array of ZC cap prices at those strikes (decimal notional,
                 NOT bp). Cap pays max((1+y_realized)^T − (1+k)^T, 0).
    real_rate : real interest rate (decimal), ideally from put-call parity.
    maturity : T in years.
    swap_rate : ZC inflation swap rate for reference.
    region : 'US' or 'EZ'.
    dense_grid_points : number of points on the evaluation grid.
    dense_grid_min_annual, dense_grid_max_annual : endpoints of dense annual-rate grid.
                                                     Default to the input strike range.

    Returns
    -------
    HRRDensityResult
    """
    strikes_annual = np.asarray(strikes_annual, dtype=float)
    cap_prices = np.asarray(cap_prices, dtype=float)
    order = np.argsort(strikes_annual)
    strikes_annual = strikes_annual[order]
    cap_prices = cap_prices[order]

    # Transform strikes to GROSS CUMULATIVE space (k = (1+annual)^T)
    strikes_gross = (1.0 + strikes_annual) ** maturity

    # Spline in GROSS strike space with natural BCs
    spline = CubicSpline(strikes_gross, cap_prices, bc_type="natural", extrapolate=False)

    # Dense grid in annual-rate space, then convert to gross
    lo = dense_grid_min_annual if dense_grid_min_annual is not None else strikes_annual[0]
    hi = dense_grid_max_annual if dense_grid_max_annual is not None else strikes_annual[-1]
    annual_grid = np.linspace(lo, hi, dense_grid_points)
    gross_grid = (1.0 + annual_grid) ** maturity

    # Evaluate a''(k) on dense grid in gross strike space
    a_dd = spline(gross_grid, nu=2)
    # a'' should be >= 0 (positive butterflies); clip numerical noise
    a_dd = np.maximum(a_dd, 0.0)

    # HRR eq. (10): density of gross inflation q_G(k) = e^{rT} * k * a''(k)
    q_density_gross = np.exp(real_rate * maturity) * gross_grid * a_dd

    # Integrate density to get partial CDF of gross inflation within the
    # strike range. Critically we do NOT renormalize to 1 within the grid:
    # any missing mass represents real tail probability above the rightmost
    # strike (or below the leftmost). Paper uses SABR to extrapolate those
    # tails; here we truncate the strike range and account for the missing
    # mass explicitly via bin_dist tail entries.
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    q_cdf_gross = np.concatenate([[0.0], np.cumsum(
        (q_density_gross[:-1] + q_density_gross[1:]) / 2 * np.diff(gross_grid)
    )])
    # q_cdf_gross[-1] = P(gross inflation lies within strike range)
    # typically ~0.95-0.99 for wide strike grids. Missing mass sits in tails.
    total_mass = float(q_cdf_gross[-1])
    # Re-project CDF back to annual-rate grid (both grids are monotonically
    # related via gross_grid = (1+annual_grid)^T so they align pointwise)
    q_cdf = q_cdf_gross

    # Density in annual-rate space (pdf of average annualized inflation):
    # pdf_annual(theta) = pdf_gross(k) * dk/dtheta = pdf_gross * T(1+theta)^(T-1)
    jacobian = maturity * (1.0 + annual_grid) ** (maturity - 1)
    pdf_annual = q_density_gross * jacobian
    # Do NOT renormalize; preserve the raw density so tail truncation is visible

    # 8-bin integration in annual-rate space (within the strike range)
    bin_dist = np.zeros(8)
    for i in range(8):
        lo_e, hi_e = _BIN_EDGES_ANNUAL[i], _BIN_EDGES_ANNUAL[i + 1]
        mask = (annual_grid > lo_e) & (annual_grid <= hi_e)
        if mask.sum() > 1:
            bin_dist[i] = trap(pdf_annual[mask], annual_grid[mask])

    # Any residual missing mass outside [annual_grid[0], annual_grid[-1]]
    # is attributed to the nearest tail bin. Do NOT extrapolate with an
    # exponential fit — SABR's far-OTM vol can be artefactual and the fit
    # then inflates tail mass (empirically seen to add 70bp of spurious
    # deflation mass at 5Y when widening grid to -5%). Instead we rely on
    # the SABR grid being wide enough that missing mass is minimal; any
    # residual goes to whichever tail it is bounded to lie in.
    missing_mass = max(0.0, 1.0 - total_mass)
    left_missing = 0.0
    right_missing = 0.0
    if missing_mass > 0:
        # Attribute using density shape at edges (sign of asymmetry), but as a
        # simple split, not via aggressive exponential extrapolation.
        pdf_edge_L = float(pdf_annual[0])
        pdf_edge_R = float(pdf_annual[-1])
        denom = pdf_edge_L + pdf_edge_R
        if denom > 0:
            left_missing = missing_mass * pdf_edge_L / denom
            right_missing = missing_mass * pdf_edge_R / denom
        else:
            left_missing = missing_mass / 2
            right_missing = missing_mass / 2
        # Drop missing mass into the outermost bins (<=-1%, >5%). If the grid
        # itself already lies inside those bins' ranges, this is still correct.
        bin_dist[0] += left_missing
        bin_dist[7] += right_missing
    # Renormalize to ensure sum = 1.0 (defensive)
    if bin_dist.sum() > 0:
        bin_dist = bin_dist / bin_dist.sum()

    # Tail probs at standard thresholds: use within-grid integral only. Any
    # residual beyond the grid is small if the extrapolation range is chosen
    # wide enough (see run_hrr.py). Attempts to patch missing mass into these
    # reads (via density-ratio split or exponential decay) systematically
    # inflated the 5Y deflation wing — the SABR extrapolation at far OTM
    # strikes doesn't behave like a clean exponential tail, so any such patch
    # imports its own bias. Keep the reads honest to what B-L computes.
    def _prob_above(theta: float) -> float:
        idx = np.searchsorted(annual_grid, theta)
        if idx <= 0:
            return float(total_mass)
        if idx >= len(q_cdf):
            return 0.0
        return float(max(0.0, total_mass - q_cdf[idx]))

    def _prob_below(theta: float) -> float:
        idx = np.searchsorted(annual_grid, theta)
        if idx <= 0:
            return 0.0
        if idx >= len(q_cdf):
            return float(total_mass)
        return float(q_cdf[idx - 1])

    prob_above = {θ: _prob_above(θ) for θ in _STANDARD_THRESHOLDS}
    prob_below = {θ: _prob_below(θ) for θ in _STANDARD_THRESHOLDS}

    return HRRDensityResult(
        region=region,
        maturity=maturity,
        real_rate=real_rate,
        swap_rate=swap_rate,
        annual_grid=annual_grid,
        gross_grid=gross_grid,
        q_cdf=q_cdf,
        q_density_annual=pdf_annual,
        bin_dist=bin_dist,
        bin_labels=_BIN_LABELS,
        prob_above=prob_above,
        prob_below=prob_below,
    )
