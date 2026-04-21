"""GMM estimation of the HRR Markov chain — Appendix C.1 + main paper Sec. 4.3.3.

Paper spec, verbatim:

  "The data consist of 21 numbers per month, 7 for each of the 3 distributions:
   the cumulative distributions q(pi_{0,5}) and q(pi_{0,10}), and the average
   forward distribution q(pi_{5,6}). These are the moments that the model
   must hit, in a GMM procedure that assigns them equal weight."
   (main paper, p. 24-25)

  "Each month, we choose one (or sometimes more) trading days that have the
   highest quality data... For the year-on-year data ... we take the
   average of these five annual distributions."
   (Appendix A.4 + Sec. 4.3.1)

  "We kept three of the parameters fixed over the whole sample, while letting
   the other three vary across months. [...] The first constant parameter
   is pmr [...] The other two are the exit probabilities for disasters, pl, ph."
   (main paper, p. 25)

  "The main model is estimated at the quarterly frequency. [...] To obtain
   monthly estimates, we re-estimate the model separately at each month,
   maximizing fit only over the three time-varying parameters, while keeping
   fixed the three constant parameters estimated with the quarterly data."
   (Appendix C, p. 7)

Moments (21):
  (i)   7 bins of Q(pi_{0,5})   — 5Y cumulative Q from ZC B-L
  (ii)  7 bins of Q(pi_{0,10})  — 10Y cumulative Q from ZC B-L
  (iii) 7 bins of avg Q(pi_{t,t+1}) for t in {5,6,7,8,9}  — 1Y fwd avg

The 8th bin is redundant (distributions sum to 1), consistent with "7 for
each" in paper's text. We use the first 7 bins for the objective.

All 21 moments get equal weight. Objective = sum of squared deviations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from inflation_disaster.models.hrr_markov import (
    HRRMarkovParams,
    cumulative_distribution,
    forward_annual_distribution,
    hrr_validate_params,
)

log = logging.getLogger("inflation_disaster.models.hrr_gmm")


@dataclass
class HRRGMMData:
    """The three target distributions (each 8-bin) per paper Sec. 4.3.3."""
    cumul_5y: np.ndarray    # shape (8,)
    cumul_10y: np.ndarray   # shape (8,)
    fwd_1y_avg: np.ndarray  # shape (8,) -- average of q(pi_{5,6})..q(pi_{9,10})
    initial_state: np.ndarray  # shape (8,)


# Default parameter bounds per paper's descriptive text (probabilities in [0,1],
# but we constrain a little tighter to avoid pathological corners)
_DEFAULT_BOUNDS = {
    "p_dh": (1e-4, 0.30),
    "p_dl": (1e-4, 0.30),
    "p_nn": (1e-3, 0.40),
    "p_h":  (0.01, 0.40),
    "p_l":  (0.01, 0.40),
    "p_mr": (0.01, 0.80),
}


def _feasible_penalty(params: HRRMarkovParams) -> float:
    """Return 0 if all probs and derived probs are non-negative, else large penalty.

    Paper footnote 17-18: derived probs p_n, p_ml, p_mh, p_m must all be
    non-negative for the transition matrix to be a valid Markov chain.
    Row 0 and row 7 self-loops (1 - 5*pl, 1 - 5*ph) must also be in [0,1].
    Also guard against negative input parameters (optimizer may propose).
    """
    # Input-level guard
    for name in ("p_dh", "p_dl", "p_nn", "p_h", "p_l", "p_mr"):
        v = getattr(params, name)
        if v < 0.0 or v > 1.0:
            return 1e8
    d = hrr_validate_params(params)
    if any(v < -1e-9 for v in d.values()):
        return 1e8
    if d["row0_self"] < 0 or d["row7_self"] < 0:
        return 1e8
    return 0.0


def _moments_from_params(
    params: HRRMarkovParams,
    initial_state: np.ndarray,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three model-implied 8-bin distributions."""
    cumul_5 = cumulative_distribution(params, initial_state, horizon=5,
                                      n_paths=n_paths, seed=seed)
    cumul_10 = cumulative_distribution(params, initial_state, horizon=10,
                                       n_paths=n_paths, seed=seed + 1)
    fwd_avg = forward_annual_distribution(params, initial_state,
                                          years=(6, 7, 8, 9, 10))
    return cumul_5, cumul_10, fwd_avg


def _objective_21_moments(
    params: HRRMarkovParams,
    data: HRRGMMData,
    n_paths: int = 80_000,
    seed: int = 42,
) -> float:
    """GMM objective: SSE across 21 moments (7 bins x 3 dists), equal weight.

    Paper Sec. 4.3.3: "the 21 numbers per month, 7 for each of the 3
    distributions [...] equal weight."
    """
    penalty = _feasible_penalty(params)
    if penalty > 0:
        return penalty

    cumul_5, cumul_10, fwd_avg = _moments_from_params(
        params, data.initial_state, n_paths, seed
    )

    # Use first 7 bins of each distribution (the 8th is redundant)
    err = 0.0
    err += np.sum((cumul_5[:7] - data.cumul_5y[:7]) ** 2)
    err += np.sum((cumul_10[:7] - data.cumul_10y[:7]) ** 2)
    err += np.sum((fwd_avg[:7] - data.fwd_1y_avg[:7]) ** 2)
    return float(err)


def hrr_gmm_fit_quarterly(
    data: HRRGMMData,
    initial_guess: dict | None = None,
    n_paths_fast: int = 60_000,
    n_paths_refine: int = 200_000,
    seed: int = 42,
) -> HRRMarkovParams:
    """Fit all 6 parameters at quarterly frequency (paper Sec. 4.3.3).

    Paper: "The main model is estimated at the quarterly frequency."
    """
    x0_dict = initial_guess or {
        "p_dh": 0.04, "p_dl": 0.03, "p_nn": 0.10,
        "p_h": 0.20,  "p_l": 0.20,  "p_mr": 0.45,
    }
    param_order = ["p_dh", "p_dl", "p_nn", "p_h", "p_l", "p_mr"]
    x0 = np.array([x0_dict[k] for k in param_order])
    bounds = [_DEFAULT_BOUNDS[k] for k in param_order]

    def _obj(x):
        p = HRRMarkovParams(**dict(zip(param_order, x)))
        return _objective_21_moments(p, data, n_paths=n_paths_fast, seed=seed)

    # Multi-start local search: paper uses GMM; we use a robust Nelder-Mead
    # sweep from several starts to avoid local minima on the MC-noisy objective.
    starts = [
        x0,
        np.array([0.02, 0.02, 0.05, 0.20, 0.20, 0.45]),
        np.array([0.06, 0.06, 0.15, 0.20, 0.20, 0.45]),
        np.array([0.03, 0.05, 0.10, 0.15, 0.15, 0.50]),
        np.array([0.05, 0.03, 0.10, 0.25, 0.25, 0.40]),
    ]
    best_cost = np.inf
    best_x = x0
    for s in starts:
        try:
            r = minimize(_obj, x0=s, method="Nelder-Mead",
                         options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-8})
            if r.fun < best_cost:
                best_cost, best_x = r.fun, r.x
        except Exception as e:
            log.warning(f"GMM start {s} failed: {e}")

    # Refine best with more MC paths
    def _obj_refine(x):
        p = HRRMarkovParams(**dict(zip(param_order, x)))
        return _objective_21_moments(p, data, n_paths=n_paths_refine, seed=seed)

    r = minimize(_obj_refine, x0=best_x, method="Nelder-Mead",
                 options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-10})
    if r.fun < best_cost:
        best_x = r.x
        best_cost = r.fun

    # Clip to bounds
    for i, (lo, hi) in enumerate(bounds):
        best_x[i] = np.clip(best_x[i], lo, hi)

    params = HRRMarkovParams(**dict(zip(param_order, best_x)))
    log.info(f"Quarterly GMM: cost={best_cost:.6e}, params={params}")
    return params


def hrr_gmm_fit_monthly(
    data: HRRGMMData,
    fixed_params: dict,  # {"p_h": ..., "p_l": ..., "p_mr": ...}
    initial_guess: dict | None = None,
    n_paths_fast: int = 60_000,
    n_paths_refine: int = 200_000,
    seed: int = 42,
) -> HRRMarkovParams:
    """Fit (p_dh, p_dl, p_nn) monthly, holding (p_h, p_l, p_mr) at their
    quarterly values (paper Appendix C, p. 7).
    """
    x0_dict = initial_guess or {"p_dh": 0.04, "p_dl": 0.03, "p_nn": 0.10}
    tv_order = ["p_dh", "p_dl", "p_nn"]
    x0 = np.array([x0_dict[k] for k in tv_order])
    bounds = [_DEFAULT_BOUNDS[k] for k in tv_order]

    def _build(x):
        return HRRMarkovParams(**dict(zip(tv_order, x)), **fixed_params)

    def _obj(x):
        return _objective_21_moments(_build(x), data, n_paths=n_paths_fast, seed=seed)

    # Multi-start search over the 3-parameter surface
    starts = [
        x0,
        np.array([0.02, 0.02, 0.05]),
        np.array([0.06, 0.05, 0.15]),
        np.array([0.03, 0.08, 0.10]),
        np.array([0.08, 0.02, 0.12]),
    ]
    best_cost = np.inf
    best_x = x0
    for s in starts:
        try:
            r = minimize(_obj, x0=s, method="Nelder-Mead",
                         options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-8})
            if r.fun < best_cost:
                best_cost, best_x = r.fun, r.x
        except Exception as e:
            log.warning(f"monthly GMM start {s} failed: {e}")

    def _obj_refine(x):
        return _objective_21_moments(_build(x), data, n_paths=n_paths_refine, seed=seed)

    r = minimize(_obj_refine, x0=best_x, method="Nelder-Mead",
                 options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-10})
    if r.fun < best_cost:
        best_x = r.x
        best_cost = r.fun

    for i, (lo, hi) in enumerate(bounds):
        best_x[i] = np.clip(best_x[i], lo, hi)

    params = _build(best_x)
    log.info(f"Monthly GMM: cost={best_cost:.6e}, params={params}")
    return params
