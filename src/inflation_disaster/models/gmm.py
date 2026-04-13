"""GMM estimator for the Markov chain inflation model.

Matches 21 moment conditions per month:
- 7 bin probabilities from 5y ZC options (cumulative 5y distribution)
- 7 bin probabilities from 10y ZC options (cumulative 10y distribution)
- 7 bin probabilities from YOY options (average forward annual distribution)

Two-stage estimation:
  Stage 1: Estimate all 6 params each quarter freely
  Stage 2: Fix p_h, p_l, p_mr at medians from Stage 1,
           re-estimate p_dh, p_dl, p_nn monthly
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams
from inflation_disaster.models.markov_chain import (
    marginal_one_year_distribution,
    simulate_cumulative_distribution,
)

log = logging.getLogger("inflation_disaster.models.gmm")


@dataclass
class GMMTargets:
    """Target moments for one month."""

    date: object
    q_5y: np.ndarray  # 8-element Q-measure 5y bin probabilities
    q_10y: np.ndarray  # 8-element Q-measure 10y bin probabilities
    q_yoy_avg: np.ndarray  # 8-element average forward YOY distribution
    initial_state: np.ndarray  # 8-element current state distribution


def _moment_conditions(
    param_vector: np.ndarray,
    targets: GMMTargets,
    constant_params: np.ndarray | None = None,
    n_paths: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Compute 21-vector of moment conditions g(theta).

    g_j = model_prob_j - empirical_prob_j

    Parameters
    ----------
    param_vector : array
        If constant_params is None: [p_dh, p_dl, p_nn, p_h, p_l, p_mr]
        If constant_params given: [p_dh, p_dl, p_nn]
    targets : GMMTargets
    constant_params : array, optional
        [p_h, p_l, p_mr] if fixing constant parameters.
    """
    if constant_params is not None:
        p_dh, p_dl, p_nn = param_vector
        p_h, p_l, p_mr = constant_params
    else:
        p_dh, p_dl, p_nn, p_h, p_l, p_mr = param_vector

    # Validate bounds
    if any(p < 0 or p > 1 for p in [p_dh, p_dl, p_nn, p_h, p_l, p_mr]):
        return np.full(21, 1e6)

    # Check feasibility
    p_n = 1.0 - 2 * p_nn - p_dl - p_dh
    if p_n < 0:
        return np.full(21, 1e6)

    try:
        params = MarkovParams(
            p_dh=p_dh, p_dl=p_dl, p_nn=p_nn,
            p_h=p_h, p_l=p_l, p_mr=p_mr,
        )
    except Exception:
        return np.full(21, 1e6)

    # Model-implied distributions
    model_5y = simulate_cumulative_distribution(
        params, targets.initial_state, horizon=5,
        n_paths=n_paths, seed=seed,
    )
    model_10y = simulate_cumulative_distribution(
        params, targets.initial_state, horizon=10,
        n_paths=n_paths, seed=seed,
    )

    # Forward YOY: average of marginal distributions at years 5-9
    # (state at start of year determines that year's inflation)
    model_yoy = np.zeros(8)
    for year in range(5, 10):
        model_yoy += marginal_one_year_distribution(
            params, targets.initial_state, year
        )
    model_yoy /= 5.0

    # We use 7 moments per distribution (drop one bin for identification,
    # since they sum to 1). Drop the last bin.
    g = np.concatenate([
        (model_5y[:7] - targets.q_5y[:7]),
        (model_10y[:7] - targets.q_10y[:7]),
        (model_yoy[:7] - targets.q_yoy_avg[:7]),
    ])

    return g


def _gmm_objective(
    param_vector: np.ndarray,
    targets: GMMTargets,
    W: np.ndarray,
    constant_params: np.ndarray | None = None,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """GMM criterion: g(theta)' W g(theta)."""
    g = _moment_conditions(
        param_vector, targets, constant_params, n_paths, seed
    )
    return float(g @ W @ g)


def estimate_single_month(
    targets: GMMTargets,
    constant_params: np.ndarray | None = None,
    n_starts: int | None = None,
    n_paths: int = 50_000,
) -> tuple[MarkovParams, float]:
    """Estimate Markov chain parameters for a single month.

    Parameters
    ----------
    targets : GMMTargets
        Empirical moment targets.
    constant_params : array, optional
        [p_h, p_l, p_mr] if fixing constant parameters.
    n_starts : int
        Number of random starting points.
    n_paths : int
        MC paths per objective evaluation.

    Returns
    -------
    params : MarkovParams
    obj_value : float
        Final GMM objective value.
    """
    if n_starts is None:
        n_starts = settings.gmm_n_starts

    # Identity weighting matrix (equal weight to all moments)
    W = np.eye(21)

    if constant_params is not None:
        # Optimize over 3 time-varying params only
        bounds = [
            (1e-6, 0.3),   # p_dh
            (1e-6, 0.3),   # p_dl
            (1e-6, 0.45),  # p_nn
        ]
        n_params = 3
    else:
        # Optimize over all 6 params
        bounds = [
            (1e-6, 0.3),   # p_dh
            (1e-6, 0.3),   # p_dl
            (1e-6, 0.45),  # p_nn
            (1e-6, 0.2),   # p_h
            (1e-6, 0.2),   # p_l
            (0.1, 0.7),    # p_mr
        ]
        n_params = 6

    best_result = None
    best_cost = np.inf

    rng = np.random.RandomState(settings.mc_seed)
    for i in range(n_starts):
        x0 = np.array([
            rng.uniform(b[0], b[1]) for b in bounds
        ])

        try:
            result = minimize(
                _gmm_objective,
                x0=x0,
                args=(targets, W, constant_params, n_paths, settings.mc_seed + i),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-10},
            )

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        except Exception as e:
            log.debug(f"Optimization start {i} failed: {e}")
            continue

    if best_result is None:
        log.warning(f"GMM estimation failed for {targets.date}")
        return MarkovParams(
            p_dh=0.05, p_dl=0.05, p_nn=0.15,
            p_h=0.20, p_l=0.20, p_mr=0.50,
        ), np.inf

    x = best_result.x
    if constant_params is not None:
        params = MarkovParams(
            p_dh=x[0], p_dl=x[1], p_nn=x[2],
            p_h=constant_params[0], p_l=constant_params[1], p_mr=constant_params[2],
        )
    else:
        params = MarkovParams(
            p_dh=x[0], p_dl=x[1], p_nn=x[2],
            p_h=x[3], p_l=x[4], p_mr=x[5],
        )

    log.info(
        f"GMM {targets.date}: obj={best_cost:.6f}, "
        f"p_dh={params.p_dh:.4f}, p_dl={params.p_dl:.4f}, p_nn={params.p_nn:.4f}"
    )

    return params, best_cost


def _estimate_month_wrapper(args):
    """Wrapper for parallel execution."""
    targets, constant_params, n_paths = args
    return estimate_single_month(targets, constant_params, n_paths=n_paths)


def estimate_full_sample(
    all_targets: list[GMMTargets],
    n_paths_stage1: int = 30_000,
    n_paths_stage2: int = 100_000,
    max_workers: int = 4,
) -> tuple[np.ndarray, list[MarkovParams]]:
    """Two-stage GMM estimation over the full sample.

    Stage 1: Estimate all 6 parameters each quarter freely.
    Stage 2: Fix p_h, p_l, p_mr at medians, re-estimate p_dh, p_dl, p_nn monthly.

    Parameters
    ----------
    all_targets : list of GMMTargets
        Monthly targets.
    n_paths_stage1, n_paths_stage2 : int
        MC paths for each stage.
    max_workers : int
        Parallel workers.

    Returns
    -------
    constant_params : array [p_h, p_l, p_mr]
    monthly_params : list of MarkovParams
    """
    n_months = len(all_targets)

    # --- Stage 1: quarterly estimation of all 6 params ---
    log.info(f"Stage 1: estimating all 6 params for {n_months // 3 + 1} quarters")
    quarterly_targets = all_targets[::3]  # every 3rd month

    stage1_params = []
    for targets in quarterly_targets:
        params, obj = estimate_single_month(
            targets, constant_params=None, n_paths=n_paths_stage1,
        )
        stage1_params.append(params)

    # Extract constant params as medians
    p_h_vals = [p.p_h for p in stage1_params]
    p_l_vals = [p.p_l for p in stage1_params]
    p_mr_vals = [p.p_mr for p in stage1_params]

    constant_params = np.array([
        np.median(p_h_vals),
        np.median(p_l_vals),
        np.median(p_mr_vals),
    ])

    log.info(
        f"Stage 1 constant params: p_h={constant_params[0]:.4f}, "
        f"p_l={constant_params[1]:.4f}, p_mr={constant_params[2]:.4f}"
    )

    # --- Stage 2: monthly estimation of 3 time-varying params ---
    log.info(f"Stage 2: estimating 3 time-varying params for {n_months} months")

    args_list = [
        (targets, constant_params, n_paths_stage2)
        for targets in all_targets
    ]

    monthly_params = []
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_estimate_month_wrapper, args_list))
        for params, obj in results:
            monthly_params.append(params)
    else:
        for args in args_list:
            params, obj = _estimate_month_wrapper(args)
            monthly_params.append(params)

    return constant_params, monthly_params
