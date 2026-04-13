"""8-state Markov chain model for inflation dynamics (equation 12).

Models inflation as an 8-state Markov chain corresponding to bins:
    {<=-1%, (-1,0], (0,1], (1,2], (2,3], (3,4], (4,5], >5%}

6 parameters:
    Time-varying: p_dh (enter high disaster), p_dl (enter deflation), p_nn (local volatility)
    Constant: p_h (exit high disaster), p_l (exit deflation), p_mr (mean reversion)

The transition matrix P is structured per equation (12) in the paper,
encoding mean-reversion in the center, disaster entry from normal states,
and disaster exit from extreme states.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

from inflation_disaster.config import settings
from inflation_disaster.data.schemas import MarkovParams

log = logging.getLogger("inflation_disaster.models.markov_chain")

N_STATES = 8
BIN_MIDPOINTS = np.array([-2.0, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0]) / 100.0


def build_transition_matrix(params: MarkovParams) -> np.ndarray:
    """Construct the 8x8 transition matrix P from equation (12).

    The structure encodes:
    - Row 0 (severe deflation): exits to normal states with prob 5*p_l
    - Row 7 (severe high inflation): exits to normal states with prob 5*p_h
    - Rows 1-6 (normal): can enter disasters, move locally, or mean-revert
    - Symmetry between above-target and below-target dynamics
    """
    p_dh = params.p_dh  # enter high-inflation disaster
    p_dl = params.p_dl  # enter deflation disaster
    p_nn = params.p_nn  # local movement (volatility)
    p_h = params.p_h    # exit high-inflation disaster (per destination)
    p_l = params.p_l    # exit deflation disaster (per destination)
    p_mr = params.p_mr  # mean reversion

    P = np.zeros((N_STATES, N_STATES))

    # Row 0: Severe deflation state (<= -1%)
    # Exits to states 1-5 with probability p_l each, stays with 1 - 5*p_l
    P[0, 0] = 1.0 - 5.0 * p_l
    P[0, 1] = p_l
    P[0, 2] = p_l
    P[0, 3] = p_l
    P[0, 4] = p_l
    P[0, 5] = p_l
    P[0, 6] = 0.0
    P[0, 7] = 0.0

    # Row 1: (-1%, 0%] -- near deflation, no mean reversion yet
    p_ml = 1.0 - p_dl - p_nn - p_mr  # stay probability
    P[1, 0] = p_dl + p_nn  # enter deflation disaster or move down
    P[1, 1] = p_ml
    P[1, 2] = p_mr
    P[1, 3] = 0.0
    P[1, 4] = 0.0
    P[1, 5] = 0.0
    P[1, 6] = 0.0
    P[1, 7] = 0.0

    # Row 2: (0%, 1%] -- below target
    p_m = 1.0 - p_dl - p_nn - p_mr - p_dh  # stay probability (accounting for asymmetry)
    # Actually per eq 12: this row has p_dl, p_nn, p_m, p_mr, 0, 0, 0, p_dh
    P[2, 0] = p_dl
    P[2, 1] = p_nn
    P[2, 2] = 1.0 - p_dl - p_nn - p_mr - p_dh
    P[2, 3] = p_mr
    P[2, 4] = 0.0
    P[2, 5] = 0.0
    P[2, 6] = 0.0
    P[2, 7] = p_dh

    # Row 3: (1%, 2%] -- at target (below)
    p_n = 1.0 - 2.0 * p_nn - p_dl - p_dh
    P[3, 0] = p_dl
    P[3, 1] = 0.0
    P[3, 2] = p_nn
    P[3, 3] = p_n
    P[3, 4] = p_nn
    P[3, 5] = 0.0
    P[3, 6] = 0.0
    P[3, 7] = p_dh

    # Row 4: (2%, 3%] -- at target (above)
    P[4, 0] = p_dl
    P[4, 1] = 0.0
    P[4, 2] = 0.0
    P[4, 3] = p_nn
    P[4, 4] = p_n
    P[4, 5] = p_nn
    P[4, 6] = 0.0
    P[4, 7] = p_dh

    # Row 5: (3%, 4%] -- above target
    P[5, 0] = p_dl
    P[5, 1] = 0.0
    P[5, 2] = 0.0
    P[5, 3] = 0.0
    P[5, 4] = p_mr
    P[5, 5] = 1.0 - p_dh - p_nn - p_mr - p_dl
    P[5, 6] = p_nn
    P[5, 7] = p_dh

    # Row 6: (4%, 5%] -- near high-inflation disaster
    p_mh = 1.0 - p_dh - p_nn - p_mr
    P[6, 0] = 0.0
    P[6, 1] = 0.0
    P[6, 2] = 0.0
    P[6, 3] = 0.0
    P[6, 4] = 0.0
    P[6, 5] = p_mr
    P[6, 6] = p_mh
    P[6, 7] = p_dh + p_nn  # enter disaster or move up

    # Row 7: Severe high inflation (> 5%)
    P[7, 0] = 0.0
    P[7, 1] = 0.0
    P[7, 2] = p_h
    P[7, 3] = p_h
    P[7, 4] = p_h
    P[7, 5] = p_h
    P[7, 6] = p_h
    P[7, 7] = 1.0 - 5.0 * p_h

    # Clip tiny negatives from rounding
    P = np.clip(P, 0.0, 1.0)

    # Renormalize rows to sum to 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1.0)

    return P


def validate_transition_matrix(P: np.ndarray, tol: float = 1e-6) -> bool:
    """Check that P is a valid stochastic matrix."""
    if P.shape != (N_STATES, N_STATES):
        return False
    if np.any(P < -tol):
        return False
    row_sums = P.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > tol):
        return False
    return True


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute stationary distribution pi such that pi @ P = pi."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    return pi


@njit
def _simulate_paths(
    P: np.ndarray,
    midpoints: np.ndarray,
    initial_state_probs: np.ndarray,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Numba-accelerated Monte Carlo simulation of Markov chain paths.

    Returns array of shape (n_paths,) containing the average annual
    inflation over n_steps years for each path.
    """
    np.random.seed(seed)
    n_states = P.shape[0]
    cum_P = np.zeros_like(P)
    for i in range(n_states):
        cum_P[i, 0] = P[i, 0]
        for j in range(1, n_states):
            cum_P[i, j] = cum_P[i, j - 1] + P[i, j]

    # CDF for initial state
    cum_init = np.zeros(n_states)
    cum_init[0] = initial_state_probs[0]
    for j in range(1, n_states):
        cum_init[j] = cum_init[j - 1] + initial_state_probs[j]

    avg_inflation = np.empty(n_paths)

    for path in range(n_paths):
        # Draw initial state
        u = np.random.random()
        state = 0
        for j in range(n_states):
            if u <= cum_init[j]:
                state = j
                break

        total_inflation = 0.0
        for step in range(n_steps):
            # Record inflation for this year
            total_inflation += midpoints[state]

            # Transition to next state
            u = np.random.random()
            for j in range(n_states):
                if u <= cum_P[state, j]:
                    state = j
                    break

        avg_inflation[path] = total_inflation / n_steps

    return avg_inflation


def simulate_cumulative_distribution(
    params: MarkovParams,
    initial_state: np.ndarray,
    horizon: int,
    n_paths: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate the distribution of average annual inflation over `horizon` years.

    Parameters
    ----------
    params : MarkovParams
    initial_state : array of shape (8,)
        Probability distribution over states at time 0.
    horizon : int
        Number of years.
    n_paths : int
        Number of MC paths.
    seed : int

    Returns
    -------
    bin_probs : array of shape (8,)
        Probability that average inflation falls in each bin.
    """
    if n_paths is None:
        n_paths = settings.mc_n_paths
    if seed is None:
        seed = settings.mc_seed

    P = build_transition_matrix(params)
    midpoints = BIN_MIDPOINTS

    avg_inflations = _simulate_paths(
        P, midpoints, initial_state, horizon, n_paths, seed
    )

    # Bin the average inflations
    bin_edges = np.array(settings.bin_edges) / 100.0
    counts = np.histogram(avg_inflations, bins=bin_edges)[0]
    probs = counts / counts.sum()

    return probs


def simulate_forward_distribution(
    params: MarkovParams,
    initial_state: np.ndarray,
    T: int = 5,
    H: int = 5,
    n_paths: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate the 5y5y forward distribution.

    First run the chain T years to get the state distribution at time T,
    then simulate H more years and compute average inflation over [T, T+H].

    Parameters
    ----------
    params : MarkovParams
    initial_state : array of shape (8,)
    T : int
        Forward start (years).
    H : int
        Forward horizon (years).

    Returns
    -------
    bin_probs : array of shape (8,)
        Probability that average inflation in [T, T+H] falls in each bin.
    """
    if n_paths is None:
        n_paths = settings.mc_n_paths
    if seed is None:
        seed = settings.mc_seed + 7919  # different seed from spot simulations

    P = build_transition_matrix(params)

    # Step 1: Get state distribution at time T
    # P^T applied to initial_state
    state_at_T = initial_state @ np.linalg.matrix_power(P, T)

    # Step 2: Simulate H years from state_at_T
    midpoints = BIN_MIDPOINTS
    avg_inflations = _simulate_paths(
        P, midpoints, state_at_T, H, n_paths, seed
    )

    # Bin the results
    bin_edges = np.array(settings.bin_edges) / 100.0
    counts = np.histogram(avg_inflations, bins=bin_edges)[0]
    probs = counts / counts.sum()

    return probs


def marginal_one_year_distribution(
    params: MarkovParams,
    initial_state: np.ndarray,
    year: int,
) -> np.ndarray:
    """State distribution at a specific future year.

    Returns P(state = j at year t | initial_state).
    """
    P = build_transition_matrix(params)
    return initial_state @ np.linalg.matrix_power(P, year)
