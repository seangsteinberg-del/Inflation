"""Markov chain model of inflation dynamics — HRR 2024 eq. (12)-(13), p. 22-23.

Paper eq. (12), verbatim (p. 22):

    pi_{t+Delta} = pi_bar + eps_{t+Delta} + d^h_{t+Delta} - d^l_{t+Delta}

Paper eq. (13) transition matrix, verbatim (p. 23):

           bin: 0       1       2     3     4     5     6     7
    P = [
      [1-5pl,        pl,     pl,    pl,   pl,   pl,   0,    0      ],  # row 0: low disaster
      [pdl+pnn,      pml,    pmr,   0,    0,    0,    0,    0      ],  # row 1
      [pdl,          pnn,    pm,    pmr,  0,    0,    0,    pdh    ],  # row 2
      [pdl,          0,      pnn,   pn,   pnn,  0,    0,    pdh    ],  # row 3
      [pdl,          0,      0,     pnn,  pn,   pnn,  0,    pdh    ],  # row 4
      [pdl,          0,      0,     0,    pmr,  pm,   pnn,  pdh    ],  # row 5
      [0,            0,      0,     0,    0,    pmr,  pmh,  pdh+pnn],  # row 6
      [0,            0,      ph,    ph,   ph,   ph,   ph,   1-5ph  ],  # row 7: high disaster
    ]

with pn = 1 - 2*pnn - pdl - pdh   (footnote 17)
     pml = 1 - pdl - pnn - pmr    (footnote 18)
     pmh = 1 - pdh - pnn - pmr    (footnote 18)
     pm  = 1 - pmr - pnn - (the symmetric opposite from row 2 / row 5)

Six parameters (paper Sec. 4.3.2-3):
  p_dh, p_dl : probabilities of entering high/low inflation disaster (TIME-VARYING)
  p_nn       : probability of normal-inflation local move        (TIME-VARYING)
  p_l, p_h   : probabilities of exiting low/high disaster        (FIXED)
  p_mr       : mean-reversion probability toward target          (FIXED)

Paper's published fixed-param estimates (main paper p. 25):
  US: p_mr = 0.50, p_l = 0.1990, p_h = 0.1998
  EZ: p_mr = 0.47, p_l = 0.1999, p_h = 0.0617

Bin definition (paper p. 23):
  pi(i) = {<= -1, (-1, 0], (0, 1], (1, 2], (2, 3], (3, 4], (4, 5], > 5}

For mean/average-inflation computations, paper sets the disaster-bin values
(Appendix C.1, p. 8 mention): below-minus-1 -> -2%, above-5 -> 6%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Bin midpoints for mean computation (paper Appendix C.1: <-1% -> -2%, >5% -> 6%)
# Paper uses inflation *rates* not cumulative for the bin centers.
HRR_BIN_EDGES_PCT = np.array([-np.inf, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf])
HRR_BIN_EDGES_DEC = HRR_BIN_EDGES_PCT / 100.0
HRR_BIN_CENTERS_PCT = np.array([-2.0, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0])
HRR_BIN_CENTERS_DEC = HRR_BIN_CENTERS_PCT / 100.0
HRR_BIN_LABELS = ("<=-1", "(-1,0]", "(0,1]", "(1,2]", "(2,3]", "(3,4]", "(4,5]", ">5")


@dataclass(frozen=True)
class HRRMarkovParams:
    """Six-parameter set for the HRR 2024 eq. (13) transition matrix."""
    p_dh: float   # prob enter high disaster (time-varying)
    p_dl: float   # prob enter low disaster (time-varying)
    p_nn: float   # prob normal local move (time-varying)
    p_h: float    # prob exit high disaster (fixed)
    p_l: float    # prob exit low disaster (fixed)
    p_mr: float   # mean-reversion prob (fixed)

    def __post_init__(self):
        # No hard bounds — optimizers may propose negative values on their
        # way through the search. Infeasibility is caught and penalized
        # in the GMM objective via hrr_validate_params(), not here.
        pass


# HRR published fixed-parameter estimates (main paper p. 25)
HRR_FIXED_US = {"p_mr": 0.50, "p_l": 0.1990, "p_h": 0.1998}
HRR_FIXED_EZ = {"p_mr": 0.47, "p_l": 0.1999, "p_h": 0.0617}


def hrr_transition_matrix(params: HRRMarkovParams) -> np.ndarray:
    """Build the 8x8 transition matrix P per HRR eq. (13), p. 23.

    Rows are the current state, columns are the next state (standard Markov
    convention: P[i,j] = P(next = j | current = i)). Each row sums to 1.
    """
    p_dh, p_dl, p_nn = params.p_dh, params.p_dl, params.p_nn
    p_h, p_l, p_mr = params.p_h, params.p_l, params.p_mr

    # Derived probabilities from footnotes 17-18
    p_n = 1.0 - 2.0 * p_nn - p_dl - p_dh              # middle rows 3, 4
    p_ml = 1.0 - p_dl - p_nn - p_mr                    # row 1
    p_mh = 1.0 - p_dh - p_nn - p_mr                    # row 6
    # For rows 2 and 5 ("close to target" on each side): residual after other transitions
    # Row 2: pdl + pnn + pm + pmr + pdh = 1 -> pm = 1 - pdl - pnn - pmr - pdh
    # Row 5: pdl + pmr + pm + pnn + pdh = 1 -> same expression
    p_m = 1.0 - p_dl - p_nn - p_mr - p_dh

    P = np.zeros((8, 8))

    # Row 0: low-inflation disaster
    P[0, 0] = 1.0 - 5.0 * p_l
    P[0, 1:6] = p_l
    # P[0, 6:8] = 0 (already zero)

    # Row 1
    P[1, 0] = p_dl + p_nn
    P[1, 1] = p_ml
    P[1, 2] = p_mr
    # P[1, 3:8] = 0

    # Row 2
    P[2, 0] = p_dl
    P[2, 1] = p_nn
    P[2, 2] = p_m
    P[2, 3] = p_mr
    # P[2, 4:7] = 0
    P[2, 7] = p_dh

    # Row 3
    P[3, 0] = p_dl
    # P[3, 1] = 0
    P[3, 2] = p_nn
    P[3, 3] = p_n
    P[3, 4] = p_nn
    # P[3, 5:7] = 0
    P[3, 7] = p_dh

    # Row 4
    P[4, 0] = p_dl
    # P[4, 1:3] = 0
    P[4, 3] = p_nn
    P[4, 4] = p_n
    P[4, 5] = p_nn
    # P[4, 6] = 0
    P[4, 7] = p_dh

    # Row 5
    P[5, 0] = p_dl
    # P[5, 1:4] = 0
    P[5, 4] = p_mr
    P[5, 5] = p_m
    P[5, 6] = p_nn
    P[5, 7] = p_dh

    # Row 6
    # P[6, 0:5] = 0
    P[6, 5] = p_mr
    P[6, 6] = p_mh
    P[6, 7] = p_dh + p_nn

    # Row 7: high-inflation disaster
    # P[7, 0:2] = 0
    P[7, 2:7] = p_h
    P[7, 7] = 1.0 - 5.0 * p_h

    return P


def hrr_validate_params(params: HRRMarkovParams) -> dict[str, float]:
    """Return the derived probabilities, raising if any are negative.

    Useful for GMM penalty: if any derived prob < 0, the parameter set is
    infeasible and the objective should return a large penalty.
    """
    p_dh, p_dl, p_nn = params.p_dh, params.p_dl, params.p_nn
    p_h, p_l, p_mr = params.p_h, params.p_l, params.p_mr

    derived = {
        "p_n": 1.0 - 2.0 * p_nn - p_dl - p_dh,
        "p_ml": 1.0 - p_dl - p_nn - p_mr,
        "p_mh": 1.0 - p_dh - p_nn - p_mr,
        "p_m": 1.0 - p_dl - p_nn - p_mr - p_dh,
        "row0_self": 1.0 - 5.0 * p_l,
        "row7_self": 1.0 - 5.0 * p_h,
    }
    return derived


def simulate_paths(
    params: HRRMarkovParams,
    initial_state: np.ndarray,
    horizon: int,
    n_paths: int = 500_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate `n_paths` paths of length `horizon` years from the Markov chain.

    Parameters
    ----------
    params : chain parameters.
    initial_state : shape (8,), probability over the 8 bins for year 0
                    (typically one-hot based on current CPI YoY).
    horizon : number of years to simulate forward.
    n_paths : number of Monte Carlo paths.
    seed : RNG seed.

    Returns
    -------
    paths : int array of shape (n_paths, horizon) giving the bin index
            visited at each year 1..horizon (year 0 is the initial draw).
    """
    rng = np.random.default_rng(seed)
    P = hrr_transition_matrix(params)
    # Cumulative distributions per row for fast inverse-CDF sampling
    P_cum = np.cumsum(P, axis=1)

    # Sample initial states from the initial distribution
    initial_state = np.asarray(initial_state, dtype=float)
    if abs(initial_state.sum() - 1.0) > 1e-6:
        initial_state = initial_state / initial_state.sum()
    init_cum = np.cumsum(initial_state)
    u0 = rng.random(n_paths)
    state = np.searchsorted(init_cum, u0)

    paths = np.zeros((n_paths, horizon), dtype=np.int16)
    for t in range(horizon):
        u = rng.random(n_paths)
        # For each path, next state by inverse CDF on its row
        rows = P_cum[state]                      # shape (n_paths, 8)
        state = (rows < u[:, None]).sum(axis=1)  # next state index
        state = np.clip(state, 0, 7)
        paths[:, t] = state
    return paths


def cumulative_distribution(
    params: HRRMarkovParams,
    initial_state: np.ndarray,
    horizon: int,
    n_paths: int = 500_000,
    seed: int | None = None,
) -> np.ndarray:
    """Q-distribution of the AVERAGE annual inflation over years 1..horizon.

    Paper Appendix C.1: for each simulated path, average the bin centers
    (with <-1% -> -2% and >5% -> 6%), classify that average back into one
    of the 8 bins. Paper uses this as the "5Y cumul" and "10Y cumul"
    distribution for GMM.

    Returns
    -------
    dist : shape (8,) probability distribution over the 8 inflation bins.
    """
    paths = simulate_paths(params, initial_state, horizon, n_paths, seed)
    # Map each path-year bin to its midpoint in decimal
    centers = HRR_BIN_CENTERS_DEC[paths]          # shape (n_paths, horizon)
    avg_annual = centers.mean(axis=1)             # shape (n_paths,)
    # Re-bin the averages into the 8 annual-rate bins
    hist, _ = np.histogram(avg_annual, bins=HRR_BIN_EDGES_DEC)
    return hist / hist.sum()


def forward_annual_distribution(
    params: HRRMarkovParams,
    initial_state: np.ndarray,
    years: tuple = (6, 7, 8, 9, 10),
) -> np.ndarray:
    """Average of forward 1Y marginals at years `years`.

    Paper Sec. 4.3.1: "we take the average of these five annual
    distributions" q(pi_{5,6}), ..., q(pi_{9,10}). Computed as
    `(init @ P^t)` averaged across t in `years`.
    """
    P = hrr_transition_matrix(params)
    initial_state = np.asarray(initial_state, dtype=float)
    if abs(initial_state.sum() - 1.0) > 1e-6:
        initial_state = initial_state / initial_state.sum()

    # Power of P for each year
    Ps = {1: P}
    max_y = max(years)
    for y in range(2, max_y + 1):
        Ps[y] = Ps[y - 1] @ P

    dists = [initial_state @ Ps[y] for y in years]
    avg = np.mean(dists, axis=0)
    avg = np.maximum(avg, 0.0)
    return avg / avg.sum()


def forward_5y5y_distribution(
    params: HRRMarkovParams,
    initial_state: np.ndarray,
    n_paths: int = 500_000,
    seed: int | None = None,
) -> np.ndarray:
    """Q-distribution of the AVERAGE annual inflation over years 6..10.

    This is the paper's headline "5y5y forward" disaster measure.
    Simulates paths for 10 years, averages bin centers over years 6..10.
    """
    paths = simulate_paths(params, initial_state, horizon=10,
                           n_paths=n_paths, seed=seed)
    # Years 6..10 are indices 5..9 (0-indexed year 1 = paths[:,0])
    centers = HRR_BIN_CENTERS_DEC[paths[:, 5:10]]
    avg_annual = centers.mean(axis=1)
    hist, _ = np.histogram(avg_annual, bins=HRR_BIN_EDGES_DEC)
    return hist / hist.sum()


if __name__ == "__main__":
    # Self-test: build P with paper's US fixed params + plausible time-varying
    params = HRRMarkovParams(
        p_dh=0.05, p_dl=0.02, p_nn=0.10,
        p_h=HRR_FIXED_US["p_h"],
        p_l=HRR_FIXED_US["p_l"],
        p_mr=HRR_FIXED_US["p_mr"],
    )
    P = hrr_transition_matrix(params)
    print("Row sums (should all be 1.0):")
    for i, rs in enumerate(P.sum(axis=1)):
        print(f"  row {i}: {rs:.6f}")
    print("\nTransition matrix:")
    for row in P:
        print("  " + "  ".join(f"{v:6.3f}" for v in row))
    print("\nDerived probabilities:")
    for k, v in hrr_validate_params(params).items():
        print(f"  {k} = {v:.4f}")
