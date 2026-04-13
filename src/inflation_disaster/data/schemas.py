"""Pydantic data models for all data structures in the pipeline.

These schemas enforce type safety and validation across the entire system.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class NumpyModel(BaseModel):
    """Base model that allows numpy arrays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OptionSurface(NumpyModel):
    """Raw option price surface for a single date and maturity."""

    date: date
    region: Literal["US", "EZ"]
    maturity: int  # years
    strikes: np.ndarray  # strike prices (gross inflation rate, e.g. 1.02 for 2%)
    cap_prices: np.ndarray  # zero-coupon cap prices
    floor_prices: np.ndarray  # zero-coupon floor prices
    swap_rate: float  # inflation swap rate for this maturity

    @field_validator("strikes", "cap_prices", "floor_prices", mode="before")
    @classmethod
    def coerce_to_array(cls, v):
        if not isinstance(v, np.ndarray):
            return np.asarray(v, dtype=np.float64)
        return v

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.cap_prices) > 0 and len(self.strikes) != len(self.cap_prices):
            raise ValueError("strikes and cap_prices must have same length")
        if len(self.floor_prices) > 0 and len(self.strikes) != len(self.floor_prices):
            raise ValueError("strikes and floor_prices must have same length")
        return self


class CleanedSurface(NumpyModel):
    """Cleaned and SABR-smoothed option surface."""

    date: date
    region: Literal["US", "EZ"]
    maturity: int
    strikes: np.ndarray  # original strike grid
    cap_prices: np.ndarray  # cleaned cap prices
    floor_prices: np.ndarray  # cleaned floor prices
    swap_rate: float
    implied_vols: np.ndarray  # SABR-smoothed implied volatilities
    real_rate: float  # put-call parity implied real rate
    quality_score: float  # 0-1, fraction of data points passing all checks

    # Fine grid for density extraction
    fine_strikes: np.ndarray  # dense strike grid for interpolation
    fine_call_prices: np.ndarray  # SABR-reconstructed call prices on fine grid


class InflationDistribution(NumpyModel):
    """Discrete probability distribution of inflation over 8 bins."""

    date: date
    region: Literal["US", "EZ"]
    horizon: int | str  # e.g. 5, 10, "5y5y", "yoy_avg"
    measure: Literal["N", "Q", "P"]  # nominal risk-neutral, real risk-neutral, physical
    bin_probabilities: np.ndarray  # 8-element probability vector

    @field_validator("bin_probabilities", mode="before")
    @classmethod
    def coerce_to_array(cls, v):
        if not isinstance(v, np.ndarray):
            return np.asarray(v, dtype=np.float64)
        return v

    @model_validator(mode="after")
    def check_valid_distribution(self):
        p = self.bin_probabilities
        if len(p) != 8:
            raise ValueError(f"Expected 8 bin probabilities, got {len(p)}")
        if np.any(p < -0.01):  # small tolerance for numerical noise
            raise ValueError(f"Negative probabilities: {p}")
        if abs(p.sum() - 1.0) > 0.05:
            raise ValueError(f"Probabilities sum to {p.sum()}, expected ~1.0")
        return self


class MarkovParams(BaseModel):
    """Parameters for the 8-state Markov chain inflation model (equation 12)."""

    # Time-varying parameters
    p_dh: float  # probability of entering high-inflation disaster
    p_dl: float  # probability of entering deflation disaster
    p_nn: float  # local diffusion probability (stochastic volatility)

    # Constant parameters
    p_h: float  # probability of exiting high-inflation disaster
    p_l: float  # probability of exiting deflation disaster
    p_mr: float  # mean-reversion probability

    @model_validator(mode="after")
    def check_valid_probabilities(self):
        for name in ["p_dh", "p_dl", "p_nn", "p_h", "p_l", "p_mr"]:
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name}={val} not in [0, 1]")

        # Check that transition matrix rows can sum to 1
        # For middle rows: p_dl + p_nn + p_n + p_nn + p_dh = 1
        # where p_n = 1 - 2*p_nn - p_dl - p_dh
        p_n = 1.0 - 2 * self.p_nn - self.p_dl - self.p_dh
        if p_n < -0.01:
            raise ValueError(
                f"Infeasible: 1 - 2*p_nn - p_dl - p_dh = {p_n} < 0"
            )
        return self


class SABRParams(BaseModel):
    """SABR stochastic volatility model parameters."""

    alpha: float  # initial vol level, > 0
    beta: float  # CEV exponent, typically 0.5 for inflation
    rho: float  # correlation, in (-1, 1)
    nu: float  # vol of vol, > 0

    @model_validator(mode="after")
    def check_bounds(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha={self.alpha} must be > 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho={self.rho} must be in (-1, 1)")
        if self.nu <= 0:
            raise ValueError(f"nu={self.nu} must be > 0")
        return self


class DisasterProbability(BaseModel):
    """Final disaster probability output for a single date."""

    date: date
    region: Literal["US", "EZ"]
    threshold: float  # disaster threshold d (2.0 or 3.0)

    # High-inflation disaster probabilities
    prob_high_n_5y: float  # N-measure, 5y horizon
    prob_high_q_5y: float  # Q-measure, 5y horizon
    prob_high_n_10y: float
    prob_high_q_10y: float
    prob_high_q_5y5y: float  # Q-measure, 5y5y forward
    prob_high_p_5y5y: float  # P-measure, 5y5y forward (final estimate)

    # Deflation disaster probabilities
    prob_low_n_5y: float
    prob_low_q_5y: float
    prob_low_n_10y: float
    prob_low_q_10y: float
    prob_low_q_5y5y: float
    prob_low_p_5y5y: float

    # Adjustment factors
    inflation_adj_5y: float  # Q/N ratio at 5y
    inflation_adj_10y: float  # Q/N ratio at 10y
    horizon_adj_high: float  # Q_5y5y / Q_10y for high inflation
    horizon_adj_low: float  # Q_5y5y / Q_10y for deflation
    risk_adj_high: float  # P/Q ratio for high inflation
    risk_adj_low: float  # P/Q ratio for deflation


class ParetoFit(BaseModel):
    """Fitted Pareto distribution parameters for consumption disasters."""

    disaster_type: Literal["high", "low", "pooled"]
    alpha: float  # tail exponent
    z_0: float  # minimum inverse consumption drop
    n_observations: int  # number of disaster episodes used in fit
    p_tilde: float  # conditional probability of output disaster given inflation disaster
    risk_adjustment_factor: float  # P/Q ratio
