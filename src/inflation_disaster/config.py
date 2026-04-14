"""Global configuration and constants for the inflation disaster model.

All numerical constants from Hilscher, Raviv, and Reis (2024).
"""

from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings. Override via environment variables prefixed INFL_."""

    model_config = SettingsConfigDict(env_prefix="INFL_")

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def historical_dir(self) -> Path:
        return self.data_dir / "historical"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    # --- Inflation bins (8 states for Markov chain) ---
    # Bins: {<=-1%, (-1,0], (0,1], (1,2], (2,3], (3,4], (4,5], >5%}
    n_bins: int = 8
    # Use large finite values instead of inf for JSON serialization compatibility
    bin_edges: tuple[float, ...] = (-1e10, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1e10)

    @property
    def bin_midpoints(self) -> np.ndarray:
        """Midpoints for each bin. Extremes use -2% and 6% per paper."""
        return np.array([-2.0, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0])

    # --- Inflation target ---
    target_inflation: float = 2.0  # percent, annual

    # --- Disaster thresholds ---
    # d=2% => disaster if average annual inflation > 4% or < 0%
    # d=3% => severe disaster if average > 5% or < -1%
    disaster_thresholds: tuple[float, ...] = (2.0, 3.0)

    # --- Option strike prices ---
    # ZC caps: 1% to 6% in 0.5% steps; ZC floors: -2% to 3% in 0.5% steps
    # Combined range for full distribution
    strike_min: float = -2.0
    strike_max: float = 6.0
    strike_step: float = 0.5

    @property
    def strikes(self) -> np.ndarray:
        n_strikes = int(round((self.strike_max - self.strike_min) / self.strike_step)) + 1
        return np.linspace(self.strike_min, self.strike_max, n_strikes)

    # --- Horizons ---
    maturities: tuple[int, ...] = (5, 10)
    t_forward: int = 5  # forward start (years)
    h_horizon: int = 5  # forward horizon (years)
    yoy_maturities: tuple[int, ...] = (5, 6, 7, 8, 9, 10)

    # --- Risk adjustment parameters (Epstein-Zin) ---
    rra: float = 3.0  # relative risk aversion
    eis: float = 1.5  # elasticity of intertemporal substitution
    discount_factor: float = 0.99

    # --- Paper's calibrated Pareto parameters ---
    # High-inflation disasters
    pareto_alpha_high: float = 5.45
    pareto_z0_high: float = 1.03
    # Deflation disasters
    pareto_alpha_low: float = 15.18
    pareto_z0_low: float = 1.06

    # --- Paper's calibrated risk adjustment factors ---
    risk_adj_high: float = 0.66  # P/Q ratio for high-inflation disasters
    risk_adj_low: float = 0.96  # P/Q ratio for deflation disasters

    # --- Paper's estimated Markov constant parameters ---
    # US
    us_p_mr: float = 0.50
    us_p_l: float = 0.1990
    us_p_h: float = 0.1998
    # EZ
    ez_p_mr: float = 0.47
    ez_p_l: float = 0.1999
    ez_p_h: float = 0.0617

    # --- Monte Carlo ---
    mc_n_paths: int = 200_000
    mc_seed: int = 42

    # --- GMM ---
    gmm_n_starts: int = 20  # multi-start optimizations

    # --- SABR defaults ---
    sabr_beta: float = 0.5  # CEV exponent, fixed for inflation options

    # --- Sample periods ---
    us_start: str = "2009-10-01"
    ez_start: str = "2011-01-01"

    # --- JST historical data ---
    jst_countries: tuple[str, ...] = (
        "Australia", "Belgium", "Canada", "Denmark", "Finland", "France",
        "Germany", "Ireland", "Italy", "Japan", "Netherlands", "Norway",
        "Portugal", "Spain", "Sweden", "Switzerland", "United Kingdom",
        "United States",
    )
    jst_start_year: int = 1875
    jst_end_year: int = 2015


# Singleton instance
settings = Settings()
