# How Likely Is an Inflation Disaster?

**A complete Python implementation of [Hilscher, Raviv, and Reis (2024)](https://r2rsquaredlse.github.io/web-inflationdisasters/) for estimating market-implied tail probabilities of inflation disasters from Bloomberg options data.**

---

## What This Does

Central banks and macro strategists track the 5-year-5-year (5y5y) forward expected inflation rate as a headline measure of whether inflation expectations are anchored. But that number is just a point estimate of the mean. This toolkit estimates the **tail probabilities** -- the market-perceived chance that inflation will be persistently above 4% or below 0% between 5 and 10 years from now.

The paper shows that naive readings of inflation option prices can grossly over- or under-state these probabilities. Three corrections are required:

| Adjustment | What it fixes | Typical magnitude |
|-----------|---------------|-------------------|
| **Inflation** (N &rarr; Q) | Nominal option payoffs don't reflect real purchasing power | 1.24x for 10y high-inflation options |
| **Horizon** (spot &rarr; 5y5y forward) | Traded 5y and 10y options don't directly give the 5y5y forward probability | 0.38x for US high inflation (sluggish inflation overstates 10y prob) |
| **Risk** (Q &rarr; P) | Risk-neutral probabilities overstate physical probabilities because disasters hurt more | 0.66x for high inflation, 0.96x for deflation |

Combined, a naive 10y reading of 14% becomes a corrected 5y5y physical probability of 4.2%.

### Key Outputs

- Monthly time series of **5y5y inflation disaster probabilities** under N (nominal risk-neutral), Q (real risk-neutral), and P (physical) measures
- For the **United States** (from Oct 2009) and **Eurozone** (from Jan 2011) through present
- Decomposition of each adjustment factor over time
- Risk-neutral density plots at any date
- Markov chain parameter estimates (perceived inflation dynamics)
- Conditional anchoring analysis (counterfactuals varying initial inflation)
- Confidence bands via delta method and bootstrap

---

## Paper Reference

> **Hilscher, J., Raviv, A., and Reis, R. (2024).** "How likely is an inflation disaster?"
> *Journal of Financial Economics* (forthcoming). [CFMDP2024-37](https://r2rsquaredlse.github.io/web-inflationdisasters/)

**Authors:** Jens Hilscher (UC Davis), Alon Raviv (Bar-Ilan University), Ricardo Reis (LSE)

**Key findings from the paper:**
1. US deflation risk in 2011-14 was **overstated** by conventional measures
2. ECB unconventional policies **lowered** the deflation disaster probability
3. Inflation expectations **deanchored** in 2021-22 (probability peaked at 10% in May 2022)
4. Expectations **reanchored** as the Fed and ECB hiked rates aggressively
5. The 2021-24 disaster left **scars** -- probabilities remain 2-3x higher than pre-2021
6. US expectations are **less sensitive** to inflation realizations than the Eurozone

---

## Architecture

```
src/inflation_disaster/
|
|-- config.py                     Global constants matching the paper exactly
|
|-- data/
|   |-- bloomberg.py              Bloomberg API (blpapi) fetcher for:
|   |                               - Zero-coupon inflation caps & floors (5y, 10y)
|   |                               - Year-on-year inflation caps & floors (yr 5-10)
|   |                               - Inflation swap rates (breakeven inflation)
|   |-- jst.py                    Jorda-Schularick-Taylor Macrohistory Database:
|   |                               - 18 countries, 1875-2015
|   |                               - Inflation & GDP disaster episode identification
|   |-- cleaning.py               Option surface quality checks:
|   |                               - Cap/floor price monotonicity
|   |                               - Butterfly spread positivity (convexity)
|   |                               - Put-call parity cross-validation
|   |-- schemas.py                Pydantic data models with validation
|
|-- models/
|   |-- sabr.py                   SABR stochastic volatility model (Hagan et al. 2002):
|   |                               - Closed-form implied vol approximation
|   |                               - Multi-start L-BFGS-B calibration
|   |                               - Price reconstruction on fine strike grids
|   |-- breeden_litzenberger.py   Risk-neutral density extraction:
|   |                               - N-density: n(k) = e^{iT} * a''(k)     [eq. 8]
|   |                               - Q-density: Q(k) = e^{rT} * k * a''(k) [eq. 10]
|   |                               - N-to-Q conversion: q = n * e^{pi-pi^e} [eq. 4]
|   |-- markov_chain.py           8-state Markov chain for inflation dynamics [eq. 12]:
|   |                               - Transition matrix with disaster entry/exit
|   |                               - Numba-JIT Monte Carlo (200K paths)
|   |                               - Cumulative and 5y5y forward distributions
|   |-- gmm.py                    Generalized Method of Moments estimator:
|   |                               - 21 moment conditions (7 bins x 3 distributions)
|   |                               - Two-stage: constant params then time-varying
|   |                               - Parallel multi-start optimization
|   |-- pareto.py                 Pareto distribution for consumption disasters:
|   |                               - MLE fitting to historical disaster data
|   |                               - Separate fits for high-inflation and deflation
|   |-- epstein_zin.py            Risk adjustment via Epstein-Zin preferences:
|   |                               - E[z^gamma] under Pareto (closed-form)
|   |                               - P/Q ratio from eq. (2)
|
|-- adjustments/
|   |-- inflation_adj.py          1st adjustment: N -> Q probabilities
|   |-- horizon_adj.py            2nd adjustment: spot -> 5y5y forward
|   |-- risk_adj.py               3rd adjustment: Q -> P physical measure
|   |-- pipeline.py               End-to-end orchestration for single dates
|
|-- analytics/
|   |-- disaster_probs.py         Final probability time series
|   |-- decomposition.py          Adjustment factor decomposition (Table 1)
|   |-- anchoring.py              Conditional anchoring analysis (Figure 5)
|
|-- visualization/
|   |-- style.py                  Consistent chart styling
|   |-- time_series.py            Figures 3, 4, 6 reproduction
|
|-- utils/
    |-- numerical.py              Spline interpolation, numerical integration
    |-- logging.py                Structured logging
```

---

## Mathematical Framework

### The Three Probabilities

The paper defines inflation disasters as:

```
Prob[pi_{T,T+H} / H > pi_bar + d]     (high-inflation disaster)
Prob[pi_{T,T+H} / H < pi_bar - d]     (deflation disaster)
```

where T=5 (forward start), H=5 (horizon), pi_bar=2% (target), d=2% (disaster threshold).

### Proposition 1 (Main Result)

The physical probability of an inflation disaster is:

```
p(pi) = n(pi)                          [from options data]
      x exp((pi - pi^e) * H)           [inflation adjustment: real payoff correction]
      x (e^{-rH} * m(pi))              [risk adjustment: marginal utility in disaster]
      x [horizon factor]               [forward probability from Markov chain]
```

### The 8-State Markov Chain (Equation 12)

Inflation is modeled as an 8-state Markov chain with bins:

```
State 0: <= -1%     (severe deflation)
State 1: (-1%, 0%]  (mild deflation)
State 2: (0%, 1%]   (below target)
State 3: (1%, 2%]   (at target, lower)
State 4: (2%, 3%]   (at target, upper)
State 5: (3%, 4%]   (above target)
State 6: (4%, 5%]   (elevated inflation)
State 7: > 5%       (severe high inflation)
```

The 8x8 transition matrix has 6 parameters:
- **p_dh, p_dl** (time-varying): probability of entering high/low inflation disaster
- **p_nn** (time-varying): local volatility (probability of moving one bin)
- **p_h, p_l** (constant): probability of exiting high/low disaster
- **p_mr** (constant): mean-reversion probability

Paper's estimated constant parameters:
| Parameter | US | EZ | Interpretation |
|-----------|----|----|----------------|
| p_mr | 0.50 | 0.47 | Strong mean reversion |
| p_l | 0.199 | 0.200 | Quick deflation exit (US) |
| p_h | 0.200 | 0.062 | Quick high-infl exit (US), persistent in EZ |

### Risk Adjustment

The P/Q ratio uses historical data from 18 countries (1875-2015):

```
P/Q = 1 / [(m_tilde - 1) * p_tilde + 1]
```

where m_tilde = E[z^gamma] under a Pareto distribution of consumption drops, and p_tilde is the conditional probability of an output disaster given an inflation disaster.

| | p_tilde | alpha | z_0 | P/Q ratio |
|---|---------|-------|-----|-----------|
| High inflation | 0.356 | 5.45 | 1.03 | **0.66** |
| Deflation | 0.085 | 15.18 | 1.06 | **0.96** |

High inflation comes with deeper recessions more often, so risk-neutral probabilities overstate physical probabilities more for high inflation than for deflation.

---

## Installation

### Prerequisites

- Python 3.10+
- Bloomberg Terminal with API access (for real data)
- `blpapi` Python package (included with Bloomberg)

### Setup

```bash
git clone https://github.com/seangsteinberg-del/Inflation.git
cd Inflation

# Install core dependencies
pip install -e ".[dev]"

# Install Bloomberg dependencies (requires Bloomberg Terminal)
pip install -e ".[bloomberg]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations, linear algebra |
| scipy | Optimization (L-BFGS-B), spline interpolation, signal processing |
| pandas | Time series data handling |
| matplotlib | Visualization |
| pydantic | Data validation and settings |
| numba | JIT compilation for Monte Carlo simulation |
| statsmodels | Statistical tools |
| blpapi | Bloomberg API access |

---

## Usage

### Quick Start: Single Date Pipeline

```python
from inflation_disaster.adjustments.pipeline import DisasterProbabilityPipeline
from inflation_disaster.data.bloomberg import BloombergFetcher
from inflation_disaster.data.cleaning import clean_surface
from datetime import date

# 1. Fetch data from Bloomberg
fetcher = BloombergFetcher()
zc_data = fetcher.fetch_zc_caps_floors("US", date(2024, 1, 1), date(2024, 1, 31))
swap_data = fetcher.fetch_inflation_swaps("US", date(2024, 1, 1), date(2024, 1, 31))

# 2. Build and clean option surfaces
surface_5y = fetcher.build_option_surface(zc_data, swap_data, date(2024, 1, 2), "US", 5)
surface_10y = fetcher.build_option_surface(zc_data, swap_data, date(2024, 1, 2), "US", 10)
cleaned_5y, _ = clean_surface(surface_5y)
cleaned_10y, _ = clean_surface(surface_10y)

# 3. Run the full pipeline
pipeline = DisasterProbabilityPipeline("US")
result = pipeline.process_single_date(
    surface_5y=cleaned_5y,
    surface_10y=cleaned_10y,
    markov_params=markov_params,  # from GMM estimation
    current_inflation_state=np.array([0,0,0,0,1,0,0,0]),  # 2-3% bin
    threshold=2.0,  # disaster = 2pp above/below target
)

print(f"P(high-inflation disaster, 5y5y): {result.prob_high_p_5y5y:.1%}")
print(f"P(deflation disaster, 5y5y):      {result.prob_low_p_5y5y:.1%}")
```

### Full Sample Estimation

```python
from inflation_disaster.models.gmm import estimate_full_sample, GMMTargets

# Build monthly targets from extracted Q-distributions
targets = [
    GMMTargets(
        date=dt,
        q_5y=q_dist_5y.bin_probabilities,
        q_10y=q_dist_10y.bin_probabilities,
        q_yoy_avg=q_yoy.bin_probabilities,
        initial_state=current_state,
    )
    for dt, q_dist_5y, q_dist_10y, q_yoy, current_state in monthly_data
]

# Two-stage GMM estimation
constant_params, monthly_params = estimate_full_sample(
    targets,
    n_paths_stage1=30_000,   # faster first pass
    n_paths_stage2=200_000,  # precise second pass
    max_workers=4,           # parallel across months
)
```

### Risk Adjustment Calibration from Historical Data

```python
from inflation_disaster.data.jst import load_jst_dataset, identify_inflation_disasters
from inflation_disaster.models.pareto import fit_pareto_separate
from inflation_disaster.models.epstein_zin import compute_risk_adjustments

# Load 18-country historical data (1875-2015)
df = load_jst_dataset()
high_disasters, low_disasters = identify_inflation_disasters(df)

# Fit Pareto distributions separately
high_fit, low_fit = fit_pareto_separate(high_disasters, low_disasters)

# Compute P/Q ratios with Epstein-Zin (RRA=3)
adj_high, adj_low = compute_risk_adjustments(high_fit, low_fit, gamma=3.0)
# adj_high ~ 0.66, adj_low ~ 0.96
```

### Visualization

```python
from inflation_disaster.visualization.time_series import (
    plot_high_inflation_probability,
    plot_deflation_probability,
    plot_density_snapshots,
    plot_markov_parameters,
)

# Figure 4a: High-inflation disaster probability (US vs EZ)
plot_high_inflation_probability(results_df, save_path="fig4a.png")

# Figure 3: Deflation probability
plot_deflation_probability(us_results, region="US", save_path="fig3_us.png")

# Figure 4c: Risk-neutral density snapshots
plot_density_snapshots({
    "Mar 2020": (grid_2020, density_2020),
    "Mar 2022": (grid_2022, density_2022),
    "Mar 2024": (grid_2024, density_2024),
}, save_path="fig4c.png")
```

---

## Data Requirements

### Bloomberg Data (Real-Time)

| Instrument | Tickers | Strikes | Maturities |
|------------|---------|---------|------------|
| ZC inflation caps | USCP / EUCP | -2% to 6% (0.5% steps) | 5y, 10y |
| ZC inflation floors | USFP / EUFP | -2% to 6% (0.5% steps) | 5y, 10y |
| YOY inflation caps | USCPYY / EUCPYY | -2% to 6% (0.5% steps) | 5y-10y |
| YOY inflation floors | USFPYY / EUFPYY | -2% to 6% (0.5% steps) | 5y-10y |
| Inflation swaps | USSWIT / EUSWI | N/A | 1,2,3,5,7,10y |

**Note:** Bloomberg ticker conventions may vary by terminal. The ticker templates in `data/bloomberg.py` follow common conventions but may need adjustment for your specific setup.

### Historical Data (One-Time)

The risk adjustment calibration requires the **Jorda-Schularick-Taylor Macrohistory Database**:

1. Download from [macrohistory.net/database](https://www.macrohistory.net/database/)
2. Place the Excel file in `data/historical/`
3. The loader auto-detects filenames (`JSTdatasetR6.xlsx`, etc.)

**Countries covered:** Australia, Belgium, Canada, Denmark, Finland, France, Germany, Ireland, Italy, Japan, Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom, United States.

---

## Configuration

All constants live in `config.py` and can be overridden via environment variables prefixed `INFL_`:

```bash
export INFL_MC_N_PATHS=500000      # More MC paths for production
export INFL_GMM_N_STARTS=30        # More optimization restarts
export INFL_RRA=4.0                # Higher risk aversion
```

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `target_inflation` | 2.0 | Central bank inflation target (%) |
| `disaster_thresholds` | (2.0, 3.0) | Disaster threshold d (pp from target) |
| `mc_n_paths` | 200,000 | Monte Carlo simulation paths |
| `gmm_n_starts` | 20 | Multi-start optimizations for GMM |
| `rra` | 3.0 | Relative risk aversion (Epstein-Zin) |
| `sabr_beta` | 0.5 | SABR CEV exponent for inflation |
| `risk_adj_high` | 0.66 | Default P/Q ratio for high inflation |
| `risk_adj_low` | 0.96 | Default P/Q ratio for deflation |

---

## Validation

The implementation has been verified against the paper's published results:

| Metric | Implementation | Paper | Match |
|--------|---------------|-------|-------|
| P/Q ratio (high inflation) | 0.663 | 0.66 | Yes |
| P/Q ratio (deflation) | 0.960 | 0.96 | Yes |
| Markov matrix rows sum to 1 | True | Required | Yes |
| Stationary distribution exists | True | Required | Yes |
| N-probabilities vs Kitsul & Wright (2013) | ~0.25pp diff | Appendix Fig 1 | Yes |

Comprehensive test suite covers:
- SABR calibration (Hagan 2002 formula, multi-start convergence)
- Breeden-Litzenberger density extraction (integral = 1, non-negativity)
- Markov chain (valid stochastic matrix, stationary distribution, MC convergence)
- GMM estimation (parameter recovery, objective function)
- All three adjustments (direction, magnitude, paper values)
- Full pipeline integration

---

## Project Structure

```
Inflation/
  README.md                        This file
  pyproject.toml                   Dependencies and project config
  .gitignore                       Excludes data files and caches
  src/inflation_disaster/          Main package (20 modules)
  tests/                           Test suite
  data/
    raw/                           Bloomberg data cache (parquet)
    processed/                     Cleaned option surfaces
    historical/                    JST Macrohistory Database
    output/                        Final results
```

---

## Performance Notes

- **GMM estimation** is the bottleneck: each month requires ~20 multi-start optimizations, each calling Monte Carlo simulation inside the optimizer loop
- **Numba JIT** accelerates the MC inner loop by ~50x vs pure Python
- **Parallelization** across months via `ProcessPoolExecutor` gives near-linear speedup
- Full sample (160 months, US): ~2-4 hours on a modern workstation with 8 cores
- Single date: ~30 seconds

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{hilscher2024inflation,
  title={How likely is an inflation disaster?},
  author={Hilscher, Jens and Raviv, Alon and Reis, Ricardo},
  journal={Journal of Financial Economics},
  year={2024},
  note={CFMDP2024-37}
}
```

---

## License

This implementation is for research and internal use. The methodology and all equations belong to Hilscher, Raviv, and Reis (2024). The Jorda-Schularick-Taylor database has its own license terms.
