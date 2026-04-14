# How Likely Is an Inflation Disaster?

**A production-grade Python implementation of [Hilscher, Raviv, and Reis (2024)](https://r2rsquaredlse.github.io/web-inflationdisasters/) for estimating market-implied tail probabilities of inflation disasters from Bloomberg options data.**

Built for the State Street Global Markets macro strategy team.

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [The Three Adjustments](#the-three-adjustments)
3. [Key Findings from the Paper](#key-findings-from-the-paper)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
7. [Data Requirements](#data-requirements)
8. [Mathematical Framework](#mathematical-framework)
9. [Architecture](#architecture)
10. [Module Reference](#module-reference)
11. [Configuration](#configuration)
12. [Testing](#testing)
13. [Performance](#performance)
14. [Validation Against the Paper](#validation-against-the-paper)
15. [Known Limitations](#known-limitations)
16. [Bug Fix History](#bug-fix-history)
17. [Citation](#citation)

---

## What This Does

Central banks and macro strategists track the **5-year-5-year (5y5y) forward expected inflation rate** as a headline measure of whether inflation expectations are anchored. The Fed and ECB cite this number in speeches, and even small moves can trigger large policy shifts -- a decline in the EZ 5y5y in 2014 justified the start of quantitative easing.

**But that number is just a point estimate of the mean.** The distribution around it could be tight or wildly dispersed. This toolkit estimates what the mean cannot: the **tail probabilities** -- the market-perceived chance that inflation will be persistently above 4% or below 0% between 5 and 10 years from now.

Specifically, we compute:

```
Prob[ average annual inflation from year 5 to year 10 > 4% ]    (high-inflation disaster)
Prob[ average annual inflation from year 5 to year 10 < 0% ]    (deflation disaster)
```

These are **forward-looking, market-implied, physical-measure** probabilities, extracted from traded inflation option prices with three critical corrections that the raw option prices do not account for.

### Why This Matters

- In **May 2022**, the probability of a 5y5y US high-inflation disaster peaked at **10%** -- while the 5y5y swap rate barely moved. The mean was anchored; the tails were not.
- In **2011-14**, conventional measures vastly overstated the probability of a US deflation trap. Our corrected estimates show the risk was much lower.
- The **Eurozone** has a structurally higher deflation disaster probability than the US (6.3% vs 2.4%), entirely explaining why the ECB faces a harder anchoring challenge.

---

## The Three Adjustments

The paper's core contribution is showing that naive readings of inflation option prices are wrong in three specific, quantifiable ways. Each adjustment has a clear economic intuition:

### Adjustment 1: Inflation (N to Q)

**Problem:** Inflation options pay in nominal dollars. A $1 payoff when inflation is 6% is worth less in real terms than a $1 payoff when inflation is 0%. Standard methods ignore this.

**Fix:** Multiply the nominal risk-neutral density by `exp((pi - pi^e) * T)` to get the real risk-neutral density.

**Magnitude:** For 10-year high-inflation options, the conventional probability understates the true probability by a factor of **1.24x**. For deflation, it overstates by **0.69x**.

**Intuition:** Markets pay less for options that pay off in high-inflation states because the payoff is worth less. Researchers must undo this effect.

### Adjustment 2: Horizon (spot to 5y5y forward)

**Problem:** Traded options give probabilities for cumulative inflation over the next 5 or 10 years from today. Policymakers want the **5y5y forward** -- what happens between years 5 and 10. These are not the same because inflation is sluggish.

**Fix:** Estimate a Markov chain model of inflation dynamics from the term structure of option prices, then simulate the forward distribution.

**Magnitude:** During 2021-23, the 10y option-implied high-inflation probability was 17.2%, but the corrected 5y5y forward was only **6.3%** (factor: 0.38x). Current high inflation contributes to the 10y probability but is expected to mean-revert before year 5.

**Intuition:** If inflation is high today, a 10-year option will pay off more easily than a 5-year-forward option, because the current high inflation gets averaged into the 10-year window but not the forward window.

### Adjustment 3: Risk (Q to P)

**Problem:** Risk-neutral probabilities overstate the physical probability of disasters because investors have higher marginal utility during disasters (they hurt more), so they pay a premium for insurance against these states.

**Fix:** Use historical data on the co-occurrence of inflation disasters and output disasters across 18 countries over 140 years to estimate how much marginal utility rises during inflation disasters. Apply Epstein-Zin preferences with relative risk aversion of 3.

**Magnitude:** For high inflation, `P/Q = 0.66` (risk-neutral overstates by 52%). For deflation, `P/Q = 0.96` (almost no overstatement).

**Intuition:** High inflation historically comes with deep recessions (1970s stagflation), so investors pay a large premium to hedge it. Deflation often occurred without severe depressions (late 19th century), so the risk premium is much smaller.

### Combined Effect

| Step | US High Inflation (median, 2021-23) |
|------|--------------------------------------|
| Naive 10y N-probability | 14.0% |
| After inflation adjustment (Q, 10y) | 17.2% |
| After horizon adjustment (Q, 5y5y) | 6.3% |
| After risk adjustment (P, 5y5y) | **4.2%** |

A naive reading of 14% becomes a corrected **4.2%**. The direction of each adjustment makes economic sense: inflation adjustment raises it (high-inflation options are underpriced in real terms), horizon adjustment lowers it (inflation is expected to mean-revert), risk adjustment lowers it further (investors overpay for disaster insurance).

---

## Key Findings from the Paper

| Finding | Detail |
|---------|--------|
| **US deflation risk overstated (2011-14)** | Conventional measures showed 15-25% deflation probability; our corrected 5y5y estimate was below 5% by end of 2012 |
| **ECB policies worked (partially)** | Unconventional policies since 2014 brought EZ deflation probability below 5%, but only temporarily |
| **Expectations deanchored in 2021-22** | US 5y5y high-inflation disaster probability rose from ~1% to 10% (May 2022 peak) |
| **Reanchored with rate hikes** | Sharp fall to 3% by end of 2022, coinciding with 400bp of Fed hikes |
| **Scars remain** | End-of-sample probabilities are 2-3x higher than pre-2021 levels |
| **US more anchored than EZ** | US disaster probability insensitive to current inflation; EZ probability depends on it |

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Bloomberg Terminal** with API access (for live data)
- **blpapi** Python package (ships with Bloomberg Terminal SDK)

### Setup

```bash
# Clone the repository
git clone https://github.com/seangsteinberg-del/Inflation.git
cd Inflation

# Install core dependencies
pip install numpy scipy pandas matplotlib pydantic pydantic-settings numba statsmodels

# Install Bloomberg dependencies (requires Bloomberg Terminal SDK)
pip install blpapi

# Install development dependencies
pip install pytest pytest-cov ruff

# Verify installation
python -c "from inflation_disaster.config import settings; print(f'OK: {settings.n_bins} bins, target={settings.target_inflation}%')"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Array operations, linear algebra, histogram binning |
| scipy | >= 1.10 | L-BFGS-B optimization, cubic spline interpolation, Butterworth filter |
| pandas | >= 2.0 | Time series data handling, parquet I/O |
| matplotlib | >= 3.7 | Publication-quality figures matching paper style |
| pydantic | >= 2.0 | Data model validation with type safety |
| pydantic-settings | >= 2.0 | Environment variable configuration |
| numba | >= 0.57 | JIT compilation for Monte Carlo inner loop (~50x speedup) |
| statsmodels | >= 0.14 | Statistical tools (used in historical data analysis) |
| blpapi | >= 3.19 | Bloomberg Terminal API for options and swap data |

---

## Quick Start

### Single Date: What Is Today's Inflation Disaster Probability?

```python
import numpy as np
from datetime import date
from inflation_disaster.adjustments.pipeline import DisasterProbabilityPipeline
from inflation_disaster.data.bloomberg import BloombergFetcher
from inflation_disaster.data.cleaning import clean_surface
from inflation_disaster.data.schemas import MarkovParams

# Step 1: Connect to Bloomberg and fetch data
fetcher = BloombergFetcher()

# Fetch zero-coupon inflation caps and floors (5y and 10y maturities)
zc_data = fetcher.fetch_zc_caps_floors("US", date(2024, 1, 1), date(2024, 1, 31))

# Fetch inflation swap rates (for expected inflation / breakeven)
swap_data = fetcher.fetch_inflation_swaps("US", date(2024, 1, 1), date(2024, 1, 31))

# Step 2: Build option surfaces for a specific date
surface_5y = fetcher.build_option_surface(zc_data, swap_data, date(2024, 1, 2), "US", 5)
surface_10y = fetcher.build_option_surface(zc_data, swap_data, date(2024, 1, 2), "US", 10)

# Step 3: Clean surfaces (monotonicity, butterfly, put-call parity checks)
cleaned_5y, diag_5y = clean_surface(surface_5y)
cleaned_10y, diag_10y = clean_surface(surface_10y)

# Step 4: Use paper's estimated Markov parameters (or estimate your own via GMM)
markov_params = MarkovParams(
    p_dh=0.05, p_dl=0.02, p_nn=0.12,  # time-varying (example values)
    p_h=0.1998, p_l=0.1990, p_mr=0.50,  # constant (paper's US estimates)
)

# Step 5: Run the full pipeline
pipeline = DisasterProbabilityPipeline("US")
result = pipeline.process_single_date(
    surface_5y=cleaned_5y,
    surface_10y=cleaned_10y,
    markov_params=markov_params,
    current_inflation_state=np.array([0, 0, 0, 0, 1, 0, 0, 0]),  # currently in 2-3% bin
    threshold=2.0,  # disaster = 2pp above/below target
)

# Step 6: Read the results
print(f"5y5y P(inflation > 4%):  {result.prob_high_p_5y5y:.1%}")
print(f"5y5y P(deflation < 0%): {result.prob_low_p_5y5y:.1%}")
print(f"Adjustment factors:")
print(f"  Inflation (N->Q, 10y): {result.inflation_adj_10y:.2f}")
print(f"  Horizon (Q 10y->5y5y): {result.horizon_adj_high:.2f}")
print(f"  Risk (Q->P, high):     {result.risk_adj_high:.2f}")

# Don't forget to close the Bloomberg session
fetcher.close()
```

---

## Full Pipeline Walkthrough

### Step-by-Step: From Raw Option Prices to Disaster Probabilities

The pipeline follows this exact sequence:

```
Bloomberg ZC caps/floors (5y, 10y)
    |
    v
[1] Data Cleaning (cleaning.py)
    - Monotonicity: cap prices decreasing in strike, floor prices increasing
    - Butterfly: c(K-dK) - 2c(K) + c(K+dK) >= 0 (no-arbitrage convexity)
    - Put-call parity: Cap(K) - Floor(K) = DF * ((1+swap)^T - (1+K)^T)
    - Select best-quality daily data per month
    |
    v
[2] SABR Calibration (sabr.py)
    - Hagan et al. (2002) closed-form implied vol approximation
    - 3 free parameters: alpha (vol level), rho (skew), nu (vol-of-vol)
    - beta fixed at 0.5 for inflation options
    - Multi-start L-BFGS-B optimization (20 starts)
    - Output: smooth implied vol smile on dense strike grid
    |
    v
[3] Density Extraction (breeden_litzenberger.py)
    - Convert SABR smile to call prices on fine grid (1500 points, -5% to 12%)
    - Fit cubic spline to call prices
    - N-density: n(k) = e^{iT} * d^2C/dk^2   [Breeden-Litzenberger, eq. 8]
    - Normalize to integrate to 1
    |
    v
[4] Inflation Adjustment (inflation_adj.py)
    - Q-density: q(k) = n(k) * exp((k - pi^e) * T)   [eq. 4]
    - Renormalize Q-density
    - Integrate over 8 bins to get discrete probabilities
    |
    v
[5] Markov Chain Estimation (markov_chain.py + gmm.py)
    - Build 8x8 transition matrix from 6 parameters [eq. 12]
    - GMM: match 21 moments (7 bins x 3 distributions)
      - 5y zero-coupon Q-distribution
      - 10y zero-coupon Q-distribution
      - Average forward YOY Q-distribution (years 5-9)
    - Two-stage: constant params (quarterly), time-varying params (monthly)
    |
    v
[6] Horizon Adjustment (horizon_adj.py)
    - Compute state distribution at year 5: pi_5 = pi_0 @ P^5
    - Monte Carlo: simulate 200K 5-year paths from pi_5
    - Bin average inflation over years 5-10 to get 5y5y forward distribution
    |
    v
[7] Risk Adjustment (risk_adj.py)
    - P(high disaster) = Q(high disaster) * 0.66
    - P(defl disaster) = Q(defl disaster) * 0.96
    - Factors from Pareto fit to 18-country historical inflation-GDP data
    |
    v
FINAL OUTPUT: DisasterProbability with all measures and decomposition
```

### Full Sample Estimation

```python
from inflation_disaster.models.gmm import estimate_full_sample, GMMTargets

# After extracting Q-distributions for every month...
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

# Two-stage GMM
constant_params, monthly_params = estimate_full_sample(
    targets,
    n_paths_stage1=30_000,   # faster for initial constant param estimation
    n_paths_stage2=200_000,  # precise for final time-varying estimation
    max_workers=4,           # parallel across months (CPU-bound)
)

# constant_params = [p_h, p_l, p_mr] (median from Stage 1)
# monthly_params = list of MarkovParams (one per month)
```

### Risk Adjustment from Historical Data

```python
from inflation_disaster.data.jst import load_jst_dataset, identify_inflation_disasters
from inflation_disaster.models.pareto import fit_pareto_separate
from inflation_disaster.models.epstein_zin import compute_risk_adjustments

# Load Jorda-Schularick-Taylor Macrohistory Database
df = load_jst_dataset()  # 18 countries, 1875-2015

# Identify inflation disasters using peak/trough cycles + Butterworth filter
high_disasters, low_disasters = identify_inflation_disasters(df)

# Fit Pareto distributions separately for high-inflation and deflation
high_fit, low_fit = fit_pareto_separate(high_disasters, low_disasters)
# high_fit: alpha=5.45, z_0=1.03, p_tilde=0.356
# low_fit:  alpha=15.18, z_0=1.06, p_tilde=0.085

# Compute P/Q ratios using Epstein-Zin preferences (RRA=3)
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

# Figure 4a: US vs EZ high-inflation disaster probability over time
plot_high_inflation_probability(results_df, save_path="fig4a.png")

# Figure 3: US deflation probability (zoomed to 2011-14)
plot_deflation_probability(us_df, region="US", save_path="fig3_us.png")

# Figure 4c: Risk-neutral density snapshots at key dates
plot_density_snapshots({
    "Mar 2020": (grid_2020, density_2020),  # pre-COVID
    "Mar 2022": (grid_2022, density_2022),  # peak inflation
    "Mar 2024": (grid_2024, density_2024),  # post-tightening
}, title="US risk-neutral densities, 10-year horizon", save_path="fig4c.png")

# Figure 6: Markov chain parameters over time
plot_markov_parameters(params_df, region="US", save_path="fig6_us.png")
```

---

## Data Requirements

### Bloomberg Data (Live)

You need a Bloomberg Terminal with API access. The following instruments are required:

| Instrument | Description | Tickers | Strikes | Maturities |
|------------|-------------|---------|---------|------------|
| **ZC inflation caps** | Zero-coupon caps on cumulative CPI inflation | `USCP{mat}{strike}` / `EUCP{mat}{strike}` | -2% to 6% in 0.5% steps (17 strikes) | 5y, 10y |
| **ZC inflation floors** | Zero-coupon floors on cumulative CPI inflation | `USFP{mat}{strike}` / `EUFP{mat}{strike}` | -2% to 6% in 0.5% steps | 5y, 10y |
| **YOY inflation caps** | Year-on-year caps for forward distributions | `USCPYY{mat}{strike}` / `EUCPYY{mat}{strike}` | -2% to 6% in 0.5% steps | 5y, 6y, 7y, 8y, 9y, 10y |
| **YOY inflation floors** | Year-on-year floors for forward distributions | `USFPYY{mat}{strike}` / `EUFPYY{mat}{strike}` | Same | Same |
| **Inflation swaps** | Breakeven inflation rates | `USSWIT{mat}` / `EUSWI{mat}` | N/A | 1, 2, 3, 5, 7, 10y |
| **Nominal yields** | For discounting | `USGG5YR` / `USGG10YR` (US), `GDBR5` / `GDBR10` (EZ) | N/A | 5y, 10y |

**Strike code convention:** `_strike_to_code(-2.0)` = `"N200"`, `_strike_to_code(0.0)` = `"000"`, `_strike_to_code(4.5)` = `"450"`

**Note on data availability:**
- US ZC data is available from October 2009. EZ from January 2011.
- After August 2021, US strike spacing changed from 0.5% to 1% increments.
- EZ lowest cap strike is 1.5% (not -2%).
- YOY maturities vary: EZ commonly only has 5y, 7y, and 10y (interpolation needed).
- All data is cached locally as Parquet files to avoid repeated Bloomberg queries.

### Historical Data (One-Time Download)

The **risk adjustment** (third adjustment) requires historical GDP and inflation data:

**Jorda-Schularick-Taylor Macrohistory Database:**
1. Go to [macrohistory.net/database](https://www.macrohistory.net/database/)
2. Download the latest Excel file (e.g., `JSTdatasetR6.xlsx`)
3. Place it in `data/historical/`
4. The loader auto-detects common filenames

**18 countries covered:** Australia, Belgium, Canada, Denmark, Finland, France, Germany, Ireland, Italy, Japan, Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom, United States.

**Time period:** 1875-2015 (140 years, ~2,500 country-year observations)

**Why this data:** The paper needs to know how often inflation disasters coincide with output disasters, and how severe the output drops are, in order to calibrate the risk adjustment. US data alone has too few inflation disaster episodes to estimate the Pareto tail reliably.

---

## Mathematical Framework

### Notation

| Symbol | Meaning |
|--------|---------|
| `pi_{T,T+H}` | Log inflation between dates T and T+H |
| `pi_bar` | Central bank inflation target (2%) |
| `d` | Disaster threshold (2pp or 3pp from target) |
| `T` | Forward start (5 years) |
| `H` | Forward horizon (5 years) |
| `n(pi)` | N-probability density (nominal risk-neutral) |
| `q(pi)` | Q-probability density (real risk-neutral, inflation-adjusted) |
| `p(pi)` | Physical probability density |
| `m(pi)` | Stochastic discount factor conditioned on inflation |
| `b(pi)` | Arrow-Debreu inflation security price |
| `a(k)` | Traded option price at strike k |
| `i` | Nominal interest rate |
| `r` | Real interest rate |
| `pi^e` | Expected inflation (breakeven: i - r) |

### Proposition 1: The Main Result

The physical probability density of inflation is:

```
p(pi_{T,T+H}) = n(pi_{T,T+H})                    [from option prices]
              * exp((pi_{T,T+H} - pi^e) * H)       [Inflation Factor]
              * e^{-r*H} * m(pi_{T,T+H})           [Risk Factor]
              * [Horizon Factor]                    [from Markov chain]
```

where the Horizon Factor involves the joint conditional distribution of annual inflation rates over the forward period.

### Equation 4: N to Q Conversion (Inflation Adjustment)

```
q(pi) = n(pi) * exp(pi - pi^e)
```

**Derivation:** The N-probability is `n(pi) = b(pi) * e^{i-pi}` where `b(pi) = p(pi) * m(pi)` is the Arrow-Debreu price. The Q-probability is `q(pi) = b(pi) * e^r`. Therefore `q(pi) = n(pi) * e^{r+pi-i} = n(pi) * e^{pi-pi^e}` since `pi^e = i - r`.

### Equations 8 and 10: Density Extraction (Breeden-Litzenberger)

From option prices `a(k)`:

```
N-distribution: N(k) = 1 + e^{iT} * a'(k)           [eq. 8]
N-density:      n(k) = e^{iT} * a''(k)

Q-distribution: Q(k) = e^{rT} * k * a''(k)           [eq. 10]
```

**Implementation:** Fit cubic spline to SABR-smoothed call prices, take analytic second derivative of the spline. Natural boundary conditions (d2C/dk2 = 0 at endpoints).

### Equation 12: The 8-State Markov Transition Matrix

```
P = [ 1-5p_l   p_l    p_l    p_l    p_l    p_l     0       0    ]   State 0: <= -1%
    [ p_dl+p_nn p_ml   p_mr    0      0      0      0       0    ]   State 1: (-1,0%]
    [ p_dl     p_nn    p_m    p_mr    0      0      0      p_dh  ]   State 2: (0,1%]
    [ p_dl      0     p_nn    p_n    p_nn    0      0      p_dh  ]   State 3: (1,2%]
    [ p_dl      0      0     p_nn    p_n    p_nn    0      p_dh  ]   State 4: (2,3%]
    [ p_dl      0      0      0     p_mr    p_m    p_nn   p_dh  ]   State 5: (3,4%]
    [  0        0      0      0      0     p_mr   p_mh  p_dh+p_nn]  State 6: (4,5%]
    [  0        0     p_h    p_h    p_h    p_h    p_h   1-5p_h   ]   State 7: > 5%
```

where:
- `p_n = 1 - 2*p_nn - p_dl - p_dh` (stay at target)
- `p_m = 1 - p_dl - p_nn - p_mr - p_dh` (stay above/below target)
- `p_ml = 1 - p_dl - p_nn - p_mr` (stay near deflation)
- `p_mh = 1 - p_dh - p_nn - p_mr` (stay near high inflation)

**Structure:** States 0 and 7 are absorbing-like disaster states that exit randomly to normal states. States 1-6 have local diffusion (p_nn), mean reversion (p_mr), and disaster entry (p_dh, p_dl). Boundary states (1, 6) cannot jump to the opposite disaster.

### Equation 2: Risk Adjustment

```
q(1) = [(m_tilde - 1) * p_tilde + 1] * p_d

=> P/Q = 1 / [(m_tilde - 1) * p_tilde + 1]
```

where:
- `m_tilde = E[z^gamma]` under the Pareto distribution of inverse consumption drops
- `p_tilde` = P(output disaster | inflation disaster) from historical data
- `gamma` = relative risk aversion (3.0)
- `z = 1/(1+g)` where g is GDP growth during the disaster

For the Pareto distribution `F(z) = 1 - (z_0/z)^alpha`:

```
E[z^gamma] = alpha * z_0^gamma / (alpha - gamma)     [requires alpha > gamma]
```

### Paper's Calibrated Parameters

| Parameter | High Inflation | Deflation | Pooled |
|-----------|---------------|-----------|--------|
| p_tilde (P(output disaster \| inflation disaster)) | 0.356 | 0.085 | 0.200 |
| alpha (Pareto tail) | 5.45 | 15.18 | 6.38 |
| z_0 (Pareto location) | 1.03 | 1.06 | 1.03 |
| m_tilde (E[z^gamma]) | 2.431 | 1.484 | -- |
| **P/Q ratio** | **0.66** | **0.96** | 0.82 |

---

## Architecture

```
src/inflation_disaster/
|
|-- config.py                         Global constants (all paper values hardcoded)
|
|-- data/
|   |-- bloomberg.py                  Bloomberg blpapi fetcher
|   |   |-- BloombergFetcher          Main class with session management
|   |   |-- fetch_zc_caps_floors()    Zero-coupon caps/floors (5y, 10y)
|   |   |-- fetch_yoy_caps_floors()   Year-on-year caps/floors (5y-10y)
|   |   |-- fetch_inflation_swaps()   Breakeven inflation rates
|   |   |-- fetch_nominal_rates()     Treasury/Bund yields for discounting
|   |   |-- build_option_surface()    Assemble OptionSurface for a date
|   |
|   |-- jst.py                       Historical data (1875-2015)
|   |   |-- load_jst_dataset()        Load JST Macrohistory Database
|   |   |-- identify_inflation_disasters()  Peak/trough disaster identification
|   |   |-- compute_disaster_statistics()   Table 1/2 summary stats
|   |
|   |-- cleaning.py                  Quality checks
|   |   |-- check_cap_monotonicity()  Caps must decrease in strike
|   |   |-- check_floor_monotonicity()  Floors must increase in strike
|   |   |-- check_butterfly()         Convexity (no-arbitrage)
|   |   |-- check_put_call_parity()   Cross-validation with swaps
|   |   |-- clean_surface()           Apply all checks, interpolate violations
|   |   |-- select_best_monthly_dates()  Pick highest-quality day per month
|   |
|   |-- schemas.py                   Pydantic models
|   |   |-- OptionSurface             Raw option data
|   |   |-- CleanedSurface            After quality checks + SABR smoothing
|   |   |-- InflationDistribution     8-bin probability vector (N, Q, or P)
|   |   |-- MarkovParams              6 Markov chain parameters
|   |   |-- SABRParams                4 SABR model parameters
|   |   |-- DisasterProbability       Full output with all measures
|   |   |-- ParetoFit                 Pareto distribution fit results
|
|-- models/
|   |-- sabr.py                      SABR stochastic vol (Hagan 2002)
|   |   |-- sabr_implied_vol()        General SABR formula (single strike)
|   |   |-- sabr_smile()              Vectorized smile computation
|   |   |-- calibrate_sabr()          Multi-start L-BFGS-B fitting
|   |   |-- black_price()             Black-76 pricing
|   |   |-- sabr_to_prices()          Reconstruct prices on fine grid
|   |
|   |-- breeden_litzenberger.py      Density extraction
|   |   |-- extract_n_distribution()  N-density via d2C/dk2 (eq. 8)
|   |   |-- extract_q_distribution()  Q-density directly (eq. 10)
|   |   |-- n_to_q_adjustment()       N->Q conversion (eq. 4)
|   |   |-- density_to_bin_probabilities()  Integrate over 8 bins
|   |   |-- extract_full_distributions()    Complete N+Q extraction
|   |
|   |-- markov_chain.py              8-state inflation dynamics (eq. 12)
|   |   |-- build_transition_matrix()  Construct 8x8 P from 6 params
|   |   |-- validate_transition_matrix()  Verify stochastic matrix
|   |   |-- stationary_distribution()  Solve pi@P=pi
|   |   |-- _simulate_paths()         Numba-JIT MC simulation
|   |   |-- simulate_cumulative_distribution()  H-year avg inflation dist
|   |   |-- simulate_forward_distribution()     5y5y forward dist
|   |   |-- marginal_one_year_distribution()    State dist at year t
|   |
|   |-- gmm.py                      GMM estimation
|   |   |-- GMMTargets               Dataclass for monthly targets
|   |   |-- _moment_conditions()      21-vector g(theta)
|   |   |-- _gmm_objective()          g'Wg criterion
|   |   |-- estimate_single_month()   Single-month estimation
|   |   |-- estimate_full_sample()    Two-stage full-sample estimation
|   |
|   |-- pareto.py                    Consumption disaster distribution
|   |   |-- fit_pareto()             MLE with bias correction
|   |   |-- fit_pareto_separate()    Separate high/low inflation fits
|   |
|   |-- epstein_zin.py              Risk adjustment
|   |   |-- expected_marginal_utility_ratio()   E[z^gamma] under Pareto
|   |   |-- risk_adjustment_factor()             P/Q from eq. (2)
|   |   |-- compute_risk_adjustments()           Both tails at once
|   |   |-- default_risk_adjustments()           Paper's calibrated values
|
|-- adjustments/
|   |-- inflation_adj.py             N -> Q conversion
|   |-- horizon_adj.py               Spot -> 5y5y forward
|   |-- risk_adj.py                  Q -> P physical measure
|   |-- pipeline.py                  DisasterProbabilityPipeline
|
|-- analytics/
|   |-- disaster_probs.py            Time series construction
|   |-- decomposition.py             Adjustment factor decomposition
|   |-- anchoring.py                 Conditional anchoring analysis
|
|-- visualization/
|   |-- style.py                     Paper-consistent matplotlib style
|   |-- time_series.py               Figures 3, 4, 6 reproduction
|
|-- utils/
    |-- numerical.py                 Splines, integration, grid construction
    |-- logging.py                   Structured logging
```

**Total: 33 Python files, ~4,500 lines of code, 67 tests**

---

## Configuration

All paper constants are in `config.py` and can be overridden via `INFL_`-prefixed environment variables:

```bash
export INFL_MC_N_PATHS=500000       # More MC paths for production
export INFL_GMM_N_STARTS=30         # More optimization restarts
export INFL_RRA=4.0                 # Higher risk aversion
export INFL_TARGET_INFLATION=2.0    # Change inflation target
```

### Complete Settings Reference

| Setting | Default | Paper Ref | Description |
|---------|---------|-----------|-------------|
| `target_inflation` | 2.0 | pi_bar | Central bank target (%) |
| `disaster_thresholds` | (2.0, 3.0) | d | Thresholds in pp from target |
| `n_bins` | 8 | eq. 12 | Markov chain states |
| `bin_edges` | (-1e10, -1, 0, ..., 5, 1e10) | | Bin boundaries (%) |
| `strike_min` / `strike_max` / `strike_step` | -2.0 / 6.0 / 0.5 | | Option strike grid |
| `maturities` | (5, 10) | T, T+H | Traded option maturities |
| `t_forward` / `h_horizon` | 5 / 5 | T, H | 5y5y forward definition |
| `mc_n_paths` | 200,000 | | Monte Carlo simulation paths |
| `mc_seed` | 42 | | Random seed for reproducibility |
| `gmm_n_starts` | 20 | | Multi-start optimizations |
| `rra` | 3.0 | gamma | Relative risk aversion |
| `eis` | 1.5 | | Elasticity of intertemporal substitution |
| `sabr_beta` | 0.5 | | CEV exponent for inflation options |
| `risk_adj_high` | 0.66 | Table 1 | Default P/Q for high inflation |
| `risk_adj_low` | 0.96 | Table 1 | Default P/Q for deflation |
| `us_p_mr` / `ez_p_mr` | 0.50 / 0.47 | Sec. 5.2.3 | Mean-reversion (constant) |
| `us_p_l` / `us_p_h` | 0.199 / 0.200 | Sec. 5.2.3 | US disaster exit probs |
| `ez_p_l` / `ez_p_h` | 0.200 / 0.062 | Sec. 5.2.3 | EZ disaster exit probs |
| `pareto_alpha_high` / `pareto_z0_high` | 5.45 / 1.03 | Table 2 | High-inflation Pareto |
| `pareto_alpha_low` / `pareto_z0_low` | 15.18 / 1.06 | Table 2 | Deflation Pareto |

---

## Testing

### Test Suite

67 tests across 6 test files covering every module:

```bash
# Run all tests
PYTHONPATH=src python -m pytest tests/ -v

# Run specific module tests
PYTHONPATH=src python -m pytest tests/test_sabr.py -v
PYTHONPATH=src python -m pytest tests/test_markov_chain.py -v
PYTHONPATH=src python -m pytest tests/test_adjustments.py -v
```

### Test Coverage by Module

| Test File | Module | Tests | What's Tested |
|-----------|--------|-------|---------------|
| `test_sabr.py` | SABR model | 9 | ATM consistency, positive vol, rho skew, calibration recovery, Black-76 put-call parity, edge cases |
| `test_markov_chain.py` | Markov chain | 14 | Stochastic matrix validity, rows sum to 1, non-negativity, disaster exit structure, row symmetry, stationary distribution, MC convergence, seed independence, dispersion with horizon |
| `test_adjustments.py` | All 3 adjustments | 18 | Inflation adj direction/magnitude, bin probability normalization, horizon adj bin mapping, risk adj paper values (0.66/0.96), Pareto MLE recovery, identical-value edge case |
| `test_schemas.py` | Pydantic models | 16 | MarkovParams feasibility (middle rows, boundary rows, disaster exits), SABRParams bounds (beta, rho, alpha), InflationDistribution validation (length, negativity, sum), horizon type |
| `test_config.py` | Configuration | 7 | Bin edges count, sorting, midpoints, strike count/endpoints/spacing, paper constants |
| `test_cleaning.py` | Data cleaning | 6 | Cap/floor monotonicity detection, butterfly convexity detection |

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single date (all 7 steps) | ~30 seconds | Dominated by MC simulation |
| GMM single month (20 starts) | ~2-3 minutes | Each start runs MC inside optimizer |
| Full sample, US (160 months) | ~3-4 hours | With 4 parallel workers |
| Full sample, EZ (160 months) | ~3-4 hours | Independent, can run in parallel with US |
| Bloomberg data fetch (one month) | ~5-10 seconds | Cached after first pull |
| JST historical data load | ~2 seconds | 2,500 rows |

**Bottleneck:** The GMM estimator calls Monte Carlo simulation inside the optimization loop. Each objective function evaluation requires 200K paths x 5 years = 1M state transitions. With 20 multi-start optimizations and ~50 iterations each, that's ~1 billion state transitions per month.

**Mitigation:** Numba JIT compilation of `_simulate_paths` provides ~50x speedup over pure Python. `ProcessPoolExecutor` parallelizes across months. Reducing `mc_n_paths` to 50K during development gives 4x speedup with modest accuracy loss.

---

## Validation Against the Paper

### Exact Numerical Matches

| Quantity | Our Value | Paper Value | Source |
|----------|-----------|-------------|--------|
| P/Q ratio, high inflation | **0.663** | 0.66 | Table 1, Panel B |
| P/Q ratio, deflation | **0.960** | 0.96 | Table 1, Panel D |
| E[z^gamma], high inflation | 2.431 | -- | Derived from alpha=5.45, z_0=1.03, gamma=3 |
| E[z^gamma], deflation | 1.484 | -- | Derived from alpha=15.18, z_0=1.06, gamma=3 |

### Structural Validation

| Property | Verified |
|----------|----------|
| Transition matrix rows sum to 1 | Yes (all 8 rows, tolerance 1e-10) |
| All transition probabilities non-negative | Yes |
| Stationary distribution exists and is unique | Yes |
| Stationary distribution peaks near 2% target | Yes (bins 3+4 > 50% mass) |
| MC simulation distributions sum to 1 | Yes (tolerance 0.01) |
| 5y distribution more concentrated than 10y | Yes (5y peak > 10y peak) |
| Forward uses different random seed than spot | Yes (verified non-identical) |
| Black-76 put-call parity holds | Yes (tolerance 1e-10) |
| SABR ATM formula matches general formula | Yes (tolerance 1e-10) |

---

## Known Limitations

1. **Bloomberg ticker conventions** may differ across terminals. The templates in `bloomberg.py` follow common patterns but may need adjustment.

2. **US data after August 2021** has reduced strike granularity (1% instead of 0.5% increments). SABR interpolation handles this but with reduced precision.

3. **EZ YOY data** is only available at 5y, 7y, and 10y maturities (not annually). Linear interpolation is used for missing years.

4. **The risk adjustment is time-invariant.** The paper estimates a single P/Q ratio from 140 years of historical data. In reality, the relationship between inflation and output disasters may vary over time.

5. **Liquidity concerns.** Post-2021, the US inter-dealer inflation options market has thinned, though dealer-to-client volume remains. Monthly frequency mitigates day-to-day noise.

6. **The `extract_q_distribution` function** (direct Q extraction via eq. 10) has a known Jacobian issue when called with annual-rate strikes. The pipeline uses the N-then-Q path instead, which is correct.

---

## Bug Fix History

| Round | Bugs Fixed | Key Fixes |
|-------|-----------|-----------|
| Round 1 | 9 | SABR denominator formula, pipeline strike units, implied real rate /T, fine grid range, MC seed reuse |
| Round 2 | 12 | Missing pandas import, Bloomberg timeout loop, date type mismatch, weighted LSQ, Pareto div-by-zero, SABRParams beta validation |
| Rounds 3-4 | 17 | np.histogram with inf edges, eigenvector sign flip, SABR log NaN guard, N-density normalization, ZC put-call parity formula, Pareto bias correction, truncated expectation normalization, bin edge double-counting, GMM year range |
| **Total** | **38** | Across 4 rounds of comprehensive auditing |

---

## Citation

```bibtex
@article{hilscher2024inflation,
  title={How likely is an inflation disaster?},
  author={Hilscher, Jens and Raviv, Alon and Reis, Ricardo},
  journal={Journal of Financial Economics},
  year={2024},
  note={CFMDP2024-37}
}
```

**Paper website with data:** [r2rsquaredlse.github.io/web-inflationdisasters](https://r2rsquaredlse.github.io/web-inflationdisasters/)

**Authors:** Jens Hilscher (UC Davis), Alon Raviv (Bar-Ilan University), Ricardo Reis (LSE)

---

## License

This implementation is for research and internal use at State Street Global Markets. The methodology and all equations are from Hilscher, Raviv, and Reis (2024). The Jorda-Schularick-Taylor Macrohistory Database has its own license terms at [macrohistory.net](https://www.macrohistory.net/).
