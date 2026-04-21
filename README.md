# How Likely Is an Inflation Disaster?

**A Python implementation of [Hilscher, Raviv, and Reis (2024)](https://r2rsquaredlse.github.io/web-inflationdisasters/) for estimating market-implied tail probabilities of inflation disasters from live Bloomberg YOY inflation options data.**

Built for the State Street Global Markets macro strategy team.

---

## What This Does

Estimates the probability that average annual inflation between 5 and 10 years from now will exceed 4% (high-inflation disaster) or fall below 0% (deflation disaster):

```
P[ avg annual inflation from year 5 to year 10 > 4% ]    high-inflation disaster
P[ avg annual inflation from year 5 to year 10 < 0% ]    deflation disaster
```

These are **physical-measure (P) probabilities** derived from traded inflation option prices, with three corrections from the paper:

1. **Inflation adjustment** (N to Q): nominal option payoffs are worth less in real terms during high inflation
2. **Horizon adjustment** (spot to 5y5y): uses an 8-state Markov chain to convert spot option-implied distributions into 5-year-forward probabilities
3. **Risk adjustment** (Q to P): investors overpay for disaster insurance; P/Q = 0.66 for high inflation, 0.96 for deflation

---

## Two Operating Modes

### Paper-Calibrated Mode (exact match)

Uses the paper's published [.dta files](https://r2rsquaredlse.github.io/web-inflationdisasters/) as calibration targets. The Markov chain parameters are found by matching the paper's P-measure 5y5y disaster probabilities. Then the pipeline can run any day using live Bloomberg CPI to update the initial state.

| Measure | Ours | Paper (Feb 2026) | Diff |
|---------|------|------------------|------|
| US P(>4%, 5y5y) | 2.58% | 2.56% | +0.02pp |
| US P(<0%, 5y5y) | 4.39% | 4.34% | +0.04pp |
| EZ P(>4%, 5y5y) | 5.34% | 5.35% | -0.01pp |
| EZ P(<0%, 5y5y) | 2.87% | 2.87% | +0.00pp |

### Self-Sufficient Mode (Bloomberg only)

Calibrates the Markov chain entirely from YOY option prices and swap rates, with no paper data. Extracts the full term structure of annual inflation distributions at maturities 2, 3, 5, 7, and 10 years, plus forward caplet-stripped distributions for years 6-10.

**High-inflation tail**: well-constrained by cap strikes at 3-6%.
- US Q(>4%, 5y) = 11.0% vs paper's 10.6% (+0.4pp)

**Deflation tail**: underdetermined because floor strikes only go down to 1%. The swap rate (which pins the distribution mean) helps but cannot fully identify tail probabilities.
- US P(<0%, 5y5y) ~2% vs paper's 4.3% (-2.3pp gap)

The deflation gap is a fundamental data limitation: the paper uses zero-coupon (ZC) options that directly price cumulative tail probabilities, while Bloomberg only provides year-on-year (YOY) options on our terminal. See [Known Limitations](#known-limitations).

---

## Quick Start

### Run the Pipeline

```bash
python run_live.py
```

Pulls ~85 live Bloomberg tickers (YOY caps/floors, inflation swaps, nominal yields, CPI) and produces disaster probability estimates for both US and EZ. Takes ~15-20 minutes due to Markov chain calibration.

### What It Does Step by Step

```
Bloomberg YOY caps/floors (2Y, 3Y, 5Y, 7Y, 10Y) + swaps + CPI
    |
    v
[1] Extract annual Q-distributions via CDF gradient approach
    - Per-year caplet/floorlet prices from YOY options / annuity
    - Survival function from cap price differences: P(pi > K)
    - CDF from floor price differences: P(pi < K)
    - Exponential tail extrapolation + swap rate anchor
    - Bin into 8 inflation states
    |
    v
[2a] Paper-calibrated Markov params (if .dta file available)
    - Grid search + Nelder-Mead, 5x500K MC, targets paper's P-measure
    - Uses paper's CPI (hard state assignment) for apples-to-apples
    |
[2b] Self-sufficient Markov params (Bloomberg only)
    - Two-stage: fix p_dh/p_nn from distribution shape, sweep p_dl
    - Full term structure matching at 5 maturities + forward
    - Swap rate constraint on distribution mean
    |
    v
[3] Monte Carlo simulation (5x500K paths, multi-seed)
    - Cumulative distributions at 5Y and 10Y
    - 5y5y forward distribution
    |
    v
[4] Risk adjustment: P = Q * 0.66 (high) or Q * 0.96 (low)
    |
    v
OUTPUT: P(>4%, 5y5y) and P(<0%, 5y5y) for US and EZ
```

---

## Data Requirements

### Bloomberg Data (Live)

| Instrument | Tickers | Strikes | Maturities |
|------------|---------|---------|------------|
| US YOY caps | `USISC{strike}{mat} Curncy` | 3%, 4%, 5%, 6% | 2, 3, 5, 7, 10Y |
| US YOY floors | `USISF{strike}{mat} Curncy` | 1%, 2% | 1, 2, 5, 7, 10Y |
| EZ YOY caps | `EUISC{strike}{mat} Curncy` | 1%, 2%, 3%, 4%, 5% | 1, 2, 3, 5, 7, 10Y |
| EZ YOY floors | `EUISF{strike}{mat} Curncy` | 1%, 2% | 1, 2, 3, 5, 7, 10Y |
| Inflation swaps | `USSWIT{mat}` / `EUSWI{mat}` | N/A | 1, 2, 3, 5, 7, 10Y |
| Nominal yields | `USGG5YR`, `USGG10YR`, `GDBR5`, `GDBR10` | N/A | 5Y, 10Y |
| CPI | `CPI YOY Index`, `ECCPEMUY Index` | N/A | N/A |

**Ticker convention:** `USISC35 Curncy` = US inflation cap, 3% strike, 5Y maturity. Integer strike and maturity appended directly.

### Paper Data (Optional, for Paper-Calibrated Mode)

Download `USwestimates.dta` and `EZwestimates.dta` from [the paper's website](https://r2rsquaredlse.github.io/web-inflationdisasters/) and place in `data/paper_data/`. Updated monthly (latest: Vintage 7, Feb 2026).

---

## The Paper's Methodology vs Ours

The paper (Appendix B) uses **two sets of option data**:

1. **Zero-coupon (ZC) caps and floors** for 5Y and 10Y cumulative distributions
2. **Year-on-year (YOY) caps and floors** for forward annual distributions (years 5-9)

Their GMM estimates 6 Markov chain parameters from 21 moments (7 bins x 3 distributions: 5Y ZC, 10Y ZC, forward YOY).

**Our approach** uses only YOY data (ZC tickers return no data on our terminal) and compensates with:

- **Full term structure**: annual distributions at 5 maturities (2, 3, 5, 7, 10Y) instead of just 2 ZC horizons
- **Swap rate matching**: pins the distribution mean, indirectly constraining the left tail
- **Forward caplet stripping**: (10Y price - 5Y price) / forward annuity for years 6-10

This gives 42+ moment conditions for 6 parameters, comparable to the paper's 21.

---

## Key Files

| File | Purpose |
|------|---------|
| `run_live.py` | Main pipeline. Pulls Bloomberg, calibrates Markov chain, computes disaster probabilities |
| `backtest.py` | 181-month historical backtest (2011-2026) |
| `visualize_results.py` | Publication-quality charts |
| `src/inflation_disaster/config.py` | All paper constants (bin edges, Pareto params, risk adjustments) |
| `src/inflation_disaster/models/markov_chain.py` | 8-state Markov chain with Numba-JIT MC simulation |
| `src/inflation_disaster/models/sabr.py` | SABR stochastic vol model (Hagan 2002) |
| `src/inflation_disaster/adjustments/horizon_adj.py` | Spot-to-5y5y forward conversion |
| `src/inflation_disaster/adjustments/risk_adj.py` | Q-to-P risk adjustment (0.66 / 0.96) |
| `src/inflation_disaster/data/surface_builder.py` | CPI state assignment, SABR surface building |
| `data/paper_data/` | Paper's published .dta files (USwestimates, EZwestimates) |

---

## The Three Adjustments

### 1. Inflation Adjustment (N to Q)

```
q(pi) = n(pi) * exp(pi - pi^e)
```

Inflation options pay nominal dollars. A $1 payoff when inflation is 6% is worth less in real terms. Median factors: 1.09 (5Y), 1.24 (10Y) for high inflation.

### 2. Horizon Adjustment (spot to 5y5y forward)

Uses the 8-state Markov chain (equation 12 of the paper) with 6 parameters:

- `p_dh`, `p_dl`: probability of entering high/low inflation disaster (time-varying)
- `p_nn`: local volatility of normal inflation (time-varying)
- `p_h`, `p_l`: probability of exiting disaster (constant)
- `p_mr`: mean reversion probability (constant)

The chain advances 5 years (matrix power P^5), then simulates 5 more years via Monte Carlo (500K paths x 5 seeds) to get the 5y5y forward distribution.

### 3. Risk Adjustment (Q to P)

```
P(high disaster) = Q(high disaster) * 0.66
P(defl disaster) = Q(defl disaster) * 0.96
```

Estimated from Pareto fits to 18-country historical data (1875-2015) on joint inflation-output disasters, using Epstein-Zin preferences with RRA = 3.

---

## Known Limitations

1. **No ZC option data on our terminal.** The paper extracts cumulative distributions from zero-coupon inflation caps/floors (Appendix B). Our Bloomberg terminal returns no data for ZC tickers (USISCD, USISCQ). We use YOY options instead, which give annual distributions. The Markov chain bridges annual to cumulative, but the deflation tail is underdetermined because YOY floor strikes (1%, 2%) don't reach the 0% deflation boundary.

2. **Deflation tail gap in self-sufficient mode.** Without ZC data or floor strikes at 0%, P(<0%, 5y5y) is systematically underestimated by ~2pp. The swap rate constrains the distribution mean but not the tails (changing P(<0%) from 0.5% to 5% barely moves the mean). Paper-calibrated mode eliminates this gap entirely.

3. **CPI date mismatch.** Live estimates use today's CPI; the paper uses the CPI from the month of their data vintage. With US CPI at 3.3% (Apr 2026) vs 2.8% (Feb 2026), the starting state shifts significantly, affecting the 5y5y forward.

4. **Risk adjustment is time-invariant.** P/Q = 0.66 and 0.96 are fixed constants from 140 years of historical data. The actual ratio may vary over time.

5. **US YOY strike coverage is sparse.** Only 4 cap strikes (3-6%) and 2 floor strikes (1-2%) per maturity. EZ has better coverage (5 cap strikes, 2 floor strikes).

---

## Paper's Constant Parameters

| Parameter | US | EZ | Description |
|-----------|----|----|-------------|
| p_mr | 0.50 | 0.47 | Mean reversion probability |
| p_l | 0.199 | 0.200 | Exit probability, deflation disaster |
| p_h | 0.200 | 0.062 | Exit probability, high-inflation disaster |
| risk_adj_high | 0.66 | 0.66 | P/Q ratio, high inflation |
| risk_adj_low | 0.96 | 0.96 | P/Q ratio, deflation |
| Pareto alpha (high) | 5.45 | 5.45 | Tail thickness, high-inflation output disasters |
| Pareto alpha (low) | 15.18 | 15.18 | Tail thickness, deflation output disasters |

---

## Installation

```bash
pip install numpy scipy pandas matplotlib pydantic pydantic-settings numba statsmodels blpapi
python -m pytest tests/ -v  # 121 tests
```

Requires Python 3.10+ and a Bloomberg Terminal with API access.

---

## Citation

```bibtex
@article{hilscher2025inflation,
  title={How likely is an inflation disaster?},
  author={Hilscher, Jens and Raviv, Alon and Reis, Ricardo},
  journal={Review of Financial Studies},
  year={2025}
}
```

**Paper website:** [r2rsquaredlse.github.io/web-inflationdisasters](https://r2rsquaredlse.github.io/web-inflationdisasters/)
