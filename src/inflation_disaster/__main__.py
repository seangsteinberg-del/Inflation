"""CLI entry point: python -m inflation_disaster

Usage:
    python -m inflation_disaster --region US --start 2009-10-01 --end 2024-04-30
    python -m inflation_disaster --region EZ --start 2011-01-01 --end 2024-04-30
    python -m inflation_disaster --region US --date 2024-01-02  # single date
    python -m inflation_disaster --calibrate-risk  # calibrate risk adj from JST data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from inflation_disaster.config import settings
from inflation_disaster.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inflation_disaster",
        description="Estimate inflation disaster probabilities from Bloomberg options data",
    )
    parser.add_argument("--region", choices=["US", "EZ"], default="US")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--date", type=str, help="Single date mode (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Disaster threshold d (pp)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--calibrate-risk", action="store_true", help="Calibrate risk adj from JST data")
    parser.add_argument("--mc-paths", type=int, default=None, help="Override MC paths")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for GMM")
    parser.add_argument("--no-bloomberg", action="store_true", help="Skip Bloomberg, use cached data")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def run_risk_calibration():
    """Calibrate risk adjustment factors from historical data."""
    from inflation_disaster.data.jst import load_jst_dataset, identify_inflation_disasters, compute_disaster_statistics
    from inflation_disaster.models.pareto import fit_pareto_separate
    from inflation_disaster.models.epstein_zin import compute_risk_adjustments

    log = logging.getLogger("inflation_disaster")
    log.info("Loading JST Macrohistory Database...")
    df = load_jst_dataset()

    log.info("Identifying inflation disasters...")
    high, low = identify_inflation_disasters(df)
    stats = compute_disaster_statistics(high, low)

    log.info(f"Found {stats['n_inflation_disasters']} inflation disasters")
    log.info(f"  High inflation: {stats['n_high_disasters']} (p_tilde={stats['p_tilde_high']:.3f})")
    log.info(f"  Deflation: {stats['n_low_disasters']} (p_tilde={stats['p_tilde_low']:.3f})")

    log.info("Fitting Pareto distributions...")
    high_fit, low_fit = fit_pareto_separate(high, low)

    log.info("Computing risk adjustment factors...")
    adj_h, adj_l = compute_risk_adjustments(high_fit, low_fit, gamma=settings.rra)

    print(f"\n{'='*60}")
    print(f"RISK ADJUSTMENT CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"High inflation:  alpha={high_fit.alpha:.2f}, z_0={high_fit.z_0:.3f}, "
          f"p_tilde={high_fit.p_tilde:.3f}, P/Q={adj_h:.3f}")
    print(f"Deflation:       alpha={low_fit.alpha:.2f}, z_0={low_fit.z_0:.3f}, "
          f"p_tilde={low_fit.p_tilde:.3f}, P/Q={adj_l:.3f}")
    print(f"Paper values:    high P/Q=0.66, low P/Q=0.96")
    print(f"{'='*60}")


def run_single_date(args):
    """Run pipeline for a single date."""
    from inflation_disaster.data.bloomberg import BloombergFetcher
    from inflation_disaster.data.cache import ResultsCache
    from inflation_disaster.adjustments.pipeline import DisasterProbabilityPipeline
    from inflation_disaster.data.schemas import MarkovParams

    log = logging.getLogger("inflation_disaster")
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    region = args.region

    log.info(f"Running single-date pipeline for {region} on {target_date}")

    # Fetch data
    fetcher = BloombergFetcher()
    start = date(target_date.year, target_date.month, 1)
    end = date(target_date.year, target_date.month, 28)

    zc_data = fetcher.fetch_zc_caps_floors(region, start, end)
    swap_data = fetcher.fetch_inflation_swaps(region, start, end)

    # Build and clean surfaces
    surface_5y = fetcher.build_option_surface(zc_data, swap_data, target_date, region, 5)
    surface_10y = fetcher.build_option_surface(zc_data, swap_data, target_date, region, 10)

    if surface_5y is None or surface_10y is None:
        log.error("Could not build option surfaces. Check Bloomberg data availability.")
        fetcher.close()
        return

    from inflation_disaster.data.surface_builder import build_ready_surface

    cleaned_5y = build_ready_surface(surface_5y)
    cleaned_10y = build_ready_surface(surface_10y)

    if cleaned_5y is None or cleaned_10y is None:
        log.error("Surfaces failed quality checks or SABR calibration.")
        fetcher.close()
        return

    # Use paper defaults for Markov params
    if region == "US":
        markov_params = MarkovParams(
            p_dh=0.05, p_dl=0.02, p_nn=0.12,
            p_h=settings.us_p_h, p_l=settings.us_p_l, p_mr=settings.us_p_mr,
        )
    else:
        markov_params = MarkovParams(
            p_dh=0.04, p_dl=0.03, p_nn=0.10,
            p_h=settings.ez_p_h, p_l=settings.ez_p_l, p_mr=settings.ez_p_mr,
        )

    # Determine current inflation state (default to 2-3% bin)
    current_state = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float)

    # Run pipeline
    pipeline = DisasterProbabilityPipeline(region)
    result = pipeline.process_single_date(
        cleaned_5y, cleaned_10y, markov_params, current_state, args.threshold,
    )

    fetcher.close()

    # Output
    print(f"\n{'='*60}")
    print(f"INFLATION DISASTER PROBABILITIES: {region} on {target_date}")
    print(f"{'='*60}")
    print(f"Threshold: {args.threshold}pp (disaster = >{settings.target_inflation + args.threshold}% or <{settings.target_inflation - args.threshold}%)")
    print(f"")
    print(f"HIGH INFLATION DISASTER (>{ settings.target_inflation + args.threshold}%):")
    print(f"  N-prob 5y:     {result.prob_high_n_5y:7.2%}")
    print(f"  Q-prob 5y:     {result.prob_high_q_5y:7.2%}  (inflation adj: {result.inflation_adj_5y:.2f}x)")
    print(f"  N-prob 10y:    {result.prob_high_n_10y:7.2%}")
    print(f"  Q-prob 10y:    {result.prob_high_q_10y:7.2%}  (inflation adj: {result.inflation_adj_10y:.2f}x)")
    print(f"  Q-prob 5y5y:   {result.prob_high_q_5y5y:7.2%}  (horizon adj: {result.horizon_adj_high:.2f}x)")
    print(f"  P-prob 5y5y:   {result.prob_high_p_5y5y:7.2%}  (risk adj: {result.risk_adj_high:.2f}x)")
    print(f"")
    print(f"DEFLATION DISASTER (<{settings.target_inflation - args.threshold}%):")
    print(f"  N-prob 5y:     {result.prob_low_n_5y:7.2%}")
    print(f"  Q-prob 5y:     {result.prob_low_q_5y:7.2%}")
    print(f"  N-prob 10y:    {result.prob_low_n_10y:7.2%}")
    print(f"  Q-prob 10y:    {result.prob_low_q_10y:7.2%}")
    print(f"  Q-prob 5y5y:   {result.prob_low_q_5y5y:7.2%}  (horizon adj: {result.horizon_adj_low:.2f}x)")
    print(f"  P-prob 5y5y:   {result.prob_low_p_5y5y:7.2%}  (risk adj: {result.risk_adj_low:.2f}x)")
    print(f"{'='*60}")

    # Save if requested
    if args.output:
        from inflation_disaster.analytics.disaster_probs import results_to_dataframe
        df = results_to_dataframe([result])
        df.to_csv(args.output, index=False)
        log.info(f"Results saved to {args.output}")

    return result


def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.mc_paths:
        settings.mc_n_paths = args.mc_paths

    if args.calibrate_risk:
        run_risk_calibration()
    elif args.date:
        run_single_date(args)
    elif args.start and args.end:
        print("Full sample mode: use the pipeline API directly for now.")
        print("Example:")
        print("  from inflation_disaster.adjustments.pipeline import DisasterProbabilityPipeline")
        print(f"  pipeline = DisasterProbabilityPipeline('{args.region}')")
        print("  # Loop over monthly dates and call pipeline.process_single_date()")
    else:
        print("Usage: python -m inflation_disaster --date 2024-01-02 --region US")
        print("       python -m inflation_disaster --calibrate-risk")
        print("       python -m inflation_disaster --help")


if __name__ == "__main__":
    main()
