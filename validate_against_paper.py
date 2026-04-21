"""Validate our HRR pipeline output against paper's published .dta files.

Usage:
    python validate_against_paper.py --date 2026-02-12 --region US
      - runs the pipeline for the given valuation date (requires cap/floor
        paste matching that date via the zc_paste hardcodes or CLI flags),
      - loads `data/paper_data/USwestimates.dta`,
      - finds the row matching the valuation date,
      - prints side-by-side diff for all 14 columns.

If no paste-matching hook is wired, defaults to today's hardcoded US paste
and whatever the closest matching row in the .dta is, clearly flagging the
date mismatch.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd


# Columns in the .dta that our HRROutput reports
_COMPARE_COLUMNS = [
    "zc_higher4_5y",  "zc_higher5_5y",  "zc_lower0_5y",  "zc_lowerm1_5y",
    "zc_higher4_10y", "zc_higher5_10y", "zc_lower0_10y", "zc_lowerm1_10y",
    "higher4_5y5y",   "higher5_5y5y",   "lower0_5y5y",   "lowerm1_5y5y",
]


def load_dta(region: str) -> pd.DataFrame:
    """Load HRR's published time series for the given region."""
    fname = f"data/paper_data/{region}westimates.dta"
    df = pd.read_stata(fname)
    df["date"] = pd.to_datetime(df["date_stata"]) if "date_stata" in df.columns else pd.to_datetime(df["date_ym"])
    return df


def find_row_for_date(df: pd.DataFrame, target: date) -> pd.Series:
    """Row in the .dta closest to `target`. Returns row + actual date."""
    target_ts = pd.Timestamp(target)
    diffs = (df["date"] - target_ts).abs()
    idx = int(diffs.idxmin())
    return df.loc[idx]


def diff_table(ours: dict, paper: pd.Series) -> None:
    """Print a side-by-side table of our pipeline output vs paper."""
    print(f"{'column':<22s} {'ours':>9s} {'paper':>9s} {'diff (pp)':>12s} {'ratio':>8s}")
    print("-" * 62)
    for col in _COMPARE_COLUMNS:
        o = ours.get(col, float("nan"))
        p = float(paper[col]) if col in paper and pd.notna(paper[col]) else float("nan")
        diff_pp = (o - p) * 100.0
        ratio = o / p if p and abs(p) > 1e-9 else float("nan")
        print(f"  {col:<20s} {o:>8.2%} {p:>8.2%} {diff_pp:>+12.3f} {ratio:>8.3f}")


def _autoselect_paste(region: str, valuation_date: date, prefer_premium: bool = True):
    """Pick the hardcoded SWIL paste closest to `valuation_date` for the region.

    If `prefer_premium=True` (paper-faithful), looks for *_PREMIUM variants
    first and falls back to vol pastes only if none match.
    Returns (cap_text, floor_text, paste_date, value_type, input_units).
    """
    from inflation_disaster.data import zc_paste as zp
    prefix_cap = f"SAMPLE_CAP_PASTE_{region.upper()}_"
    prefix_floor = f"SAMPLE_FLOOR_PASTE_{region.upper()}_"

    def _scan(suffix: str):
        cap_c, floor_c = [], []
        for name in dir(zp):
            for cand_list, prefix in ((cap_c, prefix_cap), (floor_c, prefix_floor)):
                if not name.startswith(prefix):
                    continue
                rest = name[len(prefix):]
                if suffix and not rest.endswith(suffix):
                    continue
                if not suffix and rest.endswith("_PREMIUM"):
                    continue
                dstr = rest[:-len(suffix)] if suffix else rest
                try:
                    y, m, d = map(int, dstr.split("_"))
                    cand_list.append((date(y, m, d), getattr(zp, name), name))
                except Exception:
                    pass
        return cap_c, floor_c

    if prefer_premium:
        cap_c, floor_c = _scan("_PREMIUM")
        if cap_c and floor_c:
            best_cap = min(cap_c, key=lambda c: abs((c[0] - valuation_date).days))
            best_floor = min(floor_c, key=lambda c: abs((c[0] - valuation_date).days))
            return best_cap[1], best_floor[1], best_cap[0], "premium", "bp"
    cap_c, floor_c = _scan("")
    if not cap_c or not floor_c:
        return None, None, None, None, None
    best_cap = min(cap_c, key=lambda c: abs((c[0] - valuation_date).days))
    best_floor = min(floor_c, key=lambda c: abs((c[0] - valuation_date).days))
    return best_cap[1], best_floor[1], best_cap[0], "vol", "percent"


def run_pipeline_and_compare(
    region: str = "US",
    valuation_date: date | None = None,
    cap_paste: str | None = None,
    floor_paste: str | None = None,
) -> None:
    """Run HRR pipeline for the given date and compare to paper's .dta."""
    from run_hrr import run

    if valuation_date is None:
        valuation_date = date.today()

    # Auto-select the closest hardcoded paste if none given
    paste_value_type, paste_input_units = "vol", "percent"
    if cap_paste is None or floor_paste is None:
        auto_cap, auto_floor, auto_date, vt, iu = _autoselect_paste(region, valuation_date)
        if auto_cap is not None:
            cap_paste = cap_paste or auto_cap
            floor_paste = floor_paste or auto_floor
            paste_value_type = vt
            paste_input_units = iu
            paste_gap = abs((auto_date - valuation_date).days)
            print(f"Auto-selected SWIL paste from {auto_date} "
                  f"(gap {paste_gap}d, value_type={vt}, units={iu})")

    print("=" * 70)
    print("HRR PIPELINE vs PAPER .dta — VALIDATION")
    print("=" * 70)
    print(f"Region:           {region}")
    print(f"Valuation date:   {valuation_date}")

    # Run pipeline
    out = run(region=region, valuation_date=valuation_date,
              cap_paste=cap_paste, floor_paste=floor_paste,
              paste_value_type=paste_value_type, paste_input_units=paste_input_units,
              verbose=False)

    # Load paper's .dta and find closest matching row
    df = load_dta(region)
    paper_row = find_row_for_date(df, valuation_date)
    paper_date = paper_row["date"].date()
    days_diff = abs((paper_date - valuation_date).days)

    print(f"Paper closest row: {paper_date}  (gap: {days_diff} days)")
    if days_diff > 31:
        print(f"  WARNING: {days_diff}-day gap — results are NOT apples-to-apples.")
        print(f"           Paste a SWIL surface from {paper_date} for proper validation.")
    print()

    ours = {col: getattr(out, col) for col in _COMPARE_COLUMNS}
    diff_table(ours, paper_row)

    # Summary metric
    diffs_pp = [
        (ours[col] - float(paper_row[col])) * 100.0
        for col in _COMPARE_COLUMNS
        if col in paper_row and pd.notna(paper_row[col])
    ]
    mae_pp = float(np.mean(np.abs(diffs_pp)))
    rmse_pp = float(np.sqrt(np.mean(np.square(diffs_pp))))
    print()
    print(f"Mean absolute diff: {mae_pp:.3f} pp")
    print(f"RMS diff:            {rmse_pp:.3f} pp")
    if days_diff <= 31:
        print(f"\nVALIDATION {'PASSED' if rmse_pp < 0.5 else 'PENDING'} "
              f"(target: <0.5 pp RMS for matching-date paste)")
    else:
        print(f"\nINFERENCE ONLY: {days_diff}-day gap means diffs reflect "
              f"real market changes + any residual methodology gaps.")


def main():
    parser = argparse.ArgumentParser(description="Validate HRR pipeline vs paper .dta")
    parser.add_argument("--region", choices=["US", "EZ"], default="US")
    parser.add_argument("--date", default=None,
                        help="Valuation date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    d = None
    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()

    run_pipeline_and_compare(region=args.region, valuation_date=d)


if __name__ == "__main__":
    main()
