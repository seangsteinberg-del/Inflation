"""Put-call parity real rate extraction — HRR Sec. 4.2 (Birru-Figlewski 2012).

Paper text (Sec. 4.2): "We also compare put-call-parity real rates with those
implied by the inflation swap contracts to confirm that prices are not only
consistent within the options market but also across inflation derivative
markets."

Appendix A.2: "We discount using the real interest rate which is extracted
from the put-call parity relationship of the zero coupon options (Birru and
Figlewski, 2012)."

Derivation:
  A ZC inflation cap at strike K, maturity T, pays at T:
      max((1+y_realized)^T - (1+K)^T, 0)
  A ZC inflation floor at K, T pays:
      max((1+K)^T - (1+y_realized)^T, 0)

  Put-call parity:
      cap(K,T) - floor(K,T) = DF_nominal(T) * [ E_Q[(1+y)^T] - (1+K)^T ]

  The forward of the CPI ratio under Q is the inflation swap forward:
      E_Q[(1+y)^T] = (1+swap_rate)^T

  So: cap - floor = DF_nom * [(1+swap)^T - (1+K)^T]

  Alternatively, working in real measure (paper convention), the nominal
  discount factor = e^{-i*T} and real = e^{-r*T} with i - r = swap_rate
  (Fisher). The real rate is backed out by matching the observed
  (cap - floor) spread to (1+swap)^T - (1+K)^T under the nominal discount:

      DF_nom(T) = (cap - floor) / [(1+swap)^T - (1+K)^T]
      i = -ln(DF_nom) / T
      r = i - swap_rate   (Fisher approximation to first order)

  The paper uses this "across strike prices" consistency check — if prices
  are clean, the backed-out real rate should be nearly constant across K.
  Average across strikes where the signal-to-noise is good (K close to
  forward, away from deep OTM on either side).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("inflation_disaster.pricing.put_call_parity")


@dataclass
class PutCallParityRealRate:
    """Real rate extracted from put-call parity across strikes for one T."""
    maturity: int
    swap_rate: float          # inflation swap rate, decimal
    nominal_rate: float       # i = -ln(DF_nom)/T, decimal
    real_rate: float          # r = i - swap, decimal
    per_strike_real_rate: np.ndarray  # real rate at each strike used
    strikes_used: np.ndarray   # decimal
    cross_strike_dispersion: float     # stdev of per-strike real rate


def extract_real_rate(
    strikes: np.ndarray,
    cap_prices: np.ndarray,
    floor_prices: np.ndarray,
    swap_rate: float,
    maturity: int,
    min_abs_spread: float = 1e-6,
    strike_window: tuple[float, float] | None = None,
) -> PutCallParityRealRate:
    """Extract the nominal and real rate from the put-call parity at each (K,T).

    Parameters
    ----------
    strikes : decimal strikes (e.g. 0.02 for 2%).
    cap_prices : cap premium at each strike, decimal notional.
    floor_prices : floor premium at each strike, decimal notional.
    swap_rate : ZC inflation swap rate for maturity T, decimal.
    maturity : T in years.
    min_abs_spread : exclude strikes with |(1+swap)^T - (1+K)^T| < this
                     (parity denominator too small).
    strike_window : (lo, hi) decimal window to restrict averaging to
                    strikes with best data quality. Default: (swap - 0.02,
                    swap + 0.02) which brackets the forward.

    Returns
    -------
    PutCallParityRealRate
    """
    strikes = np.asarray(strikes, dtype=float)
    cap_prices = np.asarray(cap_prices, dtype=float)
    floor_prices = np.asarray(floor_prices, dtype=float)

    F_T = (1.0 + swap_rate) ** maturity
    K_T = (1.0 + strikes) ** maturity
    spread = F_T - K_T
    diff = cap_prices - floor_prices

    valid = np.abs(spread) > min_abs_spread
    if strike_window is None:
        strike_window = (swap_rate - 0.02, swap_rate + 0.02)
    lo, hi = strike_window
    valid &= (strikes >= lo) & (strikes <= hi)

    if not np.any(valid):
        # Fall back: drop the strike window constraint and just use valid spreads
        valid = np.abs(spread) > min_abs_spread
        if not np.any(valid):
            raise ValueError(
                f"No strikes with sufficient parity signal for T={maturity}"
            )

    # DF_nominal per strike
    df_per_k = diff[valid] / spread[valid]
    # Nominal rate per strike: i = -ln(DF)/T (continuous compounding convention
    # paper uses e^{-iT} as nominal discount)
    # Guard against non-positive DF (bad data)
    df_pos = np.where(df_per_k > 0, df_per_k, np.nan)
    nominal_per_k = -np.log(df_pos) / maturity
    # Real rate per strike: Fisher i = r + swap (linearized) => r = i - swap
    real_per_k = nominal_per_k - swap_rate

    nominal_median = float(np.nanmedian(nominal_per_k))
    real_median = float(np.nanmedian(real_per_k))
    dispersion = float(np.nanstd(real_per_k, ddof=0))

    return PutCallParityRealRate(
        maturity=maturity,
        swap_rate=swap_rate,
        nominal_rate=nominal_median,
        real_rate=real_median,
        per_strike_real_rate=real_per_k,
        strikes_used=strikes[valid],
        cross_strike_dispersion=dispersion,
    )


if __name__ == "__main__":
    # Quick self-test with a synthetic example
    import sys
    T = 5
    swap = 0.025
    strikes = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04])
    # Assume true nominal rate = 4% => DF = exp(-0.04*5)
    df_true = np.exp(-0.04 * 5)
    F_T = (1 + swap) ** T
    K_T = (1 + strikes) ** T
    # Construct cap - floor = DF * (F_T - K_T)
    diff = df_true * (F_T - K_T)
    # Split: for K < F, cap is ITM and floor OTM; use typical magnitudes
    cap = np.where(K_T < F_T, diff + 0.05, 0.05)
    floor = cap - diff

    res = extract_real_rate(strikes, cap, floor, swap, T,
                            strike_window=(0.0, 0.05))
    print(f"Synthetic test: true i=4.00%, true r = 4 - 2.5 = 1.50%")
    print(f"  extracted i = {res.nominal_rate*100:.2f}%")
    print(f"  extracted r = {res.real_rate*100:.2f}%")
    print(f"  dispersion  = {res.cross_strike_dispersion*10000:.3f} bp")
