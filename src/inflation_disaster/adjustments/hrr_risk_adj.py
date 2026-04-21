"""Q -> P risk adjustment via Epstein-Zin utility + Pareto consumption disaster.

Derived from first principles per HRR Sec. 4.4 (main paper) and Appendix B.

Framework:
  Conditional on an inflation disaster, let z = 1/(1 - beta*d) be the inverse
  consumption drop. HRR fit a Pareto distribution to z on Barro (2006) output
  + Jorda-Schularick-Taylor (2016) inflation, baseline method C2.T3:

     F(z) = 1 - (z/z0)^(-alpha), z >= z0 > 1, alpha > 0

  Under Epstein-Zin utility with RRA = gamma, the marginal utility ratio in
  consumption disaster vs normal times is E[z^gamma], a Pareto moment:

     m_tilde = E[z^gamma] = alpha * z0^gamma / (alpha - gamma)     (requires alpha > gamma)

  The risk-neutral probability of an inflation disaster (Q) relates to the
  physical probability (P) via HRR eq. (3):

     q = [(m_tilde - 1) * p_tilde + 1] * p

  where p_tilde is the probability of a consumption disaster conditional on
  an inflation disaster. Therefore:

     P = Q / [(m_tilde - 1) * p_tilde + 1]

HRR's baseline (C2.T3) calibration (Internet Appendix Table 2):
  High-inflation tail: alpha_h = 5.45, z0_h = 1.03, p_tilde_h = 0.356
  Deflation tail:      alpha_l = 15.18, z0_l = 1.06, p_tilde_l = 0.085
  Epstein-Zin RRA = 3 (standard literature value per Gabaix 2012, Barro-Liao 2021)

Numerical verification:
  m_tilde_h = 5.45 * 1.03^3 / 2.45 = 2.430 -> mult_h = 1/(1.430*0.356 + 1) = 0.663
  m_tilde_l = 15.18 * 1.06^3 / 12.18 = 1.484 -> mult_l = 1/(0.484*0.085 + 1) = 0.960

These match the ~0.66 / ~0.96 multipliers reported in HRR Table 1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoDisaster:
    """Pareto-tailed consumption disaster parameters."""
    alpha: float       # tail exponent
    z0: float          # minimum of inverse consumption drop
    p_tilde: float     # P(consumption disaster | inflation disaster)
    label: str = ""    # e.g. "high", "low"


# HRR 2024/25 baseline (C2.T3) calibration — Appendix B Table 2
HRR_HIGH_DISASTER = ParetoDisaster(alpha=5.45, z0=1.03, p_tilde=0.356, label="high")
HRR_LOW_DISASTER = ParetoDisaster(alpha=15.18, z0=1.06, p_tilde=0.085, label="low")
HRR_RRA = 3.0  # Epstein-Zin coefficient of relative risk aversion


def pareto_m_tilde(pareto: ParetoDisaster, rra: float = HRR_RRA) -> float:
    """Expected marginal utility ratio m_tilde = E[z^RRA] for Pareto tail.

    For z ~ Pareto(alpha, z0) with F(z) = 1 - (z/z0)^(-alpha):
        E[z^k] = alpha * z0^k / (alpha - k)   if alpha > k, else diverges
    """
    if pareto.alpha <= rra:
        raise ValueError(
            f"Pareto alpha={pareto.alpha} must exceed RRA={rra} for finite moment"
        )
    return pareto.alpha * (pareto.z0 ** rra) / (pareto.alpha - rra)


def q_to_p_multiplier(pareto: ParetoDisaster, rra: float = HRR_RRA) -> float:
    """P/Q risk-adjustment multiplier.

    Returns
    -------
    mult : float
        P(disaster) = mult * Q(disaster)
    """
    m_tilde = pareto_m_tilde(pareto, rra=rra)
    return 1.0 / ((m_tilde - 1.0) * pareto.p_tilde + 1.0)


def apply_risk_adjustment(
    q_high: float,
    q_low: float,
    high: ParetoDisaster = HRR_HIGH_DISASTER,
    low: ParetoDisaster = HRR_LOW_DISASTER,
    rra: float = HRR_RRA,
) -> tuple[float, float]:
    """Apply Q -> P risk adjustment to a pair of tail probabilities.

    Parameters
    ----------
    q_high : Q-measure probability of high-inflation disaster.
    q_low  : Q-measure probability of deflation disaster.
    high, low : Pareto tail parameters for each disaster type.
    rra : Epstein-Zin RRA coefficient.

    Returns
    -------
    (p_high, p_low) : physical-measure probabilities.
    """
    return q_high * q_to_p_multiplier(high, rra=rra), q_low * q_to_p_multiplier(low, rra=rra)


if __name__ == "__main__":
    # Self-verification: print multipliers and match against paper
    print("HRR risk adjustment — first-principles derivation")
    print("=" * 60)
    for d in (HRR_HIGH_DISASTER, HRR_LOW_DISASTER):
        m = pareto_m_tilde(d)
        mult = q_to_p_multiplier(d)
        print(
            f"  {d.label:<5s} tail: alpha={d.alpha}, z0={d.z0}, "
            f"p_tilde={d.p_tilde} -> m_tilde={m:.4f}, P/Q mult={mult:.4f}"
        )
    print(f"  (RRA={HRR_RRA})")
    print("\nExpected: high~0.66, low~0.96 (HRR Table 1).")
