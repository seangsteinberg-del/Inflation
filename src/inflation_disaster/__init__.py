"""Inflation disaster probability estimation from options data.

Implements the methodology of Hilscher, Raviv, and Reis (2024):
"How likely is an inflation disaster?"

Three adjustments to conventional option-implied probabilities:
1. Inflation adjustment: N -> Q (nominal to real risk-neutral)
2. Horizon adjustment: spot -> 5y5y forward via Markov chain
3. Risk adjustment: Q -> P (risk-neutral to physical measure)
"""

__version__ = "0.1.0"
