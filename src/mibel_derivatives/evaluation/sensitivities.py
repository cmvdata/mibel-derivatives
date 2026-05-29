"""First-order sensitivities by finite-difference shocks.

Greeks (delta, gamma, vega, theta) are computed by re-pricing under
shocks to the relevant model input:

- delta / gamma: shock the spot (or, for forward-settled exposures,
  the relevant forward maturity bucket).
- vega: shock the diffusion volatility sigma of the spot model; for the
  Schwartz-Smith forward model, separate vegas for the short- and
  long-factor diffusion volatilities.
- theta: re-price advancing the valuation date by one calendar day
  while holding the rest of the state fixed.

The shock sizes and the choice of one-sided vs central differences
are configurable per Greek — see the calling site in each pricer.
"""

from __future__ import annotations


def finite_difference_greeks(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the finite-difference Greek calculator."""
    raise NotImplementedError("finite_difference_greeks not implemented yet")
