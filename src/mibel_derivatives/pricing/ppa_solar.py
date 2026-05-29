"""Solar PPA pricer with capture-price decomposition.

Specification (CONTEXT.md § PPA solar):

Reference plant: 100 MW in Andalucía, hourly profile simulated with an
annual seasonal component plus stochastic variability.

Structure (v1): 80% of expected generation sold at a fixed price; the
remaining 20% settled at spot. No caps or floors in v1.

Capture price hour by hour is the ratio between realised revenue
(generation x hourly spot) and a baseline revenue (generation x mean
spot). The implicit swap is priced as the present value of the cash
flows of the fixed-leg minus the spot-leg.

Cannibalisation sensitivity is computed by shocking the assumed
solar-price correlation (negative) and re-pricing.
"""

from __future__ import annotations


def price_ppa_solar(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the solar PPA pricer."""
    raise NotImplementedError("price_ppa_solar not implemented yet")
