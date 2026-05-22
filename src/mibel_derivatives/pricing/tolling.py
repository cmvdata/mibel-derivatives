"""Tolling agreement pricer for a CCGT (Castejón I reference plant).

Specification (CONTEXT.md § Tolling agreement sobre Castejón I):

Dispatch optimiser via dynamic programming over operating states
(uncoupled, starting, technical minimum, optimal output, stopping) with
temporal constraints (minimum up-time, minimum down-time). Valuation by
Longstaff–Schwartz on the spark spread (S − HR · G − CO2_cost · ER),
honouring operating constraints.

Asset parameters live in the asset_parameters table — see CONTEXT.md
for the Castejón I reference values and their public sources (Iberdrola
Environmental Declaration 2024, NREL, Aurecon/AEMO benchmarks).
"""

from __future__ import annotations


def price_tolling(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the CCGT tolling pricer."""
    raise NotImplementedError("price_tolling not implemented yet")
