"""Solar PPA pricer — compatibility shim.

The solar-PPA implementation lives in
:mod:`mibel_derivatives.products.ppa` (Pieza 5). This module was the
original scaffolding placeholder (``raise NotImplementedError``); it now
re-exports the real pricer so the historical ``pricing.ppa_solar`` import
path keeps working.

New code should import from :mod:`mibel_derivatives.products.ppa`.
"""

from __future__ import annotations

from mibel_derivatives.products.ppa import (
    PPA,
    PPAResult,
    PriceModel,
    price_ppa,
    representative_year_profile,
    simulate_price_paths,
    simulate_price_paths_from_spot,
    simulate_production_paths,
)


def price_ppa_solar(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Backwards-compatible alias for :func:`products.ppa.price_ppa`."""
    return price_ppa(*args, **kwargs)


__all__ = [
    "PPA",
    "PPAResult",
    "PriceModel",
    "price_ppa",
    "price_ppa_solar",
    "representative_year_profile",
    "simulate_price_paths",
    "simulate_price_paths_from_spot",
    "simulate_production_paths",
]
