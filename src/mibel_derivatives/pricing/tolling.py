"""CCGT tolling pricer -- compatibility shim.

The tolling-agreement implementation lives in
:mod:`mibel_derivatives.products.tolling` (Pieza 4). This module was the
original scaffolding placeholder (``raise NotImplementedError``); it now
re-exports the real pricer so the historical ``pricing.tolling`` import
path keeps working.

New code should import from :mod:`mibel_derivatives.products.tolling`.
"""

from __future__ import annotations

from mibel_derivatives.products.tolling import (
    DEFAULT_N_PATHS_DEV,
    DEFAULT_N_PATHS_REPORT,
    GJ_PER_MWH,
    AssetParameters,
    DispatchResult,
    TollingAgreement,
    TollingResult,
    heat_rate_gj_per_mwh,
    hourly_gross_margin,
    optimise_dispatch,
    price_tolling,
    spark_spread_per_mwh,
)

__all__ = [
    "DEFAULT_N_PATHS_DEV",
    "DEFAULT_N_PATHS_REPORT",
    "GJ_PER_MWH",
    "AssetParameters",
    "DispatchResult",
    "TollingAgreement",
    "TollingResult",
    "heat_rate_gj_per_mwh",
    "hourly_gross_margin",
    "optimise_dispatch",
    "price_tolling",
    "spark_spread_per_mwh",
]
