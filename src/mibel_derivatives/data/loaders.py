"""Data loaders for OMIE spot, OMIP forwards, MIBGAS and TTF inputs.

The five master tables described in CONTEXT.md (asset_parameters,
market_prices, dispatch_states, settlement_reconciliation,
compliance_log) are the canonical interface; the loaders here build
in-memory views from on-disk parquet under data/processed/.

Reuse of curated panels from sibling repo `mibel-forecasting` is
expected for OMIE day-ahead prices — the exact handoff is part of
the open implementation questions raised at scaffolding time.
"""

from __future__ import annotations


def load_market_prices(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the market-prices loader."""
    raise NotImplementedError("load_market_prices not implemented yet")
