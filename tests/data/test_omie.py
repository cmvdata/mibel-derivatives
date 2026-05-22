"""Tests for the OMIE loader (no network — reads the mibel-forecasting cache)."""

from __future__ import annotations

import datetime as dt

import pytest

from mibel_derivatives.data import omie


@pytest.mark.skipif(
    not omie.FORECASTING_CACHE_DIR.exists(),
    reason="mibel-forecasting cache not present on this machine",
)
def test_load_one_month_from_forecasting_cache() -> None:
    df = omie.load_month(2024, 1, geo=omie.GEO_ES)
    # Expect ~24 * 31 = 744 hourly rows for Spain in January 2024.
    assert 720 <= len(df) <= 768  # tolerate DST adjustment
    assert df["price_eur_mwh"].notna().any()
    # Column contract:
    assert {"dt_utc", "price_eur_mwh", "year", "month", "geo"}.issubset(df.columns)
    # UTC timestamps:
    assert all(ts.tzinfo is not None for ts in df["dt_utc"].iloc[:5])


@pytest.mark.skipif(
    not omie.FORECASTING_CACHE_DIR.exists(),
    reason="mibel-forecasting cache not present on this machine",
)
def test_build_curated_smoke_one_year(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "CURATED_DIR", tmp_path)
    df = omie.build_curated(2024, 2024, geos=(omie.GEO_ES,))
    # Roughly one year of hourly Spain data.
    assert 8700 <= len(df) <= 8800
    assert df["dt_utc"].min().year == 2024
    assert df["dt_utc"].max().year == 2024
    out = tmp_path / "omie_spot.parquet"
    assert out.exists()


def test_load_missing_month_returns_empty(tmp_path, monkeypatch) -> None:
    """Asking for an uncached month returns an empty frame with the right schema."""
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    # Pick a far-future month that no cache should have
    df = omie.load_month(2099, 12, geo=omie.GEO_ES)
    assert df.empty
    assert {"dt_utc", "price_eur_mwh"}.issubset(df.columns)


def test_build_curated_warns_on_gaps(caplog) -> None:
    """The 2019 months are not in the forecasting cache — function should warn."""
    import logging
    caplog.set_level(logging.WARNING)
    omie.build_curated(2019, 2019, geos=(omie.GEO_ES,), write=False)
    assert any("missing months" in rec.message for rec in caplog.records)
