"""Tests for the ESIOS client.

The token is read from `.env`; the online test pulls one indicator for
one month. Offline tests cover URL construction and curated config.
"""

from __future__ import annotations

import pytest

from mibel_derivatives.data import esios


def test_curated_indicators_have_unique_ids() -> None:
    ids = [ind.id for ind in esios.CURATED_INDICATORS]
    assert len(ids) == len(set(ids))
    assert 600 in ids  # OMIE day-ahead is part of the curated panel


def test_cache_path_filename_matches_forecasting_convention(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = esios.cache_path(600, 3, 2024, 1)
    assert p.name == "i600_geo3_2024_01.parquet"
    assert "esios" in p.as_posix() and "cache" in p.as_posix()


def test_cache_path_for_systemwide_indicator_uses_geo_all(tmp_path, monkeypatch) -> None:
    """System-wide indicators (geo=None) must not collide with geo=3 files."""
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p_geo3 = esios.cache_path(682, 3, 2024, 1)
    p_none = esios.cache_path(682, None, 2024, 1)
    assert p_geo3.name == "i682_geo3_2024_01.parquet"
    assert p_none.name == "i682_geoall_2024_01.parquet"
    assert p_geo3 != p_none


def test_curated_indicators_have_correct_geo_split() -> None:
    """Only the two OMIE indicators are geo-aware; the rest are system-wide."""
    by_id = {ind.id: ind for ind in esios.CURATED_INDICATORS}
    assert by_id[600].geo == esios.GEO_ES
    assert by_id[612].geo == esios.GEO_ES
    for sid in (634, 682, 683, 676, 677, 686, 687, 793, 794, 10211, 79):
        assert by_id[sid].geo is None, (
            f"indicator {sid} must be system-wide (geo=None); "
            f"got geo={by_id[sid].geo!r}"
        )


def test_resolve_token_raises_without_env(monkeypatch) -> None:
    monkeypatch.delenv("ESIOS_API_TOKEN", raising=False)
    with pytest.raises(esios.ESIOSConfigError):
        esios._resolve_token()


def test_pull_indicator_with_empty_cached_month_does_not_crash(
    tmp_path, monkeypatch,
) -> None:
    """Regression: an empty (RangeIndex) cached parquet must not break
    pull_indicator. Triggered the 2026-05-23 bulk failure where an
    indicator returned no rows for a 2019 month and the cached empty
    series degraded the post-concat index to RangeIndex, causing
    `out.index.tz` to AttributeError."""
    import pandas as pd

    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)

    # Lay down two cache files for indicator 634 / geo 3 / Jan and Feb 2019.
    # Both empty (the API-returned-zero-rows case).
    empty = pd.Series(dtype=float, name="value")  # RangeIndex on purpose
    for month in (1, 2):
        p = esios.cache_path(634, esios.GEO_ES, 2019, month)
        empty.to_frame("value").to_parquet(p)

    s = esios.pull_indicator(
        634, esios.GEO_ES, start="2019-01-01", end="2019-02-28",
    )
    assert s.empty
    assert isinstance(s.index, pd.DatetimeIndex)
    assert str(s.index.tz) == "UTC"


@pytest.mark.network
def test_lookup_indicator_600() -> None:
    meta = esios.lookup_indicator(600)
    assert meta.get("id") == 600
    assert "SPOT" in (meta.get("name") or "").upper()


@pytest.mark.network
def test_pull_one_month(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    # Pull a small window: indicator 600 Spain, January 2024.
    s = esios.pull_indicator(600, esios.GEO_ES,
                             start="2024-01-01", end="2024-01-31")
    assert len(s) >= 24 * 30  # tolerate DST + month length
    assert s.notna().any()
    # Cache file written:
    p = tmp_path / "raw" / "esios" / "cache" / "i600_geo3_2024_01.parquet"
    assert p.exists()
    # Manifest line written
    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "esios" in manifest and "600" in manifest
