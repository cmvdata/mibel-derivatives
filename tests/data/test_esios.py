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


def test_resolve_token_raises_without_env(monkeypatch) -> None:
    monkeypatch.delenv("ESIOS_API_TOKEN", raising=False)
    with pytest.raises(esios.ESIOSConfigError):
        esios._resolve_token()


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
