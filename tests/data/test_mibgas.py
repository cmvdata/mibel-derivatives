"""Tests for the MIBGAS annual XLSX downloader."""

from __future__ import annotations

import datetime as dt

import pytest

from mibel_derivatives.data import mibgas


def test_annual_url_shape() -> None:
    url = mibgas.annual_url(2024)
    assert "MIBGAS_Data_2024.xlsx" in url
    assert "AGNO_2024/XLS" in url


def test_raw_path_partitioned_by_year(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = mibgas.raw_path(2023)
    assert p.name == "MIBGAS_Data_2023.xlsx"
    assert "year=2023" in p.as_posix()


@pytest.mark.network
def test_fetch_and_parse_one_year(tmp_path, monkeypatch) -> None:
    """Live download of MIBGAS_Data_2024.xlsx, parsed end-to-end."""
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = mibgas.fetch_annual(2024)
    assert p.exists() and p.stat().st_size > 50_000

    pvb = mibgas.parse_pvb(p)
    idx = mibgas.parse_indices(p)
    assert not pvb.empty
    assert not idx.empty
    # Spot check a known product code is present
    products = set(pvb["product_code"].dropna().unique())
    assert "GDAES_D+1" in products
    # Spot check date range is in 2024 / nearby
    assert pvb["trade_date"].max() >= dt.date(2024, 12, 1)
    assert pvb["trade_date"].min() <= dt.date(2024, 6, 1)
    # Indices: at least one valid spot index point
    assert idx["mibgas_es_index_eur_mwh"].notna().any()

    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "mibgas" in manifest and "2024" in manifest
