"""Tests for the EUA primary-auction (EEX) scraper.

Offline coverage: URL composition, raw-path layout, format-boundary
between .xls / .xlsx, header normalization. The network tests pull
two sample years spanning the format boundary (2019 .xls and 2024
.xlsx) from public.eex-group.com.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mibel_derivatives.data import eua


def test_xls_boundary_2019_vs_2020() -> None:
    assert eua.annual_url(2019).endswith(".xls")
    assert eua.annual_url(2020).endswith(".xlsx")
    assert eua.annual_url(2024).endswith(".xlsx")


def test_annual_url_pattern() -> None:
    assert eua.annual_url(2024) == (
        "https://public.eex-group.com/eex/eua-auction-report/"
        "emission-spot-primary-market-auction-report-2024-data.xlsx"
    )


def test_raw_path_extension(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p_xls = eua.raw_path(2019)
    p_xlsx = eua.raw_path(2024)
    assert p_xls.relative_to(tmp_path).as_posix() == "eua/2019.xls"
    assert p_xlsx.relative_to(tmp_path).as_posix() == "eua/2024.xlsx"


def test_normalize_header_replaces_mojibake() -> None:
    assert eua._normalize_header("Auction Price �/tCO2") == "Auction Price €/tCO2"
    assert eua._normalize_header("Austria\n(AT)") == "Austria (AT)"


def test_find_column_substring() -> None:
    cols = ["Date", "Contract", "Auction Price €/tCO2", "Country"]
    assert eua._find_column(cols, "Auction Price") == "Auction Price €/tCO2"
    with pytest.raises(KeyError):
        eua._find_column(cols, "Settlement Price")


def test_build_curated_returns_empty_when_no_files(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_paths, "CURATED_DIR", tmp_path / "curated")
    df = eua.build_curated(start_year=2024, end_year=2024, write=False)
    assert df.empty
    assert "clearing_price_eur_t" in df.columns


@pytest.mark.network
def test_smoke_fetch_and_parse_2024_xlsx(tmp_path, monkeypatch) -> None:
    """Live: pull 2024 XLSX and validate the parsed schema + price range."""
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = eua.fetch_year(2024, force=True)
    assert Path(p).exists() and p.stat().st_size > 10_000
    df = eua.parse_year(p)
    # EU primary auctions run ~3x/week minus August. T3PA-only filter keeps
    # most rows; expect roughly 130-200 successful EUA auctions in 2024.
    assert 120 <= len(df) <= 220
    # Schema
    assert {
        "auction_date", "contract", "status",
        "clearing_price_eur_t", "volume_t", "cover_ratio", "country",
    }.issubset(df.columns)
    assert (df["contract"] == eua.CONTRACT_EUA).all()
    assert (df["status"] == "successful").all()
    # 2024 EUA prices roughly 50-90 €/t — well inside a wide sanity band.
    assert df["clearing_price_eur_t"].between(20, 200).all()

    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "eua" in manifest


@pytest.mark.network
def test_smoke_fetch_and_parse_2019_xls(tmp_path, monkeypatch) -> None:
    """Live: 2019 .xls round-trips via xlrd. Validates the legacy path."""
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = eua.fetch_year(2019, force=True)
    assert Path(p).exists() and str(p).endswith(".xls")
    df = eua.parse_year(p)
    # 2019 had ~140 successful T3PA auctions.
    assert 100 <= len(df) <= 220
    # 2019 EUA averaged ~25 €/t — wider band to allow swings.
    assert df["clearing_price_eur_t"].between(5, 50).all()
