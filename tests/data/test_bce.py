"""Tests for the ECB yield-curve client.

Offline coverage: series-key composition, path layout, validation,
CSV parsing with an inline fixture. The network test pulls one
(curve, tenor) for one month against `data-api.ecb.europa.eu`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mibel_derivatives.data import bce


def test_curated_set_is_2x6() -> None:
    assert bce.CURATED_CURVES == ("aaa", "all")
    assert bce.CURATED_TENORS == ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y")
    assert len(bce.CURATED_CURVES) * len(bce.CURATED_TENORS) == 12


def test_series_key_aaa_10y() -> None:
    assert bce.series_key("aaa", "10Y") == "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"


def test_series_key_all_1y() -> None:
    assert bce.series_key("all", "1Y") == "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_1Y"


def test_series_key_validates_curve_and_tenor() -> None:
    with pytest.raises(ValueError, match="curve"):
        bce.series_key("BBB", "10Y")
    with pytest.raises(ValueError, match="tenor"):
        bce.series_key("aaa", "30Y")


def test_raw_path_layout(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = bce.raw_path("aaa", "10Y")
    rel = p.relative_to(tmp_path).as_posix()
    assert rel == "bce/curve=aaa/10Y.csv"


def test_parse_series_with_fixture(tmp_path) -> None:
    # Two-row CSV mirroring the ECB Data Portal csvdata schema.
    csv_text = (
        "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
        "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2025-01-02,2.4331074802,A\n"
        "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2025-01-03,2.4502345600,A\n"
    )
    p = tmp_path / "10Y.csv"
    p.write_text(csv_text, encoding="utf-8")
    df = bce.parse_series(p, "aaa", "10Y")

    assert list(df.columns) == ["date", "curve", "tenor", "rate_pct"]
    assert len(df) == 2
    assert (df["curve"] == "aaa").all()
    assert (df["tenor"] == "10Y").all()
    assert df.iloc[0]["rate_pct"] == pytest.approx(2.4331074802)
    assert df.iloc[1]["rate_pct"] == pytest.approx(2.4502345600)


def test_parse_series_drops_missing_observations(tmp_path) -> None:
    csv_text = (
        "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
        "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2025-01-02,,M\n"
        "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2025-01-03,2.4502,A\n"
    )
    p = tmp_path / "10Y.csv"
    p.write_text(csv_text, encoding="utf-8")
    df = bce.parse_series(p, "aaa", "10Y")
    assert len(df) == 1
    assert df.iloc[0]["rate_pct"] == pytest.approx(2.4502)


def test_build_curated_skips_missing(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_paths, "CURATED_DIR", tmp_path / "curated")
    df = bce.build_curated(write=False)
    assert df.empty
    assert list(df.columns) == ["date", "curve", "tenor", "rate_pct"]


@pytest.mark.network
def test_smoke_fetch_one_series(tmp_path, monkeypatch) -> None:
    """Live: pull AAA 10Y for one month from the ECB Data Portal."""
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = bce.fetch_series(
        "aaa", "10Y",
        start_period="2025-01-01", end_period="2025-01-31", force=True,
    )
    assert Path(p).exists()
    df = bce.parse_series(p, "aaa", "10Y")
    # ECB publishes on TARGET2 business days — ~20 obs in January.
    assert 15 <= len(df) <= 23
    # Reasonable rate range (yields in % — early 2025 AAA 10Y is around 2.2-2.7).
    assert df["rate_pct"].between(-1.0, 10.0).all()
    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "bce" in manifest and "SR_10Y" in manifest
