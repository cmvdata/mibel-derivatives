"""Tests for the PVGIS hourly PV-output client.

Offline coverage: configuration sanity, path layout, JSON parsing with
an inline fixture, raddatabase validation. The online test
(`network` marker) hits re.jrc.ec.europa.eu and pulls 2023 (the upper
edge of SARAH3 coverage) for one config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mibel_derivatives.data import pvgis


def test_curated_configs_have_unique_labels() -> None:
    labels = [c.label for c in pvgis.CURATED_CONFIGS]
    assert len(labels) == len(set(labels))
    assert {c.label for c in pvgis.CURATED_CONFIGS} == {
        "fixed_35deg_south", "one_axis_horizontal_ns",
    }


def test_fixed_and_tracker_have_expected_geometry() -> None:
    assert pvgis.FIXED_35_SOUTH.trackingtype == 0
    assert pvgis.FIXED_35_SOUTH.angle == 35.0
    assert pvgis.ONE_AXIS_NS.trackingtype == 1
    assert pvgis.ONE_AXIS_NS.angle == 0.0


def test_default_raddatabase_is_sarah3() -> None:
    assert pvgis.DEFAULT_RADDATABASE == pvgis.RADDATABASE_SARAH3
    assert {
        pvgis.RADDATABASE_SARAH3, pvgis.RADDATABASE_ERA5,
    } == pvgis.ALLOWED_RADDATABASES


def test_raw_path_layout_includes_db(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = pvgis.raw_path(37.4, -5.0, 2023, pvgis.FIXED_35_SOUTH)
    rel = p.relative_to(tmp_path).as_posix()
    assert rel == (
        "pvgis/db=PVGIS-SARAH3/config=fixed_35deg_south"
        "/lat=37.4000_lon=-5.0000/2023.json"
    )


def test_raw_path_distinguishes_sarah3_and_era5(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p_sarah = pvgis.raw_path(
        37.4, -5.0, 2023, pvgis.FIXED_35_SOUTH,
        raddatabase=pvgis.RADDATABASE_SARAH3,
    )
    p_era5 = pvgis.raw_path(
        37.4, -5.0, 2023, pvgis.FIXED_35_SOUTH,
        raddatabase=pvgis.RADDATABASE_ERA5,
    )
    assert p_sarah != p_era5


def test_fetch_year_rejects_unknown_raddatabase() -> None:
    with pytest.raises(ValueError, match="raddatabase"):
        pvgis.fetch_year(
            37.4, -5.0, 2023, pvgis.FIXED_35_SOUTH, raddatabase="PVGIS-SARAH2",
        )


def test_parse_year_with_fixture(tmp_path) -> None:
    fixture = {
        "inputs": {"location": {"latitude": 37.4, "longitude": -5.0}},
        "outputs": {
            "hourly": [
                {
                    "time": "20230101:0010",
                    "P": 0.0, "G(i)": 0.0, "H_sun": -10.0,
                    "T2m": 5.7, "WS10m": 2.5, "Int": 0,
                },
                {
                    "time": "20230101:1310",
                    "P": 642.3, "G(i)": 720.1, "H_sun": 28.5,
                    "T2m": 12.4, "WS10m": 3.1, "Int": 0,
                },
            ]
        },
        "meta": {"name": "fixture"},
    }
    p = tmp_path / "2023.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    df = pvgis.parse_year(p, pvgis.FIXED_35_SOUTH)

    assert len(df) == 2
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    # Floor-to-hour: 00:10 -> 00:00; 13:10 -> 13:00
    assert df.index[0] == pd.Timestamp("2023-01-01 00:00", tz="UTC")
    assert df.index[1] == pd.Timestamp("2023-01-01 13:00", tz="UTC")
    # Schema renamed
    assert list(df.columns) == [
        "config", "p_w", "g_i_w_m2", "sun_height_deg",
        "t_air_c", "wind_10m_m_s", "is_reconstructed",
    ]
    assert (df["config"] == "fixed_35deg_south").all()
    assert df.iloc[1]["p_w"] == pytest.approx(642.3)


def test_parse_year_handles_empty_outputs(tmp_path) -> None:
    p = tmp_path / "2023.json"
    p.write_text(json.dumps({"outputs": {"hourly": []}}), encoding="utf-8")
    df = pvgis.parse_year(p, pvgis.FIXED_35_SOUTH)
    assert df.empty


def test_build_curated_skips_missing(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_paths, "CURATED_DIR", tmp_path / "curated")

    df = pvgis.build_curated(
        37.4, -5.0, start_year=2023, end_year=2023, write=False,
    )
    assert df.empty  # no files present


@pytest.mark.network
def test_smoke_download_one_year(tmp_path, monkeypatch) -> None:
    """Live: pull 2023 SARAH3 for the fixed config (one HTTP call).

    2023 is the upper edge of SARAH3 coverage as of 2026-05-22; this
    test also functions as a tripwire if JRC publishes 2024 (the
    coverage_upper assertion below).
    """
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = pvgis.fetch_year(
        pvgis.LAT_ANDALUCIA, pvgis.LON_ANDALUCIA, 2023, pvgis.FIXED_35_SOUTH,
    )
    assert Path(p).exists()
    df = pvgis.parse_year(p, pvgis.FIXED_35_SOUTH)
    # ~8760 rows for a non-leap year; tolerate gaps and DST.
    assert len(df) >= 24 * 360
    assert (df["p_w"] >= 0).all()
    # Sevilla averages ~11.5 h of usable sun/day; 4000 is a safe floor.
    assert (df["p_w"] > 0).sum() >= 4000

    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "pvgis" in manifest and "PVGIS-SARAH3" in manifest


@pytest.mark.network
def test_pvgis_2024_not_yet_available() -> None:
    """Tripwire: when JRC extends SARAH3 past 2023, this test will fail
    and signal that we can extend the bulk range. As of 2026-05-22 the
    upper bound is 2023."""
    import requests

    r = requests.get(
        pvgis.PVGIS_BASE_URL,
        params={
            "lat": "37.4", "lon": "-5.0",
            "startyear": "2024", "endyear": "2024",
            "pvcalculation": "1", "peakpower": "1", "loss": "14",
            "trackingtype": "0", "angle": "35", "aspect": "0",
            "mountingplace": "free",
            "raddatabase": pvgis.RADDATABASE_SARAH3,
            "outputformat": "json",
        },
        timeout=30,
    )
    assert r.status_code == 400
    assert "2023" in r.text  # current upper bound in the error message
