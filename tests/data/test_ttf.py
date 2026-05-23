"""Tests for the TTF (Yahoo v8 chart) prototype scraper.

Offline coverage: path layout, JSON parsing with an inline fixture,
schema and dtype checks. The network test pulls one year for TTF=F.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from mibel_derivatives.data import ttf


def test_raw_path_layout(tmp_path, monkeypatch) -> None:
    from mibel_derivatives.data import _paths

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = ttf.raw_path("TTF=F", dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    rel = p.relative_to(tmp_path).as_posix()
    assert rel == "ttf/symbol=TTF_F/2024-01-01_2024-12-31.json"


def test_parse_window_with_fixture(tmp_path) -> None:
    # Three-day fixture mirroring the Yahoo v8 chart response shape.
    ts = [
        int(time.mktime(dt.date(2024, 1, 2).timetuple())),
        int(time.mktime(dt.date(2024, 1, 3).timetuple())),
        int(time.mktime(dt.date(2024, 1, 4).timetuple())),
    ]
    fixture = {
        "chart": {
            "result": [{
                "meta": {"symbol": "TTF=F", "currency": "EUR"},
                "timestamp": ts,
                "indicators": {"quote": [{
                    "open":   [33.60, 33.10, 32.45],
                    "high":   [34.20, 33.85, 32.90],
                    "low":    [32.95, 32.40, 31.80],
                    "close":  [33.05, 32.60, 32.10],
                    "volume": [12345, 11000, 9870],
                }]},
            }],
            "error": None,
        }
    }
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    df = ttf.parse_window(p)

    assert len(df) == 3
    assert list(df.columns) == [
        "date", "open", "high", "low", "close", "volume",
    ]
    assert df.iloc[0]["close"] == pytest.approx(33.05)
    assert df["volume"].dtype.name == "Int64"


def test_parse_window_drops_rows_with_null_close(tmp_path) -> None:
    ts = [
        int(time.mktime(dt.date(2024, 1, 2).timetuple())),
        int(time.mktime(dt.date(2024, 1, 3).timetuple())),
    ]
    fixture = {
        "chart": {
            "result": [{
                "timestamp": ts,
                "indicators": {"quote": [{
                    "open": [None, 33.10], "high": [None, 33.85],
                    "low":  [None, 32.40], "close": [None, 32.60],
                    "volume": [None, 11000],
                }]},
            }],
            "error": None,
        }
    }
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    df = ttf.parse_window(p)
    assert len(df) == 1
    assert df.iloc[0]["close"] == pytest.approx(32.60)


def test_parse_window_handles_empty_result(tmp_path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"chart": {"result": []}}), encoding="utf-8")
    df = ttf.parse_window(p)
    assert df.empty
    assert "close" in df.columns


@pytest.mark.network
def test_smoke_fetch_one_year_ttf(tmp_path, monkeypatch) -> None:
    """Live: pull TTF=F for 2020 and validate schema + value sanity.

    Yahoo v8 chart is a prototype source — see the diagnostic doc and
    the module docstring for licensing caveats.
    """
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    p = ttf.fetch_window(
        ttf.TTF_SYMBOL, dt.date(2020, 1, 1), dt.date(2020, 12, 31),
        force=True,
    )
    assert Path(p).exists() and p.stat().st_size > 10_000
    df = ttf.parse_window(p)
    # Trading day count: ICE Endex runs ~250 business days/year.
    assert 220 <= len(df) <= 270
    # 2020 TTF averaged ~9.5 €/MWh (covid year), with peaks below 25.
    assert df["close"].between(2.0, 50.0).all()
    # Date range
    assert df["date"].min() >= dt.date(2020, 1, 1)
    assert df["date"].max() <= dt.date(2020, 12, 31)

    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8")
    assert "ttf" in manifest and "TTF" in manifest
