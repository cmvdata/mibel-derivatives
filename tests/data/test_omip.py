"""Tests for the OMIP scraper.

Offline tests run always; they exercise the parser against a fixture
that mirrors the real omip.pt table structure (20-cell rows, packed
contract-metadata cell, US-style decimal dot). The `network`-marked
test pulls one real page and asserts the parser produces a
non-trivial DataFrame; it is excluded from CI's fast path.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mibel_derivatives.data import omip

# 20-cell row layout matching the real OMIP HTML table. Two contracts.
_FIXTURE_HTML = """
<html><body>
<table>
<tr>
  <td></td><td></td><td>Session Info</td><td></td><td></td><td></td>
  <td>Last Deal</td><td></td><td></td><td></td>
  <td>End of day info</td><td></td><td></td><td></td>
  <td>Reference prices</td><td></td><td></td><td></td><td></td><td></td>
</tr>
<tr>
  <td>Contract name</td><td></td>
  <td>Best bid</td><td>Best Ask</td><td>Volume (MWh)</td><td></td>
  <td>Price</td><td>Time</td><td>Volume (MWh)</td><td></td>
  <td>Open Interest</td><td>Nr of Contracts</td><td>OTC volume (MWh)</td><td></td>
  <td>D</td><td>D-1</td>
  <td></td><td></td><td></td><td></td>
</tr>
<tr>
  <td>ISIN Code:PTFTO0343908Nominal Fixo MWH:744Trading last day:2024-12-31Trading quotation:EUR/MWhFTB M Jan-25</td><td></td>
  <td>n.a.</td><td>n.a.</td><td>0</td><td></td>
  <td>n.a.</td><td>n.a.</td><td>n.a.</td><td></td>
  <td>378</td><td>0</td><td>0</td><td></td>
  <td>97.50</td><td>103.25</td>
  <td></td><td></td><td></td><td></td>
</tr>
<tr>
  <td>ISIN Code:PTFTO0343916Nominal Fixo MWH:672Trading last day:2025-01-31Trading quotation:EUR/MWhFTB M Feb-25</td><td></td>
  <td>n.a.</td><td>n.a.</td><td>672</td><td></td>
  <td>85.25</td><td>11h:27m:12s</td><td>672</td><td></td>
  <td>381</td><td>1</td><td>0</td><td></td>
  <td>84.00</td><td>87.00</td>
  <td></td><td></td><td></td><td></td>
</tr>
</table>
</body></html>
"""


def test_page_url_shape() -> None:
    url = omip.page_url(dt.date(2024, 12, 30), "M")
    assert "date=2024-12-30" in url
    assert "instrument=FTB" in url
    assert "maturity=M" in url


def test_page_url_rejects_bad_maturity() -> None:
    with pytest.raises(ValueError):
        omip.page_url(dt.date(2024, 1, 2), "Q")


def test_parse_page_returns_typed_rows() -> None:
    df = omip.parse_page(_FIXTURE_HTML, dt.date(2024, 12, 30), "M")
    assert len(df) == 2
    assert list(df["contract"]) == ["FTB M Jan-25", "FTB M Feb-25"]
    assert list(df["isin"]) == ["PTFTO0343908", "PTFTO0343916"]
    assert df["reference_d_eur_mwh"].iloc[0] == pytest.approx(97.50)
    assert df["reference_d_minus_1_eur_mwh"].iloc[1] == pytest.approx(87.00)
    assert pd.isna(df["last_deal_price_eur_mwh"].iloc[0])
    assert df["last_deal_price_eur_mwh"].iloc[1] == pytest.approx(85.25)
    assert int(df["nominal_mwh"].iloc[0]) == 744
    assert int(df["session_volume_mwh"].iloc[1]) == 672
    assert df["last_trading_day"].iloc[0] == dt.date(2024, 12, 31)
    assert df["trade_date"].iloc[0] == dt.date(2024, 12, 30)
    assert df["maturity"].iloc[0] == "M"


def test_parse_page_empty_html() -> None:
    df = omip.parse_page("<html><body><p>no data</p></body></html>",
                         dt.date(2024, 1, 2), "M")
    assert df.empty


def test_split_contract_meta_unpacks_four_fields() -> None:
    text = (
        "ISIN Code:PTFTO0343940Nominal Fixo MWH:744Trading last day:2025-04-30"
        "Trading quotation:EUR/MWhFTB M May-25"
    )
    meta = omip._split_contract_meta(text)
    assert meta["isin"] == "PTFTO0343940"
    assert meta["nominal"] == "744"
    assert meta["last_trading_day"] == "2025-04-30"
    assert meta["contract"] == "FTB M May-25"


def test_trading_days_skips_weekends_and_holidays() -> None:
    days = omip.trading_days(dt.date(2024, 12, 23), dt.date(2024, 12, 27))
    assert dt.date(2024, 12, 25) not in days  # Christmas, PT+ES national
    assert dt.date(2024, 12, 28) not in days  # Saturday
    assert all(d.weekday() < 5 for d in days)


@pytest.mark.network
def test_fetch_and_parse_one_real_page(tmp_path, monkeypatch) -> None:
    """One real OMIP fetch end-to-end. Skipped in CI fast path."""
    from mibel_derivatives.data import _paths, _provenance

    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", tmp_path / "_manifest.jsonl")

    target = dt.date(2024, 12, 30)
    p = omip.fetch_page(target, "M")
    assert p.exists() and p.stat().st_size > 1024

    df = omip.parse_page(p.read_bytes(), target, "M")
    assert not df.empty
    assert (df["contract"].str.startswith("FTB M")).all()
    assert df["reference_d_eur_mwh"].notna().any()
    assert df["trade_date"].iloc[0] == target

    manifest = (tmp_path / "_manifest.jsonl").read_text(encoding="utf-8").strip()
    assert "omip" in manifest and target.isoformat() in manifest
