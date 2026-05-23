"""EU ETS EUA primary-auction price scraper (EEX).

Free public source: EEX Group publishes annual reports of every EUA /
EUAA primary-market auction at

    https://public.eex-group.com/eex/eua-auction-report/
        emission-spot-primary-market-auction-report-<YYYY>-data.<ext>

with `<ext>` = `xls` for 2017-2019, `xlsx` for 2020+.

These are the EU member states' primary auctions for EU Allowances —
the regulated price-discovery channel mandated by the ETS Directive.
The clearing price of each auction is what we want for the carbon-cost
leg of the CCGT tolling and OMIE price-stack models.

This module does NOT scrape the secondary market (ICE EUA Futures or
EEX EUA Futures); those are paywalled. Primary auction prices closely
track the secondary front-month future during liquid hours, so they
are an acceptable free proxy. The diagnostic doc compares the spread
against a reference period.

Each XLSX has one sheet ("Primary Market Auction") with a header band
of metadata rows; the column headers live on row 6 (1-indexed) and
data starts on row 7. Two contract codes appear:

- `T3PA` — Phase 3 standard EU Allowance (EUA)
- `EAA3` — Aviation EU Allowance (EUAA)

For carbon-cost work we keep T3PA only and require `status == 'successful'`.

Pipeline:
- `fetch_year(year)` writes the annual workbook under
  `data/raw/eua/<year>.<ext>`, idempotent.
- `parse_year(path)` returns a typed long-format DataFrame, one row
  per successful EUA auction.
- `build_curated(start_year, end_year)` concatenates years into
  `data/curated/eua_primary_auction.parquet`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

logger = logging.getLogger(__name__)

EEX_AUCTION_BASE_URL = "https://public.eex-group.com/eex/eua-auction-report"

CONTRACT_EUA = "T3PA"   # Phase 3+ standard EU Allowance
CONTRACT_EUAA = "EAA3"  # Aviation EU Allowance (out of scope here)

# Year .xls / .xlsx boundary (EEX migrated in 2020).
_XLSX_FROM_YEAR = 2020

# EEX header band lives at row 6 (1-indexed). pandas `header=5` reads that row
# as the column header and starts data at row 7.
_HEADER_ROW = 5
_SHEET_NAME = "Primary Market Auction"


# ---- URL / paths -----------------------------------------------------------


def _ext_for_year(year: int) -> str:
    return "xlsx" if year >= _XLSX_FROM_YEAR else "xls"


def annual_url(year: int) -> str:
    ext = _ext_for_year(year)
    return f"{EEX_AUCTION_BASE_URL}/emission-spot-primary-market-auction-report-{year}-data.{ext}"


def raw_path(year: int) -> Path:
    return _paths.raw_path("eua", filename=f"{year}.{_ext_for_year(year)}")


# ---- HTTP layer ------------------------------------------------------------


def fetch_year(
    year: int,
    *,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Download the annual EEX auction report. Idempotent on disk."""
    out = raw_path(year)
    if out.exists() and not force:
        logger.debug("EUA raw cached: %s", out)
        return out

    sess = session or ThrottledSession(min_interval_seconds=1.5)
    url = annual_url(year)
    resp = sess.get(url)
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="eua",
            url=url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={"year": year},
        )
    )
    logger.info("EUA: fetched year=%d (%d bytes)", year, len(resp.content))
    return out


# ---- Parsing ---------------------------------------------------------------


def _normalize_header(c: object) -> str:
    """Replace U+FFFD (euro sign mojibake) and collapse whitespace."""
    s = str(c).replace("�", "€").replace("\n", " ")
    return " ".join(s.split())


def _find_column(columns: list[str], substr: str) -> str:
    """Return the first column whose normalized header contains substr (case-insensitive)."""
    needle = substr.lower()
    for c in columns:
        if needle in c.lower():
            return c
    raise KeyError(f"Header {substr!r} not found in {columns!r}")


def parse_year(path: Path) -> pd.DataFrame:
    """Parse one EEX annual workbook to a typed long-format DataFrame.

    Returns one row per successful EUA primary auction with:
    auction_date, contract, status, clearing_price_eur_t, volume_t,
    cover_ratio, country.
    """
    raw = pd.read_excel(path, sheet_name=_SHEET_NAME, header=_HEADER_ROW)
    raw.columns = [_normalize_header(c) for c in raw.columns]
    cols = list(raw.columns)

    # EEX added the "Status" column in 2020; pre-2020 reports only list
    # successful auctions (failed/cancelled ones are excluded from the file).
    try:
        status_col = _find_column(cols, "Status")
        status_series = raw[status_col].astype("string")
    except KeyError:
        status_series = pd.Series(["successful"] * len(raw), dtype="string")

    out = pd.DataFrame({
        "auction_date": pd.to_datetime(
            raw[_find_column(cols, "Date")], errors="coerce"
        ).dt.date,
        "contract": raw[_find_column(cols, "Contract")].astype("string"),
        "status": status_series,
        "clearing_price_eur_t": pd.to_numeric(
            raw[_find_column(cols, "Auction Price")], errors="coerce"
        ),
        "volume_t": pd.to_numeric(
            raw[_find_column(cols, "Auction Volume")], errors="coerce"
        ),
        "cover_ratio": pd.to_numeric(
            raw[_find_column(cols, "Cover Ratio")], errors="coerce"
        ),
        "country": raw[_find_column(cols, "Country")].astype("string"),
    })
    out = out.dropna(subset=["auction_date", "clearing_price_eur_t"])
    out = out[out["status"] == "successful"]
    out = out[out["contract"] == CONTRACT_EUA]
    return out.sort_values("auction_date").reset_index(drop=True)


# ---- Curated build ---------------------------------------------------------


def build_curated(
    *,
    start_year: int,
    end_year: int,
    write: bool = True,
) -> pd.DataFrame:
    """Concatenate parsed annual reports across [start_year, end_year]."""
    rows: list[pd.DataFrame] = []
    for y in range(start_year, end_year + 1):
        p = raw_path(y)
        if not p.exists():
            continue
        df = parse_year(p)
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=[
            "auction_date", "contract", "status",
            "clearing_price_eur_t", "volume_t", "cover_ratio", "country",
        ])
    panel = pd.concat(rows, ignore_index=True)
    if write:
        out = _paths.curated_path("eua_primary_auction.parquet")
        panel.to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(panel))
    return panel


# ---- Smoke helper ----------------------------------------------------------


def smoke_download(
    years: tuple[int, ...] = (2019, 2024),
    *,
    session: ThrottledSession | None = None,
) -> list[Path]:
    """Pull two sample years spanning the .xls / .xlsx format boundary."""
    sess = session or ThrottledSession(min_interval_seconds=1.5)
    return [fetch_year(y, session=sess, force=True) for y in years]


__all__ = [
    "EEX_AUCTION_BASE_URL",
    "CONTRACT_EUA", "CONTRACT_EUAA",
    "annual_url", "raw_path",
    "fetch_year", "parse_year",
    "build_curated", "smoke_download",
]
