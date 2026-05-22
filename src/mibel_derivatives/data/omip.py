"""OMIP forward-curve scraper for the SPEL Base Futures (FTB) zone=ES strip.

URL pattern (public, parameterised — see reports/diagnostics/omip_endpoint.md):

    https://www.omip.pt/en/dados-mercado
        ?date=YYYY-MM-DD
        &product=EL
        &zone=ES
        &instrument=FTB
        &maturity={M|YR}

The real OMIP page renders one HTML table whose data rows have 20 cells.
Cells 0, 2-4, 6-8, 10-12, 14-15 carry the values; the rest are visual
separators. Cell 0 packs four pieces of metadata in one string:

    ISIN Code:PTFTO0343908Nominal Fixo MWH:744Trading last day:2024-12-31
    Trading quotation:€/MWhFTB M Jan-25

We split that with a regex. Numbers use US-style dots ("97.50"), not the
European comma seen in the synthetic fixtures of the Manus survey.

Pipeline:
- fetch_page(trade_date, maturity, force=False) — writes raw HTML.
- parse_page(html, trade_date, maturity) — returns a typed DataFrame.
- build_curated(start, end) — concatenates all raw pages in the range.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

logger = logging.getLogger(__name__)

OMIP_BASE_URL = "https://www.omip.pt/en/dados-mercado"
PRODUCT = "EL"
ZONE = "ES"
INSTRUMENT = "FTB"
MATURITIES: tuple[str, ...] = ("M", "YR")

# Column indices in the 20-cell data rows of the OMIP table.
_COL_CONTRACT_META = 0
_COL_BEST_BID = 2
_COL_BEST_ASK = 3
_COL_SESSION_VOLUME = 4
_COL_LAST_DEAL_PRICE = 6
_COL_LAST_DEAL_TIME = 7
_COL_LAST_DEAL_VOLUME = 8
_COL_OPEN_INTEREST = 10
_COL_NR_CONTRACTS = 11
_COL_OTC_VOLUME = 12
_COL_REFERENCE_D = 14
_COL_REFERENCE_D_MINUS_1 = 15

# Pieces packed into the contract-name cell.
_CONTRACT_RX = re.compile(
    r"ISIN Code:(?P<isin>\S+?)"
    r"Nominal Fixo MWH:(?P<nominal>\d+)"
    r"Trading last day:(?P<last_trading_day>\d{4}-\d{2}-\d{2})"
    r"Trading quotation:(?P<quotation>[^F]+?)"
    r"(?P<contract>FTB[^\n]+)$",
    re.UNICODE,
)


# ---- HTTP layer ------------------------------------------------------------


def page_url(trade_date: dt.date, maturity: str) -> str:
    if maturity not in MATURITIES:
        raise ValueError(f"maturity must be in {MATURITIES}, got {maturity!r}")
    params = (
        f"date={trade_date.isoformat()}"
        f"&product={PRODUCT}&zone={ZONE}&instrument={INSTRUMENT}&maturity={maturity}"
    )
    return f"{OMIP_BASE_URL}?{params}"


def raw_path(trade_date: dt.date, maturity: str) -> Path:
    return _paths.raw_path(
        "omip",
        f"maturity={maturity}",
        f"trade_date={trade_date.isoformat()}",
        filename="page.html",
    )


def fetch_page(
    trade_date: dt.date,
    maturity: str,
    *,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Download the OMIP page for a (trade_date, maturity) and write raw HTML.

    Idempotent: returns the existing file path if present and `force=False`.
    """
    out = raw_path(trade_date, maturity)
    if out.exists() and not force:
        logger.debug("OMIP raw cached: %s", out)
        return out

    sess = session or ThrottledSession()
    url = page_url(trade_date, maturity)
    resp = sess.get(url)
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="omip",
            url=url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={"trade_date": trade_date.isoformat(), "maturity": maturity},
        )
    )
    return out


# ---- Parsing ---------------------------------------------------------------


def _parse_number(text: str) -> float | None:
    """Parse "97.50" / "1,234.56" / "n.a." / "" to float or None."""
    t = (text or "").strip()
    if not t or t.lower() in {"n.a.", "n/a", "-"}:
        return None
    # Accept either US (1,234.56) or EU (1.234,56) format defensively.
    if "," in t and "." in t:
        # Heuristic: whichever appears last is the decimal separator.
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        # Only comma — assume decimal.
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _parse_int(text: str) -> int | None:
    t = (text or "").strip()
    if not t or t.lower() in {"n.a.", "n/a", "-"}:
        return None
    t = t.replace(".", "").replace(",", "")
    try:
        return int(t)
    except ValueError:
        return None


def _split_contract_meta(text: str) -> dict[str, str | None]:
    """Decompose the packed first-cell string into ISIN, nominal, last
    trading day, quotation and contract name."""
    cleaned = " ".join((text or "").split())
    m = _CONTRACT_RX.search(cleaned)
    if not m:
        return {
            "isin": None, "nominal": None, "last_trading_day": None,
            "quotation": None, "contract": None,
        }
    return {
        "isin": m.group("isin"),
        "nominal": m.group("nominal"),
        "last_trading_day": m.group("last_trading_day"),
        "quotation": m.group("quotation").strip(),
        "contract": m.group("contract").strip(),
    }


def parse_page(html: str | bytes, trade_date: dt.date, maturity: str) -> pd.DataFrame:
    """Parse OMIP HTML to a typed DataFrame; one row per contract."""
    if isinstance(html, bytes):
        soup = BeautifulSoup(html, "lxml")
    else:
        soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    rows = table.find_all("tr")
    out_rows: list[dict[str, object]] = []
    for tr in rows:
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) < 16:
            # header or separator row
            continue
        meta_cell = cells[_COL_CONTRACT_META]
        if not meta_cell.startswith("ISIN Code"):
            continue
        meta = _split_contract_meta(meta_cell)
        row: dict[str, object] = {
            "trade_date": trade_date,
            "maturity": maturity,
            "contract": meta["contract"],
            "isin": meta["isin"],
            "nominal_mwh": _parse_int(meta["nominal"] or ""),
            "last_trading_day": (
                dt.date.fromisoformat(meta["last_trading_day"])
                if meta["last_trading_day"] else None
            ),
            "quotation": meta["quotation"],
            "best_bid_eur_mwh": _parse_number(cells[_COL_BEST_BID]),
            "best_ask_eur_mwh": _parse_number(cells[_COL_BEST_ASK]),
            "session_volume_mwh": _parse_int(cells[_COL_SESSION_VOLUME]),
            "last_deal_price_eur_mwh": _parse_number(cells[_COL_LAST_DEAL_PRICE]),
            "last_deal_time": cells[_COL_LAST_DEAL_TIME] or None,
            "last_deal_volume_mwh": _parse_int(cells[_COL_LAST_DEAL_VOLUME]),
            "open_interest": _parse_int(cells[_COL_OPEN_INTEREST]),
            "nr_contracts": _parse_int(cells[_COL_NR_CONTRACTS]),
            "otc_volume_mwh": _parse_int(cells[_COL_OTC_VOLUME]),
            "reference_d_eur_mwh": _parse_number(cells[_COL_REFERENCE_D]),
            "reference_d_minus_1_eur_mwh": _parse_number(
                cells[_COL_REFERENCE_D_MINUS_1]
            ),
        }
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    for c in (
        "nominal_mwh", "session_volume_mwh", "last_deal_volume_mwh",
        "open_interest", "nr_contracts", "otc_volume_mwh",
    ):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return df


# ---- Calendar --------------------------------------------------------------


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekday-only calendar minus PT and ES public holidays."""
    import holidays

    pt = holidays.PT(years=range(start.year, end.year + 1))
    es = holidays.ES(years=range(start.year, end.year + 1))
    out: list[dt.date] = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in pt and day not in es:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


# ---- Curated build ---------------------------------------------------------


def build_curated(
    start: dt.date,
    end: dt.date,
    *,
    session: ThrottledSession | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Read all raw OMIP pages in [start, end] for both maturities and curate.

    This does NOT download — call `fetch_page` first (or `download_omip` CLI)
    to populate `data/raw/omip/`. Missing dates are skipped silently.
    """
    rows: list[pd.DataFrame] = []
    for day in trading_days(start, end):
        for maturity in MATURITIES:
            p = raw_path(day, maturity)
            if not p.exists():
                continue
            try:
                df = parse_page(p.read_bytes(), day, maturity)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("Failed to parse %s: %s", p, exc)
                continue
            if not df.empty:
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if write:
        path = _paths.curated_path("omip_forward_curve.parquet")
        out.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(out))
    return out
