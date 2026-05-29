"""MIBGAS annual XLSX downloader and parser.

Endpoint discovered 2026-05-22:

    https://www.mibgas.es/en/file-access/MIBGAS_Data_<YYYY>.xlsx?path=AGNO_<YYYY>/XLS

One workbook per calendar year. Sheets of interest:

- "Trading Data PVB" — one row per (trading day × product) for the
  Punto Virtual de Balance hub. Products include GDAES_D+1 (day-ahead),
  GMAES (month), GQES_Q+i (quarter), GYES_Y+i (year), GWDES (weekend),
  GW_BoMES (balance of month). Columns: trading day, product, place
  of delivery, area, first/last day of delivery, daily reference
  price, daily auction price, last/max/min daily prices, daily
  volume traded. This is the canonical spot+forward MIBGAS panel.
- "Indices" — the official daily MIBGAS-ES Index (volume-weighted
  spot reference) + LNG index. Used as the short-end benchmark.

The CSV variant (only present 2021-onwards) carries a single price
column and is therefore lossy; the parser always reads the XLSX.

License: see MIBGAS terms. Disclaimer sheet in the workbook states
that the data is for informational use and that MIBGAS does not
guarantee continuity of structure or format.

Pipeline:
- fetch_annual(year, force=False) — downloads one workbook to
  data/raw/mibgas/year=<YYYY>/MIBGAS_Data_<YYYY>.xlsx, manifest line
  appended. Idempotent.
- parse_pvb(path) and parse_indices(path) — read the two sheets.
- build_curated(years) — concatenate and write
  data/curated/mibgas_pvb.parquet and data/curated/mibgas_indices.parquet.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

logger = logging.getLogger(__name__)

MIBGAS_BASE_URL = "https://www.mibgas.es/en/file-access"

# MIBGAS rebrands sheet names between years. Try in order; first match wins.
_PVB_SHEET_CANDIDATES: tuple[str, ...] = (
    "Trading Data PVB",         # 2019-2020 single-hub layout
    "Trading Data PVB&VTP",     # 2024 combined PVB + VTP hub
)
_INDICES_SHEET_CANDIDATES: tuple[str, ...] = (
    "Indices",          # 2019
    "MIBGAS Indexes",   # 2024 — different column shape; reduced to ES index here
)

# Multiple aliases per logical column (old / new MIBGAS schemas).
_PVB_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "trade_date":                       ("Trading day",),
    "product_code":                     ("Product",),
    "place_of_delivery":                ("Place of delivery",),
    "area":                             ("Area",),
    "first_day_delivery":               ("First Day Delivery",),
    "last_day_delivery":                ("Last Day Delivery",),
    "daily_reference_price_eur_mwh":    ("Daily Reference Price\n[EUR/MWh]",
                                         "Reference Price\n[EUR/MWh]"),
    "daily_auction_price_eur_mwh":      ("Daily Auction Price\n[EUR/MWh]",
                                         "Auction Price\n[EUR/MWh]"),
    "last_daily_price_eur_mwh":         ("Last Daily Price\n[EUR/MWh]",
                                         "Last Price\n[EUR/MWh]"),
    "max_daily_price_eur_mwh":          ("Maximum Daily Price\n[EUR/MWh]",
                                         "Maximum Price\n[EUR/MWh]"),
    "min_daily_price_eur_mwh":          ("Minimum Daily Price\n[EUR/MWh]",
                                         "Minimum Price\n[EUR/MWh]"),
    "daily_volume_mwh":                 ("Daily Volume Traded\n[MWh]",
                                         "Volume Traded\n[MWh]"),
}

# Indices: the column we want lives under different names across years.
_INDICES_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "delivery_day":                 ("Delivery day",),
    "area":                         ("Area",),
    "mibgas_es_index_eur_mwh":      ("MIBGAS-ES Index\n[EUR/MWh]",),
    "mibgas_es_volume_mwh":         ("MIBGAS-ES Volume\n[MWh]",),
    "mibgas_es_lng_index_eur_mwh":  ("MIBGAS-ES \nLNG Index\n[EUR/MWh]",
                                     "MIBGAS\nLNG-ES Index\n[EUR/MWh]"),
    "mibgas_es_lng_volume_mwh":     ("MIBGAS-ES \nLNG Volume\n[MWh]",
                                     "MIBGAS\nLNG-ES Volume\n[MWh]"),
}


# ---- HTTP layer ------------------------------------------------------------


def annual_url(year: int) -> str:
    return f"{MIBGAS_BASE_URL}/MIBGAS_Data_{year}.xlsx?path=AGNO_{year}/XLS"


def raw_path(year: int) -> Path:
    return _paths.raw_path(
        "mibgas",
        f"year={year}",
        filename=f"MIBGAS_Data_{year}.xlsx",
    )


def fetch_annual(
    year: int,
    *,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Download one annual MIBGAS workbook (idempotent)."""
    out = raw_path(year)
    if out.exists() and not force:
        logger.debug("MIBGAS raw cached: %s", out)
        return out

    sess = session or ThrottledSession()
    url = annual_url(year)
    resp = sess.get(url)
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="mibgas",
            url=url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={"year": year},
        )
    )
    return out


# ---- Parsing ---------------------------------------------------------------


def _open_excel(path: Path) -> pd.ExcelFile:
    with path.open("rb") as fh:
        return pd.ExcelFile(io.BytesIO(fh.read()))


def _pick_sheet(xl: pd.ExcelFile, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in xl.sheet_names:
            return name
    return None


def _apply_aliases(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Rename columns according to alias lists; drop unmatched columns."""
    rename: dict[str, str] = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in df.columns:
                rename[opt] = canonical
                break
    df = df.rename(columns=rename)
    return df[[c for c in aliases if c in df.columns]].copy()


def parse_pvb(path: Path) -> pd.DataFrame:
    """Parse the PVB/VTP trading-data sheet (handles 2019-2020 and 2021+ layouts)."""
    xl = _open_excel(path)
    sheet = _pick_sheet(xl, _PVB_SHEET_CANDIDATES)
    if sheet is None:
        raise ValueError(
            f"None of {_PVB_SHEET_CANDIDATES!r} found in {path.name}; "
            f"present sheets: {xl.sheet_names!r}"
        )
    df = pd.read_excel(xl, sheet_name=sheet)
    df = _apply_aliases(df, _PVB_COLUMN_ALIASES)
    for col in ("trade_date", "first_day_delivery", "last_day_delivery"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in (
        "daily_reference_price_eur_mwh", "daily_auction_price_eur_mwh",
        "last_daily_price_eur_mwh", "max_daily_price_eur_mwh",
        "min_daily_price_eur_mwh",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "daily_volume_mwh" in df.columns:
        df["daily_volume_mwh"] = pd.to_numeric(
            df["daily_volume_mwh"], errors="coerce"
        ).astype("Int64")
    return df.dropna(subset=["trade_date", "product_code"]).reset_index(drop=True)


def parse_indices(path: Path) -> pd.DataFrame:
    """Parse the daily MIBGAS-ES spot index (handles 2019 and 2024 layouts)."""
    xl = _open_excel(path)
    sheet = _pick_sheet(xl, _INDICES_SHEET_CANDIDATES)
    if sheet is None:
        raise ValueError(
            f"None of {_INDICES_SHEET_CANDIDATES!r} found in {path.name}"
        )
    df = pd.read_excel(xl, sheet_name=sheet)
    df = _apply_aliases(df, _INDICES_COLUMN_ALIASES)
    if "delivery_day" in df.columns:
        df["delivery_day"] = pd.to_datetime(df["delivery_day"], errors="coerce").dt.date
    for col in (
        "mibgas_es_index_eur_mwh", "mibgas_es_volume_mwh",
        "mibgas_es_lng_index_eur_mwh", "mibgas_es_lng_volume_mwh",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["delivery_day"]).reset_index(drop=True)


# ---- Curated build ---------------------------------------------------------


def build_curated(
    start_year: int,
    end_year: int,
    *,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate annual workbooks already present in raw/ into two
    curated parquet files (PVB trading data and the daily index).

    Returns (pvb_df, indices_df). Missing years are skipped silently.
    """
    pvb_frames: list[pd.DataFrame] = []
    idx_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        p = raw_path(year)
        if not p.exists():
            logger.warning("MIBGAS year %d not present at %s", year, p)
            continue
        try:
            pvb_frames.append(parse_pvb(p))
        except Exception:  # pragma: no cover — defensive
            logger.exception("Failed to parse PVB %d", year)
        try:
            idx_frames.append(parse_indices(p))
        except Exception:  # pragma: no cover — defensive
            logger.exception("Failed to parse Indices %d", year)

    pvb = pd.concat(pvb_frames, ignore_index=True) if pvb_frames else pd.DataFrame()
    idx = pd.concat(idx_frames, ignore_index=True) if idx_frames else pd.DataFrame()
    if write:
        if not pvb.empty:
            pvb_path = _paths.curated_path("mibgas_pvb.parquet")
            pvb.to_parquet(pvb_path, index=False)
            logger.info("Wrote %s (%d rows)", pvb_path, len(pvb))
        if not idx.empty:
            idx_path = _paths.curated_path("mibgas_indices.parquet")
            idx.to_parquet(idx_path, index=False)
            logger.info("Wrote %s (%d rows)", idx_path, len(idx))
    return pvb, idx
