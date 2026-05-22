"""OMIE day-ahead Spain spot loader (ESIOS indicator 600, geo=3).

This module does **not** download from ESIOS. The companion repo
`mibel-forecasting` already maintains a per-(indicator, geo, year, month)
parquet cache for the same indicator at:

    C:\\Users\\Carlo\\Desktop\\Projects\\Mibel_forecasting\\data\\cache\\esios\\

We reuse that cache as the canonical source for the months it covers
(2022-2024 at the time of writing). For 2019-2021 the gap is filled
by the ESIOS API client in `esios.py`, which writes the same filename
shape into our local cache mirror under `data/raw/omie/cache_mirror/`.

A single function `build_curated(start_year, end_year)` reads everything
present in either cache and produces a curated hourly UTC parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import _paths

logger = logging.getLogger(__name__)

INDICATOR = 600
GEO_ES = 3
GEO_PT = 1
GEO_FR = 2

# Forecasting-side cache path (read-only; never written).
FORECASTING_CACHE_DIR = Path(
    r"C:\Users\Carlo\Desktop\Projects\Mibel_forecasting\data\cache\esios"
)


def _local_cache_dir() -> Path:
    """Local mirror — populated by esios.fetch_indicator(600, ...) for gaps."""
    d = _paths.raw_path("omie", filename="").parent  # ensure parent exists
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_parquet(year: int, month: int, geo: int) -> Path | None:
    """Return the cache file path if present in either cache, else None."""
    name = f"i{INDICATOR}_geo{geo}_{year}_{month:02d}.parquet"
    for root in (FORECASTING_CACHE_DIR, _local_cache_dir()):
        p = root / name
        if p.exists():
            return p
    return None


def load_month(year: int, month: int, geo: int = GEO_ES) -> pd.DataFrame:
    """Read one (year, month, geo) parquet from either cache.

    Returns columns: dt_utc (datetime, UTC), price_eur_mwh (float),
    year (int), month (int), geo (int). Empty if absent.
    """
    p = _cached_parquet(year, month, geo)
    if p is None:
        return pd.DataFrame(columns=["dt_utc", "price_eur_mwh", "year", "month", "geo"])
    raw = pd.read_parquet(p)
    df = raw.reset_index().rename(columns={"value": "price_eur_mwh"})
    df["year"] = year
    df["month"] = month
    df["geo"] = geo
    return df[["dt_utc", "price_eur_mwh", "year", "month", "geo"]]


def build_curated(
    start_year: int,
    end_year: int,
    *,
    geos: tuple[int, ...] = (GEO_ES,),
    write: bool = True,
) -> pd.DataFrame:
    """Read every (year, month, geo) in range and concatenate to a tidy
    UTC-hourly DataFrame. Missing months are logged and skipped."""
    frames: list[pd.DataFrame] = []
    missing: list[tuple[int, int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for geo in geos:
                df = load_month(year, month, geo)
                if df.empty:
                    missing.append((year, month, geo))
                else:
                    frames.append(df)

    if missing:
        gaps_by_year = {}
        for y, m, g in missing:
            gaps_by_year.setdefault((y, g), []).append(m)
        for (y, g), months in sorted(gaps_by_year.items()):
            logger.warning(
                "OMIE indicator 600 geo=%d missing months in %d: %s",
                g, y, sorted(months),
            )

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["geo", "dt_utc"]
    ).reset_index(drop=True)
    if write:
        path = _paths.curated_path("omie_spot.parquet")
        out.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(out))
    return out
