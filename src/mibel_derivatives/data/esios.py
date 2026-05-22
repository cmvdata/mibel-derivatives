"""ESIOS API client and curated indicator panel.

Token comes from `.env` (`ESIOS_API_TOKEN`); see `.env.example`. The
client caches one parquet per (indicator, geo, year, month) under
`data/raw/esios/cache/`, matching the filename convention used by the
sibling repo `mibel-forecasting` so the two caches are
interchangeable.

Curated indicators (selected for the tolling / ancillary-services
panel and verified against /indicators/<id> on 2026-05-22):

| ID    | Label                       | Unit    |
|-------|------------------------------|---------|
| 600   | omie_day_ahead_es            | €/MWh   |
| 612   | omie_intraday_s1_es          | €/MWh   |
| 634   | secondary_band_down_price    | €/MW    |
| 682   | secondary_energy_up_price    | €/MWh   |
| 683   | secondary_energy_down_price  | €/MWh   |
| 677   | tertiary_up_marginal         | €/MWh   |
| 676   | tertiary_down_marginal       | €/MWh   |
| 686   | imbalance_up_price           | €/MWh   |
| 687   | imbalance_down_price         | €/MWh   |
| 793   | tech_restrictions_pbf        | €/MWh   |
| 794   | tech_restrictions_realtime   | €/MWh   |
| 10211 | ancillary_total              | €/MWh   |
| 79    | ccgt_p48_program             | MWh     |

The curated panel joins all ten on the hourly UTC index and writes to
`data/curated/esios_panel.parquet`.

Indicator 600 also acts as the gap-fill source for `omie.py` when the
forecasting cache is missing 2019-2021 — same filename shape means
the OMIE loader picks up the freshly-pulled parquets automatically.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

load_dotenv()

logger = logging.getLogger(__name__)

ESIOS_BASE_URL = "https://api.esios.ree.es"
GEO_ES = 3
GEO_PT = 1
GEO_FR = 2


@dataclass(frozen=True)
class Indicator:
    id: int
    label: str
    unit: str
    geo: int = GEO_ES


CURATED_INDICATORS: tuple[Indicator, ...] = (
    # Day-ahead / intraday market prices
    Indicator(600,   "omie_day_ahead_es",            "EUR_per_MWh"),
    Indicator(612,   "omie_intraday_s1_es",          "EUR_per_MWh"),
    # Secondary regulation: band (capacity) + energy activation (both ways)
    Indicator(634,   "secondary_band_down_price",    "EUR_per_MW"),
    Indicator(682,   "secondary_energy_up_price",    "EUR_per_MWh"),
    Indicator(683,   "secondary_energy_down_price",  "EUR_per_MWh"),
    # Tertiary regulation: marginal activation price (programmed)
    Indicator(677,   "tertiary_up_marginal",         "EUR_per_MWh"),
    Indicator(676,   "tertiary_down_marginal",       "EUR_per_MWh"),
    # Imbalance-settlement prices (gestión desvíos)
    Indicator(686,   "imbalance_up_price",           "EUR_per_MWh"),
    Indicator(687,   "imbalance_down_price",         "EUR_per_MWh"),
    # Technical-restrictions price components and total ancillary cost
    Indicator(793,   "tech_restrictions_pbf",        "EUR_per_MWh"),
    Indicator(794,   "tech_restrictions_realtime",   "EUR_per_MWh"),
    Indicator(10211, "ancillary_total",              "EUR_per_MWh"),
    # CCGT (Castejón family) final hourly schedule — the "programa" leg of
    # settlement_reconciliation. P48 is the definitive post-intraday program.
    Indicator(79,    "ccgt_p48_program",             "MWh"),
)


class ESIOSConfigError(RuntimeError):
    """Raised when `ESIOS_API_TOKEN` is missing from the environment."""


def _resolve_token() -> str:
    token = os.environ.get("ESIOS_API_TOKEN")
    if not token:
        raise ESIOSConfigError(
            "ESIOS_API_TOKEN is not set. Copy .env.example to .env and "
            "request a token at https://api.esios.ree.es."
        )
    return token


# ---- Cache layout ----------------------------------------------------------


def cache_dir() -> Path:
    """Local cache mirror. Matches mibel-forecasting filename convention."""
    d = _paths.RAW_DIR / "esios" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(indicator_id: int, geo: int, year: int, month: int) -> Path:
    return cache_dir() / f"i{indicator_id}_geo{geo}_{year}_{month:02d}.parquet"


# ---- API calls -------------------------------------------------------------


def lookup_indicator(indicator_id: int, *, timeout: float = 20.0) -> dict:
    """Return the /indicators/<id> metadata document. No caching."""
    r = requests.get(
        f"{ESIOS_BASE_URL}/indicators/{indicator_id}",
        headers={
            "Accept": "application/json",
            "x-api-key": _resolve_token(),
            "User-Agent": "mibel-derivatives/0.1",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("indicator", {})


def _fetch_month(
    indicator_id: int,
    geo: int,
    year: int,
    month: int,
    *,
    session: ThrottledSession | None = None,
) -> pd.Series:
    """Pull one (indicator, geo, year, month) from ESIOS. UTC-hourly series."""
    last_day = (
        pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
    ).strftime("%Y-%m-%d")
    start = f"{year}-{month:02d}-01T00:00:00Z"
    end = f"{last_day}T23:59:59Z"

    url = f"{ESIOS_BASE_URL}/indicators/{indicator_id}"
    headers = {
        "Accept": "application/json",
        "x-api-key": _resolve_token(),
        "User-Agent": "mibel-derivatives/0.1",
    }
    params = {
        "start_date": start,
        "end_date": end,
        "geo_ids[]": str(geo),
    }
    # ThrottledSession.get does not take dict params + multi-value; use
    # plain requests for the GET but still bounded by the session throttle
    # if provided.
    if session is not None:
        session._wait_throttle()  # type: ignore[attr-defined]
        r = session._session.get(url, headers=headers, params=params, timeout=60)
        session._last_request_monotonic = __import__("time").monotonic()
    else:
        r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()

    payload = r.content
    append_manifest(
        record_from_download(
            source="esios",
            url=r.url,
            raw_path=cache_path(indicator_id, geo, year, month),
            http_status=r.status_code,
            payload=payload,
            params={
                "indicator": indicator_id,
                "geo": geo,
                "year": year,
                "month": month,
            },
        )
    )

    rows = r.json().get("indicator", {}).get("values", [])
    if not rows:
        return pd.Series(dtype=float, name="value")
    df = pd.DataFrame(rows)
    df["dt_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    s = df.set_index("dt_utc")["value"].astype(float)
    s = s.groupby(s.index.floor("h")).mean()
    s.name = "value"
    return s


def pull_indicator(
    indicator_id: int,
    geo: int,
    *,
    start: str | dt.date | pd.Timestamp,
    end: str | dt.date | pd.Timestamp,
    session: ThrottledSession | None = None,
    refresh: bool = False,
) -> pd.Series:
    """UTC-hourly series for (indicator, geo) over [start, end], with cache."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    months = pd.date_range(
        start_ts.tz_convert(None).replace(day=1),
        end_ts.tz_convert(None).replace(day=1),
        freq="MS",
    )

    chunks: list[pd.Series] = []
    for m in months:
        p = cache_path(indicator_id, geo, int(m.year), int(m.month))
        if p.exists() and not refresh:
            chunks.append(pd.read_parquet(p)["value"])
            continue
        s = _fetch_month(indicator_id, geo, int(m.year), int(m.month), session=session)
        s.to_frame("value").to_parquet(p)
        chunks.append(s)

    if not chunks:
        return pd.Series(dtype=float, name="value")
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out.loc[start_ts:end_ts]


# ---- Curated panel ---------------------------------------------------------


def build_curated_panel(
    start: str | dt.date | pd.Timestamp,
    end: str | dt.date | pd.Timestamp,
    *,
    indicators: tuple[Indicator, ...] = CURATED_INDICATORS,
    session: ThrottledSession | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Pull every curated indicator and join into one hourly panel.

    Result: a wide DataFrame indexed by dt_utc with one column per
    indicator (column name = indicator.label).
    """
    series: dict[str, pd.Series] = {}
    for ind in indicators:
        logger.info("Pulling ESIOS indicator %d (%s)", ind.id, ind.label)
        s = pull_indicator(ind.id, ind.geo, start=start, end=end, session=session)
        series[ind.label] = s
    if not series:
        return pd.DataFrame()
    panel = pd.concat(series.values(), axis=1)
    panel.columns = list(series.keys())
    panel.index.name = "dt_utc"
    if write:
        out = _paths.curated_path("esios_panel.parquet")
        panel.reset_index().to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows × %d indicators)", out, len(panel), panel.shape[1])
    return panel
