"""TTF Dutch natural-gas front-month settlement price (prototype source).

The Title Transfer Facility (TTF) is the Dutch virtual trading point
and the European benchmark hub for natural gas. The deep liquid
market is the ICE Endex TTF Front-Month Future; daily settlements are
disseminated by ICE under commercial licence.

**No fully free, deep-history, daily-granular feed exists for TTF.**
The candidates evaluated on 2026-05-23 were:

| Source                          | Granularity | History         | Status            |
|---------------------------------|-------------|-----------------|-------------------|
| EEX NGP TTF 60-day CSV          | daily       | last 60 days    | rolling only; SSL chain broken on `gasandregistry.eex.com` |
| Yahoo Finance v7 download API   | daily       | multi-year      | HTTP 401 (deprecated; auth required) |
| Stooq CSV download              | daily       | multi-year      | requires captcha-issued API key |
| Eurostat `nrg_pc_*`             | semestral   | multi-year      | granularity too coarse for pricing |
| ECB Data Portal CPP             | monthly     | multi-year      | series-key for gas not found |

The Yahoo Finance **v8 chart API** (`/v8/finance/chart/<symbol>`) is
still publicly reachable without authentication and serves the TTF=F
symbol — the ICE Endex TTF front-month future as listed on NYMEX —
in EUR/MWh with multi-year daily history. Sanity check: the August
2022 historical peak close prints at €339.20/MWh, matching the
widely-reported peak; first observation 2018-12-31 at €21.98/MWh
matches the late-2018 level.

**This is a prototype source.** Yahoo Finance's terms of service
restrict redistribution of its market data and do not licence the
feed for commercial production use. We use it only for backtesting
and research; a production deployment must replace it with a licensed
ICE / Refinitiv / S&P Global Commodity Insights feed. The diagnostic
doc reiterates this caveat.

Pipeline:
- `fetch_window(symbol, start, end)` writes one raw JSON per call to
  `data/raw/ttf/<symbol>/<startYYYY-MM-DD>_<endYYYY-MM-DD>.json`.
- `parse_window(path)` returns a typed daily DataFrame
  (date, open, high, low, close, volume).
- `build_curated(start, end)` pulls + concatenates + writes
  `data/curated/ttf_front_month.parquet`.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

logger = logging.getLogger(__name__)

YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
TTF_SYMBOL = "TTF=F"  # ICE Endex TTF Front-Month, NYMEX-listed, EUR/MWh

# Yahoo rejects default Python User-Agent strings; use a browser-like one.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) mibel-derivatives/0.1"
)


# ---- Filesystem layout -----------------------------------------------------


def raw_path(symbol: str, start: dt.date, end: dt.date) -> Path:
    """Raw JSON per (symbol, start, end) window."""
    safe_symbol = symbol.replace("=", "_")
    return _paths.raw_path(
        "ttf",
        f"symbol={safe_symbol}",
        filename=f"{start.isoformat()}_{end.isoformat()}.json",
    )


# ---- HTTP layer ------------------------------------------------------------


def _to_date(d: dt.date | str) -> dt.date:
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


def fetch_window(
    symbol: str,
    start: dt.date | str,
    end: dt.date | str,
    *,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Pull one Yahoo v8 chart window for the given symbol. Idempotent."""
    s = _to_date(start)
    e = _to_date(end)
    out = raw_path(symbol, s, e)
    if out.exists() and not force:
        logger.debug("TTF raw cached: %s", out)
        return out

    sess = session or ThrottledSession(
        min_interval_seconds=1.5, user_agent=_USER_AGENT,
    )
    # Yahoo interprets period1/period2 as UTC seconds. Use calendar.timegm
    # (UTC-aware) rather than time.mktime (local-time, drifts by TZ offset).
    period1 = calendar.timegm(s.timetuple())
    period2 = calendar.timegm((e + dt.timedelta(days=1)).timetuple())
    url = f"{YAHOO_CHART_BASE_URL}/{symbol}"
    params = {
        "period1": str(period1),
        "period2": str(period2),
        "interval": "1d",
        "events": "history",
    }
    resp = sess.get(url, params=params)
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="ttf",
            url=resp.url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={
                "symbol": symbol,
                "start": s.isoformat(), "end": e.isoformat(),
            },
        )
    )
    logger.info(
        "TTF: fetched %s window %s..%s (%d bytes)",
        symbol, s, e, len(resp.content),
    )
    return out


# ---- Parsing ---------------------------------------------------------------


def parse_window(path: Path) -> pd.DataFrame:
    """Parse one Yahoo v8 chart JSON to a typed daily DataFrame."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_list = payload.get("chart", {}).get("result") or []
    if not result_list:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
    result = result_list[0]
    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not ts:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame({
        "date": [dt.datetime.utcfromtimestamp(t).date() for t in ts],
        "open": quote.get("open") or [None] * len(ts),
        "high": quote.get("high") or [None] * len(ts),
        "low": quote.get("low") or [None] * len(ts),
        "close": quote.get("close") or [None] * len(ts),
        "volume": quote.get("volume") or [None] * len(ts),
    })
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


# ---- Curated build ---------------------------------------------------------


def build_curated(
    start: dt.date | str = dt.date(2019, 1, 1),
    end: dt.date | str = dt.date(2024, 12, 31),
    *,
    symbol: str = TTF_SYMBOL,
    session: ThrottledSession | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Pull one window covering [start, end] and write the curated parquet."""
    s = _to_date(start)
    e = _to_date(end)
    p = fetch_window(symbol, s, e, session=session, force=True)
    df = parse_window(p)
    df["symbol"] = symbol
    if write and not df.empty:
        out = _paths.curated_path("ttf_front_month.parquet")
        df.to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(df))
    return df


# ---- Smoke helper ----------------------------------------------------------


def smoke_download(
    *,
    start: dt.date | str = dt.date(2020, 1, 1),
    end: dt.date | str = dt.date(2020, 12, 31),
    session: ThrottledSession | None = None,
) -> Path:
    """Pull one sample year (2020 by default) for TTF=F."""
    return fetch_window(TTF_SYMBOL, start, end, session=session, force=True)


__all__ = [
    "YAHOO_CHART_BASE_URL", "TTF_SYMBOL",
    "raw_path", "fetch_window", "parse_window",
    "build_curated", "smoke_download",
]
