"""ECB euro-area sovereign zero-coupon yield-curve scraper.

The European Central Bank publishes two daily euro-area zero-coupon
yield curves derived from a Svensson model on euro-denominated central
government bonds:

- **AAA**     — only bonds rated AAA at the time of estimation.
- **All**     — all euro-area sovereign issuers regardless of rating.

For the discount factors of the MIBEL derivatives module we keep the
six on-the-run tenors used by the rate-curve workbook:
1Y, 2Y, 3Y, 5Y, 7Y, 10Y.

Source: ECB Data Portal SDMX 2.1 REST API (free, no auth). The Data
Portal replaced the old SDW host in 2024; the dataflow code is `YC`
and the series-key skeleton is

    B.U2.EUR.4F.G_N_<rating>.SV_C_YM.SR_<tenor>

with rating = A (AAA) or C (all issuers) and tenor = SR_<N>Y for spot
rate.

Pipeline:
- `fetch_series(curve, tenor)` writes one CSV under
  `data/raw/bce/curve=<curve>/<tenor>.csv`, idempotent, full history
  is overwritten on `force=True`.
- `parse_series(path, curve, tenor)` returns a typed DataFrame
  indexed by trade date with the rate in percent.
- `build_curated()` concatenates all 12 series into the long-format
  panel `data/curated/bce_yield_curve.parquet`.
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

ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data/YC"

# Curve labels and the rating code used in the series key.
_RATING_BY_CURVE: dict[str, str] = {
    "aaa": "A",   # AAA-rated euro-area sovereigns only
    "all": "C",   # All euro-area sovereign issuers
}
CURATED_CURVES: tuple[str, ...] = ("aaa", "all")
CURATED_TENORS: tuple[str, ...] = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y")


def series_key(curve: str, tenor: str) -> str:
    """ECB SDMX series key for one (curve, tenor)."""
    if curve not in _RATING_BY_CURVE:
        raise ValueError(
            f"curve must be one of {sorted(_RATING_BY_CURVE)}; got {curve!r}"
        )
    if tenor not in CURATED_TENORS:
        raise ValueError(
            f"tenor must be one of {CURATED_TENORS}; got {tenor!r}"
        )
    return f"B.U2.EUR.4F.G_N_{_RATING_BY_CURVE[curve]}.SV_C_YM.SR_{tenor}"


# ---- Filesystem layout -----------------------------------------------------


def raw_path(curve: str, tenor: str) -> Path:
    return _paths.raw_path(
        "bce", f"curve={curve}", filename=f"{tenor}.csv",
    )


# ---- HTTP layer ------------------------------------------------------------


def fetch_series(
    curve: str,
    tenor: str,
    *,
    start_period: str | None = None,
    end_period: str | None = None,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Download one (curve, tenor) series as CSV. Idempotent on disk.

    Without `start_period` / `end_period` the API returns the full
    history (back to 2004-09 for the AAA curve).
    """
    out = raw_path(curve, tenor)
    if out.exists() and not force:
        logger.debug("BCE raw cached: %s", out)
        return out

    sess = session or ThrottledSession(min_interval_seconds=1.5)
    url = f"{ECB_BASE_URL}/{series_key(curve, tenor)}"
    params: dict[str, str] = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    resp = sess.get(
        url, params=params,
        headers={"Accept": "text/csv"},
    )
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="bce",
            url=resp.url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={
                "curve": curve, "tenor": tenor,
                "start_period": start_period, "end_period": end_period,
            },
        )
    )
    logger.info(
        "BCE: fetched curve=%s tenor=%s (%d bytes)",
        curve, tenor, len(resp.content),
    )
    return out


# ---- Parsing ---------------------------------------------------------------


def parse_series(path: Path, curve: str, tenor: str) -> pd.DataFrame:
    """Parse one ECB CSV to long-format (date, curve, tenor, rate_pct)."""
    df = pd.read_csv(io.BytesIO(path.read_bytes()))
    if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.DataFrame(
            columns=["date", "curve", "tenor", "rate_pct"]
        )
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIME_PERIOD"]).dt.date,
        "curve": curve,
        "tenor": tenor,
        "rate_pct": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return out.dropna(subset=["rate_pct"]).sort_values("date").reset_index(drop=True)


# ---- Curated build ---------------------------------------------------------


def build_curated(
    *,
    curves: tuple[str, ...] = CURATED_CURVES,
    tenors: tuple[str, ...] = CURATED_TENORS,
    write: bool = True,
) -> pd.DataFrame:
    """Concatenate all parsed (curve, tenor) series into one long panel."""
    rows: list[pd.DataFrame] = []
    for curve in curves:
        for tenor in tenors:
            p = raw_path(curve, tenor)
            if not p.exists():
                continue
            df = parse_series(p, curve, tenor)
            if not df.empty:
                rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "curve", "tenor", "rate_pct"])
    panel = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "curve", "tenor"]
    ).reset_index(drop=True)
    if write:
        out = _paths.curated_path("bce_yield_curve.parquet")
        panel.to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(panel))
    return panel


# ---- Smoke helper ----------------------------------------------------------


def smoke_download(
    *,
    start_period: str = "2024-01-01",
    end_period: str = "2024-12-31",
    session: ThrottledSession | None = None,
) -> list[Path]:
    """Pull all 12 (curve, tenor) series for one year and return paths."""
    sess = session or ThrottledSession(min_interval_seconds=1.5)
    paths: list[Path] = []
    for curve in CURATED_CURVES:
        for tenor in CURATED_TENORS:
            paths.append(
                fetch_series(
                    curve, tenor,
                    start_period=start_period, end_period=end_period,
                    session=sess, force=True,
                )
            )
    return paths


__all__ = [
    "ECB_BASE_URL",
    "CURATED_CURVES", "CURATED_TENORS",
    "series_key", "raw_path",
    "fetch_series", "parse_series",
    "build_curated", "smoke_download",
]
