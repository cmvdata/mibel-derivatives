"""PVGIS hourly PV-output series from the EU JRC.

PVGIS (https://re.jrc.ec.europa.eu) is a free public service. The
`seriescalc` endpoint returns one year of hourly PV production +
ancillary meteorology (global tilted irradiance, ambient temperature,
wind speed, sun height) for a given (lat, lon, system geometry).

For the Andalucía tolling / solar-PPA study we keep two stock
configurations:

- `fixed_35deg_south` — tilt 35°, azimuth 0 (south), no tracking;
  representative of utility-scale fixed-tilt plants in southern Spain.
- `one_axis_horizontal_ns` — single horizontal axis (N-S), E-W
  tracking, horizontal mounting; the dominant utility-scale geometry
  in the Iberian peninsula.

Both use 1 kWp peak power, 14% system losses, free-standing mounting.
Output scales linearly with peakpower so 1 kWp is the canonical unit.

Lat/Lon default to (37.4, -5.0) — the centroid of Andalucía near Sevilla,
which is the geographical reference for the PPA pricing workbook.

API version: v5.3 (current production). The radiation database is
configurable; we default to `PVGIS-SARAH3` (satellite, ~5 km
resolution over Europe, 2005-2023) with `PVGIS-ERA5` as the only
other valid option (reanalysis, global, same upper bound 2023). 2024
is not yet published by JRC — see
`reports/diagnostics/pvgis_endpoint.md`.

Pipeline:
- `fetch_year(lat, lon, year, config)` — writes raw JSON to
  `data/raw/pvgis/db=<DB>/config=<label>/lat=...lon=.../<year>.json`,
  idempotent.
- `parse_year(path, config)` — typed DataFrame indexed by UTC hour.
- `build_curated(lat, lon, start_year, end_year)` — concatenates parsed
  years for both configs into `data/curated/pvgis_panel.parquet`.

The service rate-limits aggressively; the throttled session defaults
to 2 s between calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import _paths
from ._http import ThrottledSession
from ._provenance import append_manifest, record_from_download

logger = logging.getLogger(__name__)

PVGIS_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

# Andalucía reference site for the solar-PPA / tolling study.
LAT_ANDALUCIA = 37.4
LON_ANDALUCIA = -5.0

# Allowed radiation databases for v5.3 seriescalc.
# SARAH3 = MSG-satellite-derived, ~5 km, Europe-only, currently 2005-2023.
# ERA5   = ECMWF reanalysis, ~30 km, global, currently 2005-2023.
RADDATABASE_SARAH3 = "PVGIS-SARAH3"
RADDATABASE_ERA5 = "PVGIS-ERA5"
ALLOWED_RADDATABASES: frozenset[str] = frozenset({RADDATABASE_SARAH3, RADDATABASE_ERA5})
DEFAULT_RADDATABASE = RADDATABASE_SARAH3


@dataclass(frozen=True)
class PVConfig:
    """Geometry + system losses for one PV configuration."""

    label: str
    trackingtype: int       # 0 fixed, 1 horizontal-NS, 2 two-axis, 3 vertical, 4 inclined
    angle: float            # surface tilt in degrees (irrelevant for trackingtype > 0 if 0)
    aspect: float = 0.0     # azimuth (° from south, PVGIS sign convention)
    peakpower_kwp: float = 1.0
    losses_pct: float = 14.0
    mountingplace: str = "free"  # 'free' (open-rack) or 'building'


FIXED_35_SOUTH = PVConfig(
    label="fixed_35deg_south",
    trackingtype=0,
    angle=35.0,
    aspect=0.0,
)

ONE_AXIS_NS = PVConfig(
    label="one_axis_horizontal_ns",
    trackingtype=1,
    angle=0.0,
    aspect=0.0,
)

CURATED_CONFIGS: tuple[PVConfig, ...] = (FIXED_35_SOUTH, ONE_AXIS_NS)


# ---- Filesystem layout -----------------------------------------------------


def raw_path(
    lat: float,
    lon: float,
    year: int,
    config: PVConfig,
    *,
    raddatabase: str = DEFAULT_RADDATABASE,
) -> Path:
    """Raw-zone path. Includes the radiation database so SARAH3 / ERA5
    files do not collide for the same (config, year)."""
    return _paths.raw_path(
        "pvgis",
        f"db={raddatabase}",
        f"config={config.label}",
        f"lat={lat:.4f}_lon={lon:.4f}",
        filename=f"{year}.json",
    )


# ---- HTTP layer ------------------------------------------------------------


def _params(
    lat: float, lon: float, year: int, config: PVConfig, raddatabase: str,
) -> dict[str, str]:
    return {
        "lat": f"{lat}",
        "lon": f"{lon}",
        "startyear": str(year),
        "endyear": str(year),
        "pvcalculation": "1",
        "peakpower": f"{config.peakpower_kwp}",
        "loss": f"{config.losses_pct}",
        "trackingtype": str(config.trackingtype),
        "angle": f"{config.angle}",
        "aspect": f"{config.aspect}",
        "mountingplace": config.mountingplace,
        "raddatabase": raddatabase,
        "outputformat": "json",
    }


def fetch_year(
    lat: float,
    lon: float,
    year: int,
    config: PVConfig,
    *,
    raddatabase: str = DEFAULT_RADDATABASE,
    session: ThrottledSession | None = None,
    force: bool = False,
) -> Path:
    """Download one (db, config, lat, lon, year) PVGIS series. Idempotent."""
    if raddatabase not in ALLOWED_RADDATABASES:
        raise ValueError(
            f"raddatabase must be one of {sorted(ALLOWED_RADDATABASES)}; "
            f"got {raddatabase!r}"
        )
    out = raw_path(lat, lon, year, config, raddatabase=raddatabase)
    if out.exists() and not force:
        logger.debug("PVGIS raw cached: %s", out)
        return out

    sess = session or ThrottledSession(min_interval_seconds=2.0)
    params = _params(lat, lon, year, config, raddatabase)
    resp = sess.get(PVGIS_BASE_URL, params=params)
    out.write_bytes(resp.content)
    append_manifest(
        record_from_download(
            source="pvgis",
            url=resp.url,
            raw_path=out,
            http_status=resp.status_code,
            payload=resp.content,
            params={
                "lat": lat, "lon": lon, "year": year,
                "config": config.label, "raddatabase": raddatabase,
            },
        )
    )
    logger.info(
        "PVGIS: fetched %s @ (%.4f, %.4f) year=%d db=%s (%d bytes)",
        config.label, lat, lon, year, raddatabase, len(resp.content),
    )
    return out


# ---- Parsing ---------------------------------------------------------------


_OUTPUT_COLUMNS = {
    "P": "p_w",                  # PV power output, W (for 1 kWp peakpower)
    "G(i)": "g_i_w_m2",          # global irradiance on the plane of array, W/m2
    "H_sun": "sun_height_deg",   # sun height angle, °
    "T2m": "t_air_c",            # 2 m air temperature, °C
    "WS10m": "wind_10m_m_s",     # 10 m wind speed, m/s
    "Int": "is_reconstructed",   # 1 = reconstructed, 0 = measured
}


def parse_year(path: Path, config: PVConfig) -> pd.DataFrame:
    """Parse a PVGIS hourly JSON file to a typed UTC-hourly DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    hourly = raw.get("outputs", {}).get("hourly", [])
    if not hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    # PVGIS "time" format: "YYYYMMDD:HHMM" UTC. Minutes are always :10 (the
    # value is the hour-average centered at minute 10); floor to the hour
    # so the index lines up with OMIE / ESIOS hourly panels.
    df["dt_utc"] = (
        pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    )
    df = df.rename(columns=_OUTPUT_COLUMNS)
    df["config"] = config.label
    keep = ["config"] + list(_OUTPUT_COLUMNS.values())
    return df.set_index("dt_utc").sort_index()[keep]


# ---- Curated build ---------------------------------------------------------


def build_curated(
    lat: float = LAT_ANDALUCIA,
    lon: float = LON_ANDALUCIA,
    *,
    start_year: int,
    end_year: int,
    configs: tuple[PVConfig, ...] = CURATED_CONFIGS,
    raddatabase: str = DEFAULT_RADDATABASE,
    write: bool = True,
) -> pd.DataFrame:
    """Concatenate parsed PVGIS years for all configs (no downloads).

    Missing (db, config, year) triples are skipped silently — call
    `fetch_year` first to populate the raw zone.
    """
    rows: list[pd.DataFrame] = []
    for cfg in configs:
        for y in range(start_year, end_year + 1):
            p = raw_path(lat, lon, y, cfg, raddatabase=raddatabase)
            if not p.exists():
                continue
            df = parse_year(p, cfg)
            if not df.empty:
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows).sort_index()
    panel["lat"] = lat
    panel["lon"] = lon
    panel["raddatabase"] = raddatabase
    if write:
        out = _paths.curated_path("pvgis_panel.parquet")
        panel.reset_index().to_parquet(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(panel))
    return panel


# ---- Smoke helper ----------------------------------------------------------


def smoke_download(
    lat: float = LAT_ANDALUCIA,
    lon: float = LON_ANDALUCIA,
    year: int = 2023,
    *,
    raddatabase: str = DEFAULT_RADDATABASE,
    session: ThrottledSession | None = None,
) -> list[Path]:
    """Pull one reference year for both curated configs and return paths.

    Defaults to 2023, the upper edge of SARAH3 coverage as of 2026-05-22.
    """
    sess = session or ThrottledSession(min_interval_seconds=2.0)
    return [
        fetch_year(lat, lon, year, cfg, raddatabase=raddatabase, session=sess)
        for cfg in CURATED_CONFIGS
    ]


__all__ = [
    "PVGIS_BASE_URL",
    "LAT_ANDALUCIA", "LON_ANDALUCIA",
    "RADDATABASE_SARAH3", "RADDATABASE_ERA5", "ALLOWED_RADDATABASES",
    "DEFAULT_RADDATABASE",
    "PVConfig", "FIXED_35_SOUTH", "ONE_AXIS_NS", "CURATED_CONFIGS",
    "raw_path", "fetch_year", "parse_year",
    "build_curated", "smoke_download",
]
