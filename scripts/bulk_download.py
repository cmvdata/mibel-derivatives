"""One-shot bulk download for every data scraper in the repo.

Idempotent: each scraper skips inputs already present in `data/raw/`.
Re-running is safe — failures in one scraper do not abort the next.

Usage:
    python scripts/bulk_download.py            # run every scraper
    python scripts/bulk_download.py esios bce  # subset by name

Runtime is dominated by ESIOS (~25 min for 13 indicators × 72 months
at a 1.5 s throttle). Every other scraper finishes in under a minute.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from collections.abc import Callable

from mibel_derivatives.data import bce, esios, eua, pvgis, ttf
from mibel_derivatives.data._http import ThrottledSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bulk")


# ---- Per-scraper runners ---------------------------------------------------


def run_esios() -> dict[str, object]:
    """Pull 2019-2024 for every curated ESIOS indicator (UTC hourly)."""
    sess = ThrottledSession(min_interval_seconds=1.5)
    panel = esios.build_curated_panel(
        start="2019-01-01", end="2024-12-31",
        session=sess, write=True,
    )
    return {"rows": len(panel), "indicators": panel.shape[1]}


def run_pvgis() -> dict[str, object]:
    """Pull 2019-2023 for both curated PVGIS configs (Andalucía centroid)."""
    sess = ThrottledSession(min_interval_seconds=2.0)
    n = 0
    for cfg in pvgis.CURATED_CONFIGS:
        for year in range(2019, 2024):  # SARAH3 coverage ends in 2023
            pvgis.fetch_year(
                pvgis.LAT_ANDALUCIA, pvgis.LON_ANDALUCIA,
                year, cfg, session=sess,
            )
            n += 1
    panel = pvgis.build_curated(
        start_year=2019, end_year=2023, write=True,
    )
    return {"files": n, "panel_rows": len(panel)}


def run_bce() -> dict[str, object]:
    """Pull full history for every (curve, tenor) ECB yield-curve series."""
    sess = ThrottledSession(min_interval_seconds=1.5)
    n = 0
    for curve in bce.CURATED_CURVES:
        for tenor in bce.CURATED_TENORS:
            bce.fetch_series(curve, tenor, session=sess, force=True)
            n += 1
    panel = bce.build_curated(write=True)
    return {"files": n, "panel_rows": len(panel)}


def run_eua() -> dict[str, object]:
    """Pull EEX primary-auction annual workbooks 2019-2024."""
    sess = ThrottledSession(min_interval_seconds=1.5)
    n = 0
    for year in range(2019, 2025):
        eua.fetch_year(year, session=sess)
        n += 1
    panel = eua.build_curated(start_year=2019, end_year=2024, write=True)
    return {"files": n, "panel_rows": len(panel)}


def run_ttf() -> dict[str, object]:
    """Pull the 2019-2024 TTF window via Yahoo v8 chart (prototype source)."""
    sess = ThrottledSession(min_interval_seconds=1.5)
    panel = ttf.build_curated(
        start=dt.date(2019, 1, 1), end=dt.date(2024, 12, 31),
        session=sess, write=True,
    )
    return {"panel_rows": len(panel)}


RUNNERS: dict[str, Callable[[], dict[str, object]]] = {
    # Fast first so any wiring problem surfaces early; ESIOS last as it dominates.
    "ttf":   run_ttf,
    "eua":   run_eua,
    "bce":   run_bce,
    "pvgis": run_pvgis,
    "esios": run_esios,
    # TODO: integrate the three remaining scrapers as wrappers above and
    # register them here. Reference patterns:
    #   - run_omip:   per-trade-date HTML scrape over a date range; build a
    #     long-format forward curve. See src/mibel_derivatives/data/omip.py.
    #   - run_mibgas: per-year XLSX download of PVB + MIBGAS-ES, then
    #     a curated daily panel. See src/mibel_derivatives/data/mibgas.py.
    #   - run_omie:   loader reuses the mibel-forecasting OMIE cache; the
    #     bulk runner only needs to invoke build_curated and validate.
    #     See src/mibel_derivatives/data/omie.py.
}


# ---- Driver ----------------------------------------------------------------


def main(selection: list[str]) -> int:
    names = selection or list(RUNNERS)
    unknown = [n for n in names if n not in RUNNERS]
    if unknown:
        log.error("Unknown scraper(s): %s. Known: %s", unknown, list(RUNNERS))
        return 2

    log.info("Bulk download: %s", " ".join(names))
    overall_start = time.monotonic()
    summary: list[tuple[str, float, dict[str, object] | str]] = []
    rc = 0
    for name in names:
        log.info("=== %s START ===", name)
        t0 = time.monotonic()
        try:
            result = RUNNERS[name]()
            elapsed = time.monotonic() - t0
            log.info("=== %s OK in %.1fs: %s ===", name, elapsed, result)
            summary.append((name, elapsed, result))
        except Exception as exc:  # noqa: BLE001 — top-level barrier on purpose
            elapsed = time.monotonic() - t0
            log.exception("=== %s FAILED in %.1fs: %s ===", name, elapsed, exc)
            summary.append((name, elapsed, f"FAILED: {exc!r}"))
            rc = 1

    overall = time.monotonic() - overall_start
    log.info("---- Bulk summary (wall %.1fs / %.1fmin) ----", overall, overall / 60)
    for name, elapsed, result in summary:
        log.info("  %-6s %7.1fs  %s", name, elapsed, result)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
