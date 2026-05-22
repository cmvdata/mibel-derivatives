# PVGIS hourly PV-output endpoint — diagnostic

**Captured:** 2026-05-22 against `re.jrc.ec.europa.eu/api/v5_3/seriescalc`.
**Code:** `src/mibel_derivatives/data/pvgis.py`.

## Endpoint

PVGIS is the EU JRC's free Photovoltaic Geographical Information System.
The `seriescalc` endpoint returns hourly PV output for a given site +
system geometry, including ancillary meteorology.

URL template (GET, no auth required):

```
https://re.jrc.ec.europa.eu/api/v5_3/seriescalc
    ?lat=<float>
    &lon=<float>
    &startyear=<YYYY>&endyear=<YYYY>
    &pvcalculation=1
    &peakpower=<kWp>
    &loss=<pct>
    &trackingtype=<0|1|2|3|4>
    &angle=<deg>
    &aspect=<deg from south>
    &mountingplace=<free|building>
    &raddatabase=<PVGIS-SARAH3|PVGIS-ERA5>
    &outputformat=json
```

Response is a single JSON document with one hourly record per UTC
hour. The `time` field is `YYYYMMDD:HHMM` with minutes always `:10`
(the hour-average centered at minute 10); we floor to the hour so the
index lines up with OMIE / ESIOS hourly panels.

## API version and radiation databases

We pin **v5.3** (current production). v5.3 accepts only two radiation
databases — `PVGIS-SARAH2` was retired in this version:

- **`PVGIS-SARAH3`** (default): satellite-derived from EUMETSAT MSG,
  ~5 km resolution over Europe. Cleanest physics for Iberia.
- **`PVGIS-ERA5`** (fallback): ECMWF reanalysis, ~30 km, global.

`raddatabase` is exposed as a function parameter and encoded in the
raw-zone path so SARAH3 and ERA5 files for the same year do not
collide.

## Temporal coverage (verified live 2026-05-22)

| Database         | Coverage       | 2023 | 2024 |
|------------------|----------------|------|------|
| `PVGIS-SARAH3`   | **2005-2023**  | 200  | 400 (`startyear: Incorrect value. Please, enter an integer between 2005 and 2023.`) |
| `PVGIS-ERA5`     | **2005-2023**  | 200  | 400 |

**2024 is not yet available in PVGIS through either database.** JRC's
update cadence for SARAH3 is roughly 12-18 months behind realtime; the
next refresh that adds 2024 is expected in 2026 but not yet published.

Effective coverage for this module: **2019-2023** (5 years). The CONTEXT.md
study window asks for 2019-2024, so a 1-year gap remains at the top end.

### Filling the 2024 gap

Two options, both out of scope for this scraper:

1. **Copernicus CDS ERA5 reanalysis** (direct, not via PVGIS): hourly
   single-level radiation + temperature for 2024 is published with a
   ~5-day delay. Requires CDS API key and the `cdsapi` package. We
   would have to re-implement the PV physics layer (PVGIS does it for
   us); typical approach is `pvlib` ModelChain.
2. **Wait for JRC**. Most pragmatic if 2024 is not critical for early
   model iterations.

A tripwire test (`test_pvgis_2024_not_yet_available`) will fail the
day JRC publishes 2024 in SARAH3 and prompt us to extend the bulk
window.

## Curated configurations for the Andalucía PPA study

Site: lat=37.4, lon=-5.0 — Andalucía centroid near Sevilla.

| Label                       | trackingtype | tilt | azimuth | mounting |
|------------------------------|--------------|------|---------|----------|
| `fixed_35deg_south`         | 0 (fixed)    | 35°  | 0° (S)  | free     |
| `one_axis_horizontal_ns`    | 1 (1-axis NS)| 0°   | 0° (S)  | free     |

Both use peakpower = 1 kWp and 14% system losses (PVGIS default for
free-standing PV). Output scales linearly with peakpower so 1 kWp is
the canonical unit; the analytical model rescales to plant size.

`trackingtype` codes from PVGIS docs:
- 0 = fixed
- 1 = single horizontal axis (N-S axis, E-W tracking) — utility-scale standard
- 2 = two-axis (rare in MIBEL)
- 3 = vertical axis with optimal inclination
- 4 = inclined axis

We keep 0 and 1; 2-4 are out of scope for this study.

## Smoke download (2026-05-22, year=2023, SARAH3)

| Path                                                                            | Size      |
|----------------------------------------------------------------------------------|-----------|
| `data/raw/pvgis/db=PVGIS-SARAH3/config=fixed_35deg_south/lat=37.4000_lon=-5.0000/2023.json` | ~827 KB |
| `data/raw/pvgis/db=PVGIS-SARAH3/config=one_axis_horizontal_ns/lat=37.4000_lon=-5.0000/2023.json` | ~828 KB |

Sanity (parsed output):

| Metric                              | fixed_35deg_south | one_axis_horizontal_ns |
|--------------------------------------|-------------------|------------------------|
| Hourly rows                          | 8760              | 8760                   |
| Hours with P > 0                     | 4224              | 4206                   |
| Mean P when producing (W on 1 kWp)   | 403.3             | 496.4                  |
| Annual energy (kWh/kWp/yr)           | **1703.5**        | **2087.9**             |
| 1-axis uplift vs fixed               | —                 | **+22.6%**             |

Both numbers land inside the published Iberian benchmarks
(1650-1750 fixed, 2000-2100 tracker), and the +22.6% uplift sits at
the high end of the 15-25% utility-scale rule of thumb — consistent
with the very clear-sky climate of inland Andalucía.

## Cost estimate for the bulk run

- 2 configs × 1 site × 5 years (2019-2023) = **10 files**.
- ~830 KB each → ~8 MB total raw.
- Throttle 2 s between calls → ~20 s wall clock.
- Idempotent: re-running skips existing years.

## What we did NOT include and why

- **Multi-site per province**: out of scope. The Andalucía centroid is
  the reference for the PPA workbook; adding sites is a parameter
  change, no code change.
- **Bifacial gain / row-shading**: PVGIS does not model bifacial gain
  or row shading; the tolling workbook applies a constant uplift
  downstream.
- **2024 backfill via ERA5/CDS**: a separate scraper, deferred (see
  "Filling the 2024 gap" above).
- **Hourly DC vs AC**: PVGIS reports AC power after the supplied loss
  factor (14%). Sufficient for derivative pricing.

## Rate limits

JRC is undocumented but historically aggressive on burst calls. We use
a 2 s `min_interval_seconds` floor in the shared `ThrottledSession`.
