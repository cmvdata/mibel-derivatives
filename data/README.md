# `data/` — three-zone lakehouse

This directory holds every external dataset consumed by the module.
**Nothing under `raw/`, `staging/`, `curated/` or `_manifest.jsonl` is
tracked in git** (see `.gitignore`). Only this README and per-source
schema files are committed.

## Zones

| Zone | Purpose | Format | Idempotent? |
|---|---|---|---|
| `raw/` | Bytes exactly as downloaded, partitioned by source and key | HTML, JSON, CSV, ZIP | yes — write once per (source, partition key) |
| `staging/` | Parsed and typed, one parquet per source | parquet (snappy) | yes — re-derived from `raw/` |
| `curated/` | Analysis-ready tables, one parquet per logical dataset | parquet (snappy) | yes — re-derived from `staging/` |

Each scraper in `src/mibel_derivatives/data/<source>.py` exposes:

- `fetch_*` — download to `raw/` (idempotent: skips files that exist
  unless `force=True`).
- `parse_*` — read `raw/` → `staging/`.
- `build_curated_*` — read `staging/` → `curated/`.

## Partitioning

| Source | Raw partition |
|---|---|
| OMIP | `raw/omip/maturity=<M|YR>/trade_date=YYYY-MM-DD/page.html` |
| MIBGAS | `raw/mibgas/<product>/year=YYYY/<filename>.csv` |
| OMIE | `raw/omie/` — symlink-like reuse of `mibel-forecasting` cache (no copy) |
| ESIOS | `raw/esios/indicator=<id>/year=YYYY/month=MM/response.json` |
| PVGIS | `raw/pvgis/config=<fixed|tracking_1axis>/lat<>_lon<>_<start>_<end>.json` |
| BCE | `raw/bce/series=<series_key>/year=YYYY/response.csv` |

## Manifest (`_manifest.jsonl`)

Append-only JSONL — one line per artifact saved to `raw/`. Schema:

```json
{
  "timestamp_utc": "2026-05-22T11:23:00Z",
  "source": "omip",
  "url": "https://www.omip.pt/...",
  "params": {"date": "2024-12-30", "maturity": "M"},
  "http_status": 200,
  "bytes": 53241,
  "sha256": "...",
  "raw_path": "raw/omip/maturity=M/trade_date=2024-12-30/page.html"
}
```

The manifest is the audit ledger: every byte written to `raw/` is
provenance-tagged. Re-runs append new lines; nothing is overwritten.

## Datasets at a glance

| Dataset | Source | Period | License notes |
|---|---|---|---|
| OMIE day-ahead Spain (indicator 600) | reused from `mibel-forecasting` cache | 2019-01 → 2024-12 | ESIOS open data |
| OMIP forward strip (FTB M / YR) | omip.pt public web | 2019-01 → 2024-12 | public web; document non-commercial use |
| MIBGAS PVB spot + futures | mibgas.es public CSV | 2019-01 → 2024-12 | MIBGAS public data; verify before commercial use |
| ESIOS ancillary services panel (~10 ind.) | ESIOS API (token) | 2019-01 → 2024-12 | open data, requires token |
| PVGIS solar Andalucía (37.4°N, -5.0°W) | JRC PVGIS v5_2 API | 2019-01 → 2023-12 (TMY) | JRC open |
| ECB AAA and all-issuers yield curves | data-api.ecb.europa.eu (SDMX-CSV) | 2019-01 → 2024-12 | ECB open data |
| EUA CO₂ futures | partial public (EEX free quotes) | — | bulk public unavailable; see `reports/diagnostics/eua_unavailable.md` |
| TTF gas futures | partial public | — | bulk public unavailable; see `reports/diagnostics/ttf_unavailable.md` |
