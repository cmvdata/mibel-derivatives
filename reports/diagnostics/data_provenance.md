# Phase-1 dataset provenance

Phase 1 of mibel-derivatives requires eight independent datasets covering
2019-2024 for Iberian power-derivative pricing. These come from two
distinct producers — an external data drop (referred to here as the
"Manus drop") and our own in-repo scrapers under
`src/mibel_derivatives/data/`. This document records exactly which
producer supplied which curated file, the coverage observed on disk,
and the gaps that motivated each choice.

## Curated-layer manifest

| # | Source | Curated file(s) in `data/curated/` | Producer | Rows | Coverage |
|---|---|---|---|---:|---|
| 1 | OMIP — Iberian SPEL Base futures (FTB, zone=ES, M+YR) | `omip_forward_2019_2024.parquet` | Manus drop | 23 848 | 2019-01-01 → 2024-12-31 |
| 2 | MIBGAS — PVB / VTP trading data + MIBGAS-ES Index | `mibgas_pvb.parquet`, `mibgas_indices.parquet` | `data.mibgas` scraper (this repo) | 36 841 + 2 848 | 2019-2024 |
| 3 | OMIE — day-ahead Spain spot (via ESIOS indicator 600) | `omie_spot_es_2019_2024.parquet` | Manus drop | 52 611 | 2019-01-01 → 2024-12-31 |
| 4 | ESIOS — 13-indicator ancillary / imbalance / P48 panel | `esios_panel.parquet` | `data.esios` scraper, via `scripts/bulk_download.py` | 52 585 hourly × 13 cols | 2019-2024 |
| 5 | PVGIS — hourly PV output, Andalucía centroid (fixed + 1-axis) | `pvgis_panel.parquet` | `data.pvgis` scraper, via `scripts/bulk_download.py` | 87 648 | 2019-2023 (SARAH3 cutoff) |
| 6 | ECB — euro-area sovereign yield curve (AAA + all-issuers, 6 tenors) | `bce_yield_curve.parquet` | `data.bce` scraper, via `scripts/bulk_download.py` | 66 576 | 2019-2024 |
| 7 | EU ETS — EUA primary auctions (EEX annual workbooks) | `eua_primary_auction.parquet` | `data.eua` scraper, via `scripts/bulk_download.py` | 1 281 | 2019-2024 |
| 8 | TTF — front-month future (Yahoo TTF=F, prototype source) | `ttf_front_month.parquet` | `data.ttf` scraper, via `scripts/bulk_download.py` | 1 510 | 2019-2024 |

## Where each producer fits

**Manus drop (external)**

A separate provider (Manus) delivered a packaged dataset under the local
folder `mibel_datasets_2019_2024/` (gitignored). The drop contains a
curated parquet plus raw HTML/CSV/JSON evidence per source, an
`INFORME_RESUMEN_COBERTURA.md` summary and `dataset_summary.csv`. Two of
its curated parquets were imported verbatim into `data/curated/`: OMIP
forward (98.3 % non-null `reference_d_eur_mwh`) and OMIE day-ahead Spain
(52 611 hourly rows, prices -2 → 700 €/MWh, mean 85.18 — consistent with
the 2019-2024 Spanish day-ahead history).

**`scripts/bulk_download.py` (this repo)**

The bulk runner produced the curated parquets for ESIOS, PVGIS, ECB, EU
ETS EUA and TTF in a single 2026-05-23 run (22 min wall, dominated by
ESIOS at the API-mandated 1.5 s throttle). The raw artefacts live under
`data/raw/{esios,pvgis,bce,eua,ttf}/` (gitignored). Re-running the
script is idempotent — each scraper skips inputs already present.

**`data.mibgas` scraper (this repo)**

MIBGAS was run separately on 2026-05-24 after inspection showed that the
Manus drop's `mibgas_spot_2019_2024.parquet` and
`mibgas_forward_2019_2024.parquet` carried 24 112 rows with **zero
populated prices** — the row skeleton (trade_date × zone × product) was
present but every `delivery` and `price_eur_mwh` cell was null. The
in-repo scraper downloads the six annual MIBGAS workbooks
(`MIBGAS_Data_<YYYY>.xlsx`) from `mibgas.es/en/file-access`, parses the
"Trading Data PVB&VTP" and "Indices" sheets and writes two curated
parquets with real prices for daily reference, auction, last, max and
min — 36 841 PVB rows + 2 848 MIBGAS-ES index rows.

## Not integrated from the Manus drop

Manus also shipped curated parquets for ESIOS, PVGIS, ECB, EU ETS EUA
and TTF (folders `04_esios_adjustments` through `08_ttf_gas`). These
were **deliberately not promoted to `data/curated/`**: the in-repo
scrapers had already produced more complete or schema-controlled
versions in the bulk run, and the Manus drop has two notable coverage
issues that warranted keeping our own panel as the canonical curated
layer:

- `04_esios_adjustments`: 7 of the 9 requested ancillary indicators
  (635, 636, 644, 645, 685, 1739, 10211) returned 0 rows after Manus's
  monthly retries — the same indicators our `data.esios` scraper pulls
  cleanly once `geo_ids[]` is dropped for the system-wide series (see
  commit `a246967`).
- `05_pvgis_solar_andalucia`: the originally shipped 17 544-row series
  covered only 2019-2020. A documented re-extraction with PVGIS v5.3 +
  SARAH3 brings Manus's coverage to 2019-2023 (matches ours), but the
  curated parquet duplicates what our bulk already wrote.

The Manus folders for these five sources remain on disk as an external
cross-check and audit trail; they are referenced here only for
traceability and never read by the modelling code.

## Layout on disk

```
data/
├── README.md                # zone layout (committed)
├── raw/                     # all downloaded artefacts (gitignored)
│   ├── bce/                 # ECB CSV per (curve, tenor)
│   ├── esios/cache/         # monthly parquet per (indicator, geo)
│   ├── eua/                 # EEX annual workbooks
│   ├── mibgas/              # MIBGAS annual XLSX
│   ├── omie/                # local cache mirror (esios indicator 600)
│   ├── pvgis/               # per-config annual JSON
│   └── ttf/                 # Yahoo v8 chart payload
└── curated/                 # 9 parquet files listed above (gitignored)

mibel_datasets_2019_2024/    # Manus drop, gitignored, kept as immutable source
├── 01_omip/                 # → integrated into data/curated/
├── 02_mibgas/               # NOT used (empty prices); see above
├── 03_omie_esios_600/       # → integrated into data/curated/
├── 04_esios_adjustments/    # cross-check only
├── 05_pvgis_solar_andalucia # cross-check only
├── 06_ecb_yield_curves/     # cross-check only
├── 07_co2_eua/              # cross-check only
└── 08_ttf_gas/              # cross-check only
```

## How to reproduce / refresh

The two integration paths are documented in code:

- For the five HTTP-pulled sources (ESIOS, PVGIS, ECB, EU ETS EUA, TTF):
  `python scripts/bulk_download.py` re-pulls from source. Idempotent.
- For MIBGAS: `python -c "from mibel_derivatives.data import mibgas;
  [mibgas.fetch_annual(y) for y in range(2019, 2025)];
  mibgas.build_curated(2019, 2024, write=True)"` — six XLSX downloads
  followed by a sheet parse.
- For OMIP and the OMIE 15-min ESIOS panel from the Manus drop: the raw
  HTMLs (3 138 files) and per-month ESIOS JSONs (72 files) live in
  `mibel_datasets_2019_2024/{01_omip,03_omie_esios_600}/`. If a future
  refresh from source is required, `scripts/bulk_download.py` has TODO
  stubs marking where `run_omip` and `run_omie` would plug in; the
  in-repo modules `data.omip` and `data.omie` already implement the
  fetchers and parsers.

## Caveats carried into phase 2

1. **TTF** uses Yahoo's `TTF=F` chart endpoint — fine for the prototype,
   not licenced for production. Replace with ICE Endex / Refinitiv /
   S&P Commodity Insights before any redistribution. The curated
   schema (`date + OHLCV`) is source-agnostic to make this swap mechanical.
2. **OMIP** from Manus carries 13 of the 17 fields our scraper extracts.
   Missing from the integrated parquet: `best_bid_eur_mwh`,
   `best_ask_eur_mwh`, `last_deal_volume_mwh`, `quotation`. If a model
   needs bid-ask spread it must either be reconstructed from
   `reference_d` / `reference_d_minus_1`, or the Manus HTMLs reparsed
   with `data.omip.parse_page`.
3. **PVGIS** stops at 2023 — the v5.3 + SARAH3 endpoint refuses 2024
   (`Incorrect value. Please, enter an integer between 2005 and 2023`).
   This is a structural source limitation, not a scraper bug.
