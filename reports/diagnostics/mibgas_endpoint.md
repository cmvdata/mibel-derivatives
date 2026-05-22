# MIBGAS annual XLSX downloader — endpoint diagnostic

**Captured:** 2026-05-22 against `www.mibgas.es`.
**Code:** `src/mibel_derivatives/data/mibgas.py`.

## Path that didn't work

The AJAX endpoint `/en/ajax/table/daily-price/<product>/export?date=DD/MM/YYYY` returns a CSV — but it serves only **current-window** prices. Calling it with a historical date (e.g. `02/01/2024`) returns the right header but the values are dashes. Useful for a live ticker, not for back-filling 2019-2024. Discarded.

## Path that worked

Annual workbook per year, browsable through `/en/file-access?path=AGNO_<YYYY>/XLS`:

```
https://www.mibgas.es/en/file-access/MIBGAS_Data_<YYYY>.xlsx?path=AGNO_<YYYY>/XLS
```

| Year | XLSX | CSV |
|---|---|---|
| 2019 | yes | no |
| 2020 | yes | no |
| 2021–2024 | yes | yes |

The CSV variant has a single price column and is therefore lossy; the parser always reads the XLSX (~1–2 MB per year, ~10 MB for the full 2019–2024 bulk).

## Sheet evolution (2019 → 2024)

The MIBGAS schema is not stable across years. Two layouts observed:

| Year(s) | PVB sheet name | Indices sheet name | Price column names |
|---|---|---|---|
| 2019, 2020 | `Trading Data PVB` | `Indices` | `Daily Reference Price [EUR/MWh]`, `Daily Auction Price`, `Last Daily Price`, `Maximum Daily Price`, `Minimum Daily Price`, `Daily Volume Traded` |
| 2024 | `Trading Data PVB&VTP` | `MIBGAS Indexes` | `Reference Price [EUR/MWh]` (drop the "Daily " prefix), idem for auction / last / max / min / volume |

The parser handles both via per-column alias lists (`_PVB_COLUMN_ALIASES`, `_INDICES_COLUMN_ALIASES`) and picks the first existing sheet name from a candidate list. Adding a new year that introduces another rename is a one-line change.

## Products kept

The curated `mibgas_pvb` table includes every PVB-hub product the workbook publishes — among them:

| Code | Meaning |
|---|---|
| `GWDES` | Within-day |
| `GDAES_D+1` | Day-ahead (the canonical spot reference) |
| `GDAES_D+2`, `D+3` | Future days |
| `GWES_W+1` | Weekend |
| `GW_BoMES` | Balance of month |
| `GMAES`, `GMAES_M+i` | Month-ahead |
| `GQES_Q+i` | Quarter-ahead |
| `GSAES_S+i` | Semester-ahead |
| `GYES_Y+i` | Year-ahead |

The MIBGAS-ES daily spot index lives in `mibgas_indices` (separate parquet); it is the volume-weighted average across all PVB matches on the gas day.

## Cost estimate

6 downloads (2019..2024), ~12 MB total. Single-file requests over the throttled session: under a minute end-to-end.

## License note

The workbook's first sheet is a Spanish-language disclaimer stating that MIBGAS data is free to use as long as content is not altered; MIBGAS does not guarantee continuity of structure or format. Commercial uses should be cross-checked with `mibgas.es` terms.

The `_manifest.jsonl` ledger records each download with URL, SHA-256 and bytes for audit.
