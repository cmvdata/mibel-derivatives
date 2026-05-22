# ESIOS curated indicator panel — diagnostic

**Captured:** 2026-05-22 against `api.esios.ree.es` with the token in `.env`.
**Code:** `src/mibel_derivatives/data/esios.py`.

## Why a curated subset

ESIOS exposes thousands of indicators. For the tolling and ancillary
analysis this module needs a tight, hand-picked panel covering:

1. day-ahead and intraday price formation,
2. the four ancillary-services families: secondary band + secondary
   energy, tertiary, imbalance settlement, technical restrictions,
3. one aggregated cost component, and
4. the CCGT final hourly schedule needed by
   `settlement_reconciliation` (program leg).

Each indicator below was confirmed live via `/indicators/<id>` on
2026-05-22.

## Curated indicators (geo=3 Spain, hourly UTC)

| ID    | Label                          | ESIOS name                                                                    | Unit    |
|-------|--------------------------------|--------------------------------------------------------------------------------|---------|
| 600   | `omie_day_ahead_es`            | Precio mercado SPOT Diario                                                    | €/MWh   |
| 612   | `omie_intraday_s1_es`          | Precio mercado SPOT Intradiario Sesión 1                                      | €/MWh   |
| 634   | `secondary_band_down_price`    | Precio reserva de regulación secundaria a bajar                               | €/MW    |
| 682   | `secondary_energy_up_price`    | Precio de energía de regulación secundaria a subir                            | €/MWh   |
| 683   | `secondary_energy_down_price`  | Precio de energía de regulación secundaria a bajar                            | €/MWh   |
| 677   | `tertiary_up_marginal`         | Precio marginal regulación terciaria a subir de activación programada (AP)    | €/MWh   |
| 676   | `tertiary_down_marginal`       | Precio marginal regulación terciaria a bajar de activación programada (AP)    | €/MWh   |
| 686   | `imbalance_up_price`           | Precio de cobro desvíos a subir                                               | €/MWh   |
| 687   | `imbalance_down_price`         | Precio de pago desvíos a bajar                                                | €/MWh   |
| 793   | `tech_restrictions_pbf`        | Precio medio cuarto horario componente restricciones PBF (contratación libre) | €/MWh   |
| 794   | `tech_restrictions_realtime`   | Precio medio cuarto horario componente restricciones tiempo real              | €/MWh   |
| 10211 | `ancillary_total`              | Precio medio horario final suma de componentes                                | €/MWh   |
| 79    | `ccgt_p48_program`             | Generación programada P48 Ciclo combinado                                     | MWh     |

13 indicators in total.

Indicator 600 doubles as the gap-fill source for `omie.py` — the ESIOS
cache uses the same filename convention as `mibel-forecasting`
(`i<ID>_geo<GEO>_<YYYY>_<MM>.parquet`), so the OMIE loader picks up the
freshly-pulled months automatically without configuration.

## Revision history of the curated set

**2026-05-22 — added 682, 683, 79 after review feedback.**

Initial panel had only the secondary band-down price (634, €/MW) and no
secondary energy activation price. The band is paid for reserving
capacity; the energy is paid for *activating* it. For a tolling model
that values flexibility, the band price alone is half the picture.
Verified that:

- IDs 635/636 (initially proposed) return HTTP 404 — they do not exist.
- The actual secondary-energy prices are 682 (up) and 683 (down), both €/MWh.

For settlement reconciliation, the initial panel had no scheduling
leg. The CONTEXT.md `settlement_reconciliation` table is defined as
"programa vs medida vs liquidación", and the program leg is missing
from any of the price-only indicators. Added:

- 79 = Generación programada **P48** Ciclo combinado. P48 is the
  definitive post-intraday-sessions program; PHF1..PHF7 are the
  successive updates after each intraday session and would inflate the
  panel without proportionate value. If finer reconciliation is needed
  later, IDs 9 (PBF day-ahead Ciclo combinado), 114 (PHF1), 324 (PHF7)
  and 1156 (Generación medida Ciclo combinado) can be added — they all
  exist and are live.

## What we did NOT include and why

- **Secondary band upward price**: no symmetric "Precio reserva de
  regulación secundaria a subir" indicator exists in ESIOS — the
  marginal band price is published only as a downward reference. ID
  632 returns *Asignación* (allocated quantity), not a price. The
  panel keeps only 634 for band capacity.
- **PHF1..PHF7 broken down per technology**: useful for intraday-by-intraday
  reconciliation but not part of this curated panel. The P48 final program
  is sufficient for the program-vs-measure-vs-liquidation chain.
- **Generation-mix and demand series**: out of scope for this module.
- **Quarter-hour granularity**: ESIOS quarter-hour series (P48 included)
  are collapsed to hourly mean in `_fetch_month` for panel consistency.
  The raw cache files keep the original resolution.

## Cost estimate (full 2019-2024 bulk)

- 13 indicators × 72 months = 936 cache files to fetch.
- ESIOS responses average ~50 KB; total disk ~50 MB.
- With a conservative throttle (1.5 s between calls) the bulk run takes
  ~25 minutes.
- Idempotent: re-running skips months already cached.

## Token

The token lives in `.env` (`ESIOS_API_TOKEN`); request one by email to
consultasios@ree.es. Same token as `mibel-forecasting` — `.env` is
copied verbatim during repo bootstrap.
