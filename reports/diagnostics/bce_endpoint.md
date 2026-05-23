# ECB euro-area sovereign yield curves — diagnostic

**Captured:** 2026-05-23 against `data-api.ecb.europa.eu`.
**Code:** `src/mibel_derivatives/data/bce.py`.

## Endpoint

The ECB Data Portal (replacement for the old SDW host since 2024)
exposes a SDMX 2.1 REST API. The yield-curve dataflow is `YC`.

URL template (GET, no auth required):

```
https://data-api.ecb.europa.eu/service/data/YC/<series_key>
    ?format=csvdata
    [&startPeriod=YYYY-MM-DD]
    [&endPeriod=YYYY-MM-DD]
```

Series-key skeleton:

```
B.U2.EUR.4F.G_N_<rating>.SV_C_YM.SR_<tenor>
```

| Dimension       | Value                                                  |
|-----------------|--------------------------------------------------------|
| FREQ            | `B` — business-daily                                   |
| REF_AREA        | `U2` — euro area (changing composition)                |
| CURRENCY        | `EUR`                                                  |
| PROVIDER_FM     | `4F` — ECB                                             |
| INSTRUMENT_FM   | `G_N_<rating>` — government, nominal, rating A=AAA / C=All |
| PROVIDER_FM_ID  | `SV_C_YM` — Svensson estimation, yield-to-maturity     |
| DATA_TYPE_FM    | `SR_<tenor>` — spot rate at the given maturity         |

Returned CSV columns we keep: `TIME_PERIOD` → `date`, `OBS_VALUE` →
`rate_pct` (rate in percent, e.g. 2.4331 = 2.43%).

## Curated set: 2 curves × 6 tenors = 12 series

| Curve     | Rating code | Description                                              |
|-----------|-------------|----------------------------------------------------------|
| `aaa`     | `G_N_A`     | AAA-rated euro-area sovereigns only                      |
| `all`     | `G_N_C`     | All euro-area sovereign issuers regardless of rating     |

Tenors: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y — the six on-the-run pillars used by
the rate-curve workbook for discount-factor interpolation.

## Smoke download (2026-05-23, year=2024)

12 series fetched, one CSV each under
`data/raw/bce/curve=<curve>/<tenor>.csv`. 255 observations per series
(TARGET2 business days in 2024).

Annual mean rate (%) by (curve, tenor):

| Tenor | AAA   | All   | AAA→All spread |
|-------|-------|-------|----------------|
| 1Y    | 2.910 | 3.016 |  +11 bp        |
| 2Y    | 2.535 | 2.717 |  +18 bp        |
| 3Y    | 2.348 | 2.617 |  +27 bp        |
| 5Y    | 2.244 | 2.662 |  +42 bp        |
| 7Y    | 2.287 | 2.803 |  +52 bp        |
| 10Y   | 2.416 | 3.008 |  +59 bp        |

Sanity checks: (i) the AAA curve is inverted at the short end and
humped at the belly, consistent with the 2024 ECB rate-cut cycle;
(ii) the All-issuers curve sits 11-59 bp above AAA with monotonically
widening spread by maturity — the expected sovereign credit-risk
term structure. Both checks pass.

## Cost estimate for the bulk run

Per series, full history is ~250 KB CSV (AAA goes back to 2004-09,
All-issuers to ~2004-09 too). 12 series × ~250 KB ≈ **3 MB total raw**.

With 1.5 s throttle: ~20 s wall clock. Idempotent on disk; re-running
overwrites only when `force=True` is set.

## What we did NOT include and why

- **Other tenors** (3M, 6M, 15Y, 20Y, 30Y): the rate-curve workbook
  interpolates from the 6-pillar grid above; adding tenors is a list
  change, no code change.
- **Instantaneous-forward (IF) and par-yield (PY) variants** of the
  same Svensson fit: redundant for discount-factor work since we can
  derive forwards from the spot curve.
- **Real (inflation-linked) yield curve**: the ECB publishes one but
  the derivatives module uses nominal rates throughout.
- **Country-specific sovereign curves** (Spain, Germany, etc.):
  out of scope; the model uses the euro-area aggregate curves.

## Rate limits

The Data Portal does not document a hard rate limit but bursts above
~5 req/s get throttled silently. We use `min_interval_seconds=1.5` in
the shared `ThrottledSession`.
