# EU ETS EUA primary-auction price — diagnostic

**Captured:** 2026-05-23 against `public.eex-group.com`.
**Code:** `src/mibel_derivatives/data/eua.py`.

## Source choice and rationale

The CONTEXT.md classifies EUA as a "datos parciales públicos" source
because the deep liquid market is the secondary one — ICE EUA Futures
and EEX EUA Futures, both behind paywalls.

The **EEX Primary Market Auctions** are the regulated, publicly
disclosed price-discovery channel mandated by the EU ETS Directive.
EEX runs them on behalf of every member state and publishes one annual
workbook per year containing every auction's clearing price, volume,
cover ratio, bid statistics, and country-revenue split. The URL is on
the `public.eex-group.com` subdomain — no authentication, no licence
key.

Auctions take place roughly three times per week (Tue, Wed, Thu)
except August. The primary clearing price is the daily reference for
that auction and tracks the secondary front-month future within a few
basis points on any given session (settlement-vs-clearing arbitrage is
trivial for the few large compliance buyers).

For our derivatives module (carbon cost in the CCGT stack, OMIE price
formation) the primary clearing price is a sufficient daily reference.
The full secondary curve (forward-month, year-ahead) is **not**
available for free and is left out of scope; the analytical model uses
flat-forward extrapolation from spot when needed.

## Endpoint

```
https://public.eex-group.com/eex/eua-auction-report/
    emission-spot-primary-market-auction-report-<YYYY>-data.<ext>
```

| Years      | Extension | Engine        |
|------------|-----------|---------------|
| 2017-2019  | `.xls`    | `xlrd 2.x`    |
| 2020-2026  | `.xlsx`   | `openpyxl`    |

Each workbook has one sheet, `Primary Market Auction`, with a
6-row header band of metadata; column headers live on row 6 (1-indexed)
and data starts on row 7.

## Schema evolution

EEX added a `Status` column in 2020. Pre-2020 reports only list
successful auctions — failed/cancelled ones were excluded from the
file altogether. The parser treats `Status` as optional and assumes
"successful" when the column is missing (legacy years). All other
columns we keep are stable across the 2017-2026 span.

Two contract codes appear:

- `T3PA` — Phase 3+ standard EU Allowance (EUA). This is what we keep.
- `EAA3` — Aviation EU Allowance (EUAA). Out of scope — the CCGT
  model uses the industrial EUA only.

## Columns kept

| Field                  | Source header              | Type     |
|------------------------|----------------------------|----------|
| `auction_date`         | Date                       | date     |
| `contract`             | Contract                   | string   |
| `status`               | Status (optional)          | string   |
| `clearing_price_eur_t` | Auction Price €/tCO2       | float    |
| `volume_t`             | Auction Volume tCO2        | float    |
| `cover_ratio`          | Cover Ratio                | float    |
| `country`              | Country                    | string   |

The euro sign sometimes arrives as U+FFFD mojibake in the header
band; `_normalize_header` fixes both that and embedded newlines
(e.g. `Spain\n(ES)` → `Spain (ES)`).

## Smoke download (2026-05-23)

| Year | File size | Rows (EUA, successful) | Date range          | Mean €/t | Min  | Max  |
|------|-----------|------------------------|---------------------|----------|------|------|
| 2019 | 269 KB    | 208                    | 2019-01-07 → 12-16  | 24.72    | 18.35| 29.46|
| 2024 |  89 KB    | 213                    | 2024-01-15 → 12-16  | 64.75    | 49.50| 75.35|

Both annual means land within 1-2% of the widely reported EU ETS
yearly averages for 2019 (~€25/t) and 2024 (~€65/t), which validates
the parser end-to-end.

## Cost estimate for the bulk run

- 6 years (2019-2024) × ~150 KB = **~900 KB total raw**.
- Throttle 1.5 s between calls → ~10 s wall clock.
- Idempotent: re-running skips existing year files.

## What we did NOT include and why

- **Secondary futures curve** (ICE EUA, EEX EUA): paywalled. Use
  flat-forward extrapolation from primary spot in the model.
- **EUAA (aviation)**: out of scope for the industrial CCGT carbon-cost
  leg.
- **Bid-level micro-structure**: kept aggregate fields only
  (clearing price, volume, cover ratio). Per-bidder columns and
  country-revenue split are in the raw XLSX if ever needed.
- **Secondary-vs-primary spread analysis**: deferred. Worth doing
  once the secondary curve becomes available.

## Licensing note

The EEX terms of use restrict *systematic republication* of bulk
EEX market data. Internal consumption for research and pricing is
permitted. We do not redistribute the raw XLSX files; commits include
only the parser, tests and this diagnostic.
