# TTF Dutch natural-gas front-month — diagnostic

**Captured:** 2026-05-23 against `query1.finance.yahoo.com`.
**Code:** `src/mibel_derivatives/data/ttf.py`.

## Source choice and rationale

The Title Transfer Facility (TTF) is the Dutch virtual trading point
and the European benchmark hub for natural gas. The deep liquid
market is the **ICE Endex TTF Front-Month Future**; daily settlements
are disseminated by ICE under commercial licence.

There is **no fully free, deep-history, daily-granular feed for TTF.**
The candidates evaluated were:

| Source                          | Granularity | History         | Status            |
|---------------------------------|-------------|-----------------|-------------------|
| EEX NGP TTF 60-day CSV          | daily       | last 60 days    | rolling only; SSL chain broken on `gasandregistry.eex.com` (intermediate cert not served) |
| Yahoo Finance v7 `/download`    | daily       | multi-year      | HTTP 401 (deprecated; auth required) |
| Stooq CSV download              | daily       | multi-year      | requires captcha-issued API key |
| Eurostat `nrg_pc_*`             | semestral   | multi-year      | granularity too coarse for pricing |
| ECB Data Portal CPP             | monthly     | multi-year      | series-key for gas not located |
| **Yahoo Finance v8 `/chart`**   | **daily**   | **multi-year**  | **public; accepted as prototype source** |

The Yahoo Finance v8 chart endpoint is still publicly reachable
without authentication and serves the `TTF=F` symbol — the ICE Endex
TTF front-month future as listed on NYMEX — in EUR/MWh with multi-year
daily history. We use it for backtesting and research.

## Licensing caveat — read before going to production

**Yahoo Finance is a prototype source, not a licensed production feed.**
Yahoo's terms of service restrict redistribution of its market data and
do not licence the feed for commercial production use. The
implementation in `src/mibel_derivatives/data/ttf.py` and this
diagnostic explicitly mark it as prototype.

A production deployment must replace this loader with a licensed
upstream — typically ICE Direct, Refinitiv (LSEG), or S&P Global
Commodity Insights (Platts). The parser would change only at the
HTTP and decode layers; the curated schema and downstream model
contract are designed to be source-agnostic (date + open/high/low/close
+ volume).

## Endpoint

```
https://query1.finance.yahoo.com/v8/finance/chart/<symbol>
    ?period1=<UTC seconds>
    &period2=<UTC seconds>
    &interval=1d
    &events=history
```

Symbol: `TTF=F` (currency EUR, exchange NYM = NYMEX listing of the
ICE Endex TTF front-month).

Response is JSON: `chart.result[0]` carries `timestamp[]` (Unix UTC)
and `indicators.quote[0]` with parallel `open/high/low/close/volume`
arrays. We floor each timestamp to the UTC date and drop rows whose
`close` is null.

A browser-like User-Agent is required — Yahoo blocks default Python
requests UA. Throttle: 1.5 s between calls is sufficient.

Implementation note: `period1` / `period2` must be UTC seconds. We
build them with `calendar.timegm(...)`, not `time.mktime(...)` —
the latter uses local time and drifts by the host's TZ offset, which
shifts the window boundary by one day in CET.

## Smoke download (2026-05-23, 2019-01-01 → 2024-12-31)

One JSON file under
`data/raw/ttf/symbol=TTF_F/2019-01-01_2024-12-31.json`, **156 KB**.
Parsed to **1510 daily rows**, 2019-01-02 → 2024-12-31.

Annual sanity (€/MWh close):

| Year | Trading days | Mean   | Min    | Max    |
|------|--------------|--------|--------|--------|
| 2019 |   252        |  14.60 |  9.38  |  23.00 |
| 2020 |   253        |   9.61 |  3.51  |  19.15 |
| 2021 |   252        |  47.65 | 15.52  | 180.27 |
| 2022 |   251        | 133.34 | 69.79  | 339.20 |
| 2023 |   251        |  41.30 | 23.10  |  74.30 |
| 2024 |   251        |  34.65 | 22.93  |  48.89 |

All means align with widely reported TTF historical levels (~€14 in
2019 normal year, ~€10 in COVID 2020, ~€48 in 2021 pre-crisis, ~€133
in the 2022 crisis year, ~€41 in 2023 normalization, ~€35 in 2024).
The series peak prints at **€339.20/MWh on 2022-08-26**, matching the
widely reported all-time-high close.

## Cost estimate for the bulk run

- One window per call; we pull the full 2019-2024 range in **one HTTP request**.
- ~156 KB JSON for six years.
- Throttle 1.5 s → 2 s wall clock.
- Re-running with `force=True` overwrites; idempotent on disk otherwise.

## What we did NOT include and why

- **Full forward curve** (M+2, M+3, Q+i, Y+i): not exposed by Yahoo;
  symbols only carry front-month. The model uses flat-forward
  extrapolation in the meantime.
- **Day-ahead and within-day spot** (separate from the future):
  publishing channel is ICE Endex / EEX only, both paywalled.
- **Other Yahoo symbols (NBP=F, JKM=F)**: out of scope for the
  Iberian-focused module.
- **Backfilled coverage from a licensed feed**: deferred to a future
  task once a production licence exists.

## Where this fits in the model

The TTF front-month series is the **European reference for natural-gas
cost** in the CCGT tolling stack. Iberian-physical fuel cost is read
from MIBGAS PVB (a separate scraper already in the repo); PVB ≈ TTF
+ basis. The two series together let the tolling model decompose
fuel cost into (continental gas reference, Iberian basis).

Until a licensed TTF feed replaces Yahoo, downstream consumers should
attach the explicit `source = "yahoo_chart_v8_prototype"` label to
any artifact derived from this scraper.
