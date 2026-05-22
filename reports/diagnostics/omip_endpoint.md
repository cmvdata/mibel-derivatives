# OMIP forward-curve scraper — endpoint diagnostic

**Captured:** 2026-05-22 against `www.omip.pt/en/dados-mercado`.
**Code:** `src/mibel_derivatives/data/omip.py`.

## URL pattern

```
https://www.omip.pt/en/dados-mercado
  ?date=YYYY-MM-DD
  &product=EL
  &zone=ES
  &instrument=FTB
  &maturity={M|YR}
```

Parameters:

| Param | Values | Effect |
|---|---|---|
| `date` | `2019-01-02` .. `2024-12-30` (verified end-points) | Trade date — server picks D-1 if a non-business date is passed; not validated here. |
| `product` | `EL` | Electricity. The only family this module uses. |
| `zone` | `ES` | SPEL Spain. (`PT`, `FR` exist on the same page.) |
| `instrument` | `FTB` | SPEL Base Futures. (`FTP` peak exists.) |
| `maturity` | `M` / `YR` | Monthly contracts (~6 visible at any time) or yearly (~6-9 visible). |

## HTML structure

One `<table>` per page. Headers occupy two rows (column groups + sub-headers).
Data rows are **20 cells wide** with the following layout — the rest are visual separators:

| Cell idx | Field |
|---|---|
| 0 | Packed metadata: `ISIN Code:<isin>Nominal Fixo MWH:<n>Trading last day:<YYYY-MM-DD>Trading quotation:<quote><contract>` |
| 2 | Best bid (€/MWh) |
| 3 | Best ask (€/MWh) |
| 4 | Session volume (MWh) |
| 6 | Last deal price (€/MWh) — `n.a.` if no deal |
| 7 | Last deal time (`HHh:MMm:SSs` or `n.a.`) |
| 8 | Last deal volume (MWh) |
| 10 | Open interest (number) |
| 11 | Number of contracts |
| 12 | OTC volume (MWh) |
| 14 | **Reference D (€/MWh)** — this is the canonical curve point |
| 15 | Reference D-1 (€/MWh) |

Numbers are US-style decimal dot (`97.50`), not European comma despite the page being served from Portugal. The parser accepts both defensively.

## Coverage (verified via Manus survey, replicated 2026-05-22)

| Trade date | `maturity=M` rows | `maturity=YR` rows |
|---|---|---|
| 2019-01-02 | 6 | 5 |
| 2019-12-30 | 6 | 6 |
| 2020-12-30 | 6 | 9 |
| 2021-12-30 | 6 | 9 |
| 2022-12-30 | 6 | 9 |
| 2023-12-29 | 6 | 9 |
| 2024-12-30 | 6 | 9 |

Implications for Schwartz–Smith calibration: 6 monthly + ~9 yearly cross-sectional points per day; sufficient for a two-factor model but not for a fully smoothed curve.

## Cost estimate of a full 2019-2024 bulk

- Business days PT∪ES, 2019-01-01 → 2024-12-31: ~1,510.
- Two maturities per day: ~3,020 HTTP requests.
- With the default `ThrottledSession.min_interval_seconds = 1.0` and ~150 ms server latency: ~58 minutes wall-clock.
- Bytes per page: ~47 KB compressed; ~140 MB total for `data/raw/omip/`.

## Legal note

The page is public and parameterised but not a documented bulk API. Use a conservative rate, keep an audit log (`data/_manifest.jsonl`), do not redistribute the raw HTML, and review `omip.pt`'s terms of use before any commercial use of the curated curve.
