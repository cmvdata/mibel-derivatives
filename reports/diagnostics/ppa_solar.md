# Solar PPA pricer — phase-2 Pieza 5 (capture-price Monte Carlo)

Monte Carlo pricer for a utility-scale solar power-purchase agreement on
the Andalucía reference plant. Implementation in
`src/mibel_derivatives/products/ppa.py`. Validation suite:
`pytest tests/products/test_ppa.py`. Valuation workbook:
`notebooks/05_ppa_solar.ipynb`.

> **Module location.** The brief's `pricing/ppa_solar.py` was the original
> scaffolding placeholder (`raise NotImplementedError`). The pricer lives
> in the new `products/` package (`products.ppa`) alongside the other
> structured products; `pricing.ppa_solar` is kept as a thin compatibility
> shim that re-exports the public API so the historical import path still
> works. New code should import from `mibel_derivatives.products.ppa`.

## Asset spec

| Item | Value | Source |
|---|---|---|
| Technology | PV, fixed-tilt 35deg south (also one-axis N-S) | PVGIS geometry catalogue |
| Nameplate | 100 MW AC | brief |
| Site | Andalucía, lat 37.4, lon -5.0 (near Sevilla) | `data.pvgis.LAT/LON_ANDALUCIA` |
| Resource data | PVGIS SARAH3, hourly, 2019-2023 | `data/curated/pvgis_panel.parquet` |
| Mean capacity factor | 0.1916 fixed-tilt / 0.2355 one-axis | computed (panel) |
| Equiv. full-load hours | ~1679 / ~2063 h | mean CF x 8760 |

Contract (v1): **80% fixed + 20% spot**, no caps / floors, 10-year tenor.
A fraction `fixed_pct` of every generated MWh is sold at the fixed
`strike`; the remaining `spot_pct` settles at the hourly MIBEL spot. An
optional `cost_ppa` (EUR/MWh) nets O&M / offtake costs.

## Methodology

Two coupled risk factors are simulated over a representative contract year
(8760 hours), then the annual cashflow is present-valued over the tenor.

**Generation paths** (`simulate_production_paths`). The deterministic
PVGIS hourly capacity-factor profile `cf_h` (a typical year, averaged
across 2019-2023 by month-day-hour) is scaled by nameplate and multiplied
by a per-path lognormal **resource** factor (interannual irradiance
variability, default sigma 5%) and an optional per-hour lognormal noise.
Both multipliers are mean-one, so `E[g_h] = capacity * cf_h` and the plant
factor is preserved. Output is clipped to `[0, capacity]`.

**Price paths.** Two generators, same overlay:

- `simulate_price_paths` (reduced-form, default / tests) —
  `P_h = baseload * diurnal_h * (1 - beta * solar_excess_h) * noise_h`,
  where `diurnal_h` is a two-harmonic duck curve (midday trough, evening
  peak), `solar_excess_h` is the demeaned, max-normalised capacity factor,
  `beta` is the cannibalisation strength, and `noise_h` is a mean-one
  lognormal AR(1) multiplier. Self-contained, so the heavy structural fits
  stay out of the test suite.
- `simulate_price_paths_from_spot` (reported valuation) — drives
  `models.spot.simulate` (Pieza 2: mean-reverting Kou jump-diffusion) and,
  via `initial_theta`, the OMIP forward level of `models.forward`
  (Pieza 1), then applies the **same** cannibalisation overlay
  `1 - beta * solar_excess_h`. The structural spot model is solar-blind;
  the overlay is what injects the negative price/generation correlation a
  renewable-heavy merit order produces.

**Capture price.** The generation-weighted average spot,
`capture = sum_h g_h S_h / sum_h g_h`, compared to the simple baseload
mean `S_bar = mean_h S_h`. Because generation is concentrated in the
depressed midday hours, `capture < baseload`; the ratio is the **capture
rate**.

**Cashflow & discounting.** Per the brief, the per-hour payoff is

```
payoff_h = g_h * (strike*fixed_pct + spot_h*spot_pct - cost_ppa)
```

summed to an annual cashflow per path. The PV multiplies by a flat
mid-year annuity factor `sum_{y=1..D} (1+r)^-(y-0.5)`, `r = 7%` real
(Iberian utility-scale PV WACC mid-point). v1 treats each contract year as
the same annual scenario times the annuity; interannual resampling is a
documented extension.

The pricer is **injectable**: `price_ppa` consumes supplied
`price_paths` / `production_paths` verbatim when given (path-agnostic
mode, used by the tests and by an OMIP-consistent run), and only falls
back to the internal generators otherwise. A single `seed` drives the two
internal generators with independent sub-streams (price `seed`,
production `seed+1`) so the factors are not spuriously correlated.

## Capture-price validation

Historical capture (OMIE day-ahead Spain x PVGIS fixed-tilt production,
year by year) versus the simulated model, calibrated to the historical
mean baseload:

| | capture ratio |
|---|---|
| Historical 2019-2023 (mean) | ~0.74 (notebook §2) |
| Simulated (`cannibalisation=0.45`) | **0.735** |

The reduced-form `cannibalisation` coefficient is the lever that
reproduces the realised capture discount; raising it traces forward
renewable-penetration scenarios. Even at `beta = 0` the simulated ratio is
0.908, not 1 — the deterministic duck-curve shape alone depresses midday
capture; cannibalisation deepens it.

## Results (fixed-tilt, 100 MW, 80/20, strike 55 EUR/MWh, 10y, seed 2024)

`N_PATHS = 1000` smoke (2026-05-30):

```
annual generation   : 167,818 MWh
capture price        : 44.55 EUR/MWh  (baseload 60.62, ratio 0.735)
PPA value (10y PV)   : 64.51 M EUR  +/- 0.10
levelised value      : 52.91 EUR/MWh
```

### Sensitivities (notebook §4)

| Strike [EUR/MWh] | 45 | 50 | 55 | 60 | 65 |
|---|---|---|---|---|---|
| Value [M EUR] | 54.76 | 59.63 | 64.51 | 69.39 | 74.26 |

Linear and strictly increasing in the strike (the fixed leg has positive
weight) — pinned by `test_monotonicity_in_strike`.

| fixed_pct | 1.0 | 0.8 | 0.5 | 0.2 | 0.0 |
|---|---|---|---|---|---|
| Value [M EUR] | 67.06 | 64.51 | 60.69 | 56.87 | 54.32 |
| Levelised [EUR/MWh] | 55.00 | 52.91 | 49.78 | 46.64 | 44.55 |

At 100% spot the levelised value collapses to the capture price (44.55);
at 100% fixed it equals the strike (55.00). More fixed share raises value
whenever `strike > capture`.

| Cannibalisation beta | 0.0 | 0.3 | 0.45 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| Capture [EUR/MWh] | 54.51 | 47.87 | 44.55 | 41.23 | 36.81 |
| Capture ratio | 0.908 | 0.792 | 0.735 | 0.678 | 0.603 |
| Value [M EUR] | 66.94 | 65.32 | 64.51 | 63.70 | 62.62 |

Deeper cannibalisation cuts the capture price (and thus the 20% spot leg),
but the 80% fixed leg cushions the deal — value falls only ~6% across the
full beta sweep, which is precisely the price-risk insurance the fixed
share provides.

One-axis tracking (mean CF 0.2355) lifts annual generation to ~206 GWh and
the 10-year value to ~79.3 M EUR at the same capture ratio (0.733).

## Validation evidence

`tests/products/test_ppa.py` — 19 fast tests + 1 `requires_pod`/`slow`
end-to-end. Run: 19 passed (fast lane).

| Check | What it pins |
|---|---|
| Pure-fixed structure | 100% fixed ⇒ value independent of the spot path (two price scenarios price identically) and levelised value == strike. |
| Pure-spot structure | 100% spot, deterministic generation ⇒ levelised value == capture price exactly; capture < baseload. |
| Capture < baseload | Cannibalisation ⇒ `capture_ratio < 1`; flat shape + `beta=0` collapses the ratio to 1 (control). |
| Seed reproducibility | Same seed ⇒ identical value and per-path array; a different seed moves the estimate. |
| Monotonicity in strike | Value strictly increasing in the fixed strike (paths held fixed). |
| Cost pass-through | `cost_ppa` reduces value by exactly `cost * PV(generation)`. |
| Duration / annuity | Value linear in the mid-year annuity factor. |
| Production clipping & mean-one | Generation in `[0, capacity]`; mean across paths recovers `capacity * cf`. |
| Input validation | Structure must sum to 1, pct in `[0,1]`, plant factor in `(0,1)`, matched path shapes, profile in `[0,1]`. |
| Compat shim | `pricing.ppa_solar.price_ppa is products.ppa.price_ppa`. |
| Reported size (`requires_pod`) | `N_PATHS=50000`: positive value, `capture_ratio < 1`, standard error < 2% of value. |

## Reported-valuation usage

For an OMIP-consistent reported valuation (CONTEXT.md § Parámetros
numéricos: 50 000 paths), feed `price_ppa` paths from
`simulate_price_paths_from_spot` (Pieza 2 + the forward-anchored level of
Pieza 1) instead of the reduced-form generator, plus the curated PVGIS
profile. That run depends on the curated parquets (laptop/pod only), so it
belongs in the valuation notebook and the `requires_pod` test, not the CI
fast lane.

## Deferred

- **Caps / floors** — v1 has none; a collar adds a per-hour
  `min(max(S, floor), cap)` on the spot leg, a clean extension.
- **Interannual resampling** — v1 reuses one annual scenario per path
  times the annuity; drawing D independent years per path widens the tail
  and captures diversification.
- **Structural cannibalisation** — the `cannibalisation` coefficient is
  calibrated to the historical capture ratio; a penetration-driven model
  (capture as a function of installed solar GW) is the natural successor.
- **First-order Greeks** via `evaluation.sensitivities` (delta to the
  forward level, vega) by re-pricing under shocked paths — pending that
  module.
