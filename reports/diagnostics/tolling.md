# Tolling agreement pricer — Pieza 4 (CCGT Castejón I, DP dispatch)

Dynamic-programming dispatch optimiser and Monte Carlo valuation for a
tolling agreement over Iberdrola's **Castejón I** combined-cycle gas
turbine (Navarra). Implementation in
`src/mibel_derivatives/products/tolling.py`. Validation suite:
`pytest tests/products/test_tolling.py`. Notebook:
`notebooks/04_tolling_castejon.ipynb`.

The pricer is **path-agnostic**: it consumes a `(n_paths, n_hours)` array
of power prices plus deterministic gas and carbon forward curves, and
returns the optimised dispatch value with diagnostics. The Monte Carlo
power model (Pieza 1 spot, optionally anchored to the Pieza 2
Schwartz-Smith forward) lives outside the pricer, so the dispatch logic is
validated on synthetic/deterministic paths and the stochastic model never
enters the pricing unit tests — the same design used for the swing pricer.

> **Layout note.** The implementation lives in
> `src/mibel_derivatives/products/tolling.py` and the tests in
> `tests/products/test_tolling.py`, as the brief asked and matching the
> sibling Pieza 5 `products/ppa.py`. The original `pricing/tolling.py`
> scaffolding placeholder is kept as a thin **compatibility shim** that
> re-exports the real pricer, so the legacy `pricing.tolling` import path
> still resolves (the same pattern `pricing/ppa_solar.py` uses).

## Asset specification (Castejón I)

Public reference parameters (Iberdrola Environmental Declaration 2024;
Aurecon/AEMO and NREL technical benchmarks). None is contractual; every
value is a public or benchmarked estimate, versioned in
`AssetParameters`.

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Gross design power | Pmax | 386.10 MW | Iberdrola Env. Declaration 2024 |
| Technical minimum | Pmin | 120 MW (~31% Pmax) | Aurecon 1+1 benchmark |
| Heat rate full load | HR(Pmax) | 6.55 GJ/MWh | derived from ~55% efficiency |
| Heat rate at Pmin | HR(Pmin) | 7.50 GJ/MWh | Aurecon part-load extrapolation |
| Start-up hot / warm / cold | — | 50 / 80 / 110 EUR/MW | Aurecon 2023 benchmarks |
| Minimum up-time | TMO | 4 h | modern CCGT standard |
| Minimum down-time | TMA | 2 h | modern CCGT standard |
| Ramp rate | — | 8 MW/min | single-shaft CCGT standard |
| CO2 intensity | ε(CO2) | 0.2016 tCO2/MWh_th | natural-gas combustion |

Gas is priced at **MIBGAS PVB** (Castejón pays PVB, not TTF) and carbon
at the **EUA primary-auction** clearing price.

## Spark spread and unit economics

The GJ/MWh heat rate is converted to MWh_th/MWh_e by dividing by
`GJ_PER_MWH = 3.6`, so it multiplies the EUR/MWh gas and EUR/t carbon
prices directly. The clean spark spread at power `P` is

```
spark(P) = power - (HR(P)/3.6)·gas - (HR(P)/3.6)·ε(CO2)·eua    [EUR/MWh]
```

and the hourly gross margin for running at `P` is
`P·power - thermal(P)·(gas + ε(CO2)·eua)` with thermal input
`thermal(P) = P·HR(P)/3.6` MWh_th. Heat rate is linear between the Pmin
and Pmax anchors and clamped outside the band; because `HR(Pmin) > HR(Pmax)`
the per-MWh spread is **worse at part load**, so the optimiser holds Pmax
whenever running is comfortably profitable.

## Dispatch optimiser — dynamic programming

The hour-by-hour dispatch is a single-unit commitment problem solved by
**backward dynamic programming over the operating state**, then a forward
replay of the optimal policy to recover the schedule and diagnostics
(standard formulation, e.g. Tseng & Barz 2002).

**Bellman state** (during hour `t`):
- `OFF(d)` — off for `d` consecutive hours (1..warm_max, saturating to a
  "cold" bucket); or
- `ON(k, u)` — running at power level `k` (a grid of `n_power_levels`
  points in [Pmin, Pmax]) for `u` consecutive hours (1..TMO, saturating).

**Transitions** (boundary `t → t+1`), each carrying the immediate
discounted margin and any start-up cost:
- `OFF(d) → OFF(min(d+1, ·))` — stay off.
- `OFF(d) → ON(Pmin, 1)` if `d ≥ TMA` — **start**: synchronise to the
  technical minimum. The synchronisation ramp is priced into the start-up
  cost (hot/warm/cold by `d`: `d ≤ 8h` hot, `≤ 48h` warm, else cold) and
  is exempt from the operational ramp.
- `ON(k, u) → ON(k', min(u+1, ·))` for every level `k'` with
  `|P(k') − P(k)| ≤ ramp·60` — operational ramp limit.
- `ON(k, u) → OFF(1)` if `u ≥ TMO` — **stop** (ramp-exempt).

The recursion `V_t(s) = DF_t·margin_t(s) + max_{s'} [ −DF_{t+1}·startup(s→s') + V_{t+1}(s') ]`
is vectorised across paths (each state's value is an `(n_paths,)` array).
The reported per-path value is the best transition from a pre-horizon
off-state (default: cold) into hour 0. `price_tolling` chunks the paths to
bound the policy-tensor memory and averages the discounted optimal gross
margin, netting the present value of the fixed capacity fee.

**Foresight.** The DP is solved **per realised path with perfect
foresight** — the deterministic-equivalent valuation and an **upper bound**
on the non-anticipative Longstaff-Schwartz recourse value that CONTEXT.md
lists as the further extension. The gap is the value of foresight; it is
documented, not hidden, and the LSMC variant is a clean (deferred)
extension.

**Ramp at hourly resolution.** At 8 MW/min the hourly ramp budget is
480 MW > the 266 MW band, so the operational ramp is slack by default; it
binds at finer resolution or for a slower unit. The test exercises it with
a reduced ramp (1 MW/min) and checks the schedule climbs one level per
hour.

## Validation of the constraints

`tests/products/test_tolling.py` — the required six dispatch tests plus API,
input-validation, pricing-assembly and an end-to-end Monte Carlo case.

| Test | What it asserts |
|---|---|
| `test_dispatch_respects_pmin_pmax` | every on-hour power ∈ [Pmin, Pmax] and on the level grid; off-hours = 0. |
| `test_minimum_uptime_downtime` | on a cycling path, every run length ≥ TMO and every inter-run gap ≥ TMA. |
| `test_ramp_constraint` | with a slow ramp, consecutive on-hour power moves ≤ ramp·60; reaching Pmax takes strictly longer than under the fast default. |
| `test_startup_cost_charged` | a single profitable block ⇒ exactly one cold start and `startup_cost == cold·capacity`; a prohibitive start cost suppresses a marginal one-hour run. |
| `test_monotonicity_in_spark_spread` | shifting every power price up cannot lower the optimal value. |
| `test_heat_rate_degradation_at_pmin` | `HR(Pmin) > HR(Pmax)`; per-MWh spark lower at Pmin; steady-state dispatch holds Pmax. |

Supporting tests: public-API surface, frozen dataclasses, default spec
matches Castejón I, asset/shape/discount-factor validation, fixed-fee
netting + chunking invariance, discounting reduces value, and a
`monte_carlo`/`slow` end-to-end case driving `models.spot.simulate`.

### Smoke (2026-05-30)

```
ruff check src tests        -> All checks passed!
pytest -m "not slow"        -> 177 passed, 14 deselected
pytest tests/products/test_tolling.py -> 14 passed (incl. the slow MC case)
```

## Reported-valuation results (Q1 2025, N_PATHS=1000)

Power paths from the Pieza 1 spot model fit on 2023-2024 OMIE (274 jumps),
started from the last fitted slow-factor level; gas and carbon flat at
their last-30-day means (PVB 46.0 EUR/MWh, EUA 65.3 EUR/t). Fixed fee
30 000 EUR/MW/year.

| Quantity | Value |
|---|---|
| Option value (gross dispatch margin) | **41.19 EUR million** (± 3.48 MC s.e.) |
| Fixed fee PV | 2.86 EUR million |
| Net value to offtaker | 38.33 EUR million |
| Implied EUR/MW/year | ~432 600 |
| Mean capacity factor | 33.3% |
| Mean running hours / 2160 | 762 |
| Mean number of starts | 30.0 |
| Mean start-up cost | 0.86 EUR million |

The ~33% capacity factor is consistent with the ~40% profitable-day share
in the 2022-2024 history (figure `reports/figures/tolling_spark_history.png`).
The **absolute** level is a model demonstration: it reflects the 2023-2024
calibration window (simulated mean power ~119 EUR/MWh), not a market
forward for Q1 2025; a production run would anchor the level to the
Pieza 2 forward curve and use N_PATHS=50000 (`requires_pod`).

## Sensitivities (±30%, 400 paths)

Figure `reports/figures/tolling_sensitivity.png`. Option value (EUR
million):

| Shock | Gas (PVB) | Carbon (EUA) | Heat rate |
|---|---|---|---|
| −30% | 55.70 | 49.74 | 58.46 |
| −15% | 51.40 | 48.69 | 52.56 |
|   0% | 47.68 | 47.68 | 47.68 |
| +15% | 44.45 | 46.71 | 43.61 |
| +30% | 41.66 | 45.78 | 40.23 |

Value falls with gas, carbon and heat rate (each raises the fuel/carbon
cost or worsens efficiency, shrinking the spread). **Heat rate is the
dominant lever**, then gas, then carbon — the efficiency assumption drives
the toll economics, which is why it is flagged as the key model input.

## Deferred

- **Non-anticipative valuation.** Longstaff-Schwartz recourse on the
  commitment state (a lower bound; the foresight gap vs. this DP).
- **Stochastic gas/carbon.** Gas is currently a deterministic curve;
  CONTEXT.md's mean-reverting jump-diffusion gas spot with a calibrated
  power-gas correlation is the next layer.
- **Forward anchoring.** Wire the Q1 2025 level to the Pieza 2 curve via
  `spot.fit_with_forward_anchor` for a market-consistent absolute value.
