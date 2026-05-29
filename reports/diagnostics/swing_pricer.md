# Swing option pricer — phase-2 Pieza 3 (volume-grid LSMC)

Longstaff-Schwartz Monte Carlo pricer for the annual MIBEL electricity
swing option (CONTEXT.md § Swing option). Implementation in
`src/mibel_derivatives/pricing/swing.py`. Validation suite:
`pytest tests/pricing/test_swing.py`.

The pricer is **path-agnostic**: it consumes a `(n_paths, n_steps)` array
of settlement prices and the per-step discount factors, and returns the
contract value plus policy diagnostics. The Monte Carlo price model
(`models.spot.simulate`, or a forward-anchored composition) lives outside
the pricer, so the valuation logic is validated against closed-form
benchmarks with synthetic paths and the stochastic model never enters the
pricing unit tests.

## Contract

| Term | Symbol | Meaning |
|---|---|---|
| Strike | `K` | Fixed price the holder pays per MWh (call style: in the money when `price > K`). |
| Daily cap | `volume_per_right` (q) | MWh delivered by one exercised right (one decision step). |
| Annual cap | `n_rights * q` | Maximum total volume; `n_rights` (R) is the number of exercise rights. |
| Take-or-pay | `min_rights` (m) | Minimum rights that must be exercised. Default 0. |

A step is a delivery day for the annual contract, priced at that day's
settlement price; for a MIBEL valuation the daily price is the baseload
mean of the hourly spot paths.

## Resolution of the two open spec items

CONTEXT.md left two items "deferred to design phase". The conservative
reading is taken here and documented so the choice is auditable.

| Open item | Conservative resolution | Why |
|---|---|---|
| **Exercise frequency / cap basis** | One decision per step; each step delivers 0 or `q` MWh (the daily cap). Annual cap = `R · q`. | The per-step payoff `q·(S−K)` is **linear in the nominated volume**, so the optimal nomination is bang-bang (0 or the cap) except where the global budget forces the boundary. The unit-right volume grid is therefore *exact* when the annual cap is an integer multiple of the daily cap — the only case the pricer accepts (it raises otherwise). No continuous-nomination machinery is built for a problem whose optimum is bang-bang. |
| **Take-or-pay minimum + penalty** | Off by default. When set, enforced as a **hard volume floor**: the holder must exercise ≥ `m` rights, achieved by forcing exercises once `remaining_steps == remaining_required`. No penalty *price* is invented. | An unspecified under-take penalty is most conservatively read as a mandatory take: the holder bears the realised (possibly negative) intrinsic value of the forced exercises, not an arbitrary fee. A penalty-price variant is a clean extension (terminal shortfall cost) but is **deferred** until the contract specifies one. |

## Algorithm

Volume-grid Longstaff-Schwartz (Boogert & de Jong 2008, extending
Longstaff & Schwartz 2001). State = (step `t`, cumulative rights used
`c ∈ {0..R}`). Backward induction carries, for every level `c`, the
realised value-to-go along each path:

1. At step `t`, regress the carried value of each level on a polynomial
   basis of the step price (one shared design matrix per step, applied to
   all `R+1` levels at once). This estimates the continuation value
   `Ĉ(t, c)`.
2. Decide per path and level: exercise iff
   `pay_t + Ĉ(t, c+1) > Ĉ(t, c)`, where `pay_t = q·(S_t − K)·DF_t`.
   Take-or-pay forces exercise when the remaining steps equal the
   remaining required rights.
3. Update the carried value with the **realised** continuation (not the
   regression) — the standard Longstaff-Schwartz device that keeps the
   estimator unbiased.

The headline `price` is the backward estimate `mean(value at level 0,
step 0)`. A forward pass applies the fitted regressions as a fixed policy
to produce `policy_value` (a lower-bound cross-check) and
`expected_rights_used`.

Basis: monomials `[1, x, x², …]` in the per-step-normalised price
(centre + scale), default degree 3. Normalising keeps the least-squares
normal equations well conditioned across the wide MIBEL price range
(single digits to several hundred EUR/MWh in 2022).

### Complexity

Per step: one pseudo-inverse of an `(n_paths × basis)` design and two
matmuls; the level dimension is fully vectorised. An annual daily swing
(365 steps) with 2 000 paths prices in ~1.3 s; the 30-step identity
benchmark with 20 000 paths in ~0.5 s (measured 2026-05-29, see below).

## Validation evidence

`tests/pricing/test_swing.py` — 16 fast tests + 1 `monte_carlo`
end-to-end. Run: 16 passed (fast lane), 1 passed (monte_carlo).

| Check | What it pins |
|---|---|
| Single step = intrinsic | One step, one right ⇒ `price == q·E[(S−K)⁺]·DF` to floating point (terminal continuation is exactly zero). |
| Global cap non-binding | `R = n_steps` ⇒ value collapses to the sum of independent daily calls `Σ_t q·E[(S_t−K)⁺]·DF_t`. LSMC matches the brute-force benchmark to **0.00 %** (20 000 paths, 30 steps: 479.278 vs 479.278). |
| Single-right bounds | `R = 1` value bracketed by best-fixed-time (lower) and perfect-foresight (upper); strictly above the fixed-time policy. |
| Monotonicity | Price non-decreasing in `n_rights`. |
| Take-or-pay | All-OTM + `m>0` ⇒ value negative, exactly `m` rights used, optimiser picks the least-bad days (beats the worst-`m` selection). |
| Discounting | Price scales linearly in the discount factors. |
| Determinism | Pure function of the input paths — identical output on repeat. |
| End-to-end (`monte_carlo`) | Daily-aggregated paths from `models.spot.simulate`; positive value, forward policy within 10 % of the backward estimate, rights within the cap. |

### Smoke (2026-05-29)

```
[identity] P=20000 T=30 R=30
  LSMC price = 479.2780 +/- 0.9310 EUR ; benchmark 479.2780 ; rel_err 0.00%
  E[rights] = 17.52 (ITM steps)            ; wall 470 ms

[annual]  365 daily steps, 2000 paths, R=60, K=57.39 EUR/MWh (median)
  price = 45822.91 +/- 135.10 EUR
  E[rights] = 59.7/60  -> the annual cap binds, the option selects the
  best ~60 delivery days; price_swing wall 1.34 s, spot.simulate 1.33 s.
```

The annual case has the global cap binding (59.7 of 60 rights used),
which is precisely the regime the volume-grid state is needed for — a
sum-of-daily-calls shortcut would over-value it.

## Reported-valuation usage

For a reported valuation (CONTEXT.md § Parámetros numéricos: 50 000
paths), feed `price_swing`:

- paths from `models.spot.simulate` (or `fit_with_forward_anchor` for
  OMIP-curve consistency), aggregated to the delivery-step resolution;
- `discount_factors` from the EURIBOR + OIS curve (flat extrapolation
  beyond 5 y).

Both inputs depend on the curated parquets, which are laptop-only; the
reported run therefore belongs in a valuation notebook, not the test
suite.

## Deferred

- Continuous (non-bang-bang) nomination — unnecessary while the payoff is
  linear; would only matter with a convex per-step penalty.
- Take-or-pay **penalty price** (vs the hard floor implemented here) —
  add a terminal shortfall cost once a contract specifies one.
- First-order Greeks via `evaluation.sensitivities` (delta/vega by
  re-pricing under shocked paths) — pending that module's implementation.
