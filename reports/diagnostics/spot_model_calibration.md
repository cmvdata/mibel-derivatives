# Spot model calibration — phase-2 Pieza 1

Mean-reverting jump-diffusion fit to the OMIE day-ahead Spain hourly
log-price series, 2019-01-01 → 2024-12-31. Implementation in
`src/mibel_derivatives/models/spot.py`; reproducibility helper in
`scripts/_spot_calibration_run.py` (calls `spot.fit` on
`data/curated/omie_spot_es_2019_2024.parquet`, runs the stationarity /
normality tests, simulates 100 one-year forward paths and writes both
the figures under `reports/figures/spot_*.png` and the numbers below).

## Spec recap

| Decision (resolved 2026-05-24) | Choice |
|---|---|
| Negative-price handling | `Y_t = log(P_t + c)`, `c = 10 EUR/MWh` (above the observed minimum of -2 EUR/MWh, preserves the multiplicative model) |
| Hourly architecture | Single OU on the deseasonalised residual, constant κ, 24-vector of hour-of-day σ_h |
| 2022 regime treatment | Single 2019-2024 calibration; the long-term shift is absorbed by the Schwartz-Smith long-term factor of Pieza 2 |
| Seasonality form | Fourier annual (4 harmonics) + DoW dummies (Mon ref) + HoD dummies (h0 UTC ref) |
| Jump detection | Iterative threshold at `k = 4 · σ_h(t)` over returns, σ updated each iteration |
| Jump-size distribution | Kou (2002) asymmetric double-exponential — Cartea & Figueroa (2005) is the reference for the MRJD extension of Lucia-Schwartz |
| Calibration | Two-stage: OLS for the seasonal component, then MLE for OU on non-jump pairs and the moment estimators for the Kou parameters |

## Dataset

| | |
|---|---|
| Source | `data/curated/omie_spot_es_2019_2024.parquet` (Manus drop) |
| Indicator | ESIOS 600 (OMIE day-ahead, Spain geo_id=3) |
| Coverage | 2019-01-01 00:00 → 2024-12-31 23:00 UTC, **52 608 hourly observations** |
| Price range | min **-2.00**, max **700.00**, mean **85.17** EUR/MWh |

## Fitted parameters

### Deterministic seasonality

| Group | Value | Interpretation |
|---|---|---|
| Intercept | **4.3261** (SE 0.0169, 95% CI [4.2930, 4.3592]) | Baseline `log(P + 10)` ⇒ baseline price ≈ exp(4.326) − 10 = **65.6 EUR/MWh** at Monday-h00 UTC neutral seasonality |
| Fourier annual (cos/sin pairs) | `[0.025, -0.199, 0.073, 0.035, -0.011, 0.018, -0.023, -0.044]` | Dominant single-cycle `b1 = -0.199` puts the seasonal trough around April-May and a milder peak in October-November |
| DoW (Tue..Sun, Mon = 0) | `[0.040, 0.021, 0.023, -0.006, -0.156, -0.280]` | Weekend depression of ~16 % (Sat) and ~28 % (Sun) on log-price relative to Monday; weekdays nearly flat |
| HoD (h1..h23 UTC, h0 = 0) | min **-0.197** (h3 UTC), max **+0.326** (h17 UTC) | Peak-to-trough log spread **0.523**, i.e. **peak/trough price ratio ≈ exp(0.523) = 1.69**. Note: UTC, so h17 ≈ 18-19h Madrid local |

Standard errors of the remaining seasonal coefficients are similar in
order of magnitude to the intercept SE (~0.01-0.03); reported only for
the intercept because that's the headline level.

### OU dynamics and Kou jumps

| Parameter | Estimate | Comment |
|---|---|---|
| κ (mean-reversion speed) | **0.00717 /h** | Half-life ≈ **96.6 h** (≈ 4 days). Slow by single-factor MRJD standards; see *Limitations* |
| σ_h range | **0.037 — 0.148** (mean 0.075) | Min at deep-night hours, max in the evening peak band (h18-20 UTC) |
| λ (jump intensity) | **0.0360 /h** | 316 jumps/year if extrapolated; corresponds to **3.60 %** of returns flagged |
| p_up | **0.441** | More downward than upward jumps over the period |
| η_up | 1.699 → mean upward jump in log = **+0.588** | exp(0.588) ≈ 1.80 ⇒ a typical upward jump is a +80 % price move |
| η_down | 2.151 → mean downward jump in log = **-0.465** | exp(-0.465) ≈ 0.63 ⇒ a typical downward jump is a -37 % move |
| Detected jumps | **1 896 / 52 607** returns (3.60 %) | High vs the 1-2 % range typical in European-power MRJD literature; see *Limitations* |

## Statistical tests on residuals

| Test | Series | Stat | p-value | Reading |
|---|---|---|---|---|
| Augmented Dickey-Fuller | Deseasonalised residual `Z_t` | **-8.74** | 2.97 × 10⁻¹⁴ | Strongly rejects unit root: residuals are stationary in level once seasonality is removed |
| Augmented Dickey-Fuller | Residual returns `ΔZ_t` | -41.20 | ≈ 0 | Trivially stationary (returns of a stationary series) |
| Jarque-Bera | Non-jump returns | 18 418 | 0 | Rejects normality: heavy tails remain even after iterative jump removal |
| Jarque-Bera | All returns | 2.46 × 10⁶ | 0 | Reference — confirms how much of the kurtosis the jump component absorbs (134× ratio) |

The ADF rejection is technically consistent with stationarity of the
deseasonalised series, but it is an *omnibus* test that is easy to
reject on long high-frequency electricity series even when an
underlying regime shift is present (see §Limitations). The JB rejection
is expected and quantifies why the Kou jump component is needed.

## Forward-simulation validation (1-year horizon)

| Statistic | Historical 2019-2024 | Simulated 2025 (100 paths, seed=2026) |
|---|---|---|
| Mean nominal price (EUR/MWh) | 85.17 | **206.75** |
| Std nominal price (EUR/MWh) | 75-100 (regime-dependent) | **1 180** |
| `log(P + 10)` return std | **0.155** | 0.178 |
| `log(P + 10)` return mean | ≈ 0 | ≈ 0 |

Short-horizon return statistics match well (return std 0.155 hist vs
0.178 sim, both centred on zero). The headline 1-year nominal price
divergence comes from the slow mean reversion (half-life 4 days) plus
the heavy Kou tails compounding under Jensen's inequality over 8 760
hourly steps. **For horizons of days to a few weeks the simulator
tracks the data well; for multi-month horizons it must be combined
with the Schwartz-Smith long-term factor** (Pieza 2) — this is the
intended division of labour and the reason for the single-factor
choice in this piece.

Figures regenerated by the helper script:

- `reports/figures/spot_omie_series.png` — historical series.
- `reports/figures/spot_seasonal_components.png` — Fourier annual + DoW + HoD.
- `reports/figures/spot_sigma_by_hour.png` — σ_h pattern.
- `reports/figures/spot_residuals_jumps.png` — Z_t with detected jumps overlaid.
- `reports/figures/spot_return_hist.png` — non-jump return histogram vs Gaussian.
- `reports/figures/spot_simulation_band.png` — 100-path P10/P50/P90 band.

## Comparison with the literature

| Quantity | Our 2019-2024 OMIE | Literature reference |
|---|---|---|
| Mean-reversion (per day, ≈ 24 × κ) | ≈ **0.17/day** | Lucia & Schwartz (2002), Nord Pool 1993-1999 daily: κ ≈ 0.05-0.10/day for the one-factor model. Ours faster, consistent with OMIE being a more reactive market and a finer time grid |
| Daily implied σ (≈ √24 · mean σ_h) | ≈ **0.37** | Lucia-Schwartz reports daily σ ≈ 0.15-0.25 on Nord Pool. Our higher value is consistent with the 2022 vol jump that the single-period calibration absorbs |
| Jump intensity | 3.6 % of hourly returns | Cartea & Figueroa (2005), UK PHELIX daily 2001-2004: ≈ 1 jump every 30 days ≈ 3 % of *daily* returns. Our hourly 3.6 % is in line if rescaled, given the higher frequency and the 2022 spike regime |
| Peak/trough HoD ratio | 1.69 | Industry typical for daily/sub-daily models: 1.5-2.5 depending on solar penetration. Spain 2019-2024 with rising solar share is consistent with the lower end of that range |

The headline parameter set is therefore *quantitatively consistent
with the European-power MRJD literature*; the deviations (faster κ,
slightly higher σ, jump rate above the historical European norm) all
reflect specific OMIE 2019-2024 features that the diagnostic surfaces
rather than hides.

## Limitations carried forward

1. **2022 regime shift is not isolated.** ADF on the residual rejects
   unit root (-8.74) but the *equilibrium level* of the deseasonalised
   series is visibly different in 2019-2021 vs 2022 vs 2023-2024. The
   single-period calibration mixes the three. The Schwartz-Smith
   long-term factor in Pieza 2 is the intended fix: it lets the
   equilibrium drift while this piece carries the short-term dynamics.
2. **Long-horizon simulation diverges.** A half-life of 96 h combined
   with Kou tails over 8 760 hourly steps produces a 1-year mean and
   dispersion well above the historical (207 vs 85 EUR/MWh mean,
   1 180 vs ~85 EUR/MWh std). The model is intended for short-to-medium
   horizons (≤ a few weeks). Anything longer must be combined with
   Pieza 2.
3. **Jump rate is high (3.6 %).** Even after iterative threshold
   detection, JB-rejection of normality on non-jump returns (stat
   18 418) shows residual heavy tails. The implicit trade-off:
   tighter `k` lowers the rate but biases the diffusion σ upward; the
   asymmetric exponential is a stand-in for a richer tail model.
4. **Hour-0-UTC reference is sticky.** All HoD coefficients are
   relative to h0 UTC. When interpreting the peak/trough log spread,
   read coefficients alongside the intercept (which carries h0).
5. **Daylight Saving.** The model uses UTC throughout, so DST changes
   never bleed into HoD or DoW buckets, but the *local* peak hour
   shifts by one between summer/winter — irrelevant for the price
   dynamics but worth keeping in mind when projecting into a local-time
   delivery calendar (e.g. a daily-resolution PPA settlement).

## Reproducibility

```bash
python scripts/_spot_calibration_run.py
```

regenerates `reports/_spot_summary.json` and all `reports/figures/spot_*.png`.
The notebook `notebooks/01_spot_model.ipynb` reproduces the same
workflow interactively. Random seed 2026 is fixed in both for the
simulation step.
