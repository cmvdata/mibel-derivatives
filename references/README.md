# References

Working bibliography for phase 2 (modelling). PDFs of the five primary
papers are committed alongside this README; total ~4.5 MB. Additional
methodological references are listed below without PDFs. Each row of
the table records the bibliographic detail and the specific piece of
the modelling stack that the paper underpins.

| File | Authors | Year | Journal / source | Role in the project |
|---|---|---:|---|---|
| `j.1540-6261.1997.tb02721.x.pdf` | Eduardo S. Schwartz | 1997 | *The Journal of Finance* 52(3), 923-973 | Three-model framework (one-factor mean-reverting, two-factor with stochastic convenience yield, three-factor with stochastic rates) used as the **foundation for the electricity spot model** and as a comparison baseline before Schwartz-Smith. Kalman-filter parameter estimation methodology applied throughout phase 2. |
| `mnsc.46.7.893.12034.pdf` | Eduardo Schwartz, James E. Smith | 2000 | *Management Science* 46(7), 893-911 | Two-factor short-term/long-term model with directly observable factors (equivalent to Gibson-Schwartz but easier to identify). **Canonical reference for the Schwartz-Smith forward-curve model** fitted to OMIP M and YR futures. Also feeds the **tolling-agreement** valuation through its long-term equilibrium price factor. |
| `A_1013846631785.pdf` | Julio J. Lucia, Eduardo S. Schwartz | 2002 | *Review of Derivatives Research* 5, 5-50 (Kluwer) — "Electricity Prices and Power Derivatives: Evidence from the Nordic Power Exchange" | Electricity-specific spot price model with a **deterministic seasonal component plus stochastic Ornstein-Uhlenbeck factor(s)**. The seasonality decomposition (week / month / hour-of-day) and the one/two-factor variants applied to NordPool are the structural template for the MIBEL **spot model** and for the deterministic part of the **PPA forward curve**. |
| `Longstaff.pdf` | Francis A. Longstaff, Eduardo S. Schwartz | 2001 | *Review of Financial Studies* 14(1), 113-147 — "Valuing American Options by Simulation: A Simple Least-Squares Approach" | The Least-Squares Monte Carlo (LSM) regression-based estimator of conditional continuation values. **Backbone of the swing-option pricer** (multi-exercise, path-dependent) and of any **early-exercise leg in the tolling agreement**. |
| `the-iberian-electricity-market-analysis-of-the-risk-premium-3yye5p0ne4.pdf` | Márcio Ferreira, Helder Sebastião | 2018 | *Journal of Energy Markets* 11(2), 61-82 — "The Iberian electricity market: analysis of the risk premium in an illiquid market" | Empirical characterisation of the ex-post **MIBEL forward risk premium** (2006-2017): seasonality (winter ≫ summer), term-structure decay, predictability of the sign in the last 7 days before maturity. **Calibration target and sanity-check** for the risk-neutral / physical wedge implied by the Schwartz-Smith fit to OMIP futures. |

## Additional methodological references (no PDF on disk)

| Authors | Year | Citation | Role |
|---|---:|---|---|
| Álvaro Cartea, Marcelo G. Figueroa | 2005 | "Pricing in Electricity Markets: A Mean Reverting Jump Diffusion Model with Seasonality", *Applied Mathematical Finance* 12(4), 313-335 | Adds the **Kou-style asymmetric jump-diffusion** layer on top of the Lucia-Schwartz seasonal-plus-OU skeleton. Reference for `models/spot.py` jump component (Lucia-Schwartz did not model jumps). |
| Steven G. Kou | 2002 | "A Jump-Diffusion Model for Option Pricing", *Management Science* 48(8), 1086-1101 | The **asymmetric double-exponential jump-size distribution** used in `models/spot.py` `_mle_jumps`. |

## How phase 2 modules map to these papers

- `models/spot.py` — Lucia-Schwartz 2002 (seasonality + OU baseline) + Cartea-Figueroa 2005 (jump-diffusion extension) + Kou 2002 (jump-size distribution). Schwartz 1997 as the foundational OU treatment.
- `models/schwartz_smith/` — Schwartz-Smith 2000 (primary), Schwartz 1997 (precursor).
- `models/swing/` — Longstaff-Schwartz 2001.
- `models/tolling/` — Schwartz-Smith 2000 (long-term factor for fuel/power spread) + Longstaff-Schwartz 2001 (operational option exercise).
- `models/ppa/` — Lucia-Schwartz 2002 (seasonal forward curve) + Ferreira-Sebastião 2018 (MIBEL-specific risk premium correction).
