# mibel-derivatives

Pricing and sensitivities of energy derivatives on the Iberian electricity
market (MIBEL). One of five modules of the `mibel-portfolio` umbrella
repository; sibling of `mibel-forecasting`, `mibel-trading`, `mibel-risk`,
`mibel-congestion-monitor`.

## Status

**Scaffolding only.** No models implemented yet. See `CONTEXT.md` for the
full module specification and the open implementation questions raised at
scaffolding time.

## Scope

Three products from a typical utility / trading-house Middle Office book:

1. **Swing options** on MIBEL electricity, priced by Longstaff–Schwartz
   Monte Carlo.
2. **Tolling agreements** on CCGT plants, using Castejón I (Iberdrola,
   Navarra, ~370 MW net) as the reference asset. Pricing by LSMC on the
   spark spread, honouring operational constraints via dynamic
   programming.
3. **Solar PPAs** with capture-price decomposition and cannibalisation
   sensitivity. Reference asset: 100 MW PV plant in Andalucía.

Output is a set of valuations consistent with observable market prices
(OMIE spot, OMIP forward, MIBGAS) plus first-order sensitivities
(delta, gamma, vega, theta) by finite-difference shocks.

Portfolio-level risk (VaR, ES, PFE, CVA) is **out of scope** — it lives
in the sibling repo `mibel-risk`, which consumes the valuations and
sensitivities produced here.

## Repository layout

```
src/mibel_derivatives/
  data/         OMIE / OMIP / MIBGAS / TTF loaders
  models/       spot (MRJD), forward (Schwartz–Smith 2F), gas
  pricing/      swing, tolling (CCGT), ppa_solar
  calibration/  Kalman filter for Schwartz–Smith, MLE for MRJD
  evaluation/   finite-difference Greeks
tests/          pytest suite
notebooks/      calibration and valuation notebooks (one per product)
reports/        consolidated_report.md + diagnostics/
scripts/        _dump_code_bundle.py (glob-discovery audit bundle)
```

## Setup

Python 3.11 (pinned), `uv` as the package manager:

```bash
uv sync --extra dev --extra notebooks
uv run pytest -m "not slow and not monte_carlo"
```

CI runs `ruff`, `mypy` (non-strict initially) and the fast pytest subset
on every push and PR.

## Specification

See `CONTEXT.md` for:

- Spot model (mean-reverting jump-diffusion with seasonal long-run mean).
- Forward model (Schwartz–Smith two-factor + deterministic seasonality).
- Gas model (MIBGAS short/medium, TTF long-end).
- Castejón I reference parameters and their public sources.
- Solar PPA structure (80/20 fixed/spot, no caps/floors in v1).
- Five master tables for auditability (`asset_parameters`,
  `market_prices`, `dispatch_states`, `settlement_reconciliation`,
  `compliance_log`).
- Regulatory anchoring (OMIE, REE, CNMC, REMIT, BOE references).
- Bibliography (Schwartz 1997; Schwartz–Smith 2000; Lucia–Schwartz 2002;
  Longstaff–Schwartz 2001; NREL; Aurecon/AEMO; Iberdrola Environmental
  Declaration 2024).

## License

MIT.
