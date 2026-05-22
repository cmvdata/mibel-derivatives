# Consolidated valuation report — mibel-derivatives

**Status:** placeholder. Populated by the valuation notebooks once the
spot, forward and gas calibrations land. The intended structure is:

1. Market state on the valuation date (OMIE spot, OMIP forward strip,
   MIBGAS/TTF curves, EUA price, discount curve).
2. Calibrated parameters (point estimates + standard errors) for the
   spot MRJD, the Schwartz–Smith two-factor forward model and the gas
   spot MRJD.
3. Valuation table per product (swing, Castejón I tolling, solar PPA)
   with confidence intervals from the Monte Carlo standard error.
4. First-order sensitivities (delta, gamma, vega, theta) per product
   by finite-difference shocks.
5. Scenario analysis (price shocks, correlation shocks, CCGT availability
   shocks for the tolling, capture-rate shocks for the PPA).
6. Audit trail: model version, calibration window, seed, links to the
   master tables (`asset_parameters`, `market_prices`, ...).
