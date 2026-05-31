# Production Schwartz-Smith fit — summary

- **Run wall**: 472 s (7.9 min)
- **n_obs**: 25638
- **n_dates**: 2192
- **log_likelihood**: 23403.76
- **rmse_log_M**: 0.1789
- **rmse_log_YR**: 0.0425

## Stochastic factors

| Param | Estimate | Bound | At bound? |
|---|---|---|---|
| kappa | +0.3838 | [0.1, 5.0] | no |
| sigma_chi | +1.0214 | [0.05, 5.0] | no |
| sigma_xi | +0.2305 | [0.05, 2.0] | no |
| rho | -0.8058 | [-0.99, 0.99] | no |
| mu_xi (physical) | +0.0257 | [-0.5, 0.5] | no |
| mu_xi_star (Q) | -0.0088 | [-0.5, 0.5] | no |
| lambda_chi | +0.4439 | [-0.5, 0.5] | no |
| epsilon_m | +0.1808 | [0.0001, 0.5] | no |
| epsilon_yr | +0.0438 | [0.0001, 0.5] | no |
| epsilon_spot | +0.5000 | [0.0001, 0.5] | yes (upper) |

## Seasonal dummies (Feb..Dec, January reference)

| Month | s |
|---|---|
| Feb | -0.0146 |
| Mar | -0.1482 |
| Apr | -0.2572 |
| May | -0.1850 |
| Jun | -0.0860 |
| Jul | -0.0413 |
| Aug | -0.0819 |
| Sep | -0.0730 |
| Oct | -0.0868 |
| Nov | -0.0449 |
| Dec | -0.0309 |

## Filtered state at end of sample

- last trade date: `2024-12-31 00:00:00`
- chi_T = -0.7114
- xi_T  = +4.9105

## Next steps

1. Copy `data/curated/forward_fit_production.pkl` back to the laptop.
2. Run `pytest tests/models/test_forward_validation.py -v` locally;
   V5 (composition spot MAE) and V6 (composition forward) will now
   execute against this fit (the `requires_full_fit` marker no
   longer skips).
3. Review the per-bucket RMSE above; if M-bucket RMSE exceeds 0.25
   the calibration probably needs a richer model (see Limitations §
   in forward_model_calibration.md).
