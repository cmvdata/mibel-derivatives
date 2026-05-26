"""Mandatory validation tests for the Schwartz-Smith forward model.

Marked both ``slow`` and ``monte_carlo`` so they can be excluded from
the fast CI path. The per-test wall is dominated by the bounded MLE on
the full OMIP series (~1 600 trade dates × ~15 contracts each); a
single fit takes several minutes on a modern laptop.

The tests skip themselves cleanly if the curated OMIP and OMIE
parquets are not on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mibel_derivatives.models import forward


OMIP_PATH = Path("data/curated/omip_forward_2019_2024.parquet")
OMIE_PATH = Path("data/curated/omie_spot_es_2019_2024.parquet")


@pytest.fixture(scope="module")
def omip_and_omie() -> tuple[pd.DataFrame, pd.Series]:
    if not OMIP_PATH.exists():
        pytest.skip(f"{OMIP_PATH} not present; skipping forward-validation suite")
    if not OMIE_PATH.exists():
        pytest.skip(f"{OMIE_PATH} not present; skipping forward-validation suite")
    omip = pd.read_parquet(OMIP_PATH)
    omie_raw = pd.read_parquet(OMIE_PATH)
    omie_raw = omie_raw.set_index("datetime_utc").sort_index()
    hourly = omie_raw[omie_raw.index.minute == 0]["price_eur_mwh"]
    # Daily-mean spot, indexed by calendar date (tz-naive midnight).
    daily_mean = (
        hourly.tz_convert(None)
        .groupby(hourly.tz_convert(None).index.normalize())
        .mean()
        .rename("price")
    )
    daily_mean.index.name = "trade_date"
    return omip, daily_mean


@pytest.fixture(scope="module")
def omip_fit(omip_and_omie: tuple[pd.DataFrame, pd.Series]) -> forward.SSFit:
    omip, omie_daily = omip_and_omie
    return forward.fit(omip, omie_daily, max_iter=300)


@pytest.mark.slow
@pytest.mark.monte_carlo
def test_calibrated_params_inside_bounds(omip_fit: forward.SSFit) -> None:
    """V1: every numerical parameter ends inside its declared bound."""
    p = omip_fit.params
    assert forward.KAPPA_BOUNDS[0] < p.kappa < forward.KAPPA_BOUNDS[1]
    assert forward.SIGMA_CHI_BOUNDS[0] < p.sigma_chi < forward.SIGMA_CHI_BOUNDS[1]
    assert forward.SIGMA_XI_BOUNDS[0] < p.sigma_xi < forward.SIGMA_XI_BOUNDS[1]
    assert forward.RHO_BOUNDS[0] < p.rho < forward.RHO_BOUNDS[1]
    assert forward.MU_XI_BOUNDS[0] < p.mu_xi < forward.MU_XI_BOUNDS[1]
    assert forward.MU_XI_STAR_BOUNDS[0] < p.mu_xi_star < forward.MU_XI_STAR_BOUNDS[1]
    assert forward.LAMBDA_CHI_BOUNDS[0] < p.lambda_chi < forward.LAMBDA_CHI_BOUNDS[1]
    assert forward.EPSILON_BOUNDS[0] < p.epsilon_m < forward.EPSILON_BOUNDS[1]
    assert forward.EPSILON_BOUNDS[0] < p.epsilon_yr < forward.EPSILON_BOUNDS[1]


@pytest.mark.slow
@pytest.mark.monte_carlo
def test_forward_curve_fit_quality(omip_fit: forward.SSFit) -> None:
    """V2: in-sample RMSE of log F per bucket below 0.10 (≈10% in price)."""
    assert 0 < omip_fit.rmse_log_m < 0.10, f"rmse_log_m={omip_fit.rmse_log_m:.4f}"
    assert 0 < omip_fit.rmse_log_yr < 0.10, f"rmse_log_yr={omip_fit.rmse_log_yr:.4f}"


@pytest.mark.slow
@pytest.mark.monte_carlo
def test_model_spot_reproduces_omie_daily_mean(
    omip_fit: forward.SSFit,
    omip_and_omie: tuple[pd.DataFrame, pd.Series],
) -> None:
    """V3: model F(t, T=0) ≈ daily-mean OMIE on the trade dates.

    Computed across the whole window: the MAE between model-implied
    log spot and historical log daily-mean spot must be ≤ 0.10 (≈ 10%
    multiplicative).
    """
    _, omie_daily = omip_and_omie
    p = omip_fit.params
    chi = omip_fit.state_chi
    xi = omip_fit.state_xi
    common_dates = chi.index.intersection(omie_daily.index)
    chi_c = chi.loc[common_dates]
    xi_c = xi.loc[common_dates]
    omie_c = omie_daily.loc[common_dates]
    # Model spot per date: ln F(t, T=0) = chi_t + xi_t + s(month_t).
    months = pd.DatetimeIndex(common_dates).month.to_numpy()
    s_vals = np.array([
        forward._seasonal_value(p.seasonal_dummies, int(m)) for m in months
    ])
    model_log_spot = chi_c.values + xi_c.values + s_vals
    hist_log_spot = np.log(omie_c.values.astype(float))
    mae = float(np.mean(np.abs(model_log_spot - hist_log_spot)))
    assert mae < 0.10, f"MAE log spot vs OMIE daily mean = {mae:.4f}"


@pytest.mark.slow
@pytest.mark.monte_carlo
def test_implied_forward_curve_matches_observed_at_end_of_history(
    omip_fit: forward.SSFit,
    omip_and_omie: tuple[pd.DataFrame, pd.Series],
) -> None:
    """V4: at the last trade date in the sample, the model-implied
    forward curve matches the observed OMIP YR strip within 15 % per
    contract (multiplicative)."""
    omip, _ = omip_and_omie
    last_date = omip_fit.trade_dates[-1]
    p = omip_fit.params
    chi_t = float(omip_fit.state_chi.iloc[-1])
    xi_t = float(omip_fit.state_xi.iloc[-1])

    # Observed YR contracts on the last date.
    omip_last = omip[omip["trade_date"] == str(last_date.date())].copy()
    omip_last = omip_last[omip_last["maturity_bucket"] == "YR"]
    omip_last = omip_last.dropna(subset=["reference_d_eur_mwh"])
    obs_long = forward.prepare_observations(omip_last)
    if len(obs_long) == 0:
        pytest.skip("no YR contracts on the last trade date — cannot validate V4")

    rel_errs = []
    for row in obs_long.itertuples():
        model_log = forward.futures_log_price(
            p, chi_t, xi_t, float(row.tau), is_yearly=True,
        )
        rel = abs(float(row.log_F) - model_log)
        rel_errs.append(rel)
    max_rel = max(rel_errs)
    assert max_rel < 0.15, (
        f"max abs log-error vs OMIP YR on {last_date}: {max_rel:.4f}"
    )
