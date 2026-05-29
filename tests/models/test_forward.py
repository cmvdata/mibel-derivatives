"""Tests for :mod:`mibel_derivatives.models.forward` (Schwartz-Smith).

This file accumulates tests across commits H1-H4. Commit H1 covers
closed-form pricing (A(τ) and futures_log_price); subsequent commits
add Kalman filter, MLE and simulation tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_derivatives.models import forward

# ---- H1. Public API + scaffold ---------------------------------------------


def test_module_exposes_public_api() -> None:
    for name in (
        "fit", "simulate",
        "SSParams", "SSFit",
        "A_function", "futures_log_price",
        "delivery_midpoint", "tau_years",
        "KAPPA_BOUNDS", "SIGMA_CHI_BOUNDS", "SIGMA_XI_BOUNDS",
        "RHO_BOUNDS", "MU_XI_STAR_BOUNDS", "LAMBDA_CHI_BOUNDS",
        "EPSILON_BOUNDS", "EPSILON_SPOT_BOUNDS", "DAYS_PER_YEAR",
    ):
        assert hasattr(forward, name), f"forward.{name} missing"
    assert forward.DAYS_PER_YEAR == 365.25
    assert forward.KAPPA_BOUNDS == (0.1, 5.0)
    assert forward.RHO_BOUNDS == (-0.99, 0.99)


def _trivial_params(**overrides) -> forward.SSParams:
    return forward.SSParams(
        kappa=overrides.get("kappa", 1.0),
        sigma_chi=overrides.get("sigma_chi", 0.30),
        sigma_xi=overrides.get("sigma_xi", 0.15),
        rho=overrides.get("rho", 0.0),
        mu_xi=overrides.get("mu_xi", 0.0),
        mu_xi_star=overrides.get("mu_xi_star", 0.0),
        lambda_chi=overrides.get("lambda_chi", 0.0),
        epsilon_m=overrides.get("epsilon_m", 0.02),
        epsilon_yr=overrides.get("epsilon_yr", 0.03),
        epsilon_spot=overrides.get("epsilon_spot", 0.005),
        seasonal_dummies=overrides.get("seasonal_dummies", np.zeros(11)),
    )


def test_ssparams_is_frozen() -> None:
    p = _trivial_params()
    with pytest.raises(Exception):
        p.kappa = 2.0  # frozen dataclass


# ---- H1. A(τ) closed-form --------------------------------------------------


def test_A_function_zero_tau_is_zero() -> None:
    """At τ=0 the deterministic correction is zero (no time elapsed)."""
    val = forward.A_function(
        0.0, kappa=1.0, sigma_chi=0.3, sigma_xi=0.2, rho=0.5,
        mu_xi_star=0.05, lambda_chi=0.02,
    )
    assert abs(float(val)) < 1e-12


def test_A_function_no_uncertainty_no_premia_is_zero_drift_times_tau() -> None:
    """With σ = 0 and zero risk premia, A(τ) = μ_ξ* · τ."""
    tau = 2.5
    val = float(forward.A_function(
        tau, kappa=1.0, sigma_chi=0.0, sigma_xi=0.0, rho=0.0,
        mu_xi_star=0.04, lambda_chi=0.0,
    ))
    assert abs(val - 0.04 * tau) < 1e-12


def test_A_function_pure_long_term_variance_grows_linearly() -> None:
    """σ_χ = 0 and λ_χ = 0 leave A(τ) = μ_ξ*·τ + 0.5 σ_ξ²·τ. Long-term
    factor variance contributes a Jensen 1/2 σ² term linear in τ."""
    tau = 1.0
    sigma_xi = 0.20
    val = float(forward.A_function(
        tau, kappa=1.5, sigma_chi=0.0, sigma_xi=sigma_xi, rho=0.0,
        mu_xi_star=0.0, lambda_chi=0.0,
    ))
    expected = 0.5 * sigma_xi**2 * tau
    assert abs(val - expected) < 1e-12


def test_A_function_pure_short_term_variance_saturates() -> None:
    """σ_ξ = 0 and λ_χ = 0 reduce A(τ) to 0.5 (1-e^{-2κτ}) σ_χ²/(2κ),
    which saturates at σ_χ²/(4κ) for large τ."""
    sigma_chi = 0.40
    kappa = 1.0
    val_large = float(forward.A_function(
        100.0, kappa=kappa, sigma_chi=sigma_chi, sigma_xi=0.0, rho=0.0,
        mu_xi_star=0.0, lambda_chi=0.0,
    ))
    expected_limit = sigma_chi**2 / (4.0 * kappa)
    assert abs(val_large - expected_limit) < 1e-6


def test_A_function_rejects_nonpositive_kappa() -> None:
    with pytest.raises(ValueError, match="kappa"):
        forward.A_function(
            1.0, kappa=0.0, sigma_chi=0.3, sigma_xi=0.2, rho=0.0,
            mu_xi_star=0.0, lambda_chi=0.0,
        )


def test_A_function_vectorised_matches_scalar() -> None:
    taus = np.array([0.1, 0.5, 1.0, 2.0])
    kwargs = dict(kappa=1.0, sigma_chi=0.3, sigma_xi=0.2, rho=0.4,
                  mu_xi_star=0.05, lambda_chi=0.02)
    vec = forward.A_function(taus, **kwargs)
    for i, t in enumerate(taus):
        scalar = float(forward.A_function(float(t), **kwargs))
        assert abs(float(vec[i]) - scalar) < 1e-12


# ---- H1. futures_log_price -------------------------------------------------


def test_futures_log_price_at_zero_tau_is_spot_plus_seasonal() -> None:
    """At τ = 0: ln F(t, t) = χ_t + ξ_t + s(month_of_t). A(0) = 0."""
    seas = np.linspace(0.05, -0.05, 11)  # arbitrary monthly pattern Feb..Dec
    params = _trivial_params(seasonal_dummies=seas)
    # Test for each delivery month
    for m in (1, 2, 6, 12):
        v = forward.futures_log_price(
            params, chi=0.1, xi=4.0, tau=0.0, delivery_month=m,
        )
        expected_seas = 0.0 if m == 1 else float(seas[m - 2])
        assert abs(v - (0.1 + 4.0 + expected_seas)) < 1e-12


def test_futures_log_price_yearly_uses_mean_seasonal() -> None:
    seas = np.linspace(0.05, -0.05, 11)
    expected_yearly = (0.0 + seas.sum()) / 12.0
    params = _trivial_params(seasonal_dummies=seas)
    v = forward.futures_log_price(
        params, chi=0.0, xi=4.0, tau=0.0, is_yearly=True,
    )
    assert abs(v - (4.0 + expected_yearly)) < 1e-12


def test_futures_log_price_short_term_decays_exponentially() -> None:
    """The χ contribution to log F decays as e^{-κτ} so the price for a
    long-dated contract is dominated by ξ + A(τ)."""
    params = _trivial_params(
        kappa=2.0, sigma_chi=0.0, sigma_xi=0.0, rho=0.0,
        mu_xi_star=0.0, lambda_chi=0.0,
    )
    chi, xi = 0.5, 4.0
    short = forward.futures_log_price(
        params, chi=chi, xi=xi, tau=0.1, delivery_month=1,
    )
    long_ = forward.futures_log_price(
        params, chi=chi, xi=xi, tau=10.0, delivery_month=1,
    )
    # Short-dated contains ~0.5 · e^{-0.2} = 0.41 of χ; long-dated ~0
    assert abs(short - (xi + chi * np.exp(-2.0 * 0.1))) < 1e-12
    assert abs(long_ - xi) < 1e-5  # χ contribution effectively zero


def test_futures_log_price_requires_delivery_or_yearly() -> None:
    params = _trivial_params()
    with pytest.raises(ValueError):
        forward.futures_log_price(params, chi=0.0, xi=4.0, tau=1.0)
    with pytest.raises(ValueError):
        forward.futures_log_price(
            params, chi=0.0, xi=4.0, tau=1.0,
            delivery_month=3, is_yearly=True,
        )


# ---- H1. Date helpers ------------------------------------------------------


def test_delivery_midpoint_basic() -> None:
    start = pd.Timestamp("2024-02-01")
    end = pd.Timestamp("2024-02-29")  # 2024 leap year
    mid = forward.delivery_midpoint(start, end)
    # Midpoint of 29 days inclusive: floor of 14 days after start.
    assert mid == pd.Timestamp("2024-02-15")


def test_delivery_midpoint_rejects_inverted() -> None:
    with pytest.raises(ValueError):
        forward.delivery_midpoint(pd.Timestamp("2024-02-15"), pd.Timestamp("2024-02-01"))


def test_tau_years_uses_365_25() -> None:
    t = pd.Timestamp("2024-01-01")
    d = pd.Timestamp("2025-01-01")
    tau = forward.tau_years(t, d)
    expected = 366 / 365.25  # 2024 is a leap year (366 days)
    assert abs(tau - expected) < 1e-12


# ---- H1. fit / simulate stubs still raise ---------------------------------


def test_fit_rejects_empty_observations() -> None:
    """fit() must surface degenerate inputs rather than silently fitting."""
    # Build an OMIP DataFrame whose every row has a missing price.
    empty = pd.DataFrame({
        "trade_date": ["2024-01-02"],
        "maturity_bucket": ["M"],
        "contract": ["FTB M Feb-24"],
        "reference_d_eur_mwh": [np.nan],
    })
    with pytest.raises(ValueError, match="no observations"):
        forward.fit(empty)


def test_simulate_returns_tuple() -> None:
    chi, xi = forward.simulate(
        _trivial_params(),
        initial_chi=0.0, initial_xi=4.0,
        start=pd.Timestamp("2025-01-01"),
        n_days=10, n_paths=5, seed=1,
    )
    assert chi.shape == (5, 10) and xi.shape == (5, 10)


# ---- H2. Observation preparation -------------------------------------------


def _make_omip_fixture() -> pd.DataFrame:
    """Two trade dates × handful of contracts for prepare_observations tests."""
    rows = [
        # Trade date 2024-01-02: two M contracts + two YR contracts.
        ("2024-01-02", "M",  "FTB M Feb-24",  60.0),
        ("2024-01-02", "M",  "FTB M Mar-24",  55.0),
        ("2024-01-02", "YR", "FTB YR-25",     70.0),
        ("2024-01-02", "YR", "FTB YR-26",     65.0),
        # Trade date 2024-01-03.
        ("2024-01-03", "M",  "FTB M Feb-24",  61.0),
        ("2024-01-03", "M",  "FTB M Mar-24",  56.0),
        ("2024-01-03", "YR", "FTB YR-25",     71.0),
    ]
    return pd.DataFrame(rows, columns=[
        "trade_date", "maturity_bucket", "contract", "reference_d_eur_mwh",
    ])


def test_prepare_observations_long_format_shape_and_columns() -> None:
    omip = _make_omip_fixture()
    obs = forward.prepare_observations(omip)
    expected_cols = {
        "trade_date", "bucket", "delivery_month",
        "is_yearly", "tau", "log_F",
    }
    assert expected_cols.issubset(set(obs.columns))
    assert len(obs) == len(omip)
    # M contracts: bucket="M", is_yearly=False.
    assert (obs.loc[obs["bucket"] == "M", "is_yearly"] == False).all()
    assert (obs.loc[obs["bucket"] == "YR", "is_yearly"] == True).all()
    # tau values are positive.
    assert (obs["tau"] > 0).all()
    # log_F = ln(reference_d_eur_mwh) — total sum and individual values match.
    assert abs(obs["log_F"].sum() - np.log(omip["reference_d_eur_mwh"]).sum()) < 1e-10


def test_prepare_observations_decodes_contract_to_delivery_month() -> None:
    omip = _make_omip_fixture()
    obs = forward.prepare_observations(omip)
    feb = obs[obs["bucket"] == "M"].iloc[0]
    # First M contract is "FTB M Feb-24" → February (month=2).
    assert int(feb["delivery_month"]) == 2


def test_prepare_observations_appends_spot_when_omie_provided() -> None:
    omip = _make_omip_fixture()
    omie = pd.Series(
        [62.5, 63.0],
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="trade_date"),
        name="price",
    )
    obs = forward.prepare_observations(omip, omie)
    spot = obs[obs["bucket"] == "SPOT"]
    assert len(spot) == 2
    assert (spot["tau"] == 0.0).all()
    np.testing.assert_allclose(
        spot["log_F"].values, np.log(omie.values), atol=1e-12,
    )


# ---- H2. Kalman filter -----------------------------------------------------


def _synthetic_ss_observations(
    params: forward.SSParams,
    n_dates: int,
    *,
    initial_chi: float,
    initial_xi: float,
    contracts_per_date: int = 4,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic OMIP-like observations from known states.

    Returns:
        obs_long_df,  chi_true (n_dates,), xi_true (n_dates,)
    """
    rng = np.random.default_rng(seed=seed)
    trade_dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")

    # Simulate (chi, xi) trajectory under physical measure.
    dt = 1.0 / forward.DAYS_PER_YEAR  # ~1 business day in years
    phi = float(np.exp(-params.kappa * dt))
    q11 = params.sigma_chi**2 * (1 - np.exp(-2*params.kappa*dt)) / (2*params.kappa)
    q22 = params.sigma_xi**2 * dt
    q12 = params.rho * params.sigma_chi * params.sigma_xi * (1 - np.exp(-params.kappa*dt)) / params.kappa
    Q = np.array([[q11, q12], [q12, q22]])
    L = np.linalg.cholesky(Q)
    chi_true = np.empty(n_dates)
    xi_true = np.empty(n_dates)
    chi_true[0], xi_true[0] = initial_chi, initial_xi
    for t in range(1, n_dates):
        innov = L @ rng.standard_normal(2)
        chi_true[t] = phi * chi_true[t-1] + innov[0]
        xi_true[t] = xi_true[t-1] + params.mu_xi * dt + innov[1]

    rows = []
    for i, td in enumerate(trade_dates):
        # contracts_per_date = 2 monthly + 2 yearly.
        # Monthly: delivery 30, 90 days out.
        for tau_days, m in ((30, ((td.month % 12) + 1)), (90, ((td.month + 2) % 12) + 1)):
            tau = tau_days / forward.DAYS_PER_YEAR
            true_logF = (
                np.exp(-params.kappa * tau) * chi_true[i]
                + xi_true[i]
                + float(forward.A_function(
                    tau, kappa=params.kappa,
                    sigma_chi=params.sigma_chi, sigma_xi=params.sigma_xi,
                    rho=params.rho, mu_xi_star=params.mu_xi_star,
                    lambda_chi=params.lambda_chi,
                ))
                + forward._seasonal_value(params.seasonal_dummies, m)
            )
            noisy = true_logF + params.epsilon_m * rng.standard_normal()
            rows.append({
                "trade_date": td.normalize(),
                "bucket": "M",
                "delivery_month": m,
                "is_yearly": False,
                "tau": tau,
                "log_F": noisy,
            })
        # Yearly: 1y and 3y out (midpoint).
        for tau_yr in (1.0, 3.0):
            true_logF = (
                np.exp(-params.kappa * tau_yr) * chi_true[i]
                + xi_true[i]
                + float(forward.A_function(
                    tau_yr, kappa=params.kappa,
                    sigma_chi=params.sigma_chi, sigma_xi=params.sigma_xi,
                    rho=params.rho, mu_xi_star=params.mu_xi_star,
                    lambda_chi=params.lambda_chi,
                ))
                + forward._seasonal_yearly(params.seasonal_dummies)
            )
            noisy = true_logF + params.epsilon_yr * rng.standard_normal()
            rows.append({
                "trade_date": td.normalize(),
                "bucket": "YR",
                "delivery_month": 0,
                "is_yearly": True,
                "tau": tau_yr,
                "log_F": noisy,
            })
    obs = pd.DataFrame(rows)
    return obs, chi_true, xi_true


def test_kalman_filter_recovers_latent_states_from_synthetic() -> None:
    """Kalman filter on synthetic obs with KNOWN params should produce
    state estimates close to the true (chi_t, xi_t) trajectory."""
    params = _trivial_params(
        kappa=1.5, sigma_chi=0.3, sigma_xi=0.15, rho=0.2,
        mu_xi=0.02, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.005, epsilon_yr=0.01,
        seasonal_dummies=np.linspace(0.05, -0.05, 11),
    )
    obs, chi_true, xi_true = _synthetic_ss_observations(
        params, n_dates=400, initial_chi=0.1, initial_xi=4.0, seed=42,
    )
    res = forward._kalman_filter(obs, params)
    # Drop the first 50 dates as filter burn-in (especially for xi with
    # diffuse prior). Then state estimates should track truth tightly.
    chi_est = res["chi_series"].values
    xi_est = res["xi_series"].values
    err_chi = np.abs(chi_est[50:] - chi_true[50:])
    err_xi = np.abs(xi_est[50:] - xi_true[50:])
    assert err_chi.mean() < 0.05, f"chi mean abs err = {err_chi.mean():.4f}"
    assert err_xi.mean() < 0.10, f"xi mean abs err = {err_xi.mean():.4f}"


def test_kalman_filter_log_likelihood_is_finite() -> None:
    params = _trivial_params(
        kappa=1.5, sigma_chi=0.3, sigma_xi=0.15, rho=0.2,
        mu_xi=0.02, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.01, epsilon_yr=0.02,
    )
    obs, _, _ = _synthetic_ss_observations(
        params, n_dates=100, initial_chi=0.0, initial_xi=4.0, seed=1,
    )
    res = forward._kalman_filter(obs, params, return_states=False)
    assert np.isfinite(res["log_likelihood"])
    assert res["n_obs"] == 100 * 4   # 4 contracts per date
    assert res["n_dates"] == 100
    assert res["rmse_log_m"] > 0 and res["rmse_log_m"] < 0.05
    assert res["rmse_log_yr"] > 0 and res["rmse_log_yr"] < 0.05


def test_kalman_filter_better_params_yield_higher_likelihood() -> None:
    """Among the true params and a perturbed set, the true params must
    produce a higher Kalman log-likelihood (model evidence)."""
    params_true = _trivial_params(
        kappa=1.5, sigma_chi=0.3, sigma_xi=0.15, rho=0.2,
        mu_xi=0.02, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.005, epsilon_yr=0.01,
    )
    obs, _, _ = _synthetic_ss_observations(
        params_true, n_dates=200, initial_chi=0.0, initial_xi=4.0, seed=7,
    )
    ll_true = forward._kalman_filter(obs, params_true, return_states=False)["log_likelihood"]
    # Perturb kappa significantly.
    params_wrong = _trivial_params(
        kappa=3.5, sigma_chi=0.3, sigma_xi=0.15, rho=0.2,
        mu_xi=0.02, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.005, epsilon_yr=0.01,
    )
    ll_wrong = forward._kalman_filter(obs, params_wrong, return_states=False)["log_likelihood"]
    assert ll_true > ll_wrong


# ---- H3. Bounded MLE -------------------------------------------------------


# NOTE: a synthetic end-to-end MLE recovery test would be the natural
# place to assert that the optimiser converges interior with reasonable
# parameter recovery. Empirically, 20-parameter L-BFGS-B on 50 synthetic
# dates × 4 contracts does NOT converge within practical iteration caps
# (the σ_ξ / ρ / risk-premium directions are weakly identified on short
# series); even with a warm start within ±10 % of truth the optimiser
# hits the iteration limit before satisfying the L-BFGS-B convergence
# tolerance. The meaningful test for end-to-end correctness is on real
# OMIP data — see tests/models/test_forward_validation.py, which fits
# the full 1500+ trade dates with max_iter=300 and validates the
# resulting calibration against forward-curve RMSE, spot anchoring and
# implied curve at end of history.


@pytest.mark.slow
def test_mle_flags_bound_active_kappa(caplog) -> None:
    """Generate obs with kappa OUTSIDE KAPPA_BOUNDS = [0.1, 5.0]. The
    bounded MLE pushes kappa to its lower bound and now logs a WARNING
    (was: raise) so downstream validation can route the model through
    the Pieza 1 / Pieza 2 composition pipeline. The validation V1 test
    on real OMIP catches the bound-active case via an interior-bounds
    assertion on the returned params."""
    import logging
    params_outside = _trivial_params(
        kappa=0.05,         # outside [0.1, 5.0]
        sigma_chi=0.25, sigma_xi=0.10, rho=0.0,
        mu_xi=0.0, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.005, epsilon_yr=0.01,
    )
    obs, _, _ = _synthetic_ss_observations(
        params_outside, n_dates=50,
        initial_chi=0.0, initial_xi=4.0, seed=11,
    )
    warm = forward.SSParams(
        kappa=forward.KAPPA_BOUNDS[0] + 1e-3,
        sigma_chi=0.25, sigma_xi=0.10, rho=0.0,
        mu_xi=0.0, mu_xi_star=0.0, lambda_chi=0.0,
        epsilon_m=0.005, epsilon_yr=0.01, epsilon_spot=0.005,
        seasonal_dummies=np.zeros(11),
    )
    with caplog.at_level(logging.WARNING, logger="mibel_derivatives.models.forward"):
        fit_res = forward._fit_from_obs(obs, initial_params=warm, max_iter=50)
    # κ̂ lands at the lower bound within tolerance.
    assert abs(fit_res.params.kappa - forward.KAPPA_BOUNDS[0]) < 1e-3
    # And the warning recorded it.
    assert any("kappa" in rec.message and "bound" in rec.message for rec in caplog.records)


# ---- H4. Simulation --------------------------------------------------------


def test_simulate_shape_and_initial_values() -> None:
    params = _trivial_params(kappa=1.5, sigma_chi=0.0, sigma_xi=0.0)  # zero noise
    chi, xi = forward.simulate(
        params, initial_chi=0.2, initial_xi=4.0,
        start=pd.Timestamp("2025-01-01"),
        n_days=10, n_paths=5, seed=42,
    )
    assert chi.shape == (5, 10)
    assert xi.shape == (5, 10)
    # t=0 reflects the initial values across all paths.
    np.testing.assert_allclose(chi[:, 0], 0.2, atol=1e-12)
    np.testing.assert_allclose(xi[:, 0], 4.0, atol=1e-12)
    # With zero noise, chi decays exponentially toward 0; xi stays at 4
    # up to the 1e-12 Cholesky jitter that lets the decomposition admit
    # a singular Q.
    assert chi[0, -1] < chi[0, 0]
    assert abs(xi[0, -1] - 4.0) < 1e-4


def test_simulate_reproducible_with_seed() -> None:
    params = _trivial_params()
    a_chi, a_xi = forward.simulate(
        params, initial_chi=0.0, initial_xi=4.0,
        start=pd.Timestamp("2025-01-01"), n_days=100, n_paths=20, seed=7,
    )
    b_chi, b_xi = forward.simulate(
        params, initial_chi=0.0, initial_xi=4.0,
        start=pd.Timestamp("2025-01-01"), n_days=100, n_paths=20, seed=7,
    )
    c_chi, c_xi = forward.simulate(
        params, initial_chi=0.0, initial_xi=4.0,
        start=pd.Timestamp("2025-01-01"), n_days=100, n_paths=20, seed=99,
    )
    np.testing.assert_array_equal(a_chi, b_chi)
    np.testing.assert_array_equal(a_xi, b_xi)
    assert not np.array_equal(a_chi, c_chi)


def test_simulate_long_run_chi_variance_matches_theory() -> None:
    """Pure OU on chi: long-run variance is sigma_chi^2 / (2 kappa)."""
    params = _trivial_params(
        kappa=2.0, sigma_chi=0.4, sigma_xi=0.0, rho=0.0,
        mu_xi=0.0, mu_xi_star=0.0,
    )
    n_days = int(forward.DAYS_PER_YEAR * 5)  # 5 years daily
    chi, _ = forward.simulate(
        params, initial_chi=0.0, initial_xi=0.0,
        start=pd.Timestamp("2025-01-01"),
        n_days=n_days, n_paths=500, seed=2026,
    )
    burn = n_days // 2
    var_observed = float(chi[:, burn:].var())
    var_theoretical = params.sigma_chi**2 / (2.0 * params.kappa)
    rel = abs(var_observed - var_theoretical) / var_theoretical
    assert rel < 0.15, f"Var(chi) obs={var_observed:.4f} theory={var_theoretical:.4f}"


def test_simulate_xi_drifts_with_mu_xi() -> None:
    """Pure xi RW: time-T mean drifts by mu_xi · T (years)."""
    params = _trivial_params(
        kappa=1.0, sigma_chi=0.0, sigma_xi=0.0,  # no noise
        mu_xi=0.10,                              # 0.10 per year
    )
    n_days = int(forward.DAYS_PER_YEAR)
    _, xi = forward.simulate(
        params, initial_chi=0.0, initial_xi=0.0,
        start=pd.Timestamp("2025-01-01"),
        n_days=n_days, n_paths=10, seed=1,
    )
    # n_days - 1 drift steps of mu_xi · dt each.
    expected = (n_days - 1) * params.mu_xi / forward.DAYS_PER_YEAR
    np.testing.assert_allclose(xi[:, -1], expected, atol=1e-3)


def test_implied_forward_curve_matches_pointwise_pricing() -> None:
    params = _trivial_params(
        kappa=1.5, sigma_chi=0.3, sigma_xi=0.2, rho=0.3,
        mu_xi=0.02, mu_xi_star=0.01, lambda_chi=0.01,
        seasonal_dummies=np.linspace(0.05, -0.05, 11),
    )
    chi, xi = 0.1, 4.0
    taus = np.array([0.1, 0.5, 1.0, 2.0])
    months = np.array([2, 6, 9, 12])
    is_yr = np.array([False, False, False, False])
    curve = forward.implied_forward_curve(
        params, chi, xi, taus,
        delivery_months=months, is_yearly=is_yr,
    )
    for i, (t, m) in enumerate(zip(taus, months)):
        pointwise = forward.futures_log_price(params, chi, xi, float(t), delivery_month=int(m))
        assert abs(curve[i] - pointwise) < 1e-12


def test_simulate_rejects_nonpositive_dimensions() -> None:
    params = _trivial_params()
    with pytest.raises(ValueError):
        forward.simulate(
            params, initial_chi=0.0, initial_xi=4.0,
            start=pd.Timestamp("2025-01-01"),
            n_days=0, n_paths=5,
        )
