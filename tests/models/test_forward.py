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
        "EPSILON_BOUNDS", "DAYS_PER_YEAR",
    ):
        assert hasattr(forward, name), f"forward.{name} missing"
    assert forward.DAYS_PER_YEAR == 365.25
    assert forward.KAPPA_BOUNDS == (0.5, 5.0)
    assert forward.RHO_BOUNDS == (-0.99, 0.99)


def _trivial_params(**overrides) -> forward.SSParams:
    return forward.SSParams(
        kappa=overrides.get("kappa", 1.0),
        sigma_chi=overrides.get("sigma_chi", 0.30),
        sigma_xi=overrides.get("sigma_xi", 0.15),
        rho=overrides.get("rho", 0.0),
        mu_xi_star=overrides.get("mu_xi_star", 0.0),
        lambda_chi=overrides.get("lambda_chi", 0.0),
        epsilon_m=overrides.get("epsilon_m", 0.02),
        epsilon_yr=overrides.get("epsilon_yr", 0.03),
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


def test_fit_signature_raises_until_implemented() -> None:
    with pytest.raises(NotImplementedError):
        forward.fit(pd.DataFrame(), pd.Series(dtype=float))


def test_simulate_signature_raises_until_implemented() -> None:
    with pytest.raises(NotImplementedError):
        forward.simulate(
            _trivial_params(),
            initial_chi=0.0, initial_xi=4.0,
            start=pd.Timestamp("2025-01-01"),
            n_days=10, n_paths=5,
        )
