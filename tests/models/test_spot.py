"""Tests for :mod:`mibel_derivatives.models.spot`.

This file accumulates tests across commits A-E. Commit A introduces only
the API-contract tests below; subsequent commits add tests for each
calibration stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_derivatives.models import spot


# ---- A. Public API contract ------------------------------------------------


def test_module_exposes_public_api() -> None:
    for name in (
        "fit", "simulate",
        "Seasonality", "SpotModelParams", "SpotModelFit",
        "PRICE_SHIFT", "FOURIER_HARMONICS",
        "JUMP_THRESHOLD_K", "JUMP_MAX_ITER",
    ):
        assert hasattr(spot, name), f"spot.{name} missing"
    assert spot.PRICE_SHIFT == 10.0
    assert spot.FOURIER_HARMONICS == 4
    assert spot.JUMP_THRESHOLD_K == 4.0


def test_fit_signature_raises_until_implemented() -> None:
    """Commit A: the public API exists but the body is a placeholder.
    This test will be replaced by a real recovery test in commit B/D."""
    with pytest.raises(NotImplementedError):
        spot.fit(pd.Series(dtype=float))


def test_params_dataclass_is_frozen_and_constructs() -> None:
    seasonality = spot.Seasonality(
        intercept=0.0,
        fourier_coefs=np.zeros(8),
        dow_coefs=np.zeros(6),
        hod_coefs=np.zeros(23),
        fourier_harmonics=4,
    )
    params = spot.SpotModelParams(
        seasonality=seasonality,
        price_shift=10.0,
        kappa=0.1,
        sigma_by_hour=np.full(24, 0.05),
        jump_intensity=0.001,
        jump_p_up=0.5,
        jump_eta_up=10.0,
        jump_eta_down=10.0,
    )
    with pytest.raises(Exception):
        params.kappa = 0.2  # frozen dataclass


def test_simulate_signature_raises_until_implemented() -> None:
    seasonality = spot.Seasonality(
        intercept=0.0,
        fourier_coefs=np.zeros(8),
        dow_coefs=np.zeros(6),
        hod_coefs=np.zeros(23),
        fourier_harmonics=4,
    )
    params = spot.SpotModelParams(
        seasonality=seasonality,
        price_shift=10.0,
        kappa=0.1,
        sigma_by_hour=np.full(24, 0.05),
        jump_intensity=0.001,
        jump_p_up=0.5,
        jump_eta_up=10.0,
        jump_eta_down=10.0,
    )
    with pytest.raises(NotImplementedError):
        spot.simulate(
            params,
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            n_hours=24,
            n_paths=10,
        )


# ---- B. Seasonality --------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="h", tz="UTC")


def test_seasonal_design_matrix_shape_and_intercept() -> None:
    idx = _hourly_utc("2020-01-01", 24 * 7)
    X = spot._seasonal_design_matrix(idx, harmonics=4)
    # 1 intercept + 2*4 Fourier + 6 DoW + 23 HoD = 38 cols
    assert X.shape == (24 * 7, 38)
    assert list(X.columns)[0] == "intercept"
    assert (X["intercept"] == 1.0).all()
    # Exactly one of {dow_1..dow_6} is non-zero per row (or all zero on Monday)
    dow_block = X[[f"dow_{d}" for d in range(1, 7)]].to_numpy()
    assert set(dow_block.sum(axis=1).round(6).tolist()) <= {0.0, 1.0}
    # Same for hod (zero on hour 0 UTC, one elsewhere)
    hod_block = X[[f"hod_{h}" for h in range(1, 24)]].to_numpy()
    assert set(hod_block.sum(axis=1).round(6).tolist()) <= {0.0, 1.0}


def test_seasonal_design_matrix_requires_tz() -> None:
    idx = pd.date_range("2020-01-01", periods=24, freq="h", tz=None)
    with pytest.raises(ValueError):
        spot._seasonal_design_matrix(idx)


def test_fit_seasonality_recovers_known_coefficients() -> None:
    """Synthetic series y = X β + ε with known β; OLS must recover β."""
    rng = np.random.default_rng(seed=42)
    n_hours = 24 * 365 * 2  # 2 years hourly
    idx = _hourly_utc("2020-01-01", n_hours)
    X = spot._seasonal_design_matrix(idx, harmonics=4).to_numpy()

    true_intercept = 3.5
    true_fourier = np.array([0.20, -0.10, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
    true_dow = np.array([0.05, 0.07, 0.06, 0.04, 0.02, -0.10])
    true_hod = np.linspace(-0.30, 0.50, 23)
    true_beta = np.concatenate([[true_intercept], true_fourier, true_dow, true_hod])

    y = X @ true_beta + 0.02 * rng.standard_normal(n_hours)
    log_prices = pd.Series(y, index=idx, name="log_p")

    seasonality, fitted = spot._fit_seasonality(log_prices, harmonics=4)

    assert seasonality.fourier_harmonics == 4
    assert np.isclose(seasonality.intercept, true_intercept, atol=0.01)
    np.testing.assert_allclose(seasonality.fourier_coefs, true_fourier, atol=0.01)
    np.testing.assert_allclose(seasonality.dow_coefs, true_dow, atol=0.01)
    np.testing.assert_allclose(seasonality.hod_coefs, true_hod, atol=0.01)
    assert (log_prices - fitted).std() < 0.03


def test_seasonal_predict_matches_fit_on_same_index() -> None:
    rng = np.random.default_rng(seed=7)
    idx = _hourly_utc("2021-03-01", 24 * 90)
    log_prices = pd.Series(
        rng.standard_normal(len(idx)) * 0.1 + 3.5,
        index=idx, name="log_p",
    )
    seasonality, fitted = spot._fit_seasonality(log_prices, harmonics=3)
    predicted = spot._seasonal_predict(seasonality, idx)
    np.testing.assert_allclose(predicted.to_numpy(), fitted.to_numpy(), atol=1e-10)


def test_design_matrix_buckets_correctly() -> None:
    """A hand-picked timestamp must land in the right DoW / HoD column."""
    # 2024-01-03 is a Wednesday (DoW = 2). 14:00 UTC → HoD = 14.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-03 14:00", tz="UTC")])
    X = spot._seasonal_design_matrix(idx, harmonics=4)
    assert X["intercept"].iloc[0] == 1.0
    assert X["dow_2"].iloc[0] == 1.0
    for d in (1, 3, 4, 5, 6):
        assert X[f"dow_{d}"].iloc[0] == 0.0
    assert X["hod_14"].iloc[0] == 1.0
    for h in (1, 2, 3, 13, 15, 23):
        assert X[f"hod_{h}"].iloc[0] == 0.0


def test_predict_repeats_annual_cycle() -> None:
    """Pure Fourier + HoD seasonality (no DoW): predicting on the same
    day-of-year four years apart returns near-identical hourly profiles."""
    rng = np.random.default_rng(seed=11)
    idx_train = _hourly_utc("2020-01-01", 24 * 365 * 3)
    X_train = spot._seasonal_design_matrix(idx_train, harmonics=4).to_numpy()
    true_beta = np.concatenate([
        [3.0],
        np.array([0.15, -0.05, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(6),                               # no DoW effect
        np.linspace(-0.2, 0.4, 23),
    ])
    y = X_train @ true_beta + 0.01 * rng.standard_normal(len(idx_train))
    seas, _ = spot._fit_seasonality(
        pd.Series(y, index=idx_train), harmonics=4,
    )
    # Predict 7 days of Jan in two different years (DST does not start in Jan).
    p_2024 = spot._seasonal_predict(seas, _hourly_utc("2024-01-08", 24 * 7))
    p_2028 = spot._seasonal_predict(seas, _hourly_utc("2028-01-08", 24 * 7))
    np.testing.assert_allclose(p_2024.to_numpy(), p_2028.to_numpy(), atol=0.02)
