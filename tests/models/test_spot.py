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
