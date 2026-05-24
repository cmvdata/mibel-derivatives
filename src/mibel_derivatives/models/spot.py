"""Mean-reverting jump-diffusion model for the hourly MIBEL spot log-price.

Specification (CONTEXT.md § Pieza 1, with the four blocking decisions
resolved on 2026-05-24):

    Y_t = log(P_t + c)                                  c = 10 EUR/MWh
    Y_t = f(t) + Z_t
    f(t) = alpha + Σ_{k=1..K} (a_k cos(2π k·doy/365.25)
                              + b_k sin(2π k·doy/365.25))
           + Σ_{d=1..6} γ_d 1{dow(t) = d}
           + Σ_{h=1..23} δ_h 1{hod(t) = h}
    dZ_t = -κ Z_t dt + σ_{h(t)} dW_t + J_t dN_t
    N_t ~ Poisson(λ)                       # hourly intensity
    J_t ~ AsymmetricDoubleExponential(p_up, η_up, η_down)   # Kou (2002)

Following Lucia & Schwartz (2002) for the seasonal-plus-OU baseline and
Cartea & Figueroa (2005) for the jump component (Lucia-Schwartz did not
model jumps). The shift c=10 EUR/MWh handles the occasional negative
day-ahead prices observed in OMIE 2019-2024 (minimum -2 EUR/MWh) and
preserves the multiplicative structure of the log-model.

The hourly architecture uses a single OU on the deseasonalised residual
with constant κ and a 24-vector of hour-of-day-specific volatilities
σ_h (resolved on 2026-05-24 to capture intraday heteroskedasticity
without proliferating mean-reversion parameters).

Calibration is two-stage:

    1. OLS of log(P + c) on the seasonal design matrix → f̂(t).
    2. Residuals Z_t = Y_t − f̂(t); compute hourly returns ΔZ_t.
    3. Iterative threshold jump detection: a return is flagged when
       |ΔZ_t − mean_h(ΔZ)| > k · σ_h(ΔZ_non-jump); k = 4 by default.
       Iterate until the flagged set stops changing (or max_iter).
    4. MLE for the OU on non-jump observations — closed-form AR(1)
       equivalent gives κ̂; σ̂_h is the std of non-jump ΔZ by hour.
    5. Poisson rate λ̂ = (# jumps) / (# hours observed). Asymmetric
       double-exponential parameters from the empirical distribution of
       positive and negative jump magnitudes.

The 2022 gas-crisis regime shift is intentionally not handled in this
piece — a single 2019-2024 calibration is used, and the long-term level
shift is absorbed by the Schwartz-Smith long-term factor of Pieza 2
(see reports/diagnostics/spot_model_calibration.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_SHIFT: float = 10.0
FOURIER_HARMONICS: int = 4
JUMP_THRESHOLD_K: float = 4.0
JUMP_MAX_ITER: int = 10


# ---- Parameter containers --------------------------------------------------


@dataclass(frozen=True)
class Seasonality:
    """OLS coefficients for the deterministic component f(t).

    The design matrix has one intercept, ``2 * fourier_harmonics`` Fourier
    columns (alternating cos / sin), six day-of-week dummies (Monday is
    the omitted reference category) and 23 hour-of-day dummies (hour 0
    UTC is the omitted reference). All coefficients are stored aligned
    with that order so :func:`_seasonal_predict` can rebuild f̂(t).
    """

    intercept: float
    fourier_coefs: np.ndarray
    dow_coefs: np.ndarray
    hod_coefs: np.ndarray
    fourier_harmonics: int


@dataclass(frozen=True)
class SpotModelParams:
    """All fitted parameters of the MRJD spot model."""

    seasonality: Seasonality
    price_shift: float
    kappa: float
    sigma_by_hour: np.ndarray
    jump_intensity: float
    jump_p_up: float
    jump_eta_up: float
    jump_eta_down: float


@dataclass(frozen=True)
class SpotModelFit:
    """Calibration result: params plus per-observation diagnostics."""

    params: SpotModelParams
    log_price_index: pd.DatetimeIndex
    seasonal_fitted: pd.Series
    residuals: pd.Series
    residual_returns: pd.Series
    jumps_mask: pd.Series
    jump_sizes: pd.Series
    n_obs: int
    n_jumps: int


# ---- Public API ------------------------------------------------------------


def fit(
    prices: pd.Series,
    *,
    price_shift: float = PRICE_SHIFT,
    fourier_harmonics: int = FOURIER_HARMONICS,
    jump_threshold_k: float = JUMP_THRESHOLD_K,
    jump_max_iter: int = JUMP_MAX_ITER,
) -> SpotModelFit:
    """Fit the MRJD model to a UTC-hourly price series.

    Parameters
    ----------
    prices
        Hourly nominal day-ahead prices in EUR/MWh, indexed by a
        UTC-localised ``DatetimeIndex``. Gaps are tolerated; non-finite
        values are dropped before fitting.
    price_shift
        Additive constant c used in ``log(P + c)`` to admit negative
        prices.
    fourier_harmonics
        Number of annual Fourier harmonics in the seasonal component.
    jump_threshold_k
        Sigma-multiplier for the iterative jump detector.
    jump_max_iter
        Maximum iterations of the jump detector.
    """
    raise NotImplementedError


def simulate(
    params: SpotModelParams,
    start: pd.Timestamp,
    n_hours: int,
    n_paths: int,
    *,
    initial_residual: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate ``n_paths`` forward nominal-price trajectories of length
    ``n_hours`` from ``start`` (UTC).

    The returned array has shape ``(n_paths, n_hours)`` in EUR/MWh
    (the shift c is removed before returning, so negative samples are
    possible when the simulated log-shifted price is below ``log(c)``).
    """
    raise NotImplementedError


# ---- Internals -------------------------------------------------------------


def _seasonal_design_matrix(
    timestamps: pd.DatetimeIndex,
    harmonics: int = FOURIER_HARMONICS,
) -> pd.DataFrame:
    """Build the OLS design matrix for f(t).

    Columns, in this fixed order so the coefficient vector can be sliced
    back into a :class:`Seasonality`:

    1. ``intercept`` (constant 1).
    2. ``2 * harmonics`` Fourier columns ``cos_k`` / ``sin_k`` for
       ``k = 1..harmonics`` using ``angle = 2π · day_of_year / 365.25``.
    3. Six day-of-week dummies ``dow_1..dow_6`` (Monday = 0 is the
       omitted reference).
    4. Twenty-three hour-of-day dummies ``hod_1..hod_23`` (hour 0 UTC
       is the omitted reference).

    ``timestamps`` must be tz-aware; UTC is assumed for the hour-of-day
    bucketing so DST does not bleed into the seasonal coefficients.
    """
    if timestamps.tz is None:
        raise ValueError("timestamps must be tz-aware (expected UTC)")

    n = len(timestamps)
    cols: dict[str, np.ndarray] = {"intercept": np.ones(n)}

    doy = np.asarray(timestamps.dayofyear, dtype=float)
    angle = 2.0 * np.pi * doy / 365.25
    for k in range(1, harmonics + 1):
        cols[f"cos_{k}"] = np.cos(k * angle)
        cols[f"sin_{k}"] = np.sin(k * angle)

    dow = np.asarray(timestamps.dayofweek)
    for d in range(1, 7):  # Monday = 0 is the reference
        cols[f"dow_{d}"] = (dow == d).astype(float)

    hod = np.asarray(timestamps.hour)
    for h in range(1, 24):  # hour 0 UTC is the reference
        cols[f"hod_{h}"] = (hod == h).astype(float)

    return pd.DataFrame(cols, index=timestamps)


def _fit_seasonality(
    log_prices: pd.Series,
    harmonics: int = FOURIER_HARMONICS,
) -> tuple[Seasonality, pd.Series]:
    """Solve the OLS problem and return (Seasonality, fitted f̂(t))."""
    X = _seasonal_design_matrix(log_prices.index, harmonics)
    y = log_prices.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)

    cursor = 0
    intercept = float(beta[cursor]); cursor += 1
    fourier_coefs = np.asarray(beta[cursor:cursor + 2 * harmonics]).copy()
    cursor += 2 * harmonics
    dow_coefs = np.asarray(beta[cursor:cursor + 6]).copy(); cursor += 6
    hod_coefs = np.asarray(beta[cursor:cursor + 23]).copy(); cursor += 23
    if cursor != len(beta):
        raise RuntimeError(
            f"Design matrix has {len(beta)} cols but slicing consumed {cursor}"
        )

    seasonality = Seasonality(
        intercept=intercept,
        fourier_coefs=fourier_coefs,
        dow_coefs=dow_coefs,
        hod_coefs=hod_coefs,
        fourier_harmonics=harmonics,
    )
    fitted = pd.Series(
        X.to_numpy() @ beta, index=log_prices.index, name="seasonal_fitted",
    )
    return seasonality, fitted


def _seasonal_predict(
    seasonality: Seasonality,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """Evaluate f̂(t) on a given timestamp index."""
    X = _seasonal_design_matrix(timestamps, seasonality.fourier_harmonics)
    beta = np.concatenate([
        np.array([seasonality.intercept]),
        seasonality.fourier_coefs,
        seasonality.dow_coefs,
        seasonality.hod_coefs,
    ])
    return pd.Series(
        X.to_numpy() @ beta, index=timestamps, name="seasonal_predicted",
    )


def _detect_jumps(
    residual_returns: pd.Series,
    k: float = JUMP_THRESHOLD_K,
    max_iter: int = JUMP_MAX_ITER,
) -> pd.Series:
    """Iterative threshold jump detector with σ updated by hour-of-day.

    Returns a boolean ``Series`` aligned with ``residual_returns`` whose
    True entries correspond to flagged jumps.
    """
    raise NotImplementedError


def _mle_ou(
    residuals: pd.Series,
    jump_mask: pd.Series,
    dt_hours: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Closed-form MLE for the OU process on non-jump observations.

    Returns ``(kappa, sigma_by_hour)`` where ``sigma_by_hour`` is a
    length-24 vector indexed by hour-of-day in UTC.
    """
    raise NotImplementedError


def _mle_jumps(
    jump_returns: pd.Series,
    n_total_hours: int,
) -> tuple[float, float, float, float]:
    """MLE for the Poisson rate and Kou asymmetric double-exponential.

    Returns ``(intensity, p_up, eta_up, eta_down)``.
    """
    raise NotImplementedError
