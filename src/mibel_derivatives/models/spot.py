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
    s = prices.dropna().astype(float)
    if s.index.tz is None:
        raise ValueError("prices must have a UTC-tz-aware DatetimeIndex")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if len(s) < 24 * 30:
        raise ValueError(
            f"need at least 30 days of hourly data, got {len(s)} observations"
        )

    log_p = np.log(s + price_shift).rename("log_p")

    seasonality, seasonal_fitted = _fit_seasonality(
        log_p, harmonics=fourier_harmonics,
    )
    residuals = (log_p - seasonal_fitted).rename("residuals")
    residual_returns = residuals.diff().dropna().rename("residual_returns")

    jumps_mask = _detect_jumps(
        residual_returns,
        k=jump_threshold_k,
        max_iter=jump_max_iter,
    )
    jump_sizes = residual_returns[jumps_mask].rename("jump_sizes")

    kappa, sigma_by_hour = _mle_ou(residuals, jumps_mask)
    intensity, p_up, eta_up, eta_down = _mle_jumps(
        jump_sizes, n_total_hours=len(residual_returns),
    )

    params = SpotModelParams(
        seasonality=seasonality,
        price_shift=price_shift,
        kappa=kappa,
        sigma_by_hour=sigma_by_hour,
        jump_intensity=intensity,
        jump_p_up=p_up,
        jump_eta_up=eta_up,
        jump_eta_down=eta_down,
    )
    return SpotModelFit(
        params=params,
        log_price_index=s.index,
        seasonal_fitted=seasonal_fitted,
        residuals=residuals,
        residual_returns=residual_returns,
        jumps_mask=jumps_mask,
        jump_sizes=jump_sizes,
        n_obs=len(s),
        n_jumps=int(jumps_mask.sum()),
    )


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

    At each iteration the per-hour mean μ_h and standard deviation σ_h
    are computed from the currently un-flagged residual returns; a return
    is flagged when ``|ΔZ_t − μ_{h(t)}| > k · σ_{h(t)}``. Iteration stops
    when the flagged set stabilises or ``max_iter`` is reached. Hours
    with fewer than five non-jump observations are skipped (no jumps
    declared for that bucket on this iteration).
    """
    if residual_returns.index.tz is None:
        raise ValueError("residual_returns must be UTC-tz-aware")

    hours = np.asarray(residual_returns.index.hour)
    values = residual_returns.to_numpy(dtype=float)
    is_jump = np.zeros(len(values), dtype=bool)

    for _ in range(max_iter):
        new_is_jump = np.zeros_like(is_jump)
        for h in range(24):
            mask_h = hours == h
            mask_h_nonjump = mask_h & ~is_jump
            if int(mask_h_nonjump.sum()) < 5:
                continue
            mu_h = float(values[mask_h_nonjump].mean())
            sd_h = float(values[mask_h_nonjump].std(ddof=1))
            if sd_h <= 0.0 or not np.isfinite(sd_h):
                continue
            new_is_jump[mask_h] = np.abs(values[mask_h] - mu_h) > k * sd_h
        if np.array_equal(new_is_jump, is_jump):
            break
        is_jump = new_is_jump

    return pd.Series(is_jump, index=residual_returns.index, name="is_jump")


def _mle_ou(
    residuals: pd.Series,
    jump_mask: pd.Series,
    dt_hours: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Closed-form MLE for the OU process on non-jump observations.

    The OU SDE ``dZ_t = -κ Z_t dt + σ_h(t) dW_t`` discretised at one-hour
    intervals is an AR(1) through the origin,

        Z_{t+1} = φ · Z_t + η_t,    φ = exp(-κ Δt),
        Var(η_t) = σ_{h(t+1)}^2 · (1 - φ^2) / (2 κ).

    The estimator is two-step:

    * ``φ̂ = Σ Z_t Z_{t+1} / Σ Z_t^2`` over pairs whose return was NOT
      flagged as a jump, then ``κ̂ = -log φ̂ / Δt`` (clipped into a
      numerically safe interior of (0, 1) before the log).
    * Innovations ``η̂_t = Z_{t+1} − φ̂ Z_t`` are bucketed by the
      destination hour-of-day; per-hour variance is inverted into the
      instantaneous σ via ``σ_h^2 = Var(η̂)_h · 2 κ̂ / (1 − φ̂^2)``.
      Hours with fewer than five non-jump pairs are filled with the
      cross-hour median.

    ``jump_mask`` is aligned with the returns index (``residuals.index[1:]``)
    so its length must be ``len(residuals) - 1``.
    """
    z = residuals.to_numpy(dtype=float)
    if len(z) < 24:
        raise ValueError("residuals must have at least 24 observations")
    z_t = z[:-1]
    z_tp1 = z[1:]
    if len(jump_mask) != len(z_tp1):
        raise ValueError(
            f"jump_mask length {len(jump_mask)} != residuals length-1 "
            f"({len(z_tp1)})"
        )
    nonjump = ~jump_mask.to_numpy()
    z_t_nj = z_t[nonjump]
    z_tp1_nj = z_tp1[nonjump]

    denom = float((z_t_nj * z_t_nj).sum())
    if denom <= 0:
        raise ValueError("Degenerate residual series (zero variance)")
    phi = float((z_t_nj * z_tp1_nj).sum() / denom)
    phi = float(np.clip(phi, 1e-6, 1.0 - 1e-6))
    kappa = -np.log(phi) / dt_hours

    eta = z_tp1_nj - phi * z_t_nj
    factor = (2.0 * kappa) / (1.0 - phi**2)
    hours_nj = np.asarray(residuals.index[1:].hour)[nonjump]
    sigma_by_hour = np.full(24, np.nan)
    for h in range(24):
        eta_h = eta[hours_nj == h]
        if len(eta_h) < 5:
            continue
        sigma_by_hour[h] = np.sqrt(float(eta_h.var(ddof=1)) * factor)
    overall = float(np.nanmedian(sigma_by_hour))
    sigma_by_hour = np.where(np.isnan(sigma_by_hour), overall, sigma_by_hour)
    return kappa, sigma_by_hour


def _mle_jumps(
    jump_returns: pd.Series,
    n_total_hours: int,
) -> tuple[float, float, float, float]:
    """MLE for the Poisson rate and Kou (2002) asymmetric double-exponential.

    Closed-form moment estimators:

        λ̂      = #jumps / n_total_hours
        p̂_up   = #(J > 0) / #jumps
        η̂_up   = 1 / mean(J | J > 0)
        η̂_down = 1 / mean(-J | J < 0)

    When a side is empty (all jumps share the same sign), the missing
    rate is set to 1.0 as a neutral default and the side probability
    truncates accordingly.
    """
    if n_total_hours <= 0:
        raise ValueError("n_total_hours must be positive")
    n_jumps = len(jump_returns)
    intensity = n_jumps / n_total_hours
    if n_jumps == 0:
        return intensity, 0.5, 1.0, 1.0

    j = jump_returns.to_numpy(dtype=float)
    j_up = j[j > 0]
    j_down = j[j < 0]
    p_up = float(len(j_up)) / float(n_jumps)
    eta_up = float(1.0 / j_up.mean()) if len(j_up) >= 1 else 1.0
    eta_down = float(1.0 / (-j_down.mean())) if len(j_down) >= 1 else 1.0
    return intensity, p_up, eta_up, eta_down
