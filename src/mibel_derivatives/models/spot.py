"""Mean-reverting jump-diffusion model for the hourly MIBEL spot log-price.

Specification (CONTEXT.md § Pieza 1, reespec 2026-05-24):

    Y_t = log(P_t + c)                                  c = 10 EUR/MWh
    Y_t = θ_t + s(t) + Z_t

where the three components are, from slowest to fastest:

  • θ_t  — causal EMA of Y_t with ``span = EMA_SPAN`` hours (~30 days).
    Absorbs the multi-week / regime-shift component (e.g. the 2022 gas
    crisis) so that κ̂ is not contaminated by long-memory drift.
  • s(t) — deterministic seasonality, fit by OLS on X_t := Y_t − θ_t:
        s(t) = α + Σ_{k=1..K} (a_k cos + b_k sin)(2π k·doy/365.25)
               + Σ_{d=1..6} γ_d 1{dow=d}
               + Σ_{h=1..23} δ_h 1{hod=h}.
  • Z_t — fast OU residual with hour-of-day heteroskedastic vol and Kou
    asymmetric double-exponential jumps:
        dZ_t = -κ Z_t dt + σ_{h(t)} dW_t + J_t dN_t,
        N_t ~ Poisson(λ),
        J_t ~ AsymmetricDoubleExponential(p_up, η_up, η_down).

Calibration pipeline (in :func:`fit`):

    1.  θ̂_t = ``Y_t.ewm(span=EMA_SPAN, adjust=False).mean()`` (causal,
        backward-looking).
    2.  X_t = Y_t − θ̂_t.
    3.  OLS of X_t on the seasonal design matrix → ŝ(t).
    4.  Z_t = X_t − ŝ(t); ΔZ_t are the hourly returns of Z.
    5.  Refined threshold jump detection: dynamic k by hour-of-day
        (``k_peak`` ≈ 6.0 in 18-22 UTC, ``k_base`` ≈ 4.5 elsewhere) with
        an absolute amplitude floor |ΔZ| ≥ ``JUMP_AMPLITUDE_MIN`` = 0.30.
        Iterated until the flagged set stabilises.
    6.  Bounded MLE on the OU (κ, σ_h) using
        ``scipy.optimize.minimize(method='L-BFGS-B')`` with the bounds
        documented below; ``RuntimeError`` if a parameter touches a bound
        or the optimiser fails.
    7.  Bounded MLE on the Kou jump distribution (λ, p_up, η_up, η_down).
    8.  Slow-factor RW parameters drift μ̂, σ̂ from the increments
        Δθ̂_t (used by :func:`simulate` to evolve θ_t forward).

Simulation (in :func:`simulate`) is the inverse composition: a slow RW
for θ_t plus the deterministic ŝ plus the OU+Kou for Z_t, then
``P_t = exp(θ_t + s(t) + Z_t) − c``.

Bounds enforced by the calibration:

  • κ        ∈ [KAPPA_BOUNDS_LO, KAPPA_BOUNDS_HI] = [0.05, 0.20] /h
    (half-life 3.5 h - 14 h), the plausible electricity short-term
    range once the slow factor has been removed.
  • λ        ∈ [LAMBDA_BOUNDS_LO, LAMBDA_BOUNDS_HI] = [0.008, 0.025] /h
    (≈ 70 - 220 jumps / year), the European-power MRJD literature
    range.
  • η_up, η_down ∈ [ETA_BOUNDS_LO, ETA_BOUNDS_HI] = [0.8, 4.0]; the
    upper bound bounds the upward MGF, the lower bound preserves heavy
    tails when the data demands them.

References: Lucia & Schwartz (2002) for the deterministic-seasonal +
OU baseline, Cartea & Figueroa (2005) for the jump-diffusion extension,
Kou (2002) for the asymmetric double-exponential jump distribution.

Overlap with Pieza 2 (Schwartz-Smith): the slow factor θ_t here is a
**purely backward-looking statistic** of the spot series, with no
anchoring to the OMIP forward curve. It is the standalone replacement
for the Schwartz-Smith long-term factor L_t when a forward-curve fit
is not available; once Pieza 2 lands, L_t replaces θ_t for any
valuation that needs to reproduce the live OMIP curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

PRICE_SHIFT: float = 10.0
FOURIER_HARMONICS: int = 4
EMA_SPAN: int = 720

# Refined-detection knobs (Commit G2 wires these into _detect_jumps).
JUMP_THRESHOLD_K_BASE: float = 4.5
JUMP_THRESHOLD_K_PEAK: float = 6.0
JUMP_PEAK_HOURS_UTC: tuple[int, ...] = (18, 19, 20, 21, 22)
JUMP_AMPLITUDE_MIN: float = 0.30
JUMP_MAX_ITER: int = 10

# Bounded-MLE parameter ranges (Commit G3 wires these into the MLE).
KAPPA_BOUNDS: tuple[float, float] = (0.05, 0.20)
LAMBDA_BOUNDS: tuple[float, float] = (0.008, 0.025)
ETA_BOUNDS: tuple[float, float] = (0.8, 4.0)
P_UP_BOUNDS: tuple[float, float] = (0.01, 0.99)
SIGMA_LOWER_BOUND: float = 1e-4


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
class SlowFactorParams:
    """Arithmetic random-walk parameters for the slow factor θ_t.

    Estimated from the historical first differences ``Δθ̂_t``: the drift
    is the sample mean and ``sigma`` the sample standard deviation. The
    walk is stationary in increments, NOT in level — over horizons of a
    few weeks it behaves as a slow trend; over long horizons it
    diverges and must be re-anchored by Pieza 2's L_t.
    """

    drift: float
    sigma: float


@dataclass(frozen=True)
class SpotModelParams:
    """All fitted parameters of the slow-fast MRJD spot model."""

    seasonality: Seasonality
    slow_factor: SlowFactorParams
    price_shift: float
    ema_span: int
    kappa: float
    sigma_by_hour: np.ndarray
    jump_intensity: float
    jump_p_up: float
    jump_eta_up: float
    jump_eta_down: float


@dataclass(frozen=True)
class SpotModelFit:
    """Calibration result: params plus per-observation diagnostics.

    Series fields:
        theta_series       — causal EMA of log(P+c), the slow factor θ̂_t.
        x_series           — log(P+c) − θ̂_t, the post-slow residual X_t.
        seasonal_fitted    — ŝ(t) fit on X_t.
        residuals          — Z_t = X_t − ŝ(t), the fast OU+jumps residual.
        residual_returns   — ΔZ_t, the one-hour innovations of Z_t.
        jumps_mask         — bool on residual_returns.index.
        jump_sizes         — the values of ΔZ_t at flagged jump points.
    """

    params: SpotModelParams
    log_price_index: pd.DatetimeIndex
    theta_series: pd.Series
    x_series: pd.Series
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
    ema_span: int = EMA_SPAN,
    jump_k_base: float = JUMP_THRESHOLD_K_BASE,
    jump_k_peak: float = JUMP_THRESHOLD_K_PEAK,
    jump_peak_hours: tuple[int, ...] = JUMP_PEAK_HOURS_UTC,
    jump_amplitude_min: float = JUMP_AMPLITUDE_MIN,
    jump_max_iter: int = JUMP_MAX_ITER,
) -> SpotModelFit:
    """Fit the slow-fast MRJD model to a UTC-hourly price series.

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
        Number of annual Fourier harmonics in the deterministic
        component.
    ema_span
        Span (in hours) of the causal EMA used for the slow factor θ_t.
    jump_k_base, jump_k_peak, jump_peak_hours, jump_amplitude_min
        Refined-detection knobs forwarded to :func:`_detect_jumps`. The
        peak-hour threshold ``jump_k_peak`` applies to the UTC hours in
        ``jump_peak_hours`` (default 18-22), ``jump_k_base`` everywhere
        else; ``jump_amplitude_min`` is the absolute-log-return floor
        below which σ-threshold breaches are ignored.
    jump_max_iter
        Maximum iterations of the jump detector.

    The first ``ema_span`` observations are dropped from every
    downstream calibration step to avoid the EMA warm-up bias; the
    returned ``theta_series`` and ``x_series`` still cover the full
    input window so the diagnostic plots can show the warm-up phase.
    """
    s = prices.dropna().astype(float)
    if s.index.tz is None:
        raise ValueError("prices must have a UTC-tz-aware DatetimeIndex")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if len(s) < ema_span + 24 * 30:
        raise ValueError(
            f"need at least ema_span + 30 days = {ema_span + 24*30} "
            f"hourly observations, got {len(s)}"
        )

    log_p = np.log(s + price_shift).rename("log_p")
    theta_series = log_p.ewm(span=ema_span, adjust=False).mean().rename("theta")
    x_series = (log_p - theta_series).rename("x")

    # Drop the EMA warm-up. Use post-warmup slices for every estimator,
    # but keep the full theta_series / x_series in the result for plots.
    log_p_pw = log_p.iloc[ema_span:]
    x_series_pw = x_series.iloc[ema_span:]

    seasonality, seasonal_fitted_pw = _fit_seasonality(
        x_series_pw, harmonics=fourier_harmonics,
    )
    seasonal_fitted = seasonal_fitted_pw  # only post-warmup is meaningful

    residuals = (x_series_pw - seasonal_fitted_pw).rename("residuals")
    residual_returns = residuals.diff().dropna().rename("residual_returns")

    jumps_mask = _detect_jumps(
        residual_returns,
        k_base=jump_k_base,
        k_peak=jump_k_peak,
        peak_hours=jump_peak_hours,
        amplitude_min=jump_amplitude_min,
        max_iter=jump_max_iter,
    )
    jump_sizes = residual_returns[jumps_mask].rename("jump_sizes")

    kappa, sigma_by_hour = _mle_ou(residuals, jumps_mask)
    intensity, p_up, eta_up, eta_down = _mle_jumps(
        jump_sizes, n_total_hours=len(residual_returns),
    )

    delta_theta = theta_series.iloc[ema_span:].diff().dropna()
    slow_factor = SlowFactorParams(
        drift=float(delta_theta.mean()),
        sigma=float(delta_theta.std(ddof=1)),
    )

    params = SpotModelParams(
        seasonality=seasonality,
        slow_factor=slow_factor,
        price_shift=price_shift,
        ema_span=ema_span,
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
        theta_series=theta_series,
        x_series=x_series,
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

    Discretisation:

        Z_{t+1} = φ Z_t + σ_{h(t+1)} √((1−φ²)/(2κ)) · ε_t  +  J_t · B_t

    where ε_t ~ N(0, 1), B_t ~ Bernoulli(λ) and J_t is drawn from the
    Kou asymmetric double-exponential (Bernoulli is exact for
    Poisson-per-hour with ``λ Δt ≪ 1``; for ``λ Δt = 0.01`` the
    probability of two jumps in the same hour is below 5e-5 and is
    ignored). Reproducible given ``seed``.
    """
    if start.tz is None:
        raise ValueError("start must be a UTC-tz-aware Timestamp")
    if n_hours <= 0 or n_paths <= 0:
        raise ValueError("n_hours and n_paths must be positive")

    rng = np.random.default_rng(seed=seed)
    idx = pd.date_range(start=start, periods=n_hours, freq="h")
    if idx.tz is None:
        idx = idx.tz_localize(start.tz)

    seasonal_t = _seasonal_predict(params.seasonality, idx).to_numpy()

    kappa = params.kappa
    phi = float(np.exp(-kappa))
    factor = float(np.sqrt((1.0 - phi**2) / (2.0 * kappa)))
    hours = np.asarray(idx.hour)
    sigma_step = params.sigma_by_hour[hours] * factor  # (n_hours,)

    normals = rng.standard_normal((n_paths, n_hours))
    jump_occurs = rng.random((n_paths, n_hours)) < params.jump_intensity
    n_jumps_total = int(jump_occurs.sum())
    jump_grid = np.zeros((n_paths, n_hours))
    if n_jumps_total > 0:
        is_up = rng.random(n_jumps_total) < params.jump_p_up
        sizes_up = rng.exponential(1.0 / params.jump_eta_up, n_jumps_total)
        sizes_down = -rng.exponential(1.0 / params.jump_eta_down, n_jumps_total)
        jump_sizes_flat = np.where(is_up, sizes_up, sizes_down)
        jump_grid[jump_occurs] = jump_sizes_flat

    z = np.empty((n_paths, n_hours))
    z[:, 0] = initial_residual
    for t in range(1, n_hours):
        z[:, t] = (
            phi * z[:, t - 1]
            + sigma_step[t] * normals[:, t]
            + jump_grid[:, t]
        )

    log_p = seasonal_t[np.newaxis, :] + z
    return np.exp(log_p) - params.price_shift


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
    k_base: float = JUMP_THRESHOLD_K_BASE,
    k_peak: float = JUMP_THRESHOLD_K_PEAK,
    peak_hours: tuple[int, ...] = JUMP_PEAK_HOURS_UTC,
    amplitude_min: float = JUMP_AMPLITUDE_MIN,
    max_iter: int = JUMP_MAX_ITER,
) -> pd.Series:
    """Iterative threshold jump detector with σ updated by hour-of-day,
    a peak-hour-specific threshold and an absolute amplitude floor.

    At each iteration the per-hour mean μ_h and standard deviation σ_h
    are computed from the currently un-flagged returns. A return at
    hour ``h(t)`` is flagged a jump when BOTH:

        |ΔZ_t − μ_{h(t)}| > k_h · σ_{h(t)}    (dynamic σ-threshold)
        |ΔZ_t|             > amplitude_min     (absolute floor in log)

    where ``k_h = k_peak`` if ``h ∈ peak_hours`` else ``k_base``. The
    peak-hour threshold targets the evening price band (default
    18-22 UTC ≈ 20-24 local in CET summer) where intraday volatility
    is structurally higher and the base threshold over-flags ordinary
    returns. The amplitude floor knocks out any sub-0.30-log "jumps"
    that the threshold step might pick up on quiet periods.

    Iteration stops when the flagged set stabilises or ``max_iter`` is
    reached. Hours with fewer than five non-jump observations are
    skipped on the current iteration.
    """
    if residual_returns.index.tz is None:
        raise ValueError("residual_returns must be UTC-tz-aware")

    hours = np.asarray(residual_returns.index.hour)
    values = residual_returns.to_numpy(dtype=float)
    is_jump = np.zeros(len(values), dtype=bool)
    peak_set = set(int(h) for h in peak_hours)

    abs_floor_mask = np.abs(values) > amplitude_min

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
            k_h = k_peak if h in peak_set else k_base
            threshold_breached = np.abs(values - mu_h) > k_h * sd_h
            new_is_jump[mask_h] = mask_h[mask_h] & threshold_breached[mask_h] & abs_floor_mask[mask_h]
        if np.array_equal(new_is_jump, is_jump):
            break
        is_jump = new_is_jump

    return pd.Series(is_jump, index=residual_returns.index, name="is_jump")


def _mle_ou(
    residuals: pd.Series,
    jump_mask: pd.Series,
    dt_hours: float = 1.0,
    *,
    kappa_bounds: tuple[float, float] = KAPPA_BOUNDS,
    sigma_lower: float = SIGMA_LOWER_BOUND,
) -> tuple[float, np.ndarray]:
    """Bounded MLE for the OU process on non-jump observations.

    Optimises the discrete AR(1) Gaussian log-likelihood over
    ``(κ, σ_0, σ_1, …, σ_23)`` with ``scipy.optimize.minimize(method=
    'L-BFGS-B')`` and the project bounds (``KAPPA_BOUNDS`` on κ,
    ``[sigma_lower, +∞)`` on each σ_h).

    Per-observation density:

        Z_{t+1} | Z_t  ~  N(φ Z_t, σ_{h(t+1)}^2 · (1 − φ^2) / (2 κ)),
        φ = exp(-κ · dt_hours).

    A closed-form AR(1) estimate of (κ, σ_h) is used as the warm start
    (clipped into the bounds). The function raises ``RuntimeError`` if
    the optimiser does not converge OR if κ̂ touches either of its
    bounds within tolerance 1e-3, with the bound-active path naming
    which side was hit so the caller can decide whether to widen the
    bound or re-examine the slow-factor extraction.
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
    hours_nj = np.asarray(residuals.index[1:].hour)[nonjump]

    # Warm start: closed-form AR(1) clipped into bounds.
    denom = float((z_t_nj * z_t_nj).sum())
    if denom <= 0:
        raise ValueError("Degenerate residual series (zero variance)")
    phi0_raw = float((z_t_nj * z_tp1_nj).sum() / denom)
    phi0 = float(np.clip(
        phi0_raw,
        np.exp(-kappa_bounds[1] * dt_hours),
        np.exp(-kappa_bounds[0] * dt_hours),
    ))
    kappa0 = -np.log(phi0) / dt_hours
    eta0 = z_tp1_nj - phi0 * z_t_nj
    factor0 = (2.0 * kappa0) / (1.0 - phi0**2)
    sigma0 = np.full(24, np.nan)
    for h in range(24):
        eh = eta0[hours_nj == h]
        if len(eh) >= 5:
            sigma0[h] = np.sqrt(float(eh.var(ddof=1)) * factor0)
    overall = float(np.nanmedian(sigma0))
    sigma0 = np.where(np.isnan(sigma0), overall, sigma0)
    sigma0 = np.clip(sigma0, sigma_lower * 1.1, None)

    bounds = [kappa_bounds] + [(sigma_lower, None)] * 24
    x0 = np.concatenate([[kappa0], sigma0])

    diff_innov = z_tp1_nj - 0.0  # placeholder; computed inside neg_log_lik

    def neg_log_lik(x: np.ndarray) -> float:
        kappa = x[0]
        sigma_h = x[1:]
        phi = float(np.exp(-kappa * dt_hours))
        sig2 = sigma_h**2 * (1.0 - phi**2) / (2.0 * kappa)
        sig2_obs = sig2[hours_nj]
        innov = z_tp1_nj - phi * z_t_nj
        return float(
            0.5 * np.sum(np.log(2.0 * np.pi * sig2_obs) + innov**2 / sig2_obs)
        )

    result = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"OU MLE did not converge: {result.message}")

    kappa_hat = float(result.x[0])
    sigma_hat = np.asarray(result.x[1:], dtype=float)

    tol = 1e-3
    lo, hi = kappa_bounds
    if abs(kappa_hat - lo) < tol:
        raise RuntimeError(
            f"OU MLE: kappa_hat={kappa_hat:.6f} hit lower bound {lo}. "
            "Series may have memory longer than the bound allows; "
            "check the slow-factor extraction (EMA_SPAN) or widen "
            "KAPPA_BOUNDS."
        )
    if abs(kappa_hat - hi) < tol:
        raise RuntimeError(
            f"OU MLE: kappa_hat={kappa_hat:.6f} hit upper bound {hi}. "
            "Series mean-reverts faster than the bound allows; widen "
            "KAPPA_BOUNDS."
        )
    return kappa_hat, sigma_hat


def _mle_jumps(
    jump_returns: pd.Series,
    n_total_hours: int,
    *,
    lambda_bounds: tuple[float, float] = LAMBDA_BOUNDS,
    eta_bounds: tuple[float, float] = ETA_BOUNDS,
    p_up_bounds: tuple[float, float] = P_UP_BOUNDS,
) -> tuple[float, float, float, float]:
    """Bounded MLE for the Poisson rate + Kou asymmetric double-exponential.

    Joint optimisation over ``(λ, p_up, η_up, η_down)`` of the
    Poisson-times-Kou log-likelihood

        log L = n_jumps · log λ − λ · n_total_hours
              + n_up   · log(p_up · η_up)     − η_up   · Σ J_up
              + n_down · log((1−p_up)·η_down) − η_down · Σ |J_down|

    via ``scipy.optimize.minimize(method='L-BFGS-B')`` with the project
    bounds (LAMBDA_BOUNDS on λ, ETA_BOUNDS on both η, P_UP_BOUNDS on
    p_up). Raises ``RuntimeError`` on non-convergence OR if any of
    λ, η_up, η_down touches a bound within tolerance 1e-3 (p_up is
    informational only — hitting its bounds means a one-sided sample,
    which is realistic).

    When no jumps were detected, returns the lower bound for λ and
    neutral defaults for the Kou parameters (η = 1.0, p_up = 0.5);
    the caller may want to treat this as a degenerate calibration.
    """
    if n_total_hours <= 0:
        raise ValueError("n_total_hours must be positive")
    n_jumps = len(jump_returns)
    if n_jumps == 0:
        return lambda_bounds[0], 0.5, 1.0, 1.0

    j = jump_returns.to_numpy(dtype=float)
    j_up = j[j > 0]
    j_down = j[j < 0]
    n_up = int(len(j_up))
    n_down = int(len(j_down))
    sum_up = float(j_up.sum())
    sum_down_abs = float((-j_down).sum())

    # Warm start (clipped into bounds, interior margin 1e-3 to avoid
    # starting AT a bound which can keep L-BFGS-B pinned).
    margin = 1e-3
    lam0 = float(np.clip(
        n_jumps / n_total_hours,
        lambda_bounds[0] + margin, lambda_bounds[1] - margin,
    ))
    p_up0 = float(np.clip(
        n_up / n_jumps,
        p_up_bounds[0] + margin, p_up_bounds[1] - margin,
    ))
    eta_up0 = (
        float(np.clip(
            1.0 / (sum_up / max(n_up, 1)),
            eta_bounds[0] + margin, eta_bounds[1] - margin,
        ))
        if n_up >= 1 else (eta_bounds[0] + eta_bounds[1]) / 2.0
    )
    eta_down0 = (
        float(np.clip(
            1.0 / (sum_down_abs / max(n_down, 1)),
            eta_bounds[0] + margin, eta_bounds[1] - margin,
        ))
        if n_down >= 1 else (eta_bounds[0] + eta_bounds[1]) / 2.0
    )

    bounds = [lambda_bounds, p_up_bounds, eta_bounds, eta_bounds]
    x0 = np.array([lam0, p_up0, eta_up0, eta_down0])

    def neg_log_lik(x: np.ndarray) -> float:
        lam, p_up, eta_up, eta_down = x
        ll = n_jumps * np.log(lam) - lam * n_total_hours
        if n_up >= 1:
            ll = ll + n_up * np.log(p_up * eta_up) - eta_up * sum_up
        if n_down >= 1:
            ll = ll + n_down * np.log((1.0 - p_up) * eta_down) - eta_down * sum_down_abs
        return -float(ll)

    result = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Kou MLE did not converge: {result.message}")

    lam_hat, p_up_hat, eta_up_hat, eta_down_hat = (float(v) for v in result.x)
    tol = 1e-3
    for name, val, (lo, hi) in [
        ("lambda", lam_hat, lambda_bounds),
        ("eta_up", eta_up_hat, eta_bounds),
        ("eta_down", eta_down_hat, eta_bounds),
    ]:
        if abs(val - lo) < tol:
            raise RuntimeError(
                f"Kou MLE: {name}={val:.6f} hit lower bound {lo}; "
                f"sample is at the edge of the assumed parameter space."
            )
        if abs(val - hi) < tol:
            raise RuntimeError(
                f"Kou MLE: {name}={val:.6f} hit upper bound {hi}; "
                f"sample is at the edge of the assumed parameter space."
            )
    return lam_hat, p_up_hat, eta_up_hat, eta_down_hat
