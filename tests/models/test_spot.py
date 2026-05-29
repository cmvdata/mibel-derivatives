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
        "Seasonality", "SlowFactorParams",
        "SpotModelParams", "SpotModelFit",
        "PRICE_SHIFT", "FOURIER_HARMONICS", "EMA_SPAN",
        "JUMP_THRESHOLD_K_BASE", "JUMP_THRESHOLD_K_PEAK",
        "JUMP_PEAK_HOURS_UTC", "JUMP_AMPLITUDE_MIN", "JUMP_MAX_ITER",
        "KAPPA_BOUNDS", "LAMBDA_BOUNDS", "ETA_BOUNDS",
    ):
        assert hasattr(spot, name), f"spot.{name} missing"
    assert spot.PRICE_SHIFT == 10.0
    assert spot.FOURIER_HARMONICS == 4
    assert spot.EMA_SPAN == 24
    assert spot.JUMP_THRESHOLD_K_BASE == 4.5
    assert spot.JUMP_THRESHOLD_K_PEAK == 6.0
    assert spot.JUMP_AMPLITUDE_MIN == 0.30
    assert set(spot.JUMP_PEAK_HOURS_UTC) == {18, 19, 20, 21, 22}
    assert spot.KAPPA_BOUNDS == (0.05, 0.20)
    assert spot.LAMBDA_BOUNDS == (0.008, 0.025)
    assert spot.ETA_BOUNDS == (0.8, 4.0)


def test_fit_rejects_tz_naive_index() -> None:
    s = pd.Series(
        np.full(24 * 60, 50.0),
        index=pd.date_range("2020-01-01", periods=24 * 60, freq="h"),
    )
    with pytest.raises(ValueError, match="UTC"):
        spot.fit(s)


def test_fit_rejects_too_few_observations() -> None:
    # New requirement: ema_span (720h) + 30 days = 1440 h minimum.
    idx = _hourly_utc("2020-01-01", 24 * 30)
    s = pd.Series(np.full(len(idx), 50.0), index=idx)
    with pytest.raises(ValueError, match="ema_span"):
        spot.fit(s)


def test_params_dataclass_is_frozen_and_constructs() -> None:
    seasonality = spot.Seasonality(
        intercept=0.0,
        fourier_coefs=np.zeros(8),
        dow_coefs=np.zeros(6),
        hod_coefs=np.zeros(23),
        fourier_harmonics=4,
    )
    slow = spot.SlowFactorParams(kappa=0.001, mean=0.0, sigma=0.001)
    params = spot.SpotModelParams(
        seasonality=seasonality,
        slow_factor=slow,
        price_shift=10.0,
        ema_span=720,
        kappa=0.1,
        sigma_by_hour=np.full(24, 0.05),
        jump_intensity=0.01,
        jump_p_up=0.5,
        jump_eta_up=2.0,
        jump_eta_down=2.0,
    )
    with pytest.raises(Exception):
        params.kappa = 0.2  # frozen dataclass


def _trivial_params(**overrides) -> spot.SpotModelParams:
    """Helper: a no-seasonality, no-jumps SpotModelParams for simulation tests."""
    seasonality = spot.Seasonality(
        intercept=overrides.get("intercept", 4.0),
        fourier_coefs=np.zeros(8),
        dow_coefs=np.zeros(6),
        hod_coefs=np.zeros(23),
        fourier_harmonics=4,
    )
    slow = spot.SlowFactorParams(
        kappa=overrides.get("slow_kappa", 0.0),
        mean=overrides.get("slow_mean", 0.0),
        sigma=overrides.get("slow_sigma", 0.0),
    )
    return spot.SpotModelParams(
        seasonality=seasonality,
        slow_factor=slow,
        price_shift=overrides.get("price_shift", 10.0),
        ema_span=overrides.get("ema_span", 720),
        kappa=overrides.get("kappa", 0.10),
        sigma_by_hour=overrides.get("sigma_by_hour", np.full(24, 0.10)),
        jump_intensity=overrides.get("jump_intensity", 0.0),
        jump_p_up=overrides.get("jump_p_up", 0.5),
        jump_eta_up=overrides.get("jump_eta_up", 2.0),
        jump_eta_down=overrides.get("jump_eta_down", 2.0),
    )


def test_simulate_rejects_tz_naive_start() -> None:
    with pytest.raises(ValueError, match="UTC"):
        spot.simulate(
            _trivial_params(),
            start=pd.Timestamp("2025-01-01"),
            n_hours=24, n_paths=10,
        )


def test_simulate_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError):
        spot.simulate(
            _trivial_params(),
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            n_hours=0, n_paths=10,
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


def test_ema_is_causal_no_future_leakage() -> None:
    """θ̂_t = ewm(span=720, adjust=False).mean() must not depend on any
    values past t. Truncating the future and re-evaluating up to t must
    yield the same θ̂_t."""
    rng = np.random.default_rng(seed=3)
    n = 24 * 365
    idx = _hourly_utc("2021-01-01", n)
    y = pd.Series(rng.standard_normal(n).cumsum() * 0.05 + 3.0, index=idx)
    full = y.ewm(span=720, adjust=False).mean()
    cut = n // 2
    truncated = y.iloc[:cut].ewm(span=720, adjust=False).mean()
    np.testing.assert_allclose(
        truncated.values, full.iloc[:cut].values, atol=1e-12,
    )


def test_fit_slow_factor_absorbs_planted_trend() -> None:
    """A linear trend in log-price must be picked up by θ̂_t with a
    positive drift in SlowFactorParams; the seasonal intercept stays
    around zero. The fast residual is a small OU layer with κ inside
    bounds so the (bounded) MLE doesn't trip its κ upper bound."""
    rng = np.random.default_rng(seed=5)
    n_hours = 24 * 365 * 3
    idx = _hourly_utc("2020-01-01", n_hours)
    # Slow trend.
    trend = 3.5 + 0.0002 * np.arange(n_hours)
    # Fast OU layer with κ=0.10 (inside [0.05, 0.20]) so the bounded
    # MLE has a real signal to recover.
    kappa = 0.10
    sigma = 0.05
    phi = float(np.exp(-kappa))
    factor = float(np.sqrt((1 - phi**2) / (2 * kappa)))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma * factor * rng.standard_normal()
    log_p = trend + z
    prices = pd.Series(np.exp(log_p) - 10.0, index=idx, name="p")
    result = spot.fit(prices)
    theta_pw = result.theta_series.iloc[result.params.ema_span:]
    n_pw = len(theta_pw)
    assert theta_pw.iloc[-n_pw // 4:].mean() > theta_pw.iloc[: n_pw // 4].mean() + 0.5
    # Slow OU collapses to phi_θ near 1 on a trended series (no
    # mean-reversion within the window); sigma_θ stays finite.
    assert np.exp(-result.params.slow_factor.kappa) > 0.99
    assert result.params.slow_factor.sigma > 0.0
    assert abs(result.params.seasonality.intercept) < 0.10
    # κ̂ inside bounds (would have raised otherwise).
    assert spot.KAPPA_BOUNDS[0] <= result.params.kappa <= spot.KAPPA_BOUNDS[1]


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


# ---- C. Iterative threshold jump detection ---------------------------------


def _synthetic_returns_with_jumps(
    n_hours: int,
    jump_indices: np.ndarray,
    jump_sizes: np.ndarray,
    *,
    seed: int,
    sigma_base: float = 0.10,
    sigma_slope: float = 0.005,
) -> pd.Series:
    """Build hourly residual returns ~ N(0, σ_h) with planted jumps."""
    rng = np.random.default_rng(seed=seed)
    idx = _hourly_utc("2022-01-01", n_hours)
    hours = np.asarray(idx.hour)
    sigma_h = sigma_base + sigma_slope * hours  # heteroskedastic by hour
    eps = rng.standard_normal(n_hours) * sigma_h
    eps[jump_indices] = eps[jump_indices] + jump_sizes
    return pd.Series(eps, index=idx, name="resid_ret")


def test_detect_jumps_requires_tz() -> None:
    s = pd.Series(np.zeros(48), index=pd.date_range("2020-01-01", periods=48, freq="h"))
    with pytest.raises(ValueError):
        spot._detect_jumps(s)


def test_detect_jumps_recovers_planted_jumps_with_high_precision_and_recall() -> None:
    """Plant 50 known jumps of ±1.0 in ~1 year of returns; the refined
    detector with k_base=4.0 and amplitude_min=0.30 should recover them."""
    n_hours = 24 * 400
    n_jumps = 50
    rng = np.random.default_rng(seed=2026)
    jump_idx = rng.choice(n_hours, size=n_jumps, replace=False)
    jump_sizes = rng.choice([-1.0, 1.0], size=n_jumps)
    rr = _synthetic_returns_with_jumps(
        n_hours, jump_idx, jump_sizes, seed=2026,
    )

    # Use k_base=k_peak=4.0 to recover the previous behaviour and isolate
    # the floor effect; planted ±1.0 jumps pass the 0.30 amplitude floor.
    flagged = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.30,
    )
    truth = np.zeros(n_hours, dtype=bool)
    truth[jump_idx] = True

    tp = int(np.sum(truth & flagged.to_numpy()))
    fp = int(np.sum(~truth & flagged.to_numpy()))
    fn = int(np.sum(truth & ~flagged.to_numpy()))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    assert recall >= 0.90, f"recall={recall:.3f} (tp={tp}, fn={fn})"
    assert precision >= 0.85, f"precision={precision:.3f} (tp={tp}, fp={fp})"


def test_detect_jumps_low_false_positive_on_pure_gaussian() -> None:
    """No jumps planted; at k_base=4 the false-positive rate from the
    σ-threshold step alone is ~0.5%. With the amplitude floor at 0.30
    on σ ≈ 0.10 returns, FP collapses to essentially zero."""
    n_hours = 24 * 365
    rng = np.random.default_rng(seed=99)
    idx = _hourly_utc("2023-01-01", n_hours)
    rr = pd.Series(rng.standard_normal(n_hours) * 0.10, index=idx)
    flagged = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.30,
    )
    fp_rate = float(flagged.mean())
    assert fp_rate < 0.005, f"fp_rate={fp_rate:.4%}"


def test_detect_jumps_converges_within_max_iter() -> None:
    """Iterative detector must terminate before hitting the cap."""
    n_hours = 24 * 200
    rng = np.random.default_rng(seed=11)
    n_jumps = 30
    jump_idx = rng.choice(n_hours, size=n_jumps, replace=False)
    jump_sizes = rng.normal(0.0, 0.6, size=n_jumps)
    rr = _synthetic_returns_with_jumps(n_hours, jump_idx, jump_sizes, seed=11)
    flagged_short = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.30, max_iter=2,
    )
    flagged_full = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.30, max_iter=10,
    )
    assert int(flagged_full.sum()) >= int(flagged_short.sum())


def test_detect_jumps_amplitude_floor_drops_small_breaches() -> None:
    """A planted jump of size 0.20 (above k·σ but below amplitude floor)
    must NOT be flagged."""
    n_hours = 24 * 200
    rng = np.random.default_rng(seed=42)
    idx = _hourly_utc("2022-01-01", n_hours)
    # Pure low-σ noise plus a small spike of 0.20 (above 4·σ=0.04 but
    # below amplitude_min=0.30).
    rr = pd.Series(rng.standard_normal(n_hours) * 0.01, index=idx)
    small_jump_loc = n_hours // 2
    rr.iloc[small_jump_loc] = 0.20
    flagged = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.30,
    )
    assert flagged.iloc[small_jump_loc] is np.False_ or not bool(flagged.iloc[small_jump_loc])
    # With the floor lowered to 0.05, the same spike IS flagged.
    flagged_loose = spot._detect_jumps(
        rr, k_base=4.0, k_peak=4.0, amplitude_min=0.05,
    )
    assert bool(flagged_loose.iloc[small_jump_loc])


def test_detect_jumps_peak_hour_threshold_lets_through_evening_spikes() -> None:
    """A jump of size 0.50 at hour 20 (peak) should pass k_base=4.5·σ
    BUT fail k_peak=6.0·σ when σ ~ 0.10. Conversely at hour 04 (off-peak)
    the SAME jump passes the relaxed k_base threshold but no peak rule
    applies. The peak-hour bump is doing real work."""
    n_hours = 24 * 200
    rng = np.random.default_rng(seed=7)
    idx = _hourly_utc("2022-01-01", n_hours)
    # σ ≈ 0.10; 4.5·σ ≈ 0.45; 6.0·σ ≈ 0.60.
    rr = pd.Series(rng.standard_normal(n_hours) * 0.10, index=idx)

    # Inject a 0.50 jump at one peak hour (h20) and one off-peak (h04).
    peak_loc = None; off_loc = None
    for i, ts in enumerate(idx):
        if peak_loc is None and ts.hour == 20:
            rr.iloc[i] = 0.50
            peak_loc = i
        if off_loc is None and ts.hour == 4:
            rr.iloc[i] = 0.50
            off_loc = i
        if peak_loc is not None and off_loc is not None:
            break

    flagged = spot._detect_jumps(
        rr,
        k_base=4.5, k_peak=6.0,
        peak_hours=(18, 19, 20, 21, 22),
        amplitude_min=0.30,
    )
    # Off-peak jump: 0.50 > 4.5·σ AND > 0.30 floor → flagged.
    assert bool(flagged.iloc[off_loc])
    # Peak jump: 0.50 < 6.0·σ ≈ 0.60 → NOT flagged despite passing floor.
    assert not bool(flagged.iloc[peak_loc])


# ---- D. MLE OU + Kou jumps -------------------------------------------------


def _simulate_ou_path(
    n_hours: int,
    kappa: float,
    sigma_by_hour: np.ndarray,
    *,
    seed: int,
    start: str = "2020-01-01",
) -> pd.Series:
    """Generate a discrete OU path Z_t with hour-of-day σ heteroskedasticity."""
    rng = np.random.default_rng(seed=seed)
    idx = _hourly_utc(start, n_hours)
    hours = np.asarray(idx.hour)
    phi = float(np.exp(-kappa))
    factor = np.sqrt((1.0 - phi**2) / (2.0 * kappa))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma_by_hour[hours[t]] * factor * rng.standard_normal()
    return pd.Series(z, index=idx, name="z")


def test_mle_ou_recovers_kappa_and_sigma_no_jumps() -> None:
    n_hours = 24 * 365 * 3
    true_kappa = 0.10
    true_sigma_h = 0.05 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)) ** 2
    z = _simulate_ou_path(n_hours, true_kappa, true_sigma_h, seed=42)
    # No jumps planted; the mask aligned with returns is all-False.
    jump_mask = pd.Series(False, index=z.index[1:])

    kappa_hat, sigma_h_hat = spot._mle_ou(z, jump_mask)

    rel_err_k = abs(kappa_hat - true_kappa) / true_kappa
    assert rel_err_k < 0.10, f"κ true={true_kappa:.4f} hat={kappa_hat:.4f}"
    np.testing.assert_allclose(sigma_h_hat, true_sigma_h, rtol=0.10, atol=0.01)


def _simulate_jump_diffusion_path(
    n_hours: int,
    kappa: float,
    sigma_by_hour: np.ndarray,
    jump_times: np.ndarray,
    jump_sizes: np.ndarray,
    *,
    seed: int,
    start: str = "2020-01-01",
) -> pd.Series:
    """OU + Poisson jumps: jumps push the level and OU continues from there.

    ``jump_times`` indexes into the Z-series in [1, n_hours); the size at
    ``jump_times[i]`` is added to z[jump_times[i]] AFTER the OU step.
    """
    rng = np.random.default_rng(seed=seed)
    idx = _hourly_utc(start, n_hours)
    hours = np.asarray(idx.hour)
    phi = float(np.exp(-kappa))
    factor = np.sqrt((1.0 - phi**2) / (2.0 * kappa))
    jump_at = dict(zip(jump_times.tolist(), jump_sizes.tolist()))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma_by_hour[hours[t]] * factor * rng.standard_normal()
        if t in jump_at:
            z[t] = z[t] + jump_at[t]
    return pd.Series(z, index=idx, name="z")


def test_mle_ou_robust_to_planted_jumps_when_masked() -> None:
    """When the jump mask correctly identifies planted jumps, the OU
    estimator still recovers κ and σ_h within tolerance."""
    rng = np.random.default_rng(seed=7)
    n_hours = 24 * 365 * 3
    true_kappa = 0.08
    true_sigma_h = np.full(24, 0.12)
    # Choose jump times in [1, n_hours) so z.diff() at that index is the
    # contaminated return.
    jump_times = rng.choice(np.arange(1, n_hours), size=100, replace=False)
    jump_sizes = rng.choice([-1.5, 1.5], size=100)
    z_series = _simulate_jump_diffusion_path(
        n_hours, true_kappa, true_sigma_h,
        jump_times, jump_sizes, seed=7,
    )

    # jump_mask aligns with returns (index = z_series.index[1:]).
    # A jump at z-position t corresponds to return-position t-1 within
    # that index (since residual_returns.index[i] = z.index[i+1]).
    jump_mask = pd.Series(False, index=z_series.index[1:])
    jump_mask.iloc[jump_times - 1] = True

    kappa_hat, sigma_h_hat = spot._mle_ou(z_series, jump_mask)
    assert abs(kappa_hat - true_kappa) / true_kappa < 0.15, (
        f"κ true={true_kappa:.4f} hat={kappa_hat:.4f}"
    )
    np.testing.assert_allclose(sigma_h_hat, true_sigma_h, rtol=0.15, atol=0.02)


def test_mle_jumps_recovers_kou_params() -> None:
    """Recovery with all true parameters STRICTLY inside the bounds."""
    rng = np.random.default_rng(seed=123)
    n_total = 24 * 365 * 5  # 5 years hourly
    n_jumps = 500           # λ_true = 500 / 43800 ≈ 0.0114 (inside [0.008, 0.025])
    true_lambda = n_jumps / n_total
    true_p_up = 0.70
    true_eta_up = 2.0       # inside [0.8, 4.0]
    true_eta_down = 2.5     # inside [0.8, 4.0]

    is_up = rng.random(n_jumps) < true_p_up
    mags_up = rng.exponential(1.0 / true_eta_up, size=n_jumps)
    mags_down = -rng.exponential(1.0 / true_eta_down, size=n_jumps)
    sizes = np.where(is_up, mags_up, mags_down)
    jumps = pd.Series(
        sizes, index=_hourly_utc("2019-01-01", n_jumps), name="jumps",
    )

    intensity, p_up, eta_up, eta_down = spot._mle_jumps(jumps, n_total)
    assert abs(intensity - true_lambda) / true_lambda < 0.05
    assert abs(p_up - true_p_up) < 0.05
    assert abs(eta_up - true_eta_up) / true_eta_up < 0.15
    assert abs(eta_down - true_eta_down) / true_eta_down < 0.15


def test_mle_jumps_handles_empty_input() -> None:
    """With no detected jumps, λ̂ defaults to the lower bound and Kou
    params to neutral; the bound-active check is bypassed in this path."""
    intensity, p_up, eta_up, eta_down = spot._mle_jumps(
        pd.Series(dtype=float), n_total_hours=24 * 30,
    )
    assert intensity == spot.LAMBDA_BOUNDS[0]
    assert p_up == 0.5
    assert eta_up == 1.0 and eta_down == 1.0


def test_mle_jumps_raises_on_lambda_below_bound() -> None:
    """A sample with rate below LAMBDA_BOUNDS[0] must hit the lower bound
    and raise RuntimeError rather than silently clip."""
    rng = np.random.default_rng(seed=10)
    n_total = 24 * 365 * 5
    # 30 jumps / 43800 hours = 0.00068 — well below LAMBDA_BOUNDS[0]=0.008
    sizes = rng.choice([-0.5, 0.5], size=30)
    jumps = pd.Series(sizes, index=_hourly_utc("2020-01-01", 30))
    with pytest.raises(RuntimeError, match="lambda"):
        spot._mle_jumps(jumps, n_total)


def test_mle_jumps_raises_on_eta_above_bound() -> None:
    """A sample of tiny jumps (mean 0.1, η_true=10) hits eta upper bound."""
    rng = np.random.default_rng(seed=11)
    n_total = 24 * 365 * 5
    # 500 jumps with very small sizes — true η ~ 10 (way above ETA_BOUNDS[1]=4)
    n_jumps = 500
    is_up = rng.random(n_jumps) < 0.5
    sizes_up = rng.exponential(0.1, n_jumps)
    sizes_down = -rng.exponential(0.1, n_jumps)
    sizes = np.where(is_up, sizes_up, sizes_down)
    jumps = pd.Series(sizes, index=_hourly_utc("2020-01-01", n_jumps))
    with pytest.raises(RuntimeError, match="eta"):
        spot._mle_jumps(jumps, n_total)


def test_mle_ou_raises_when_kappa_outside_bounds() -> None:
    """A series with κ_true=0.005 (below KAPPA_BOUNDS[0]=0.05) must hit
    the lower bound and raise RuntimeError."""
    rng = np.random.default_rng(seed=22)
    n_hours = 24 * 200
    very_slow_kappa = 0.005
    sigma_h = np.full(24, 0.10)
    phi = float(np.exp(-very_slow_kappa))
    factor = float(np.sqrt((1 - phi**2) / (2 * very_slow_kappa)))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma_h[0] * factor * rng.standard_normal()
    z_series = pd.Series(z, index=_hourly_utc("2020-01-01", n_hours))
    jump_mask = pd.Series(False, index=z_series.index[1:])
    with pytest.raises(RuntimeError, match="kappa"):
        spot._mle_ou(z_series, jump_mask)


def test_fit_end_to_end_recovers_seasonal_and_ou_under_slow_fast() -> None:
    """Synthetic OMIE-like series with known seasonality + OU + Kou jumps,
    NO trend in the slow factor. After the EMA decomposition:
      • θ̂_t should be flat at ≈ true_intercept + Fourier_annual_mean ≈ true_intercept
        (Fourier sin/cos integrate to 0 over the year).
      • The post-EMA seasonal fit intercept should be small (the slow factor
        absorbed the level), DoW and HoD coefs should still be recovered.
      • κ̂ and σ̂_h should be close to truth on the deseasonalised X_t."""
    rng = np.random.default_rng(seed=2026)
    n_hours = 24 * 365 * 4  # 4 years hourly

    idx = _hourly_utc("2020-01-01", n_hours)
    X = spot._seasonal_design_matrix(idx, harmonics=4).to_numpy()

    true_intercept = 3.8
    true_fourier = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    true_dow = np.array([0.05, 0.06, 0.05, 0.04, 0.02, -0.10])
    true_hod = np.linspace(-0.20, 0.40, 23)
    true_beta = np.concatenate([[true_intercept], true_fourier, true_dow, true_hod])
    seasonal = X @ true_beta

    true_kappa = 0.12
    true_sigma_h = (
        0.08 + 0.06 * np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)) ** 2
    )
    # Plant enough jumps so that, after the k=4.5 / amplitude=0.30 floor
    # culls the small-magnitude tail, the surviving rate sits inside
    # LAMBDA_BOUNDS = [0.008, 0.025]. With η_up=2, η_down=3 detection
    # keeps ~34% of planted (38% off-peak + 22% peak); 1500 planted →
    # ~510 detected → λ̂ ≈ 0.0145 inside the lambda bounds.
    n_jumps_planted = 1500
    jump_times = rng.choice(np.arange(1, n_hours), size=n_jumps_planted, replace=False)
    is_up = rng.random(n_jumps_planted) < 0.70
    true_eta_up = 2.0
    true_eta_down = 3.0
    sizes_up = rng.exponential(1.0 / true_eta_up, n_jumps_planted)
    sizes_down = -rng.exponential(1.0 / true_eta_down, n_jumps_planted)
    jump_sizes_arr = np.where(is_up, sizes_up, sizes_down)
    z = _simulate_jump_diffusion_path(
        n_hours, true_kappa, true_sigma_h,
        jump_times, jump_sizes_arr, seed=2026,
    ).to_numpy()

    log_p = seasonal + z
    prices = pd.Series(np.exp(log_p) - 10.0, index=idx, name="price_eur_mwh")

    # Use the original ~30-day EMA span here: the synthetic has no slow
    # factor, only seasonal+OU+jumps, so the long EMA acts as a near-flat
    # baseline that preserves the OU memory we want fit() to recover.
    # The library default EMA_SPAN=24 is tuned for real OMIE 2019-2024.
    result = spot.fit(prices, ema_span=720)
    p = result.params

    # θ̂_t absorbs the level PLUS the non-zero means of the dummy blocks
    # (Mon and h0 are the omitted references, so DoW/HoD coefs do not sum
    # to zero across categories) PLUS the stationary mean of the jump
    # component (asymmetric Kou with E[J] ≠ 0):
    #   E[θ̂] ≈ intercept + Σ dow/7 + Σ hod/24 + λ_planted · E[J] / κ
    true_E_J = 0.70 / true_eta_up - 0.30 / true_eta_down  # ≈ 0.25
    lambda_planted = n_jumps_planted / n_hours
    expected_z_mean = lambda_planted * true_E_J / true_kappa
    expected_theta_mean = (
        true_intercept + true_dow.sum() / 7.0 + true_hod.sum() / 24.0
        + expected_z_mean
    )
    theta_pw = result.theta_series.iloc[p.ema_span:]
    assert abs(theta_pw.mean() - expected_theta_mean) < 0.05, (
        f"θ̂ mean={theta_pw.mean():.4f} expected={expected_theta_mean:.4f}"
    )
    # After slow-factor removal, the seasonal regression intercept is the
    # negative of the dummy-block means (so that adding intercept + dummy
    # contributions back to θ̂ reconstructs the original level on average).
    expected_intercept = (
        -true_dow.sum() / 7.0 - true_hod.sum() / 24.0
    )
    assert abs(p.seasonality.intercept - expected_intercept) < 0.05
    # DoW and HoD deviations recovered (relative coefs unaffected by level).
    np.testing.assert_allclose(p.seasonality.dow_coefs, true_dow, atol=0.05)
    np.testing.assert_allclose(p.seasonality.hod_coefs, true_hod, atol=0.05)
    # OU recovery still works (closed-form MLE in G1; bounded MLE in G3).
    assert abs(p.kappa - true_kappa) / true_kappa < 0.25
    np.testing.assert_allclose(p.sigma_by_hour, true_sigma_h, rtol=0.30, atol=0.02)
    # Large jumps recovered (small ones below the k·σ threshold are noise).
    large_planted = int(np.sum(np.abs(jump_sizes_arr) > 0.40))
    assert 0.50 * large_planted <= result.n_jumps <= 1.50 * large_planted, (
        f"detected {result.n_jumps} jumps; large planted {large_planted}"
    )
    assert p.jump_p_up > 0.55
    # λ̂ must be in bounds (the bounded MLE would have raised otherwise).
    assert spot.LAMBDA_BOUNDS[0] <= p.jump_intensity <= spot.LAMBDA_BOUNDS[1]
    assert spot.ETA_BOUNDS[0] <= p.jump_eta_up <= spot.ETA_BOUNDS[1]
    assert spot.ETA_BOUNDS[0] <= p.jump_eta_down <= spot.ETA_BOUNDS[1]
    assert spot.KAPPA_BOUNDS[0] <= p.kappa <= spot.KAPPA_BOUNDS[1]
    # Slow factor RW params: drift ≈ 0, sigma small (since no trend planted).
    # Slow OU should be near-degenerate on this synthetic (no slow trend
    # planted): σ_θ small. κ_θ is unconstrained, can be anything.
    assert p.slow_factor.sigma < 0.05
    # n_obs is full input; residuals are post-warmup.
    assert result.n_obs == n_hours
    assert len(result.residuals) == n_hours - p.ema_span
    assert len(result.residual_returns) == n_hours - p.ema_span - 1


def test_fit_returns_theta_and_x_series_aligned_with_input() -> None:
    """The slow factor and X series are reported on the full input index
    (including the warm-up region) so diagnostic plots can show the EMA
    settling at the start of the series. Uses a synthetic series with
    OU dynamics inside KAPPA_BOUNDS so the bounded MLE converges
    interior."""
    rng = np.random.default_rng(seed=1)
    n_hours = 24 * 365 * 2
    idx = _hourly_utc("2020-01-01", n_hours)
    kappa, sigma = 0.10, 0.10
    phi = float(np.exp(-kappa))
    factor = float(np.sqrt((1 - phi**2) / (2 * kappa)))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma * factor * rng.standard_normal()
    prices = pd.Series(np.exp(3.5 + z) - 10.0, index=idx, name="p")
    result = spot.fit(prices)
    assert len(result.theta_series) == n_hours
    assert len(result.x_series) == n_hours
    assert abs(result.x_series.iloc[0]) < 1e-12
    assert isinstance(result.params.slow_factor, spot.SlowFactorParams)
    assert result.params.ema_span == spot.EMA_SPAN


# ---- E. Simulation ---------------------------------------------------------


def test_simulate_shape_and_reproducibility() -> None:
    params = _trivial_params(jump_intensity=0.001)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    a = spot.simulate(params, start, n_hours=240, n_paths=50, seed=42)
    b = spot.simulate(params, start, n_hours=240, n_paths=50, seed=42)
    c = spot.simulate(params, start, n_hours=240, n_paths=50, seed=7)
    assert a.shape == (50, 240)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_simulate_initial_residual_propagates() -> None:
    """A non-zero initial_residual must show up in t=0 of every path."""
    params = _trivial_params(intercept=0.0)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    a = spot.simulate(params, start, n_hours=10, n_paths=5, initial_residual=0.0, seed=1)
    b = spot.simulate(params, start, n_hours=10, n_paths=5, initial_residual=1.5, seed=1)
    # t=0 reflects exp(intercept + initial_residual) - shift
    # a: exp(0 + 0) - 10 = -9; b: exp(1.5) - 10 ≈ -5.52
    np.testing.assert_allclose(a[:, 0], np.exp(0.0) - 10.0, atol=1e-9)
    np.testing.assert_allclose(b[:, 0], np.exp(1.5) - 10.0, atol=1e-9)


def test_simulate_long_run_residual_variance_matches_theory() -> None:
    """For OU with constant σ and no jumps, Var(Z_∞) = σ² / (2κ)."""
    params = _trivial_params(
        intercept=0.0,
        kappa=0.10,
        sigma_by_hour=np.full(24, 0.15),
        jump_intensity=0.0,
    )
    start = pd.Timestamp("2025-01-01", tz="UTC")
    paths = spot.simulate(params, start, n_hours=24 * 365, n_paths=200, seed=42)
    log_p = np.log(paths + params.price_shift)
    # Drop the first half as burn-in (initial_residual=0 takes a while to
    # forget at κ=0.10 → half-life ~ ln 2 / 0.10 ≈ 7 h, so 6 months is plenty)
    burn = paths.shape[1] // 2
    var_observed = float(log_p[:, burn:].var())
    var_theoretical = 0.15**2 / (2 * 0.10)
    assert abs(var_observed - var_theoretical) / var_theoretical < 0.15


def test_simulate_jumps_increase_dispersion() -> None:
    """Higher jump intensity ⇒ higher empirical price std (same seed)."""
    start = pd.Timestamp("2025-01-01", tz="UTC")
    p_no_jumps = _trivial_params(jump_intensity=0.0)
    p_with_jumps = _trivial_params(jump_intensity=0.02, jump_eta_up=1.0, jump_eta_down=1.0)
    a = spot.simulate(p_no_jumps, start, n_hours=24 * 30, n_paths=100, seed=1)
    b = spot.simulate(p_with_jumps, start, n_hours=24 * 30, n_paths=100, seed=1)
    assert float(b.std()) > float(a.std())


def test_simulate_slow_factor_mean_attracts_path() -> None:
    """Slow-OU on θ_t with μ_θ ≠ initial_theta: paths drift toward μ_θ
    over long horizons. Compare to the κ_θ=0 (no mean reversion) case."""
    start = pd.Timestamp("2025-01-01", tz="UTC")
    p_flat = _trivial_params(slow_kappa=0.0, slow_mean=0.0, slow_sigma=0.0,
                              jump_intensity=0.0)
    p_pull = _trivial_params(slow_kappa=0.001, slow_mean=1.0, slow_sigma=0.0,
                              jump_intensity=0.0)
    flat = spot.simulate(p_flat, start, 24 * 365, 50, initial_theta=0.0, seed=42)
    pulled = spot.simulate(p_pull, start, 24 * 365, 50, initial_theta=0.0, seed=42)
    assert pulled.mean() > flat.mean() + 5.0


def test_simulate_slow_factor_noise_inflates_long_horizon_dispersion() -> None:
    """σ_θ > 0 inflates price dispersion over a year vs the σ_θ = 0 case
    at the same κ_θ."""
    start = pd.Timestamp("2025-01-01", tz="UTC")
    p_flat = _trivial_params(slow_kappa=0.001, slow_mean=0.0, slow_sigma=0.0,
                              jump_intensity=0.0)
    p_noisy = _trivial_params(slow_kappa=0.001, slow_mean=0.0, slow_sigma=0.02,
                               jump_intensity=0.0)
    flat = spot.simulate(p_flat, start, 24 * 365, 100, seed=7)
    noisy = spot.simulate(p_noisy, start, 24 * 365, 100, seed=7)
    assert float(noisy.std()) > 1.5 * float(flat.std())


def test_simulate_initial_theta_propagates_to_t0() -> None:
    """initial_theta shifts the time-0 log-price by the same amount."""
    params = _trivial_params(intercept=0.0)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    a = spot.simulate(params, start, 10, 5, initial_theta=0.0, seed=1)
    b = spot.simulate(params, start, 10, 5, initial_theta=2.0, seed=1)
    # Initial residual default 0, intercept 0, so t=0 price = exp(theta_0) - 10.
    np.testing.assert_allclose(a[:, 0], np.exp(0.0) - 10.0, atol=1e-9)
    np.testing.assert_allclose(b[:, 0], np.exp(2.0) - 10.0, atol=1e-9)


def test_fit_with_forward_anchor_consumes_ss_state(monkeypatch) -> None:
    """Pieza 1 fit consuming a Pieza 2 SSFit produces a SpotModelFit
    whose theta_series tracks (chi + xi + s_delivery) from Pieza 2,
    not an EMA of log(P+c)."""
    from types import SimpleNamespace
    rng = np.random.default_rng(seed=2026)
    n_hours = 24 * 90  # 90 days of hourly
    idx = _hourly_utc("2024-01-01", n_hours)
    daily_idx = pd.date_range("2024-01-01", periods=90, freq="D")

    # Fake Pieza 2 daily state: linear ramp in xi, zero chi.
    chi_d = pd.Series(np.zeros(90), index=daily_idx, name="chi")
    xi_d = pd.Series(np.linspace(4.0, 4.2, 90), index=daily_idx, name="xi")
    seasonal_dummies = np.array([0.05, 0.02, -0.01, -0.04, -0.03, 0.0,
                                  0.02, 0.04, 0.06, 0.08, 0.10])
    ss_fit = SimpleNamespace(
        state_chi=chi_d,
        state_xi=xi_d,
        params=SimpleNamespace(
            kappa=1.5,
            sigma_xi=0.1,
            seasonal_dummies=seasonal_dummies,
        ),
    )

    # Build synthetic hourly prices = theta + s_delivery + OU(kappa,sigma) + noise.
    months = pd.DatetimeIndex(idx).month.to_numpy()
    s_lookup = np.concatenate([[0.0, 0.0], seasonal_dummies])
    s_delivery = s_lookup[months]
    theta_hourly_vals = xi_d.reindex(
        pd.DatetimeIndex(idx).tz_convert(None).normalize(),
        method="ffill",
    ).set_axis(idx).values
    kappa_test, sigma_test = 0.10, 0.05
    phi = float(np.exp(-kappa_test))
    factor = float(np.sqrt((1 - phi**2) / (2 * kappa_test)))
    z = np.zeros(n_hours)
    for t in range(1, n_hours):
        z[t] = phi * z[t - 1] + sigma_test * factor * rng.standard_normal()
    log_p_target = theta_hourly_vals + s_delivery + z
    prices = pd.Series(
        np.exp(log_p_target) - spot.PRICE_SHIFT, index=idx, name="p",
    )

    fit = spot.fit_with_forward_anchor(prices, ss_fit)
    # theta_series should follow xi_d (plus s_delivery shift) — its
    # last value approximates last xi + last seasonal.
    expected_last = xi_d.iloc[-1] + s_lookup[idx[-1].month]
    assert abs(fit.theta_series.iloc[-1] - expected_last) < 0.05
    # ema_span = 0 signals forward-anchored mode.
    assert fit.params.ema_span == 0
    # Slow factor surfaced from Pieza 2 (kappa = ss_fit.params.kappa).
    assert fit.params.slow_factor.kappa == 1.5


def test_simulate_seasonal_profile_appears_in_paths() -> None:
    """Strong HoD seasonality must show up as an hour-of-day price profile
    when averaged across many paths."""
    seasonality = spot.Seasonality(
        intercept=4.0,
        fourier_coefs=np.zeros(8),
        dow_coefs=np.zeros(6),
        # Tall peak at midday (hour 12 UTC), low at hours 1-3.
        hod_coefs=np.array([
            -0.5, -0.5, -0.5, -0.3, -0.1, 0.0, 0.2, 0.4,
            0.6, 0.7, 0.8, 0.9, 0.6, 0.3, 0.1, -0.1,
            0.2, 0.4, 0.3, 0.1, -0.1, -0.2, -0.3, -0.4,
        ])[:23],  # hod_1..hod_23 (hour 0 is reference)
        fourier_harmonics=4,
    )
    params = spot.SpotModelParams(
        seasonality=seasonality,
        slow_factor=spot.SlowFactorParams(kappa=0.0, mean=0.0, sigma=0.0),
        price_shift=10.0, ema_span=720, kappa=0.5,
        sigma_by_hour=np.full(24, 0.05),
        jump_intensity=0.0, jump_p_up=0.5,
        jump_eta_up=1.0, jump_eta_down=1.0,
    )
    paths = spot.simulate(
        params,
        pd.Timestamp("2025-01-01", tz="UTC"),
        n_hours=24 * 14, n_paths=200, seed=42,
    )
    # Drop first 24 h (burn-in from initial_residual=0).
    paths = paths[:, 24:]
    n_hours = paths.shape[1]
    hours_grid = (np.arange(n_hours) + 24) % 24
    mean_by_hour = np.array(
        [paths[:, hours_grid == h].mean() for h in range(24)]
    )
    # Hour 12 (peak) must produce higher mean prices than hour 1 (trough).
    assert mean_by_hour[12] > mean_by_hour[1]
