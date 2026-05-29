"""Tests for :mod:`mibel_derivatives.pricing.swing`.

The swing pricer is path-agnostic, so most of these tests feed synthetic
price paths with a known distribution and check the Longstaff-Schwartz
value against a closed-form / brute-force benchmark that the LSMC must
reproduce. A single end-to-end case (marked ``monte_carlo``) drives the
real spot model so the integration path is exercised off the CI fast
lane.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from mibel_derivatives.pricing import swing

# ---- helpers ---------------------------------------------------------------


def _iid_lognormal_paths(
    n_paths: int,
    n_steps: int,
    *,
    s0: float = 50.0,
    drift: float = 0.0,
    sigma: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Independent lognormal settlement prices per step.

    Independence across steps is fine for the benchmark tests: they only
    rely on the per-step marginal distribution, not on any time
    dependence. ``models.spot.simulate`` supplies the realistic, serially
    dependent paths in the integration test."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    return s0 * np.exp(drift + sigma * z)


# ---- A. Public API contract ------------------------------------------------


def test_module_exposes_public_api() -> None:
    for name in ("price_swing", "SwingTerms", "SwingResult", "DEFAULT_BASIS_DEGREE"):
        assert hasattr(swing, name), f"swing.{name} missing"
    assert swing.DEFAULT_BASIS_DEGREE == 3


def test_swing_terms_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=10)
    assert terms.min_rights == 0
    with pytest.raises(FrozenInstanceError):
        terms.strike = 60.0


# ---- B. Input validation ---------------------------------------------------


def test_rejects_non_2d_paths() -> None:
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=5)
    with pytest.raises(ValueError, match="2D"):
        swing.price_swing(np.zeros(10), terms)


def test_rejects_non_finite_paths() -> None:
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=5)
    paths = _iid_lognormal_paths(50, 10)
    paths[3, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        swing.price_swing(paths, terms)


def test_rejects_bad_discount_factor_shape() -> None:
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=5)
    paths = _iid_lognormal_paths(50, 10)
    with pytest.raises(ValueError, match="discount_factors"):
        swing.price_swing(paths, terms, discount_factors=np.ones(9))


def test_rejects_nonpositive_discount_factor() -> None:
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=5)
    paths = _iid_lognormal_paths(50, 10)
    df = np.ones(10)
    df[2] = -0.1
    with pytest.raises(ValueError, match="positive"):
        swing.price_swing(paths, terms, discount_factors=df)


def test_rejects_bad_rights_and_volume() -> None:
    paths = _iid_lognormal_paths(50, 10)
    with pytest.raises(ValueError, match="n_rights"):
        swing.price_swing(paths, swing.SwingTerms(50.0, 1.0, 0))
    with pytest.raises(ValueError, match="volume_per_right"):
        swing.price_swing(paths, swing.SwingTerms(50.0, 0.0, 5))
    with pytest.raises(ValueError, match="min_rights"):
        swing.price_swing(paths, swing.SwingTerms(50.0, 1.0, 5, min_rights=6))


# ---- C. Closed-form benchmarks ---------------------------------------------


def test_single_step_equals_intrinsic_expectation() -> None:
    """With one step and one right the swing is a one-shot call: its value
    is exactly the discounted mean intrinsic ``q * E[(S - K)^+] * DF``.
    The terminal continuation is zero, so the LSMC regression is exact and
    the equality holds to floating point."""
    paths = _iid_lognormal_paths(5000, 1, s0=50.0, sigma=0.5, seed=1)
    q, K, df = 2.0, 50.0, 0.95
    terms = swing.SwingTerms(strike=K, volume_per_right=q, n_rights=1)
    res = swing.price_swing(paths, terms, discount_factors=np.array([df]))
    intrinsic = q * df * np.maximum(paths[:, 0] - K, 0.0).mean()
    assert res.price == pytest.approx(intrinsic, rel=1e-12)


def test_discount_factor_scales_price_linearly() -> None:
    paths = _iid_lognormal_paths(4000, 12, sigma=0.4, seed=2)
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=12)
    full = swing.price_swing(paths, terms, discount_factors=np.ones(12))
    half = swing.price_swing(paths, terms, discount_factors=np.full(12, 0.5))
    assert half.price == pytest.approx(0.5 * full.price, rel=1e-9)


def test_global_cap_nonbinding_reduces_to_sum_of_daily_calls() -> None:
    """When the number of rights equals the number of steps the annual cap
    never binds, so the optimal policy is myopic (exercise every
    in-the-money step) and the value collapses to the sum of independent
    daily call payoffs ``sum_t q * E[(S_t - K)^+] * DF_t``. The LSMC must
    reproduce this brute-force benchmark."""
    n_paths, n_steps = 5000, 30
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=55.0, sigma=0.45, seed=7)
    q, K = 1.0, 50.0
    rng = np.random.default_rng(0)
    df = np.cumprod(np.full(n_steps, 0.999)) * (1 - 1e-6 * rng.random(n_steps))
    terms = swing.SwingTerms(strike=K, volume_per_right=q, n_rights=n_steps)
    res = swing.price_swing(paths, terms, discount_factors=df)

    benchmark = float(
        (q * df[None, :] * np.maximum(paths - K, 0.0)).sum(axis=1).mean()
    )
    assert res.price == pytest.approx(benchmark, rel=0.01)
    # The forward policy must land essentially on the same value.
    assert res.policy_value == pytest.approx(benchmark, rel=0.03)
    # Expected exercises ~ count of in-the-money steps per path.
    itm_per_path = (paths > K).sum(axis=1).mean()
    assert res.expected_rights_used == pytest.approx(itm_per_path, rel=0.05)


def test_single_right_within_perfect_foresight_bounds() -> None:
    """One right over many steps: the value is bracketed below by the best
    single fixed exercise time ``q * max_t E[(S_t-K)^+]`` and above by the
    perfect-foresight payoff ``q * E[max_t (S_t-K)^+]``."""
    n_paths, n_steps = 6000, 20
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=50.0, sigma=0.5, seed=11)
    q, K = 1.0, 50.0
    terms = swing.SwingTerms(strike=K, volume_per_right=q, n_rights=1)
    res = swing.price_swing(paths, terms)

    intrinsic = np.maximum(paths - K, 0.0)
    lower = q * intrinsic.mean(axis=0).max()          # best fixed time
    upper = q * intrinsic.max(axis=1).mean()          # perfect foresight
    assert lower - 3 * res.std_error <= res.price <= upper + 3 * res.std_error
    # A real swing right is worth strictly more than committing to one date.
    assert res.price > lower


def test_price_monotone_increasing_in_rights() -> None:
    """More exercise rights cannot reduce the value (superset of feasible
    policies)."""
    n_paths, n_steps = 5000, 24
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=52.0, sigma=0.5, seed=5)
    K = 50.0
    prices = [
        swing.price_swing(
            paths, swing.SwingTerms(strike=K, volume_per_right=1.0, n_rights=r),
        ).price
        for r in (1, 3, 6, 12)
    ]
    for lo, hi in pairwise(prices):
        assert hi >= lo - 1e-6, f"non-monotone: {prices}"


def test_more_rights_never_below_single_step_zero() -> None:
    """A swing with all out-of-the-money paths and no take-or-pay is worth
    >= 0 (never exercise) and the policy uses ~zero rights."""
    n_paths, n_steps = 3000, 15
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=10.0, sigma=0.2, seed=9)
    K = 1000.0  # nothing is ever in the money
    res = swing.price_swing(
        paths, swing.SwingTerms(strike=K, volume_per_right=1.0, n_rights=5),
    )
    assert res.price == pytest.approx(0.0, abs=1e-9)
    assert res.expected_rights_used == pytest.approx(0.0, abs=1e-9)


# ---- D. Take-or-pay --------------------------------------------------------


def test_take_or_pay_forces_minimum_exercises_at_a_loss() -> None:
    """When every step is out of the money (K above all prices) and a
    take-or-pay minimum is set, the holder is forced to exercise exactly
    ``min_rights`` times, all at a loss, so the value is negative and the
    rights used equal the minimum."""
    n_paths, n_steps = 3000, 20
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=30.0, sigma=0.3, seed=13)
    K = 200.0  # always out of the money
    q = 1.0
    free = swing.price_swing(paths, swing.SwingTerms(K, q, n_rights=10))
    forced = swing.price_swing(
        paths, swing.SwingTerms(K, q, n_rights=10, min_rights=4),
    )
    assert free.price == pytest.approx(0.0, abs=1e-9)
    assert forced.price < 0.0
    assert forced.price < free.price
    assert forced.expected_rights_used == pytest.approx(4.0, abs=1e-9)


def test_take_or_pay_picks_least_bad_days() -> None:
    """Forced exercises should be the least-unfavourable steps: the loss is
    bounded below by forcing the worst ``min_rights`` days and above (less
    negative) by the best ``min_rights`` days. The optimal policy must beat
    the worst-case selection."""
    n_paths, n_steps = 4000, 18
    paths = _iid_lognormal_paths(n_paths, n_steps, s0=40.0, sigma=0.35, seed=17)
    K, q, m = 300.0, 1.0, 3  # always OTM (max path << 300), must take 3
    res = swing.price_swing(
        paths, swing.SwingTerms(K, q, n_rights=8, min_rights=m),
    )
    payoff = q * (paths - K)  # all negative
    worst = np.sort(payoff, axis=1)[:, :m].sum(axis=1).mean()   # m most negative
    best = np.sort(payoff, axis=1)[:, -m:].sum(axis=1).mean()   # m least negative
    assert res.expected_rights_used == pytest.approx(float(m), abs=1e-9)
    assert worst < res.price <= best + 3 * (res.std_error + 1e-9)
    # The optimiser should be much closer to the best (least-bad) selection.
    assert res.price > 0.5 * best + 0.5 * worst


# ---- E. Determinism --------------------------------------------------------


def test_pricer_is_deterministic() -> None:
    paths = _iid_lognormal_paths(2000, 14, seed=21)
    terms = swing.SwingTerms(strike=50.0, volume_per_right=1.0, n_rights=6)
    a = swing.price_swing(paths, terms)
    b = swing.price_swing(paths, terms)
    assert a.price == b.price
    np.testing.assert_array_equal(a.per_path_value, b.per_path_value)


# ---- F. End-to-end with the real spot model --------------------------------


@pytest.mark.slow
@pytest.mark.monte_carlo
def test_end_to_end_with_spot_simulation() -> None:
    """Drive the swing pricer with daily-aggregated paths from the Pieza 1
    spot model. Sanity-only: a positive value, the forward policy close to
    the backward estimate, and rights used within the cap."""
    from mibel_derivatives.models import spot

    seasonality = spot.Seasonality(
        intercept=3.9,
        fourier_coefs=np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dow_coefs=np.array([0.03, 0.04, 0.03, 0.02, 0.0, -0.08]),
        hod_coefs=np.linspace(-0.20, 0.40, 23),
        fourier_harmonics=4,
    )
    params = spot.SpotModelParams(
        seasonality=seasonality,
        slow_factor=spot.SlowFactorParams(kappa=0.001, mean=0.0, sigma=0.01),
        price_shift=10.0,
        ema_span=24,
        kappa=0.12,
        sigma_by_hour=np.full(24, 0.10),
        jump_intensity=0.01,
        jump_p_up=0.6,
        jump_eta_up=2.0,
        jump_eta_down=2.5,
    )
    n_paths, n_days = 400, 90
    hourly = spot.simulate(
        params,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        n_hours=24 * n_days,
        n_paths=n_paths,
        initial_theta=3.9,
        seed=2026,
    )
    # Aggregate hourly nominal prices to a daily baseload price per path.
    daily = hourly.reshape(n_paths, n_days, 24).mean(axis=2)

    strike = float(np.median(daily))
    df = np.exp(-0.03 * np.arange(n_days) / 365.25)  # flat 3% annual
    terms = swing.SwingTerms(
        strike=strike, volume_per_right=1.0, n_rights=30, min_rights=0,
    )
    res = swing.price_swing(daily, terms, discount_factors=df)

    assert res.price > 0.0
    assert 0 < res.expected_rights_used <= terms.n_rights
    assert res.policy_value == pytest.approx(res.price, rel=0.10)
    assert res.per_path_value.shape == (n_paths,)
