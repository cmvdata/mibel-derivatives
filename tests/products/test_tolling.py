"""Tests for :mod:`mibel_derivatives.products.tolling`.

The tolling pricer optimises the CCGT dispatch by dynamic programming
over the operating state and then averages the discounted optimal margin
over Monte Carlo power paths. Like the swing pricer it is path-agnostic,
so these tests feed deterministic / synthetic price paths and check the
DISPATCH CONSTRAINTS (Pmin/Pmax band, min up/down time, ramp, start-up
cost, heat-rate degradation) and the economic monotonicity directly. One
end-to-end case (marked ``monte_carlo``) drives the Pieza 1 spot model.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from mibel_derivatives.products import tolling

# ---- helpers ---------------------------------------------------------------


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, length) of each maximal run of True in a 1D mask."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def _flat(n_paths: int, n_hours: int, value: float) -> np.ndarray:
    return np.full((n_paths, n_hours), value, dtype=float)


DEV_ASSET = tolling.AssetParameters()


# ---- A. Public API contract ------------------------------------------------


def test_module_exposes_public_api() -> None:
    for name in (
        "AssetParameters",
        "TollingAgreement",
        "TollingResult",
        "DispatchResult",
        "price_tolling",
        "optimise_dispatch",
        "heat_rate_gj_per_mwh",
        "spark_spread_per_mwh",
        "hourly_gross_margin",
    ):
        assert hasattr(tolling, name), f"tolling.{name} missing"


def test_dataclasses_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    asset = tolling.AssetParameters()
    with pytest.raises(FrozenInstanceError):
        asset.pmax_mw = 400.0  # type: ignore[misc]

    agreement = tolling.TollingAgreement(asset=asset)
    assert agreement.effective_capacity_mw == asset.pmax_mw
    with pytest.raises(FrozenInstanceError):
        agreement.capacity_mw = 100.0  # type: ignore[misc]


def test_default_asset_matches_castejon_spec() -> None:
    a = tolling.AssetParameters()
    assert a.pmax_mw == pytest.approx(386.10)
    assert a.pmin_mw == pytest.approx(120.0)
    assert a.heat_rate_full_gj_per_mwh == pytest.approx(6.55)
    assert a.heat_rate_min_gj_per_mwh == pytest.approx(7.5)
    assert a.ramp_mw_per_min == pytest.approx(8.0)


# ---- B. Input validation ---------------------------------------------------


def test_rejects_bad_asset_parameters() -> None:
    with pytest.raises(ValueError, match="pmin_mw < pmax_mw"):
        tolling.AssetParameters(pmin_mw=400.0, pmax_mw=386.1)
    with pytest.raises(ValueError, match="part load"):
        tolling.AssetParameters(heat_rate_full_gj_per_mwh=7.0, heat_rate_min_gj_per_mwh=6.5)


def test_optimise_dispatch_validates_shapes() -> None:
    power = _flat(2, 10, 100.0)
    with pytest.raises(ValueError, match="shape"):
        tolling.optimise_dispatch(power, np.zeros(9), np.zeros(10), DEV_ASSET)
    with pytest.raises(ValueError, match="non-finite"):
        bad = power.copy()
        bad[0, 0] = np.nan
        tolling.optimise_dispatch(bad, np.zeros(10), np.zeros(10), DEV_ASSET)
    with pytest.raises(ValueError, match="discount_factors"):
        tolling.optimise_dispatch(
            power, np.zeros(10), np.zeros(10), DEV_ASSET,
            discount_factors=np.full(10, -1.0),
        )


# ---- C. Dispatch constraints (the required six) ----------------------------


def test_dispatch_respects_pmin_pmax() -> None:
    """Every dispatched (on) hour sits inside [Pmin, Pmax]; off hours are 0."""
    a = tolling.AssetParameters()
    power = _flat(4, 72, 200.0)  # spark spread strongly positive -> always on
    gas = np.full(72, 25.0)
    eua = np.full(72, 60.0)
    res = tolling.optimise_dispatch(power, gas, eua, a)

    sched = res.power_schedule
    on = sched > 0.0
    assert (sched[on] >= a.pmin_mw - 1e-9).all()
    assert (sched[on] <= a.pmax_mw + 1e-9).all()
    # Off hours are exactly zero; on hours land on a discrete level.
    assert np.isclose(sched[~on], 0.0).all()
    levels = a.power_levels
    for p in np.unique(sched[on]):
        assert np.isclose(levels, p).any(), f"power {p} not on the level grid"


def test_minimum_uptime_downtime() -> None:
    """Run lengths respect TMO and the gaps between runs respect TMA, on a
    fluctuating path that forces the unit to cycle."""
    a = tolling.AssetParameters(min_uptime_h=4, min_downtime_h=3)
    rng = np.random.default_rng(7)
    n_hours = 240
    # Mean near break-even so the optimal schedule genuinely cycles.
    gas = np.full(n_hours, 30.0)
    eua = np.full(n_hours, 70.0)
    breakeven = float(tolling.spark_spread_per_mwh(a, 0.0, 30.0, 70.0)) * -1  # ~ HR*fuel
    power = (breakeven + rng.normal(0.0, 25.0, size=(3, n_hours))).clip(min=0.0)
    res = tolling.optimise_dispatch(power, gas, eua, a)

    for p in range(power.shape[0]):
        on = res.power_schedule[p] > 0.0
        runs = _runs(on)
        for _start, length in runs:
            assert length >= a.min_uptime_h, f"run length {length} < TMO {a.min_uptime_h}"
        # interior off-gaps (between two runs) must respect min downtime
        for (s0, l0), (s1, _l1) in pairwise(runs):
            gap = s1 - (s0 + l0)
            assert gap >= a.min_downtime_h, f"off-gap {gap} < TMA {a.min_downtime_h}"


def test_ramp_constraint() -> None:
    """With a slow ramp the unit cannot jump levels between consecutive
    on-hours; consecutive power moves stay within the hourly ramp budget."""
    a = tolling.AssetParameters(n_power_levels=7)
    n_hours = 36
    power = _flat(1, n_hours, 220.0)  # strongly profitable -> wants Pmax asap
    gas = np.full(n_hours, 25.0)
    eua = np.full(n_hours, 60.0)
    slow = 1.0  # MW/min -> 60 MW/h, less than the (266/6 ~ 44) two-level jump
    res = tolling.optimise_dispatch(power, gas, eua, a, ramp_mw_per_min=slow)

    sched = res.power_schedule[0]
    ramp_h = slow * 60.0
    on = sched > 0.0
    for t in range(1, n_hours):
        if on[t] and on[t - 1]:
            assert abs(sched[t] - sched[t - 1]) <= ramp_h + 1e-6
    # The slow ramp forces a gradual climb: it should take several hours to
    # reach Pmax, unlike the default fast ramp which reaches it in one step.
    fast = tolling.optimise_dispatch(power, gas, eua, a)
    t_fast = int(np.argmax(fast.power_schedule[0] >= a.pmax_mw - 1e-6))
    t_slow = int(np.argmax(sched >= a.pmax_mw - 1e-6))
    assert t_slow > t_fast


def test_startup_cost_charged() -> None:
    """A single profitable block incurs exactly one (cold) start, and the
    reported start-up cost equals the cold cost per MW times capacity."""
    a = tolling.AssetParameters()
    n_hours = 24
    power = _flat(1, n_hours, 200.0)
    gas = np.full(n_hours, 25.0)
    eua = np.full(n_hours, 60.0)
    cap = a.pmax_mw
    res = tolling.optimise_dispatch(power, gas, eua, a, capacity_mw=cap)

    assert res.n_starts[0] == 1
    expected = a.startup_cost_cold_eur_per_mw * cap  # df == 1, cold initial state
    assert res.startup_cost[0] == pytest.approx(expected)

    # A high enough start-up cost suppresses a marginally profitable single
    # hour: 1 h of margin < cold-start cost -> the unit stays off.
    a_pricey = tolling.AssetParameters(
        startup_cost_cold_eur_per_mw=1e6, min_uptime_h=1, min_downtime_h=1,
    )
    marginal = _flat(1, 1, 200.0)
    off = tolling.optimise_dispatch(marginal, np.full(1, 25.0), np.full(1, 60.0), a_pricey)
    assert off.running_hours[0] == 0
    assert off.value[0] == pytest.approx(0.0)


def test_monotonicity_in_spark_spread() -> None:
    """Shifting every hourly power price up (a uniform rise in the spark
    spread) cannot lower the optimal dispatch value."""
    a = tolling.AssetParameters()
    rng = np.random.default_rng(11)
    n_hours = 120
    gas = np.full(n_hours, 30.0)
    eua = np.full(n_hours, 70.0)
    base = (60.0 + rng.normal(0.0, 30.0, size=(5, n_hours))).clip(min=0.0)
    res_lo = tolling.optimise_dispatch(base, gas, eua, a)
    res_hi = tolling.optimise_dispatch(base + 15.0, gas, eua, a)
    assert (res_hi.value >= res_lo.value - 1e-6).all()


def test_heat_rate_degradation_at_pmin() -> None:
    """Part-load efficiency is worse at Pmin, so the per-MWh spark spread is
    lower there; when running is comfortably profitable the optimiser holds
    Pmax (the efficient point) rather than Pmin."""
    a = tolling.AssetParameters()
    hr_min = float(tolling.heat_rate_gj_per_mwh(a, a.pmin_mw))
    hr_max = float(tolling.heat_rate_gj_per_mwh(a, a.pmax_mw))
    assert hr_min > hr_max
    assert hr_min == pytest.approx(7.5)
    assert hr_max == pytest.approx(6.55)

    spark_pmin = float(tolling.spark_spread_per_mwh(a, 90.0, 30.0, 70.0, power_mw=a.pmin_mw))
    spark_pmax = float(tolling.spark_spread_per_mwh(a, 90.0, 30.0, 70.0, power_mw=a.pmax_mw))
    assert spark_pmax > spark_pmin  # better heat rate at full load

    # Steady-state dispatch under a profitable price holds Pmax, not Pmin.
    n_hours = 24
    res = tolling.optimise_dispatch(
        _flat(1, n_hours, 130.0), np.full(n_hours, 30.0), np.full(n_hours, 70.0), a,
    )
    steady = res.power_schedule[0, 5:]  # skip the start/ramp transient
    assert np.isclose(steady, a.pmax_mw).all()


# ---- D. Pricing assembly ---------------------------------------------------


def test_price_tolling_nets_fixed_fee() -> None:
    a = tolling.AssetParameters()
    n_paths, n_hours = 8, 72
    power = _flat(n_paths, n_hours, 180.0)
    gas = np.full(n_hours, 25.0)
    eua = np.full(n_hours, 60.0)
    agreement = tolling.TollingAgreement(asset=a, fixed_fee_eur_per_mw_year=30_000.0)
    res = tolling.price_tolling(agreement, power, gas, eua)

    assert res.option_value > 0.0
    assert res.fixed_fee_pv > 0.0
    assert res.net_value == pytest.approx(res.option_value - res.fixed_fee_pv)
    assert 0.0 <= res.mean_capacity_factor <= 1.0
    assert res.n_paths == n_paths and res.n_hours == n_hours
    # Chunking must not change the answer.
    res_chunked = tolling.price_tolling(agreement, power, gas, eua, chunk_size=3)
    assert res_chunked.option_value == pytest.approx(res.option_value)


def test_discounting_reduces_value() -> None:
    a = tolling.AssetParameters()
    n_hours = 96
    power = _flat(2, n_hours, 180.0)
    gas = np.full(n_hours, 25.0)
    eua = np.full(n_hours, 60.0)
    undiscounted = tolling.optimise_dispatch(power, gas, eua, a)
    df = np.exp(-0.03 * np.arange(n_hours) / 8760.0)
    discounted = tolling.optimise_dispatch(power, gas, eua, a, discount_factors=df)
    assert (discounted.value < undiscounted.value).all()


# ---- E. End-to-end with the Pieza 1 spot model -----------------------------


@pytest.mark.monte_carlo
@pytest.mark.slow
def test_end_to_end_with_spot_simulation() -> None:
    """Drive the tolling pricer with hourly paths from the Pieza 1 spot
    model. Sanity-only: a positive option value and a sane capacity
    factor with flat gas/carbon curves."""
    import pandas as pd

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
    n_paths, n_hours = 60, 24 * 20
    power = spot.simulate(
        params,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        n_hours=n_hours,
        n_paths=n_paths,
        initial_theta=3.9,
        seed=2026,
    )
    a = tolling.AssetParameters()
    gas = np.full(n_hours, 28.0)
    eua = np.full(n_hours, 70.0)
    agreement = tolling.TollingAgreement(asset=a, fixed_fee_eur_per_mw_year=25_000.0)
    res = tolling.price_tolling(agreement, power, gas, eua, chunk_size=30)

    assert res.option_value > 0.0
    assert 0.0 <= res.mean_capacity_factor <= 1.0
    assert res.mean_running_hours <= n_hours
    assert res.std_error >= 0.0
