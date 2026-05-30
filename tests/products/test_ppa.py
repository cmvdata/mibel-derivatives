"""Tests for :mod:`mibel_derivatives.products.ppa` (solar PPA, Pieza 5).

The pricer's valuation arithmetic (structure split, capture price,
discounting) is path-agnostic: most tests feed deterministic or seeded
synthetic paths and pin a closed-form identity the Monte Carlo must
reproduce. The five headline cases requested in the brief are grouped
under section C. The full reported-valuation size (N_PATHS=50000) runs
only under ``requires_pod`` + ``slow``, off the CI fast lane.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest

from mibel_derivatives.products import ppa as ppa_mod
from mibel_derivatives.products.ppa import (
    PPA,
    PriceModel,
    price_ppa,
    simulate_price_paths,
    simulate_production_paths,
)

# ---- helpers ---------------------------------------------------------------


def _solar_profile(n_days: int = 30, peak_cf: float = 0.7) -> np.ndarray:
    """A clean daytime solar bump tiled over ``n_days``.

    Zero overnight, peaking at solar noon (hour 12). Mean capacity factor
    lands near the Andalucía fixed-tilt ~0.18, which keeps the synthetic
    profile economically representative without needing the parquet."""
    hod = np.arange(24)
    bump = np.clip(np.sin((hod - 6) / 12.0 * np.pi), 0.0, None)
    return np.tile(peak_cf * bump, n_days)


def _ppa(**overrides) -> PPA:
    base = dict(
        capacity_mw=100.0,
        plant_factor=0.20,
        fixed_pct=0.80,
        spot_pct=0.20,
        strike=55.0,
        duration_years=10,
    )
    base.update(overrides)
    return PPA(**base)


# ---- A. Public API contract ------------------------------------------------


def test_module_exposes_public_api() -> None:
    for name in (
        "PPA",
        "PriceModel",
        "PPAResult",
        "price_ppa",
        "simulate_price_paths",
        "simulate_production_paths",
        "simulate_price_paths_from_spot",
        "representative_year_profile",
        "DEFAULT_N_PATHS",
    ):
        assert hasattr(ppa_mod, name), f"ppa.{name} missing"
    assert ppa_mod.DEFAULT_N_PATHS == 1000


def test_ppa_is_frozen() -> None:
    ppa = _ppa()
    with pytest.raises(FrozenInstanceError):
        ppa.strike = 99.0


def test_compat_shim_reexports_pricer() -> None:
    from mibel_derivatives.pricing import ppa_solar

    assert ppa_solar.price_ppa is price_ppa
    # The historical placeholder alias now delegates instead of raising.
    assert ppa_solar.price_ppa_solar.__doc__ is not None


# ---- B. Input validation ---------------------------------------------------


def test_rejects_structure_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1"):
        _ppa(fixed_pct=0.8, spot_pct=0.1)


def test_rejects_pct_out_of_range() -> None:
    with pytest.raises(ValueError, match="fixed_pct"):
        _ppa(fixed_pct=1.2, spot_pct=-0.2)


def test_rejects_bad_plant_factor() -> None:
    with pytest.raises(ValueError, match="plant_factor"):
        _ppa(plant_factor=1.5)


def test_rejects_one_path_array_only() -> None:
    gen = np.ones((4, 24))
    with pytest.raises(ValueError, match="both"):
        price_ppa(_ppa(), production_paths=gen)


def test_rejects_mismatched_path_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        price_ppa(_ppa(), price_paths=np.ones((4, 24)), production_paths=np.ones((4, 12)))


def test_rejects_profile_out_of_range() -> None:
    with pytest.raises(ValueError, match="capacity factors"):
        simulate_price_paths(PriceModel(), np.array([0.5, 1.4, 0.2]), 3)


def test_production_clipped_to_capacity() -> None:
    profile = _solar_profile()
    gen = simulate_production_paths(
        profile, 100.0, 200, resource_sigma=0.3, hourly_sigma=0.3, seed=1
    )
    assert gen.max() <= 100.0 + 1e-9
    assert gen.min() >= 0.0


def test_production_is_mean_one_in_expectation() -> None:
    profile = _solar_profile()
    gen = simulate_production_paths(profile, 100.0, 5000, resource_sigma=0.1, seed=7)
    expected = 100.0 * profile
    # Mean across paths recovers capacity*cf (mean-one lognormal multiplier).
    np.testing.assert_allclose(gen.mean(axis=0), expected, rtol=0.05, atol=0.2)


# ---- C. The five headline cases from the brief -----------------------------


def test_pure_fixed_structure_is_deterministic_in_price() -> None:
    """100% fixed: the payoff is generation * strike, independent of the
    realised spot path. Two different price scenarios must price the same,
    and the levelised value must equal the strike (zero cost)."""
    ppa = _ppa(fixed_pct=1.0, spot_pct=0.0, strike=50.0)
    profile = _solar_profile()
    gen = simulate_production_paths(profile, ppa.capacity_mw, 64, seed=0)

    price_a = simulate_price_paths(PriceModel(), profile, 64, seed=1)
    price_b = simulate_price_paths(PriceModel(baseload=120.0), profile, 64, seed=2)

    r_a = price_ppa(ppa, price_paths=price_a, production_paths=gen)
    r_b = price_ppa(ppa, price_paths=price_b, production_paths=gen)

    assert r_a.value == r_b.value  # exactly: spot leg carries zero weight
    assert r_a.value_per_mwh == pytest.approx(50.0, rel=1e-9)


def test_pure_spot_structure_equals_capture_price() -> None:
    """100% spot, zero cost: the levelised value is exactly the capture
    price. With deterministic generation across paths the per-path weights
    are equal, so value_per_mwh coincides with the reported capture mean."""
    ppa = _ppa(fixed_pct=0.0, spot_pct=1.0)
    profile = _solar_profile()
    n_paths = 128
    # Deterministic generation: every path identical -> equal weights.
    gen = np.tile(ppa.capacity_mw * profile, (n_paths, 1))
    price = simulate_price_paths(PriceModel(), profile, n_paths, seed=3)

    r = price_ppa(ppa, price_paths=price, production_paths=gen)

    assert r.value_per_mwh == pytest.approx(r.capture_price_mean, rel=1e-9)
    # Sanity: a pure-spot deal pays the capture price, below baseload.
    assert r.value_per_mwh < r.baseload_price_mean


def test_capture_price_below_baseload() -> None:
    """Solar generation is concentrated when price is depressed, so the
    generation-weighted capture price falls below the baseload average."""
    ppa = _ppa()
    profile = _solar_profile()
    model = PriceModel(cannibalisation=0.5, diurnal_amplitude=0.3)

    r = price_ppa(ppa, pv_profile=profile, price_model=model, n_paths=300, seed=11)

    assert r.capture_ratio < 1.0
    assert r.capture_price_mean < r.baseload_price_mean

    # Control: remove cannibalisation AND the diurnal shape and the capture
    # discount disappears (capture ~ baseload).
    flat = PriceModel(cannibalisation=0.0, diurnal_amplitude=0.0, sigma=0.1)
    r_flat = price_ppa(ppa, pv_profile=profile, price_model=flat, n_paths=300, seed=11)
    assert r_flat.capture_ratio == pytest.approx(1.0, abs=0.03)


def test_seed_reproducibility() -> None:
    ppa = _ppa()
    profile = _solar_profile()
    model = PriceModel()

    r1 = price_ppa(ppa, pv_profile=profile, price_model=model, n_paths=200, seed=42)
    r2 = price_ppa(ppa, pv_profile=profile, price_model=model, n_paths=200, seed=42)
    r3 = price_ppa(ppa, pv_profile=profile, price_model=model, n_paths=200, seed=43)

    assert r1.value == r2.value
    np.testing.assert_array_equal(r1.per_path_value, r2.per_path_value)
    assert r1.value != r3.value  # a different seed moves the estimate


def test_monotonicity_in_strike() -> None:
    """Value to the generator is strictly increasing in the fixed strike
    (the fixed leg has positive weight). Hold the paths fixed so only the
    strike varies."""
    profile = _solar_profile()
    n_paths = 100
    gen = simulate_production_paths(profile, 100.0, n_paths, seed=5)
    price = simulate_price_paths(PriceModel(), profile, n_paths, seed=6)

    values = []
    for strike in (20.0, 40.0, 60.0, 80.0, 100.0):
        ppa = _ppa(strike=strike)
        values.append(price_ppa(ppa, price_paths=price, production_paths=gen).value)

    assert all(lo < hi for lo, hi in pairwise(values))


# ---- D. Cashflow / discounting identities ----------------------------------


def test_cost_reduces_value_one_for_one() -> None:
    """cost_ppa nets a fixed EUR/MWh from every generated MWh, so the value
    drops by cost * PV(total generation)."""
    profile = _solar_profile()
    gen = simulate_production_paths(profile, 100.0, 80, seed=0)
    price = simulate_price_paths(PriceModel(), profile, 80, seed=1)

    r0 = price_ppa(_ppa(cost_ppa=0.0), price_paths=price, production_paths=gen)
    r5 = price_ppa(_ppa(cost_ppa=5.0), price_paths=price, production_paths=gen)

    pv_gen = r0.value / r0.value_per_mwh
    assert r0.value - r5.value == pytest.approx(5.0 * pv_gen, rel=1e-9)


def test_value_scales_with_duration_annuity() -> None:
    """Doubling-style check: the value is linear in the mid-year annuity
    factor, so a longer contract is worth strictly more (positive net
    margin) and the ratio matches the annuity ratio."""
    from mibel_derivatives.products.ppa import _annuity_factor

    profile = _solar_profile()
    gen = simulate_production_paths(profile, 100.0, 60, seed=0)
    price = simulate_price_paths(PriceModel(), profile, 60, seed=1)

    r10 = price_ppa(_ppa(duration_years=10), price_paths=price, production_paths=gen)
    r20 = price_ppa(_ppa(duration_years=20), price_paths=price, production_paths=gen)

    ratio = _annuity_factor(0.07, 20) / _annuity_factor(0.07, 10)
    assert r20.value / r10.value == pytest.approx(ratio, rel=1e-9)


# ---- E. Reported-valuation size (pod only) ---------------------------------


@pytest.mark.slow
@pytest.mark.requires_pod
def test_reported_valuation_size() -> None:
    """End-to-end at the reported Monte Carlo size (N_PATHS=50000). Pins a
    tight standard error and the expected capture/baseload ordering."""
    ppa = _ppa()
    profile = _solar_profile(n_days=60)
    model = PriceModel(cannibalisation=0.5)

    r = price_ppa(ppa, pv_profile=profile, price_model=model, n_paths=50_000, seed=2024)

    assert r.n_paths == 50_000
    assert r.value > 0.0
    assert r.capture_ratio < 1.0
    # Standard error should be a small fraction of the value at 50k paths.
    assert r.std_error / r.value < 0.02
