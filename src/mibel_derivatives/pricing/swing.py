"""Swing option pricer (Longstaff-Schwartz Monte Carlo).

Specification (CONTEXT.md § Swing option):

Annual MIBEL electricity contract with a fixed strike, a daily exercise
cap, and an annual maximum volume. Pricing via Longstaff-Schwartz Monte
Carlo with the cumulative-volume state added to the regression (the
Boogert & de Jong 2008 volume-grid extension of Longstaff & Schwartz
2001).

Resolution of the two open spec items (CONTEXT.md left these to the
design phase; the conservative reading is taken here):

1. **Exercise frequency / cap basis.** One exercise decision per step;
   each step delivers either zero or ``volume_per_right`` MWh (the daily
   cap). A column of ``price_paths`` is one exercise step (a delivery
   day for the annual contract), priced at that step's settlement price.
   The annual maximum volume is ``n_rights * volume_per_right`` MWh.
   The per-step payoff is linear in the nominated volume, so the optimal
   nomination is bang-bang (0 or the cap) except where the global budget
   forces the boundary — the unit-right grid is therefore exact when the
   annual cap is an integer multiple of the daily cap, which is the only
   case this pricer accepts.

2. **Take-or-pay minimum.** Off by default (``min_rights = 0``). When a
   minimum is set it is enforced as a HARD volume floor: the holder must
   exercise at least ``min_rights`` times before maturity, achieved by
   forcing exercises once the remaining steps equal the remaining
   required rights. No under-take penalty *price* is invented — the
   conservative reading of an unspecified penalty is a mandatory take,
   which costs the holder the realised (possibly negative) intrinsic
   value of the forced exercises rather than an arbitrary fee.

The pricer is deliberately decoupled from path generation: it consumes a
``(n_paths, n_steps)`` array of settlement prices. For a MIBEL valuation
those paths come from ``models.spot.simulate`` (hourly) aggregated to the
delivery-step resolution, or from a forward-anchored composition; the
discount factors come from the EURIBOR + OIS curve (CONTEXT.md
§ Parámetros numéricos). Keeping the pricer path-agnostic lets it be
validated against closed-form benchmarks with synthetic paths and keeps
the Monte Carlo model out of the pricing unit tests.

References: Longstaff & Schwartz (2001) for the least-squares Monte Carlo
continuation regression; Boogert & de Jong (2008), "Gas Storage
Valuation Using a Monte Carlo Method", for the cumulative-volume grid
that turns a single optimal-stopping problem into the swing's
optimal-control problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BASIS_DEGREE: int = 3


@dataclass(frozen=True)
class SwingTerms:
    """Contract terms of an annual swing option (call style: the holder
    pays ``strike`` and receives the spot, so a step is in the money when
    ``price > strike``).

    Attributes
    ----------
    strike
        Fixed strike K in EUR/MWh.
    volume_per_right
        Volume delivered by one exercised right, in MWh — the daily
        exercise cap.
    n_rights
        Number of exercise rights R. The annual maximum volume is
        ``n_rights * volume_per_right`` MWh.
    min_rights
        Take-or-pay minimum number of rights that MUST be exercised over
        the life of the contract. Default 0 (no minimum). Must satisfy
        ``0 <= min_rights <= n_rights``.
    """

    strike: float
    volume_per_right: float
    n_rights: int
    min_rights: int = 0


@dataclass(frozen=True)
class SwingResult:
    """Valuation output of :func:`price_swing`.

    Attributes
    ----------
    price
        Longstaff-Schwartz value estimate in EUR (mean across paths of the
        discounted optimal cashflow with no rights used at the first step).
    std_error
        Monte Carlo standard error of ``price``.
    per_path_value
        Per-path discounted optimal value, shape ``(n_paths,)``.
    policy_value
        Realised mean value of the estimated exercise policy applied
        forward on the same paths. A lower-bound cross-check on ``price``;
        a large gap flags an unstable continuation regression.
    expected_rights_used
        Mean number of rights exercised under the estimated policy. Useful
        for sanity checks (e.g. converges to the count of in-the-money
        steps when the global cap never binds).
    n_paths, n_steps, n_rights
        Echoed problem dimensions.
    """

    price: float
    std_error: float
    per_path_value: np.ndarray
    policy_value: float
    expected_rights_used: float
    n_paths: int
    n_steps: int
    n_rights: int


def _polynomial_basis(prices: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix of monomials ``[1, x, x^2, ..., x^degree]`` in the
    per-step-normalised price, shape ``(n_paths, degree + 1)``.

    Normalising (centre + scale) the price before raising it to powers
    keeps the least-squares normal equations well conditioned across the
    wide price range MIBEL spans (single digits to several hundred
    EUR/MWh in the 2022 crisis)."""
    x = np.asarray(prices, dtype=float)
    mu = float(x.mean())
    sd = float(x.std())
    if sd <= 0.0 or not np.isfinite(sd):
        sd = 1.0
    xn = (x - mu) / sd
    cols = [np.ones_like(xn)]
    power = np.ones_like(xn)
    for _ in range(degree):
        power = power * xn
        cols.append(power)
    return np.column_stack(cols)


def price_swing(
    price_paths: np.ndarray,
    terms: SwingTerms,
    *,
    discount_factors: np.ndarray | None = None,
    basis_degree: int = DEFAULT_BASIS_DEGREE,
) -> SwingResult:
    """Price an annual swing option by volume-grid Longstaff-Schwartz.

    Parameters
    ----------
    price_paths
        Settlement prices in EUR/MWh, shape ``(n_paths, n_steps)``. Each
        column is one exercise step (e.g. a delivery day). Must be finite.
    terms
        :class:`SwingTerms` contract definition.
    discount_factors
        Per-step discount factors from the valuation date to each
        exercise step, shape ``(n_steps,)``. Strictly positive. Default
        is all ones (undiscounted); supply the EURIBOR + OIS factors for
        a reported valuation.
    basis_degree
        Degree of the price polynomial in the continuation regression.

    Returns
    -------
    SwingResult
        Price (EUR), MC standard error, per-path values, and policy
        diagnostics.

    The recursion carries, for each cumulative-volume level
    ``c in {0, ..., R}``, the realised value-to-go along every path.
    Going backward, the continuation value of each level is estimated by
    regressing that carried value on the price basis (one shared design
    matrix per step). The exercise decision compares the estimated
    continuation of staying at level ``c`` against the immediate payoff
    plus the estimated continuation of moving to ``c + 1``; the carried
    value is then updated with the REALISED (not regressed) continuation,
    which is the standard Longstaff-Schwartz device that keeps the
    estimator unbiased.
    """
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError(f"price_paths must be 2D (n_paths, n_steps), got {paths.shape}")
    n_paths, n_steps = paths.shape
    if n_paths < 1 or n_steps < 1:
        raise ValueError("price_paths must have at least one path and one step")
    if not np.isfinite(paths).all():
        raise ValueError("price_paths contains non-finite values")
    if basis_degree < 1:
        raise ValueError("basis_degree must be >= 1")

    R = int(terms.n_rights)
    m = int(terms.min_rights)
    q = float(terms.volume_per_right)
    K = float(terms.strike)
    if R < 1:
        raise ValueError("n_rights must be >= 1")
    if q <= 0.0:
        raise ValueError("volume_per_right must be > 0")
    if not 0 <= m <= R:
        raise ValueError(f"min_rights must be in [0, n_rights], got {m} (n_rights={R})")

    if discount_factors is None:
        df = np.ones(n_steps, dtype=float)
    else:
        df = np.asarray(discount_factors, dtype=float)
        if df.shape != (n_steps,):
            raise ValueError(
                f"discount_factors must have shape ({n_steps},), got {df.shape}"
            )
        if not np.isfinite(df).all() or (df <= 0.0).any():
            raise ValueError("discount_factors must be finite and strictly positive")

    rows = np.arange(n_paths)
    # value[:, c] = realised value-to-go from the next step given c rights
    # used. Terminal (after the last step): zero for every level.
    value = np.zeros((n_paths, R + 1), dtype=float)
    betas: list[np.ndarray] = [np.empty((basis_degree + 1, R + 1)) for _ in range(n_steps)]

    for t in range(n_steps - 1, -1, -1):
        x = paths[:, t]
        design = _polynomial_basis(x, basis_degree)
        beta, *_ = np.linalg.lstsq(design, value, rcond=None)  # (B, R+1)
        betas[t] = beta
        cont = design @ beta  # (n_paths, R+1) estimated continuation per level
        pay = q * (x - K) * df[t]  # (n_paths,) immediate payoff if exercised

        # Vectorised decision over the exercisable levels c = 0..R-1.
        cont_stay = cont[:, :R]                       # don't exercise -> stay at c
        cont_exercise = pay[:, None] + cont[:, 1:R + 1]  # exercise -> move to c+1
        exercise = cont_exercise > cont_stay          # (n_paths, R)

        # Take-or-pay forcing: at step t there are (n_steps - t) steps left
        # (this one included). Level c still needs (m - c) rights; if that
        # equals or exceeds the steps left, exercise is mandatory.
        c_levels = np.arange(R)
        required_more = m - c_levels
        force = (required_more >= (n_steps - t)) & (required_more > 0)
        exercise = exercise | force[None, :]

        new_value = np.empty_like(value)
        new_value[:, :R] = np.where(
            exercise, pay[:, None] + value[:, 1:R + 1], value[:, :R],
        )
        new_value[:, R] = value[:, R]  # all rights used -> nothing more to gain
        value = new_value

    per_path_value = value[:, 0]
    price = float(per_path_value.mean())
    std_error = float(per_path_value.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

    policy_value, expected_rights_used = _evaluate_policy(
        paths, betas, df, q, K, R, m, rows,
    )

    return SwingResult(
        price=price,
        std_error=std_error,
        per_path_value=per_path_value,
        policy_value=policy_value,
        expected_rights_used=expected_rights_used,
        n_paths=n_paths,
        n_steps=n_steps,
        n_rights=R,
    )


def _evaluate_policy(
    paths: np.ndarray,
    betas: list[np.ndarray],
    df: np.ndarray,
    q: float,
    K: float,
    R: int,
    m: int,
    rows: np.ndarray,
) -> tuple[float, float]:
    """Apply the estimated continuation regressions forward as a policy.

    Returns ``(mean realised value, mean rights used)``. This is the
    lower-bound estimator (a fixed policy evaluated on the paths) and the
    headline ``price`` is the backward estimator; the two bracket the true
    value and a large gap signals an unstable regression."""
    n_paths, n_steps = paths.shape
    basis_degree = betas[0].shape[0] - 1
    used = np.zeros(n_paths, dtype=int)
    realised = np.zeros(n_paths, dtype=float)
    for t in range(n_steps):
        x = paths[:, t]
        design = _polynomial_basis(x, basis_degree)
        cont = design @ betas[t]  # (n_paths, R+1)
        pay = q * (x - K) * df[t]
        can_exercise = used < R
        next_level = np.minimum(used + 1, R)
        cont_stay = cont[rows, used]
        cont_exercise = pay + cont[rows, next_level]
        exercise = can_exercise & (cont_exercise > cont_stay)
        required_more = m - used
        force = can_exercise & (required_more >= (n_steps - t)) & (required_more > 0)
        exercise = exercise | force
        realised += np.where(exercise, pay, 0.0)
        used = used + exercise.astype(int)
    return float(realised.mean()), float(used.mean())
