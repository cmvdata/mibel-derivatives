"""Schwartz-Smith two-factor forward-curve model for MIBEL.

Specification (CONTEXT.md § Pieza 2, with the four decisions resolved
on 2026-05-25):

    ln(S_t) = chi_t + xi_t + s(month(t))                         (spot)
    dchi_t  = -kappa · chi_t · dt + sigma_chi · dW_chi
    dxi_t   = mu_xi · dt + sigma_xi · dW_xi
    corr(dW_chi, dW_xi) = rho

where chi_t is the short-term mean-reverting deviation and xi_t the
long-term equilibrium level (random walk with drift). Risk-neutral
versions of the drifts:

    dchi_t  = (-kappa · chi_t - lambda_chi) · dt + sigma_chi · dW_chi*
    dxi_t   = mu_xi_star · dt + sigma_xi · dW_xi*

with mu_xi_star = mu_xi - lambda_xi, lambda_chi a short-term risk
premium, and (lambda_chi, mu_xi_star) the two risk-neutral parameters
that load the forward curve.

Closed-form price for a futures contract maturing at calendar date T,
observed at trade date t, with delivery-month seasonal s_m for the
delivery month of T:

    ln F(t, T) = e^{-kappa·(T-t)} · chi_t + xi_t
               + A(T - t)
               + s_delivery(T)
    A(tau)     = mu_xi_star · tau
               - (1 - e^{-kappa·tau}) · lambda_chi / kappa
               + 0.5 · [(1 - e^{-2·kappa·tau}) · sigma_chi^2 / (2 kappa)
                       + sigma_xi^2 · tau
                       + 2 · (1 - e^{-kappa·tau}) · rho · sigma_chi · sigma_xi / kappa]

Following Schwartz & Smith (2000) for the two-factor + closed-form
futures, with the four decisions adopted on 2026-05-25:

  1. Seasonal s(T) = 11 monthly dummies (January = reference). For an
     OMIP YR contract delivering Jan-Dec the model uses the mean of the
     12 dummies as the seasonal load.
  2. Spot OMIE used as the T = 0 observation in the Kalman filter is
     the daily mean of the hourly OMIE prices on the trade date (24-h
     base-load average).
  3. Two measurement-noise levels: epsilon_M for monthly contracts,
     epsilon_YR for yearly contracts. Reflects the OMIP liquidity gap
     between front-month and far-dated.
  4. Pieza 1 / Pieza 2 integration: when this module is fit, Pieza 1's
     slow factor theta_t becomes (chi_t + xi_t + s_delivery(t)) so that
     all derivative valuations consume a single, OMIP-consistent slow
     factor. Pieza 1's intraday seasonality s_HoD + DoW and the fast
     OU+Kou residual remain as an upper layer on top.

Calibration via Kalman filter + bounded MLE (commits H2 + H3); state
simulation (commit H4); validation tests on real OMIP forward
(commit H5).

Time scale convention: years (academic standard). T - t in years,
kappa in /year, sigma_chi and sigma_xi in /sqrt(year), drifts and
risk premia in /year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---- Constants and bounds --------------------------------------------------


# Bounds enforced in the MLE (commit H3); annual time scale.
KAPPA_BOUNDS: tuple[float, float] = (0.5, 5.0)            # per year, half-life [0.14, 1.4] y
SIGMA_CHI_BOUNDS: tuple[float, float] = (0.05, 2.0)       # per sqrt(year)
SIGMA_XI_BOUNDS: tuple[float, float] = (0.05, 1.0)        # per sqrt(year)
RHO_BOUNDS: tuple[float, float] = (-0.99, 0.99)
MU_XI_STAR_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # risk-neutral drift, per year
LAMBDA_CHI_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # short-term risk premium, per year
EPSILON_BOUNDS: tuple[float, float] = (1e-4, 0.50)        # measurement noise std

# Year length used in T-t conversion.
DAYS_PER_YEAR: float = 365.25


# ---- Parameter containers --------------------------------------------------


@dataclass(frozen=True)
class SSParams:
    """All fitted parameters of the Schwartz-Smith forward model.

    The eight numerical parameters (kappa, sigma_chi, sigma_xi, rho,
    mu_xi_star, lambda_chi, epsilon_M, epsilon_YR) plus the eleven
    monthly-dummy seasonals form the full parameter vector estimated
    by the Kalman + MLE step.
    """

    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float
    mu_xi_star: float
    lambda_chi: float
    epsilon_m: float
    epsilon_yr: float
    seasonal_dummies: np.ndarray  # shape (11,), Feb..Dec; January is reference (0)


@dataclass(frozen=True)
class SSFit:
    """Result of fitting the Schwartz-Smith model to OMIP forward data.

    Series fields are indexed by the calendar trade dates:
        state_chi   - filtered chi_t (short-term factor)
        state_xi    - filtered xi_t (long-term factor)
        state_cov   - 2x2 covariance of (chi_t, xi_t) per trade date,
                      flattened to shape (n_dates, 4): [c11, c12, c21, c22].
    """

    params: SSParams
    trade_dates: pd.DatetimeIndex
    state_chi: pd.Series
    state_xi: pd.Series
    state_cov: np.ndarray  # (n_dates, 4)
    log_likelihood: float
    n_obs: int        # total log F observations across all dates
    n_dates: int      # number of distinct trade dates
    rmse_log_m: float  # RMSE of log F residuals on M bucket
    rmse_log_yr: float  # RMSE of log F residuals on YR bucket


# ---- Closed-form pricing ---------------------------------------------------


def _seasonal_value(seasonal_dummies: np.ndarray, delivery_month: int) -> float:
    """Return s_delivery for a calendar month in [1, 12].

    With 11 dummies indexed 0..10 corresponding to February..December
    and January as the reference (value 0), the seasonal value is:

        s(month=1)        = 0.0
        s(month=2..12)    = seasonal_dummies[month - 2]
    """
    if not 1 <= delivery_month <= 12:
        raise ValueError(f"delivery_month must be in [1, 12], got {delivery_month}")
    if delivery_month == 1:
        return 0.0
    return float(seasonal_dummies[delivery_month - 2])


def _seasonal_yearly(seasonal_dummies: np.ndarray) -> float:
    """Mean of the twelve monthly seasonal values - the loading for an
    OMIP YR contract that delivers across all twelve months."""
    return float((0.0 + seasonal_dummies.sum()) / 12.0)


def A_function(
    tau: np.ndarray | float,
    *,
    kappa: float,
    sigma_chi: float,
    sigma_xi: float,
    rho: float,
    mu_xi_star: float,
    lambda_chi: float,
) -> np.ndarray | float:
    """Schwartz-Smith deterministic term in the futures-price formula.

        A(tau) = mu_xi_star * tau
               - (1 - exp(-kappa*tau)) * lambda_chi / kappa
               + 0.5 * [
                   (1 - exp(-2 kappa tau)) * sigma_chi^2 / (2 kappa)
                   + sigma_xi^2 * tau
                   + 2 * (1 - exp(-kappa tau)) * rho * sigma_chi * sigma_xi / kappa
                 ]

    ``tau`` is the time to delivery in years; ``kappa`` is in /year and
    the sigmas are in /sqrt(year).
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    tau_arr = np.asarray(tau, dtype=float)
    e_neg_kt = np.exp(-kappa * tau_arr)
    e_neg_2kt = np.exp(-2.0 * kappa * tau_arr)
    term_drift = mu_xi_star * tau_arr
    term_lambda = -(1.0 - e_neg_kt) * lambda_chi / kappa
    term_var = 0.5 * (
        (1.0 - e_neg_2kt) * sigma_chi**2 / (2.0 * kappa)
        + sigma_xi**2 * tau_arr
        + 2.0 * (1.0 - e_neg_kt) * rho * sigma_chi * sigma_xi / kappa
    )
    return term_drift + term_lambda + term_var


def futures_log_price(
    params: SSParams,
    chi: float,
    xi: float,
    tau: float,
    delivery_month: int | None = None,
    is_yearly: bool = False,
) -> float:
    """Closed-form log futures price.

        ln F(t, T) = e^{-kappa·tau} · chi_t + xi_t + A(tau) + s_delivery

    ``tau`` is years to maturity (T - t). For a monthly contract pass
    ``delivery_month`` in [1, 12]; for a yearly contract set
    ``is_yearly=True`` and the seasonal loading is the cross-month mean.
    """
    if delivery_month is None and not is_yearly:
        raise ValueError("must provide either delivery_month or is_yearly=True")
    if is_yearly and delivery_month is not None:
        raise ValueError("delivery_month must be None when is_yearly=True")

    a_val = float(A_function(
        tau,
        kappa=params.kappa,
        sigma_chi=params.sigma_chi,
        sigma_xi=params.sigma_xi,
        rho=params.rho,
        mu_xi_star=params.mu_xi_star,
        lambda_chi=params.lambda_chi,
    ))
    if is_yearly:
        s_val = _seasonal_yearly(params.seasonal_dummies)
    else:
        s_val = _seasonal_value(params.seasonal_dummies, int(delivery_month))
    e_neg_kt = float(np.exp(-params.kappa * tau))
    return e_neg_kt * chi + xi + a_val + s_val


# ---- Delivery-date midpoint helper -----------------------------------------


def delivery_midpoint(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """Midpoint of a delivery period [start, end] (inclusive). Used as
    the canonical T in F(t, T) for OMIP base-load contracts that pay
    the average over the period (Schwartz-Smith literature standard)."""
    if end < start:
        raise ValueError(f"end {end} before start {start}")
    return start + (end - start) / 2


def tau_years(trade_date: pd.Timestamp, delivery_date: pd.Timestamp) -> float:
    """Years between trade and delivery using DAYS_PER_YEAR = 365.25."""
    delta_days = (delivery_date - trade_date).days
    return float(delta_days) / DAYS_PER_YEAR


# ---- Public API (filled in commits H2-H4) ----------------------------------


def fit(
    omip_forward: pd.DataFrame,
    omie_daily_mean: pd.Series,
) -> SSFit:
    """Fit the Schwartz-Smith model to OMIP forward data, anchored by
    daily-mean OMIE spot. Implementation in commits H2 (Kalman filter)
    and H3 (bounded MLE).
    """
    raise NotImplementedError("forward.fit is implemented across commits H2 + H3")


def simulate(
    params: SSParams,
    initial_chi: float,
    initial_xi: float,
    start: pd.Timestamp,
    n_days: int,
    n_paths: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate (chi, xi) daily-step paths under the physical measure.

    Returns ``(chi_paths, xi_paths)`` both shape ``(n_paths, n_days)``.
    Implementation in commit H4.
    """
    raise NotImplementedError("forward.simulate is implemented in commit H4")
