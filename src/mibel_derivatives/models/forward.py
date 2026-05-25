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
MU_XI_BOUNDS: tuple[float, float] = (-0.50, 0.50)         # physical drift, per year
MU_XI_STAR_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # risk-neutral drift, per year
LAMBDA_CHI_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # short-term risk premium, per year
EPSILON_BOUNDS: tuple[float, float] = (1e-4, 0.50)        # measurement noise std

# Year length used in T-t conversion.
DAYS_PER_YEAR: float = 365.25


# ---- Parameter containers --------------------------------------------------


@dataclass(frozen=True)
class SSParams:
    """All fitted parameters of the Schwartz-Smith forward model.

    Nine numerical parameters and eleven monthly-dummy seasonals form
    the full parameter vector estimated by the Kalman + MLE step:

      - ``kappa``        — OU mean-reversion rate of chi (/year)
      - ``sigma_chi``    — diffusion coefficient of chi (/sqrt(year))
      - ``sigma_xi``     — diffusion coefficient of xi (/sqrt(year))
      - ``rho``          — correlation between chi and xi shocks
      - ``mu_xi``        — physical drift of xi (/year) — Kalman state
                           transition uses this
      - ``mu_xi_star``   — risk-neutral drift of xi (/year) — A(tau)
                           uses this. ``lambda_xi = mu_xi - mu_xi_star``
                           is the implied long-term risk premium
      - ``lambda_chi``   — short-term risk premium (/year) — A(tau) uses
                           this
      - ``epsilon_m``    — measurement noise std for monthly contracts
      - ``epsilon_yr``   — measurement noise std for yearly contracts
      - ``seasonal_dummies`` — 11-vector of monthly dummies Feb..Dec
                              (January = reference)
    """

    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float
    mu_xi: float
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


# ---- Observation preparation -----------------------------------------------


def prepare_observations(
    omip_forward: pd.DataFrame,
    omie_daily_mean: pd.Series | None = None,
) -> pd.DataFrame:
    """Reshape the curated OMIP DataFrame into long format consumable by
    the Kalman filter.

    Output columns:
        trade_date            (pd.Timestamp, normalised to midnight UTC)
        bucket                ("M" or "YR")
        delivery_month        (int 1..12 for M; ignored for YR)
        is_yearly             (bool)
        tau                   (years to delivery midpoint, float)
        log_F                 (float, ln of reference_d_eur_mwh)

    OMIP rows with missing ``reference_d_eur_mwh`` are dropped (≈ 1.7%
    of the curated parquet). Each M contract is given a delivery
    midpoint of the calendar month it names; each YR contract is given
    a delivery midpoint of July 2nd of the named year. The convention
    is consistent with the Schwartz-Smith literature for base-load
    average contracts.

    If ``omie_daily_mean`` is provided, its values are inserted as
    pseudo-observations at ``tau = 0`` with bucket = "SPOT", treated by
    the Kalman filter as an additional row in the observation vector.
    """
    df = omip_forward.copy()
    df = df.dropna(subset=["reference_d_eur_mwh"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # --- Decode contract → (delivery_month, delivery_midpoint, is_yearly).
    contract = df["contract"].astype(str)
    is_m = contract.str.startswith("FTB M ")
    is_yr = contract.str.startswith("FTB YR-")
    if not (is_m | is_yr).all():
        bad = contract[~(is_m | is_yr)].iloc[0]
        raise ValueError(f"unrecognised contract format: {bad!r}")

    delivery_midpoints: list[pd.Timestamp] = []
    delivery_months: list[int] = []
    yearly_flags: list[bool] = []
    _months_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    for c in contract:
        if c.startswith("FTB M "):
            # e.g. "FTB M Feb-19" → February 2019.
            tail = c.removeprefix("FTB M ")
            mon, yy = tail.split("-")
            month_num = _months_map[mon]
            year_full = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)
            start = pd.Timestamp(year_full, month_num, 1)
            end = (start + pd.offsets.MonthEnd(0)).normalize()
            mid = delivery_midpoint(start, end)
            delivery_midpoints.append(mid)
            delivery_months.append(month_num)
            yearly_flags.append(False)
        else:
            # e.g. "FTB YR-26" → 2026 entire year, midpoint July 2nd.
            yy = c.removeprefix("FTB YR-")
            year_full = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)
            delivery_midpoints.append(pd.Timestamp(year_full, 7, 2))
            delivery_months.append(0)
            yearly_flags.append(True)
    df["delivery_midpoint"] = pd.to_datetime(delivery_midpoints)
    df["delivery_month"] = delivery_months
    df["is_yearly"] = yearly_flags
    df["bucket"] = np.where(df["is_yearly"], "YR", "M")
    df["tau"] = (
        (df["delivery_midpoint"] - df["trade_date"]).dt.days
    ) / DAYS_PER_YEAR
    # Drop already-expired contracts (negative tau) — possible at the
    # very end of a contract's life in the source data.
    df = df[df["tau"] > 0.0].reset_index(drop=True)
    df["log_F"] = np.log(df["reference_d_eur_mwh"].astype(float))

    obs = df[[
        "trade_date", "bucket", "delivery_month",
        "is_yearly", "tau", "log_F",
    ]].copy()

    if omie_daily_mean is not None:
        spot_df = omie_daily_mean.dropna().to_frame("price")
        spot_df["log_F"] = np.log(spot_df["price"].astype(float))
        spot_df = spot_df.reset_index().rename(columns={
            spot_df.index.name or "index": "trade_date",
        })
        spot_df["trade_date"] = pd.to_datetime(spot_df["trade_date"]).dt.tz_localize(None)
        spot_df["bucket"] = "SPOT"
        spot_df["delivery_month"] = 0  # ignored
        spot_df["is_yearly"] = False
        spot_df["tau"] = 0.0
        spot_df = spot_df[[
            "trade_date", "bucket", "delivery_month",
            "is_yearly", "tau", "log_F",
        ]]
        obs = pd.concat([obs, spot_df], ignore_index=True)

    obs = obs.sort_values(["trade_date", "bucket", "tau"]).reset_index(drop=True)
    return obs


# ---- Kalman filter ---------------------------------------------------------


def _build_measurement_matrices(
    params: SSParams,
    day_obs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (H_t, d_t, R_t) for a single trade-date slice of obs.

    H_t : (m, 2)  measurement loadings on (chi, xi)
    d_t : (m,)    deterministic part of log F = A(tau) + s_delivery
    R_t : (m,)    diagonal measurement-noise variances
    """
    m = len(day_obs)
    H = np.empty((m, 2))
    d = np.empty(m)
    R = np.empty(m)
    taus = day_obs["tau"].to_numpy(dtype=float)
    a_vals = np.asarray(A_function(
        taus,
        kappa=params.kappa, sigma_chi=params.sigma_chi,
        sigma_xi=params.sigma_xi, rho=params.rho,
        mu_xi_star=params.mu_xi_star, lambda_chi=params.lambda_chi,
    ))
    eps_m_var = params.epsilon_m ** 2
    eps_yr_var = params.epsilon_yr ** 2
    # SPOT uses epsilon_m by convention (front-month liquidity).
    for i, row in enumerate(day_obs.itertuples()):
        H[i, 0] = float(np.exp(-params.kappa * row.tau))
        H[i, 1] = 1.0
        if row.is_yearly:
            s_val = _seasonal_yearly(params.seasonal_dummies)
        elif row.bucket == "SPOT":
            month_t = pd.Timestamp(row.trade_date).month
            s_val = _seasonal_value(params.seasonal_dummies, month_t)
        else:
            s_val = _seasonal_value(params.seasonal_dummies, int(row.delivery_month))
        d[i] = float(a_vals[i]) + s_val
        if row.bucket == "YR":
            R[i] = eps_yr_var
        else:
            R[i] = eps_m_var
    return H, d, R


def _kalman_filter(
    obs: pd.DataFrame,
    params: SSParams,
    *,
    initial_chi: float = 0.0,
    initial_xi: float | None = None,
    initial_p_chi: float | None = None,
    initial_p_xi: float = 1.0,
    return_states: bool = True,
) -> dict:
    """Run the Schwartz-Smith Kalman filter on long-format observations.

    Returns a dict with keys:
        log_likelihood : float (sum over trade dates)
        n_obs          : total number of scalar observations
        n_dates        : number of distinct trade dates
        chi_series     : pd.Series of filtered chi per trade date (if return_states)
        xi_series      : pd.Series of filtered xi  per trade date (if return_states)
        cov_array      : np.ndarray (n_dates, 4) flattened 2×2 covariance
                          [[p_chi_chi, p_chi_xi], [p_xi_chi, p_xi_xi]]
                          (if return_states)
        rmse_log_m     : RMSE of post-filter residuals on M bucket
        rmse_log_yr    : RMSE of post-filter residuals on YR bucket
    """
    if initial_p_chi is None:
        initial_p_chi = params.sigma_chi ** 2 / (2.0 * params.kappa)
    if initial_xi is None:
        # First trade-date's mean spot or first long-dated obs.
        first = obs[obs["trade_date"] == obs["trade_date"].iloc[0]]
        spot_first = first[first["bucket"] == "SPOT"]
        if len(spot_first) > 0:
            initial_xi = float(spot_first["log_F"].iloc[0])
        else:
            initial_xi = float(first["log_F"].mean())

    x = np.array([initial_chi, initial_xi], dtype=float)
    P = np.array([[initial_p_chi, 0.0], [0.0, initial_p_xi]], dtype=float)

    grouped = obs.groupby("trade_date", sort=True)
    trade_dates = sorted(obs["trade_date"].unique())
    n_dates = len(trade_dates)

    log_lik = 0.0
    n_obs_total = 0
    chi_arr = np.empty(n_dates)
    xi_arr = np.empty(n_dates)
    cov_arr = np.empty((n_dates, 4))
    rmse_m_terms: list[float] = []
    rmse_yr_terms: list[float] = []

    kappa = params.kappa
    sigma_chi = params.sigma_chi
    sigma_xi = params.sigma_xi
    rho = params.rho
    mu_xi = params.mu_xi
    log_2pi = float(np.log(2.0 * np.pi))

    prev_date: pd.Timestamp | None = None
    for i, td in enumerate(trade_dates):
        # Predict step. Step size dt in years.
        if prev_date is None:
            dt = 0.0  # no prediction on the first step
        else:
            dt = (td - prev_date).days / DAYS_PER_YEAR
        if dt > 0.0:
            phi = float(np.exp(-kappa * dt))
            F = np.array([[phi, 0.0], [0.0, 1.0]], dtype=float)
            c = np.array([0.0, mu_xi * dt], dtype=float)
            q11 = sigma_chi ** 2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            q22 = sigma_xi ** 2 * dt
            q12 = (
                rho * sigma_chi * sigma_xi
                * (1.0 - np.exp(-kappa * dt)) / kappa
            )
            Q = np.array([[q11, q12], [q12, q22]], dtype=float)
            x = F @ x + c
            P = F @ P @ F.T + Q

        # Update step.
        day_obs = grouped.get_group(td)
        H, d_vec, R_diag = _build_measurement_matrices(params, day_obs)
        z = day_obs["log_F"].to_numpy(dtype=float)
        v = z - (H @ x + d_vec)
        S = H @ P @ H.T + np.diag(R_diag)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Kalman update at {td}: S not invertible: {exc}") from exc
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError(f"Kalman update at {td}: log|S| not positive (sign={sign})")
        m = len(z)
        ll_t = -0.5 * (m * log_2pi + log_det + float(v @ S_inv @ v))
        log_lik += ll_t
        n_obs_total += m

        K = P @ H.T @ S_inv
        x = x + K @ v
        P = (np.eye(2) - K @ H) @ P

        if return_states:
            chi_arr[i] = x[0]
            xi_arr[i] = x[1]
            cov_arr[i] = P.flatten()

        # Track post-filter residuals per bucket for RMSE.
        z_pred = H @ x + d_vec
        v_post = z - z_pred
        for j, row in enumerate(day_obs.itertuples()):
            if row.bucket == "M":
                rmse_m_terms.append(float(v_post[j]) ** 2)
            elif row.bucket == "YR":
                rmse_yr_terms.append(float(v_post[j]) ** 2)
        prev_date = td

    rmse_m = float(np.sqrt(np.mean(rmse_m_terms))) if rmse_m_terms else float("nan")
    rmse_yr = float(np.sqrt(np.mean(rmse_yr_terms))) if rmse_yr_terms else float("nan")

    out = {
        "log_likelihood": float(log_lik),
        "n_obs": n_obs_total,
        "n_dates": n_dates,
        "rmse_log_m": rmse_m,
        "rmse_log_yr": rmse_yr,
    }
    if return_states:
        idx = pd.DatetimeIndex(trade_dates)
        out["chi_series"] = pd.Series(chi_arr, index=idx, name="chi")
        out["xi_series"] = pd.Series(xi_arr, index=idx, name="xi")
        out["cov_array"] = cov_arr
    return out


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
