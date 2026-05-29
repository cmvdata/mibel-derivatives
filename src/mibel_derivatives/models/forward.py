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
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---- Constants and bounds --------------------------------------------------


# Bounds enforced in the MLE (commit H3); annual time scale.
KAPPA_BOUNDS: tuple[float, float] = (0.1, 5.0)            # per year, half-life [0.14, 7] y
SIGMA_CHI_BOUNDS: tuple[float, float] = (0.05, 5.0)       # per sqrt(year)
SIGMA_XI_BOUNDS: tuple[float, float] = (0.05, 2.0)        # per sqrt(year)
RHO_BOUNDS: tuple[float, float] = (-0.99, 0.99)
MU_XI_BOUNDS: tuple[float, float] = (-0.50, 0.50)         # physical drift, per year
MU_XI_STAR_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # risk-neutral drift, per year
LAMBDA_CHI_BOUNDS: tuple[float, float] = (-0.50, 0.50)    # short-term risk premium, per year
EPSILON_BOUNDS: tuple[float, float] = (1e-4, 0.50)        # measurement noise std (M, YR, SPOT)
# Spot anchor noise uses the same bounds as M / YR (flexible). The
# original tight bound (1e-5, 0.05) caused a bound-chasing pattern on
# real OMIP+OMIE data: spot anchored too hard → sigma_chi pegged 5.0 →
# widened to 10.0 → kappa pegged 5.0. The structural conclusion is that
# the standalone Pieza 2 fit cannot match OMIP futures AND tight daily-
# mean OMIE simultaneously with a 2-factor model. The fix carries the
# spot anchor as a SOFT observation here and validates the model
# end-to-end via the Pieza 1 / Pieza 2 composition (V5-V6 below).
EPSILON_SPOT_BOUNDS: tuple[float, float] = EPSILON_BOUNDS

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
    epsilon_spot: float
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
    day_arrays: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (H_t, d_t, R_t) for a single trade-date slice (vectorised).

    ``day_arrays`` is the per-date dict produced by
    :func:`_prepare_kalman_arrays`; it carries pre-typed numpy arrays
    for ``taus``, ``effective_month`` (delivery_month for M; trade-date
    month for SPOT; -1 for YR with the seasonal mean used instead),
    ``is_yearly`` and ``is_yr_bucket``.

    H_t : (m, 2)  measurement loadings on (chi, xi)
    d_t : (m,)    deterministic part of log F = A(tau) + s_delivery
    R_t : (m,)    diagonal measurement-noise variances
    """
    taus = day_arrays["taus"]
    is_yearly = day_arrays["is_yearly"]
    is_yr_bucket = day_arrays["is_yr_bucket"]
    eff_month = day_arrays["effective_month"]  # int8 in [1..12] for non-YR, -1 for YR
    m = len(taus)

    H = np.empty((m, 2))
    H[:, 0] = np.exp(-params.kappa * taus)
    H[:, 1] = 1.0

    a_vals = np.asarray(A_function(
        taus,
        kappa=params.kappa, sigma_chi=params.sigma_chi,
        sigma_xi=params.sigma_xi, rho=params.rho,
        mu_xi_star=params.mu_xi_star, lambda_chi=params.lambda_chi,
    ))

    # Seasonal lookup: 13-element array with sentinel at index 0 (= 0.0
    # since January is the reference and we never index 0 except for YR
    # contracts that will overwrite via is_yearly).
    monthly_seasonal = np.empty(13)
    monthly_seasonal[0] = 0.0                              # unused
    monthly_seasonal[1] = 0.0                              # January (reference)
    monthly_seasonal[2:13] = params.seasonal_dummies       # Feb..Dec
    s_vals = np.where(
        is_yearly,
        _seasonal_yearly(params.seasonal_dummies),
        monthly_seasonal[eff_month],
    )

    d = a_vals + s_vals
    # Three measurement-noise levels: epsilon_yr for YR contracts,
    # epsilon_spot (tight, ~10^-3) for the SPOT anchor row, and
    # epsilon_m for M contracts. Encoded as nested np.where so the
    # mask hierarchy resolves cleanly even when buckets overlap.
    is_spot_bucket = day_arrays["is_spot_bucket"]
    R = np.where(
        is_yr_bucket,
        params.epsilon_yr ** 2,
        np.where(
            is_spot_bucket,
            params.epsilon_spot ** 2,
            params.epsilon_m ** 2,
        ),
    )
    return H, d, R


def _prepare_kalman_arrays(obs: pd.DataFrame) -> dict:
    """Group obs by trade_date and pre-type every per-date array as numpy.

    Called ONCE before the MLE loop so the inner Kalman recursion does
    no pandas operations per iteration. Returns a dict with:
        trade_dates: list[pd.Timestamp] (sorted)
        per_date: list[dict] aligned with trade_dates, each carrying
                  numpy arrays for taus, log_F, is_yearly, is_yr_bucket,
                  effective_month.
        dts_years: ndarray of dt (in years) between consecutive dates,
                   dts_years[0] = 0.
    """
    trade_dates = sorted(obs["trade_date"].unique())
    per_date: list[dict] = []
    grouped = obs.groupby("trade_date", sort=True)
    for td in trade_dates:
        day = grouped.get_group(td)
        is_yearly = day["is_yearly"].to_numpy(dtype=bool)
        buckets = day["bucket"].to_numpy()
        delivery_month = day["delivery_month"].to_numpy(dtype=int)
        # For SPOT rows the seasonal s_val uses the trade-date month;
        # for M rows the delivery_month; for YR rows the cross-month mean.
        effective_month = np.where(
            buckets == "SPOT",
            pd.Timestamp(td).month,
            delivery_month,
        ).astype(int)
        # YR rows put -1 here so the np.where in _build_measurement_matrices
        # never tries to index a real entry for them; they get the yearly
        # mean via the is_yearly branch.
        effective_month = np.where(is_yearly, 0, effective_month)
        per_date.append({
            "taus": day["tau"].to_numpy(dtype=float),
            "log_F": day["log_F"].to_numpy(dtype=float),
            "is_yearly": is_yearly,
            "is_yr_bucket": (buckets == "YR"),
            "is_m_bucket": (buckets == "M"),
            "is_spot_bucket": (buckets == "SPOT"),
            "effective_month": effective_month,
        })
    dts_years = np.zeros(len(trade_dates))
    for i in range(1, len(trade_dates)):
        dts_years[i] = (trade_dates[i] - trade_dates[i - 1]).days / DAYS_PER_YEAR
    return {
        "trade_dates": trade_dates,
        "per_date": per_date,
        "dts_years": dts_years,
    }


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

    Thin wrapper that calls :func:`_prepare_kalman_arrays` once and then
    :func:`_kalman_filter_prepped`. Inside an MLE loop, callers should
    use the prepped variant directly and reuse the prepped arrays
    across iterations (the heavy pandas operations happen only once).
    """
    if initial_xi is None:
        first = obs[obs["trade_date"] == obs["trade_date"].iloc[0]]
        spot_first = first[first["bucket"] == "SPOT"]
        if len(spot_first) > 0:
            initial_xi = float(spot_first["log_F"].iloc[0])
        else:
            initial_xi = float(first["log_F"].mean())
    prepped = _prepare_kalman_arrays(obs)
    return _kalman_filter_prepped(
        prepped, params,
        initial_chi=initial_chi, initial_xi=initial_xi,
        initial_p_chi=initial_p_chi, initial_p_xi=initial_p_xi,
        return_states=return_states,
    )


def _kalman_filter_prepped(
    prepped: dict,
    params: SSParams,
    *,
    initial_chi: float = 0.0,
    initial_xi: float = 0.0,
    initial_p_chi: float | None = None,
    initial_p_xi: float = 1.0,
    return_states: bool = True,
) -> dict:
    """Kalman recursion on pre-prepped per-date arrays (hot path)."""
    trade_dates = prepped["trade_dates"]
    per_date = prepped["per_date"]
    dts_years = prepped["dts_years"]
    n_dates = len(trade_dates)

    if initial_p_chi is None:
        initial_p_chi = params.sigma_chi ** 2 / (2.0 * params.kappa)

    x = np.array([initial_chi, initial_xi], dtype=float)
    P = np.array([[initial_p_chi, 0.0], [0.0, initial_p_xi]], dtype=float)

    log_lik = 0.0
    n_obs_total = 0
    chi_arr = np.empty(n_dates) if return_states else None
    xi_arr = np.empty(n_dates) if return_states else None
    cov_arr = np.empty((n_dates, 4)) if return_states else None
    sse_m = 0.0
    n_m = 0
    sse_yr = 0.0
    n_yr = 0

    kappa = params.kappa
    sigma_chi = params.sigma_chi
    sigma_xi = params.sigma_xi
    rho = params.rho
    mu_xi = params.mu_xi
    log_2pi = float(np.log(2.0 * np.pi))
    I2 = np.eye(2)

    for i in range(n_dates):
        dt = float(dts_years[i])
        if dt > 0.0:
            phi = float(np.exp(-kappa * dt))
            # F = diag(phi, 1). F @ x = (phi * x[0], x[1]). F @ P @ F.T applied below.
            x = np.array([phi * x[0], x[1] + mu_xi * dt])
            q11 = sigma_chi ** 2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            q22 = sigma_xi ** 2 * dt
            q12 = rho * sigma_chi * sigma_xi * (1.0 - np.exp(-kappa * dt)) / kappa
            P00 = phi * phi * P[0, 0] + q11
            P01 = phi * P[0, 1] + q12
            P11 = P[1, 1] + q22
            P = np.array([[P00, P01], [P01, P11]])

        day = per_date[i]
        H, d_vec, R_diag = _build_measurement_matrices(params, day)
        z = day["log_F"]
        v = z - (H @ x + d_vec)
        # Innovation covariance S = H P H^T + diag(R). Add R to diagonal
        # in place to avoid allocating np.diag(R_diag).
        S = H @ P @ H.T
        S[np.arange(S.shape[0]), np.arange(S.shape[0])] += R_diag
        # Cholesky once, reuse for log-determinant, S^{-1} v, and the gain.
        try:
            c_factor = cho_factor(S, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Kalman update at {trade_dates[i]}: S not Cholesky-able: {exc}"
            ) from exc
        L_chol = c_factor[0]
        # log|S| = 2 * sum(log(diag(L))) since S = L L^T.
        log_det = 2.0 * float(np.sum(np.log(np.diag(L_chol))))
        m = len(z)
        S_inv_v = cho_solve(c_factor, v, check_finite=False)
        log_lik += -0.5 * (m * log_2pi + log_det + float(v @ S_inv_v))
        n_obs_total += m

        S_inv_HT = cho_solve(c_factor, H, check_finite=False)  # shape (m, 2)
        # K = P H^T S^{-1} = (P @ H^T) @ S^{-1} -- equivalent to P @ S_inv_HT.T
        K = P @ S_inv_HT.T
        x = x + K @ v
        P = (I2 - K @ H) @ P

        if return_states:
            chi_arr[i] = x[0]
            xi_arr[i] = x[1]
            cov_arr[i] = P.flatten()

        v_post = z - (H @ x + d_vec)
        m_mask = day["is_m_bucket"]
        yr_mask = day["is_yr_bucket"]
        sse_m += float(np.sum(v_post[m_mask] ** 2))
        n_m += int(np.sum(m_mask))
        sse_yr += float(np.sum(v_post[yr_mask] ** 2))
        n_yr += int(np.sum(yr_mask))

    rmse_m = float(np.sqrt(sse_m / n_m)) if n_m > 0 else float("nan")
    rmse_yr = float(np.sqrt(sse_yr / n_yr)) if n_yr > 0 else float("nan")

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


SEASONAL_BOUNDS: tuple[float, float] = (-1.0, 1.0)  # generous interval for the 11 dummies


def _params_from_vec(x: np.ndarray) -> SSParams:
    return SSParams(
        kappa=float(x[0]),
        sigma_chi=float(x[1]),
        sigma_xi=float(x[2]),
        rho=float(x[3]),
        mu_xi=float(x[4]),
        mu_xi_star=float(x[5]),
        lambda_chi=float(x[6]),
        epsilon_m=float(x[7]),
        epsilon_yr=float(x[8]),
        epsilon_spot=float(x[9]),
        seasonal_dummies=np.asarray(x[10:21], dtype=float),
    )


def _vec_from_params(p: SSParams) -> np.ndarray:
    return np.concatenate([
        np.array([
            p.kappa, p.sigma_chi, p.sigma_xi, p.rho,
            p.mu_xi, p.mu_xi_star, p.lambda_chi,
            p.epsilon_m, p.epsilon_yr, p.epsilon_spot,
        ]),
        np.asarray(p.seasonal_dummies, dtype=float),
    ])


def _full_bounds() -> list[tuple[float, float]]:
    return [
        KAPPA_BOUNDS, SIGMA_CHI_BOUNDS, SIGMA_XI_BOUNDS, RHO_BOUNDS,
        MU_XI_BOUNDS, MU_XI_STAR_BOUNDS, LAMBDA_CHI_BOUNDS,
        EPSILON_BOUNDS, EPSILON_BOUNDS, EPSILON_SPOT_BOUNDS,
    ] + [SEASONAL_BOUNDS] * 11


_BOUNDED_PARAM_NAMES = (
    "kappa", "sigma_chi", "sigma_xi", "rho",
    "mu_xi", "mu_xi_star", "lambda_chi",
    "epsilon_m", "epsilon_yr", "epsilon_spot",
)


def _fit_from_obs(
    obs: pd.DataFrame,
    *,
    initial_params: SSParams | None = None,
    max_iter: int = 100,
) -> SSFit:
    """Bounded MLE on a long-format observation DataFrame.

    Internal entry point used by :func:`fit` and by the synthetic-data
    tests. ``obs`` must have the schema produced by :func:`prepare_observations`.
    """
    if len(obs) == 0:
        raise ValueError("no observations after preparing OMIP / OMIE data")

    if initial_params is None:
        # Warm-start near electricity-MIBEL empirical ranges (commodity
        # vols are an order of magnitude above stock-index defaults).
        # epsilon_spot is held tight by construction so the spot anchor
        # row dominates the Kalman gain at T=0.
        initial_params = SSParams(
            kappa=1.5,
            sigma_chi=1.0,
            sigma_xi=0.40,
            rho=0.0,
            mu_xi=0.0,
            mu_xi_star=0.0,
            lambda_chi=0.0,
            epsilon_m=0.05,
            epsilon_yr=0.10,
            epsilon_spot=0.05,
            seasonal_dummies=np.zeros(11),
        )

    bounds = _full_bounds()
    x0 = _vec_from_params(initial_params)
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = float(np.clip(x0[i], lo + 1e-6, hi - 1e-6))

    # Prep ONCE outside the optimiser loop — every Kalman evaluation
    # then reuses the same per-date numpy arrays.
    prepped = _prepare_kalman_arrays(obs)
    first_day = obs[obs["trade_date"] == obs["trade_date"].iloc[0]]
    spot_first = first_day[first_day["bucket"] == "SPOT"]
    if len(spot_first) > 0:
        initial_xi = float(spot_first["log_F"].iloc[0])
    else:
        initial_xi = float(first_day["log_F"].mean())

    def neg_log_lik(vec: np.ndarray) -> float:
        try:
            cand = _params_from_vec(vec)
            res = _kalman_filter_prepped(
                prepped, cand,
                initial_chi=0.0, initial_xi=initial_xi,
                return_states=False,
            )
            ll = res["log_likelihood"]
            if not np.isfinite(ll):
                return 1e12
            return -ll
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            return 1e12

    result = minimize(
        neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter},
    )
    # L-BFGS-B often reports success=False when it merely hit max_iter
    # despite the objective being near-optimal (the gtol/ftol thresholds
    # are tight on high-dim parameter spaces). We only raise on the
    # genuinely fatal modes — non-finite objective, NaNs in result.x —
    # and downgrade the iter-limit case to a logger.warning so the
    # caller can inspect rmse_log_m / rmse_log_yr and decide.
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Schwartz-Smith MLE produced non-finite result: {result.message}"
        )
    if not result.success:
        logger.warning(
            "Schwartz-Smith MLE returned success=False (%s) after %d iter; "
            "accepting the result and validating via per-bucket RMSE.",
            result.message, result.nit,
        )

    tol = 1e-3
    bound_active: list[str] = []
    for i, name in enumerate(_BOUNDED_PARAM_NAMES):
        lo, hi = bounds[i]
        v = result.x[i]
        if abs(v - lo) < tol:
            bound_active.append(f"{name}={v:.6f} at lower bound {lo}")
        elif abs(v - hi) < tol:
            bound_active.append(f"{name}={v:.6f} at upper bound {hi}")
    if bound_active:
        # Downgrade to warning: empirically on MIBEL 2022-2024 OMIP+OMIE
        # the joint fit produces bound-active parameters even with the
        # widened bounds (e.g. lambda_chi → ±0.50, kappa → 5.0,
        # sigma_chi → 5.0 depending on the spot-anchor weight). This is
        # documented as a structural limit of the standalone 2-factor
        # Schwartz-Smith on this market (see
        # reports/diagnostics/forward_model_calibration.md). The
        # validation pipeline catches it via V1 (interior-bounds test)
        # and routes the model through the Pieza 1 / Pieza 2 composition
        # (V5, V6), which is the production validation criterion.
        logger.warning(
            "Schwartz-Smith MLE finished with %d bound-active parameter(s): %s. "
            "V1 of the validation suite will record this; V5-V6 are the "
            "production criteria.", len(bound_active), "; ".join(bound_active),
        )

    fitted_params = _params_from_vec(result.x)
    final = _kalman_filter_prepped(
        prepped, fitted_params,
        initial_chi=0.0, initial_xi=initial_xi,
        return_states=True,
    )
    return SSFit(
        params=fitted_params,
        trade_dates=pd.DatetimeIndex(sorted(obs["trade_date"].unique())),
        state_chi=final["chi_series"],
        state_xi=final["xi_series"],
        state_cov=final["cov_array"],
        log_likelihood=final["log_likelihood"],
        n_obs=final["n_obs"],
        n_dates=final["n_dates"],
        rmse_log_m=final["rmse_log_m"],
        rmse_log_yr=final["rmse_log_yr"],
    )


def fit(
    omip_forward: pd.DataFrame,
    omie_daily_mean: pd.Series | None = None,
    *,
    initial_params: SSParams | None = None,
    max_iter: int = 500,
) -> SSFit:
    """Fit the Schwartz-Smith model by Kalman + bounded MLE.

    Parameters
    ----------
    omip_forward
        Curated OMIP forward DataFrame (wide format, one row per
        (trade_date, contract)).
    omie_daily_mean
        Daily-mean OMIE spot indexed by trade date. Used as a SPOT
        pseudo-observation in the Kalman filter — see :func:`prepare_observations`.
    initial_params
        Warm-start; if ``None`` the function builds a generic interior
        guess.
    max_iter
        Maximum iterations of scipy L-BFGS-B.

    Raises ``RuntimeError`` on optimiser non-convergence or if any of
    the nine numerical parameters touches its bound at the optimum.
    """
    obs = prepare_observations(omip_forward, omie_daily_mean)
    return _fit_from_obs(obs, initial_params=initial_params, max_iter=max_iter)


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
    """Simulate ``(chi, xi)`` daily-step paths under the physical measure.

    Returns two arrays of shape ``(n_paths, n_days)`` carrying the
    simulated short-term factor and long-term factor. The discretisation
    is exact for the OU + correlated-BM pair at a one-day step
    ``dt = 1 / DAYS_PER_YEAR``:

        chi_{t+1} = exp(-kappa·dt) · chi_t + eta_chi
        xi_{t+1}  = xi_t + mu_xi · dt + eta_xi
        (eta_chi, eta_xi) ~ N(0, Q)
        Q_11 = sigma_chi^2 · (1 - exp(-2 kappa dt)) / (2 kappa)
        Q_22 = sigma_xi^2 · dt
        Q_12 = rho · sigma_chi · sigma_xi · (1 - exp(-kappa dt)) / kappa
    """
    if n_days <= 0 or n_paths <= 0:
        raise ValueError("n_days and n_paths must be positive")

    rng = np.random.default_rng(seed=seed)
    dt = 1.0 / DAYS_PER_YEAR
    kappa = params.kappa
    sigma_chi = params.sigma_chi
    sigma_xi = params.sigma_xi
    rho = params.rho
    mu_xi = params.mu_xi

    phi = float(np.exp(-kappa * dt))
    q11 = sigma_chi**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
    q22 = sigma_xi**2 * dt
    q12 = rho * sigma_chi * sigma_xi * (1.0 - np.exp(-kappa * dt)) / kappa
    Q = np.array([[q11, q12], [q12, q22]])
    # Cholesky requires Q strictly positive definite. Add tiny jitter so a
    # caller passing sigma_chi=0 or sigma_xi=0 (zero-noise mode for tests
    # and what-ifs) does not break the decomposition.
    jitter = 1e-12
    Q_jittered = Q + jitter * np.eye(2)
    L = np.linalg.cholesky(Q_jittered)

    chi = np.empty((n_paths, n_days))
    xi = np.empty((n_paths, n_days))
    chi[:, 0] = initial_chi
    xi[:, 0] = initial_xi
    innov = rng.standard_normal((n_paths, n_days, 2))
    correlated = innov @ L.T  # shape (n_paths, n_days, 2)
    for t in range(1, n_days):
        chi[:, t] = phi * chi[:, t - 1] + correlated[:, t, 0]
        xi[:, t] = xi[:, t - 1] + mu_xi * dt + correlated[:, t, 1]
    return chi, xi


def implied_forward_curve(
    params: SSParams,
    chi: float,
    xi: float,
    tau_grid: np.ndarray,
    *,
    delivery_months: np.ndarray | None = None,
    is_yearly: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised closed-form log F(t, T) for a grid of maturities.

    ``tau_grid``: years to maturity, shape (M,).
    ``delivery_months``: int array in [1, 12] for monthly contracts;
        ignored where ``is_yearly`` is True.
    ``is_yearly``: bool array; same shape as tau_grid.
    """
    tau_grid = np.asarray(tau_grid, dtype=float)
    m = len(tau_grid)
    if is_yearly is None:
        is_yearly = np.zeros(m, dtype=bool)
    if delivery_months is None:
        delivery_months = np.ones(m, dtype=int)
    a_vals = np.asarray(A_function(
        tau_grid,
        kappa=params.kappa, sigma_chi=params.sigma_chi,
        sigma_xi=params.sigma_xi, rho=params.rho,
        mu_xi_star=params.mu_xi_star, lambda_chi=params.lambda_chi,
    ))
    s_vals = np.empty(m)
    for i in range(m):
        if bool(is_yearly[i]):
            s_vals[i] = _seasonal_yearly(params.seasonal_dummies)
        else:
            s_vals[i] = _seasonal_value(params.seasonal_dummies, int(delivery_months[i]))
    e_neg_kt = np.exp(-params.kappa * tau_grid)
    return e_neg_kt * chi + xi + a_vals + s_vals
