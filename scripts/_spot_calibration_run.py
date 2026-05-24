"""Calibration helper for the phase-2 spot model.

Loads the curated OMIE day-ahead Spain series, fits the MRJD model,
runs ADF + Jarque-Bera diagnostics, simulates 100 forward paths and
saves figures plus a JSON summary that the calibration report quotes.
Intended for ad-hoc execution before regenerating the diagnostic doc;
not part of the production pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.tsa.stattools import adfuller

from mibel_derivatives.models import spot


def main() -> None:
    df = pd.read_parquet("data/curated/omie_spot_es_2019_2024.parquet")
    df = df.set_index("datetime_utc").sort_index()
    hourly = df[df.index.minute == 0]["price_eur_mwh"].rename("price_eur_mwh")
    hourly.index.name = "dt_utc"

    fit = spot.fit(hourly)
    p = fit.params
    nj = fit.residual_returns[~fit.jumps_mask]

    adf_r = adfuller(fit.residuals.values, regression="c", autolag="AIC")
    adf_d = adfuller(fit.residual_returns.values, regression="c", autolag="AIC")
    jb_nj = sstats.jarque_bera(nj.values)
    jb_all = sstats.jarque_bera(fit.residual_returns.values)

    sim_start = hourly.index[-1] + pd.Timedelta("1h")
    last_resid = float(fit.residuals.iloc[-1])
    sim_paths = spot.simulate(
        p, sim_start, n_hours=24 * 365, n_paths=100,
        initial_residual=last_resid, seed=2026,
    )
    sim_log = np.log(sim_paths + p.price_shift)
    sim_log_returns = np.diff(sim_log, axis=1).ravel()
    hist_returns = np.log(hourly + p.price_shift).diff().dropna().values

    X = spot._seasonal_design_matrix(hourly.index, harmonics=4).values
    y = np.log(hourly + p.price_shift).values
    beta = np.concatenate([
        [p.seasonality.intercept],
        p.seasonality.fourier_coefs,
        p.seasonality.dow_coefs,
        p.seasonality.hod_coefs,
    ])
    resid_full = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid_full.var(ddof=k))
    XtX_inv = np.linalg.inv(X.T @ X)
    se_coefs = np.sqrt(sigma2 * np.diag(XtX_inv))
    intercept_se = float(se_coefs[0])

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly.index, hourly.values, lw=0.3)
    ax.set_title("OMIE day-ahead Spain (ESIOS 600), 2019-2024 hourly")
    ax.set_ylabel("EUR/MWh"); ax.set_xlabel("UTC date"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "spot_omie_series.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    doys = np.arange(1, 366)
    ang = 2 * np.pi * doys / 365.25
    fy = np.zeros_like(doys, dtype=float)
    for kh in range(1, 5):
        fy = fy + p.seasonality.fourier_coefs[2 * (kh - 1)] * np.cos(kh * ang)
        fy = fy + p.seasonality.fourier_coefs[2 * kh - 1] * np.sin(kh * ang)
    axes[0].plot(doys, fy); axes[0].set_title("Fourier annual")
    axes[0].set_xlabel("day of year"); axes[0].grid(alpha=0.3)
    dow_full = np.concatenate([[0.0], p.seasonality.dow_coefs])
    axes[1].bar(range(7), dow_full,
                tick_label=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_title("Day-of-week"); axes[1].grid(alpha=0.3, axis="y")
    hod_full = np.concatenate([[0.0], p.seasonality.hod_coefs])
    axes[2].bar(range(24), hod_full)
    axes[2].set_title("Hour-of-day (UTC)"); axes[2].set_xlabel("hour UTC")
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "spot_seasonal_components.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(24), p.sigma_by_hour)
    ax.set_title(f"OU sigma by hour-of-day (UTC); kappa={p.kappa:.4f}/h")
    ax.set_xlabel("hour UTC"); ax.set_ylabel("sigma"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "spot_sigma_by_hour.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(fit.residuals.index, fit.residuals.values, lw=0.3, label="residuals")
    jr_idx = fit.residual_returns.index[fit.jumps_mask.values]
    ax.scatter(jr_idx, fit.residuals.reindex(jr_idx).values,
               s=4, color="red", label=f"jumps n={fit.n_jumps}", zorder=3)
    ax.set_title("Deseasonalised residuals Z_t with detected jumps")
    ax.set_ylabel("log(P+10) residual"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "spot_residuals_jumps.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    nj_v = nj.values
    ax.hist(nj_v, bins=120, density=True, alpha=0.6, label="non-jump returns")
    xs = np.linspace(nj_v.min(), nj_v.max(), 400)
    ax.plot(xs, sstats.norm.pdf(xs, nj_v.mean(), nj_v.std()), "k", label="N fit")
    ax.set_yscale("log"); ax.set_title("Non-jump return distribution (log y)")
    ax.set_xlabel("delta Z_t"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "spot_return_hist.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    sim_idx = pd.date_range(sim_start, periods=sim_paths.shape[1], freq="h")
    mean_p = sim_paths.mean(axis=0)
    p10 = np.percentile(sim_paths, 10, axis=0)
    p90 = np.percentile(sim_paths, 90, axis=0)
    ax.fill_between(sim_idx, p10, p90, alpha=0.2, label="P10-P90")
    ax.plot(sim_idx, mean_p, label="mean", lw=0.6)
    ax.set_title("Simulated 2025 paths (n=100) from end-2024 state")
    ax.set_ylabel("EUR/MWh"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "spot_simulation_band.png", dpi=110)
    plt.close(fig)

    summary = {
        "n_obs": int(fit.n_obs),
        "date_min": str(hourly.index.min()),
        "date_max": str(hourly.index.max()),
        "price_min": float(hourly.min()),
        "price_max": float(hourly.max()),
        "price_mean": float(hourly.mean()),
        "intercept": float(p.seasonality.intercept),
        "intercept_se": float(intercept_se),
        "intercept_ci95": [
            float(p.seasonality.intercept - 1.96 * intercept_se),
            float(p.seasonality.intercept + 1.96 * intercept_se),
        ],
        "fourier_coefs": p.seasonality.fourier_coefs.tolist(),
        "dow_coefs": p.seasonality.dow_coefs.tolist(),
        "hod_coefs_min": float(p.seasonality.hod_coefs.min()),
        "hod_coefs_max": float(p.seasonality.hod_coefs.max()),
        "peak_minus_trough_log": float(
            p.seasonality.hod_coefs.max() - p.seasonality.hod_coefs.min()
        ),
        "kappa": float(p.kappa),
        "kappa_halflife_hours": float(np.log(2) / p.kappa),
        "sigma_min": float(p.sigma_by_hour.min()),
        "sigma_max": float(p.sigma_by_hour.max()),
        "sigma_mean": float(p.sigma_by_hour.mean()),
        "jump_intensity_per_hour": float(p.jump_intensity),
        "jump_intensity_per_year": float(p.jump_intensity * 8760),
        "jump_p_up": float(p.jump_p_up),
        "jump_eta_up": float(p.jump_eta_up),
        "jump_eta_down": float(p.jump_eta_down),
        "mean_jump_up_log": float(1.0 / p.jump_eta_up),
        "mean_jump_down_log": float(1.0 / p.jump_eta_down),
        "n_jumps": int(fit.n_jumps),
        "jump_pct": float(fit.n_jumps / fit.n_obs * 100),
        "residual_std": float(fit.residuals.std()),
        "return_std_all": float(fit.residual_returns.std()),
        "return_std_nonjump": float(nj.std()),
        "adf_residuals_stat": float(adf_r[0]),
        "adf_residuals_pvalue": float(adf_r[1]),
        "adf_returns_stat": float(adf_d[0]),
        "adf_returns_pvalue": float(adf_d[1]),
        "jb_nonjump_stat": float(jb_nj.statistic),
        "jb_nonjump_pvalue": float(jb_nj.pvalue),
        "jb_all_stat": float(jb_all.statistic),
        "jb_all_pvalue": float(jb_all.pvalue),
        "sim_mean_2025": float(sim_paths.mean()),
        "sim_std_2025": float(sim_paths.std()),
        "hist_returns_mean": float(hist_returns.mean()),
        "hist_returns_std": float(hist_returns.std()),
        "sim_returns_mean": float(sim_log_returns.mean()),
        "sim_returns_std": float(sim_log_returns.std()),
    }
    Path("reports/_spot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
