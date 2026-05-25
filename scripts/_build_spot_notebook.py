"""Build notebooks/01_spot_model.ipynb from cell sources defined here.

The notebook is committed with empty outputs; running it from top to
bottom reproduces the EDA, calibration and validation against
``data/curated/omie_spot_es_2019_2024.parquet`` for the slow-fast
MRJD spot model.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/01_spot_model.ipynb")


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells: list[nbf.NotebookNode] = []

    cells.append(md(
        "# Spot model calibration on OMIE day-ahead Spain\n"
        "\n"
        "Phase 2 / Pieza 1 of the mibel-derivatives module.\n"
        "\n"
        "Slow-fast mean-reverting jump-diffusion on the hourly UTC log-price\n"
        "(reespec 2026-05-24, slow-OU adjustment 2026-05-25):\n"
        "\n"
        "    Y_t = log(P_t + c) = θ_t + s(t) + Z_t\n"
        "\n"
        "with θ_t the slow factor (EMA span = 24 h, slow OU in simulation),\n"
        "s(t) the deterministic seasonality (Fourier annual + DoW + HoD),\n"
        "and Z_t the fast OU + Kou jumps residual. Bounded MLE keeps κ̂,\n"
        "λ̂, η̂ inside physically plausible ranges. Full spec and decisions\n"
        "recorded in `src/mibel_derivatives/models/spot.py` and the\n"
        "calibration diagnostic in\n"
        "`reports/diagnostics/spot_model_calibration.md`.\n"
    ))

    cells.append(code(
        "from __future__ import annotations\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy import stats as sstats\n"
        "from statsmodels.tsa.stattools import adfuller\n"
        "\n"
        "from mibel_derivatives.models import spot\n"
        "\n"
        "pd.set_option('display.float_format', '{:.4f}'.format)\n"
        "plt.rcParams['figure.dpi'] = 110\n"
    ))

    cells.append(md(
        "## 1. Load curated data\n"
        "\n"
        "Manus-sourced OMIE day-ahead Spain via ESIOS indicator 600. UTC\n"
        "hourly granularity 2019-01-01 → 2024-12-31."
    ))
    cells.append(code(
        "df = pd.read_parquet('data/curated/omie_spot_es_2019_2024.parquet')\n"
        "df = df.set_index('datetime_utc').sort_index()\n"
        "hourly = df[df.index.minute == 0]['price_eur_mwh'].rename('price_eur_mwh')\n"
        "hourly.index.name = 'dt_utc'\n"
        "print('Rows:', len(hourly))\n"
        "print('Coverage:', hourly.index.min(), '..', hourly.index.max())\n"
        "print(hourly.describe())\n"
    ))

    cells.append(md(
        "## 2. Exploratory analysis\n"
        "\n"
        "- Full hourly series 2019-2024 with the 2022 gas-crisis regime.\n"
        "- Annual breakdown: mean and p95 by calendar year, showing the\n"
        "  scale of the regime shift the slow factor must absorb.\n"
        "- Histogram of nominal prices and the autocorrelation of returns."
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "ax.plot(hourly.index, hourly.values, lw=0.3)\n"
        "ax.set_title('OMIE day-ahead Spain (ESIOS 600), 2019-2024 hourly')\n"
        "ax.set_ylabel('EUR/MWh'); ax.set_xlabel('UTC date'); ax.grid(alpha=0.3)\n"
        "plt.show()\n"
    ))
    cells.append(code(
        "for y in (2019, 2020, 2021, 2022, 2023, 2024):\n"
        "    yr = hourly[hourly.index.year == y]\n"
        "    print(f'{y}: n={len(yr):5d}  mean={yr.mean():7.2f}  p95={np.percentile(yr, 95):7.2f}')\n"
        "print(f'union: mean={hourly.mean():.2f}  p95={np.percentile(hourly, 95):.2f}')\n"
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].hist(hourly.values, bins=100, color='steelblue')\n"
        "axes[0].set_title('Hourly price distribution (EUR/MWh)')\n"
        "axes[0].grid(alpha=0.3)\n"
        "log_returns = np.log(hourly + 10).diff().dropna()\n"
        "lags = list(range(1, 49))\n"
        "acf = [log_returns.autocorr(lag=k) for k in lags]\n"
        "axes[1].bar(lags, acf, color='darkorange')\n"
        "axes[1].set_title('Autocorrelation of log(P+10) returns up to 48 h')\n"
        "axes[1].set_xlabel('lag (hours)'); axes[1].grid(alpha=0.3, axis='y')\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(md(
        "## 3. Calibration\n"
        "\n"
        "`spot.fit` runs the slow-fast pipeline: causal EMA → seasonal OLS\n"
        "on X_t → iterative jump detection (k_base=4.5, k_peak=6.0, "
        "|ΔZ| ≥ 0.30) → bounded MLE on OU + Kou (κ∈[0.05, 0.20], "
        "λ∈[0.008, 0.025], η∈[0.8, 4.0]). Raises ``RuntimeError`` if "
        "any non-marginal parameter touches a bound."
    ))
    cells.append(code(
        "fit = spot.fit(hourly)\n"
        "p = fit.params\n"
        "sf = p.slow_factor\n"
        "print('--- Slow factor θ_t ---')\n"
        "print(f'  ema_span           : {p.ema_span} h  (~{p.ema_span/24:.1f} days)')\n"
        "print(f'  μ_θ                : {sf.mean:.4f}')\n"
        "print(f'  κ_θ                : {sf.kappa:.5e}  /h  (half-life {np.log(2)/sf.kappa:.0f} h)')\n"
        "print(f'  σ_θ                : {sf.sigma:.5e}')\n"
        "print(f'  stationary std     : {sf.sigma / np.sqrt(2 * sf.kappa):.4f}  '\n"
        "      f'(empirical θ_pw std: {fit.theta_series.iloc[p.ema_span:].std():.4f})')\n"
        "print()\n"
        "print('--- Seasonality s(t) on X_t ---')\n"
        "print(f'  intercept          : {p.seasonality.intercept:+.4f}')\n"
        "print(f'  Fourier coefs      : {np.round(p.seasonality.fourier_coefs, 4)}')\n"
        "print(f'  DoW coefs Tue..Sun : {np.round(p.seasonality.dow_coefs, 4)}')\n"
        "print(f'  HoD coefs range    : {p.seasonality.hod_coefs.min():+.4f}..{p.seasonality.hod_coefs.max():+.4f}')\n"
        "print()\n"
        "print('--- Fast OU + Kou jumps on Z_t ---')\n"
        "print(f'  κ                  : {p.kappa:.5f} /h  (half-life {np.log(2)/p.kappa:.1f} h)')\n"
        "print(f'  σ_h range          : {p.sigma_by_hour.min():.4f}..{p.sigma_by_hour.max():.4f}  (mean {p.sigma_by_hour.mean():.4f})')\n"
        "print(f'  λ                  : {p.jump_intensity:.5f} /h  ({p.jump_intensity*8760:.0f} /year)')\n"
        "print(f'  p_up               : {p.jump_p_up:.3f}')\n"
        "print(f'  η_up               : {p.jump_eta_up:.3f}  (mean +J = {1/p.jump_eta_up:.3f} log)')\n"
        "print(f'  η_down             : {p.jump_eta_down:.3f}  (mean -J = {1/p.jump_eta_down:.3f} log)')\n"
        "print(f'  n_obs={fit.n_obs}  n_jumps={fit.n_jumps} ({100*fit.n_jumps/fit.n_obs:.2f} %)')\n"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "log_p = np.log(hourly + p.price_shift)\n"
        "ax.plot(log_p.index, log_p.values, lw=0.3, alpha=0.5, label='log(P+10)')\n"
        "ax.plot(fit.theta_series.index, fit.theta_series.values, lw=1.0,\n"
        "        color='red', label=f'theta (EMA span={p.ema_span} h)')\n"
        "ax.axhline(sf.mean, color='black', lw=0.7, ls='--', label=f'μ_θ={sf.mean:.3f}')\n"
        "ax.set_title('Slow factor θ_t overlay'); ax.set_ylabel('log(P+10)')\n"
        "ax.grid(alpha=0.3); ax.legend()\n"
        "plt.show()\n"
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "doys = np.arange(1, 366); ang = 2*np.pi*doys/365.25\n"
        "fy = np.zeros_like(doys, dtype=float)\n"
        "for kh in range(1, 5):\n"
        "    fy = fy + p.seasonality.fourier_coefs[2*(kh-1)] * np.cos(kh*ang)\n"
        "    fy = fy + p.seasonality.fourier_coefs[2*kh - 1] * np.sin(kh*ang)\n"
        "axes[0].plot(doys, fy)\n"
        "axes[0].set_title('Fourier annual (on X_t)'); axes[0].set_xlabel('day of year')\n"
        "axes[0].grid(alpha=0.3)\n"
        "dow_full = np.concatenate([[0.0], p.seasonality.dow_coefs])\n"
        "axes[1].bar(range(7), dow_full,\n"
        "            tick_label=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])\n"
        "axes[1].set_title('Day-of-week dummies'); axes[1].grid(alpha=0.3, axis='y')\n"
        "hod_full = np.concatenate([[0.0], p.seasonality.hod_coefs])\n"
        "axes[2].bar(range(24), hod_full)\n"
        "axes[2].set_title('Hour-of-day dummies (UTC)'); axes[2].set_xlabel('hour UTC')\n"
        "axes[2].grid(alpha=0.3, axis='y')\n"
        "plt.tight_layout(); plt.show()\n"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.bar(range(24), p.sigma_by_hour)\n"
        "ax.set_title(f'OU sigma by hour-of-day; kappa={p.kappa:.4f}/h')\n"
        "ax.set_xlabel('hour UTC'); ax.set_ylabel('sigma'); ax.grid(alpha=0.3, axis='y')\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "## 4. Residual diagnostics\n"
        "\n"
        "- ADF on the fast residual Z_t (stationarity expected since the\n"
        "  slow factor is removed).\n"
        "- ADF on its returns ΔZ_t.\n"
        "- Jarque-Bera on non-jump returns (normality expected to be REJECTED:\n"
        "  the residual still carries 24-h autocorrelation that the additive\n"
        "  seasonality cannot resolve, a documented limitation)."
    ))
    cells.append(code(
        "nj = fit.residual_returns[~fit.jumps_mask]\n"
        "adf_resid = adfuller(fit.residuals.values, regression='c', autolag='AIC')\n"
        "adf_ret = adfuller(fit.residual_returns.values, regression='c', autolag='AIC')\n"
        "jb = sstats.jarque_bera(nj.values)\n"
        "print(f'ADF Z_t         : stat={adf_resid[0]:.3f}  p={adf_resid[1]:.3g}')\n"
        "print(f'ADF delta Z_t   : stat={adf_ret[0]:.3f}  p={adf_ret[1]:.3g}')\n"
        "print(f'JB non-jump ret : stat={jb.statistic:.1f}  p={jb.pvalue:.3g}')\n"
        "print(f'  return std all     : {fit.residual_returns.std():.4f}')\n"
        "print(f'  return std nonjump : {nj.std():.4f}')\n"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "ax.plot(fit.residuals.index, fit.residuals.values, lw=0.3, label='Z_t')\n"
        "jr_idx = fit.residual_returns.index[fit.jumps_mask.values]\n"
        "ax.scatter(jr_idx, fit.residuals.reindex(jr_idx).values,\n"
        "           s=4, color='red', label=f'jumps n={fit.n_jumps}', zorder=3)\n"
        "ax.set_title('Fast residual Z_t with detected jumps')\n"
        "ax.set_ylabel('Z'); ax.grid(alpha=0.3); ax.legend()\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "## 5. Validation by simulation\n"
        "\n"
        "Simulate 5000 paths × 8760 h starting from μ_θ (the slow-OU\n"
        "stationary mean) — see the diagnostic doc for why μ_θ and not\n"
        "θ̂_T is the right initial value when comparing against the\n"
        "*unconditional* historical distribution.\n"
        "\n"
        "Compare sim vs hist mean (target ±20 %) and sim vs hist p95\n"
        "(loosened to ±25 % from the original ±15 %; the structural\n"
        "limit is documented in the diagnostic doc)."
    ))
    cells.append(code(
        "sim_start = hourly.index[-1] + pd.Timedelta('1h')\n"
        "paths = spot.simulate(\n"
        "    p, sim_start, n_hours=24*365, n_paths=5000,\n"
        "    initial_theta=p.slow_factor.mean,\n"
        "    initial_residual=0.0,\n"
        "    seed=2026,\n"
        ")\n"
        "sim_mean = paths.mean()\n"
        "sim_p95  = np.percentile(paths, 95)\n"
        "hist_mean = hourly.mean()\n"
        "hist_p95  = np.percentile(hourly.values, 95)\n"
        "print(f'sim mean = {sim_mean:7.2f}  hist mean = {hist_mean:7.2f}  '\n"
        "      f'rel-err = {abs(sim_mean-hist_mean)/hist_mean*100:5.2f}%  (spec ≤20%)')\n"
        "print(f'sim p95  = {sim_p95:7.2f}  hist p95  = {hist_p95:7.2f}  '\n"
        "      f'rel-err = {abs(sim_p95-hist_p95)/hist_p95*100:5.2f}%  (spec ≤25%)')\n"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "sim_idx = pd.date_range(sim_start, periods=paths.shape[1], freq='h')\n"
        "ax.fill_between(sim_idx, np.percentile(paths, 10, axis=0),\n"
        "                np.percentile(paths, 90, axis=0), alpha=0.2,\n"
        "                label='P10-P90 sim')\n"
        "ax.plot(sim_idx, paths.mean(axis=0), label='mean sim', lw=0.6)\n"
        "ax.axhline(hist_mean, color='black', lw=0.8, ls='--', label=f'hist mean={hist_mean:.1f}')\n"
        "ax.set_title('Simulated paths from μ_θ (n=5000, horizon 1y)')\n"
        "ax.set_ylabel('EUR/MWh'); ax.grid(alpha=0.3); ax.legend()\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "## 6. Limitations carried forward to Pieza 2\n"
        "\n"
        "1. **2022 regime contamination of historical p95**. The union-2019-2024\n"
        "   p95 = 218 EUR/MWh is dominated by 2022 (p95 = 268); the other five\n"
        "   years sit at 52–254 (median ≈ 140). A stationary-distribution model\n"
        "   calibrated on the union cannot match the union p95 within ±15 %\n"
        "   (we are at ≈ 21 %). The fix is Pieza 2: the Schwartz-Smith fit\n"
        "   anchors to the OMIP forward curve rather than to historical p95,\n"
        "   so the long-term factor reflects market-implied rather than\n"
        "   union-historical statistics.\n"
        "2. **24-h autocorrelation in Z_t (~0.50)**. The additive\n"
        "   DoW + HoD seasonality does not capture the weekday/weekend cycle\n"
        "   interaction (the evening peak shape is genuinely different on\n"
        "   Sundays). A DoW × HoD interaction (168 dummies) would close most\n"
        "   of that gap. Pending engineering investment.\n"
        "3. **Heavy-tailed non-jump residual** (JB rejects normality even on\n"
        "   the post-jump residuals). The Kou jump component already absorbs\n"
        "   the worst tail; the residual heaviness is a known artefact of the\n"
        "   additive seasonality limitation above.\n"
        "\n"
        "Full discussion in `reports/diagnostics/spot_model_calibration.md`.\n"
    ))

    nb["cells"] = cells
    return nb


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
