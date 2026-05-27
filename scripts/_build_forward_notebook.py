"""Build notebooks/02_forward_model.ipynb from cell sources defined here.

Mirror of scripts/_build_spot_notebook.py but for Pieza 2 (Schwartz-
Smith forward-curve model). The notebook is committed with empty
outputs; running it on the curated OMIP + OMIE parquets reproduces
the calibration and the diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/02_forward_model.ipynb")


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
        "# Schwartz-Smith forward-curve calibration on OMIP\n"
        "\n"
        "Phase 2 / Pieza 2 of the mibel-derivatives module.\n"
        "\n"
        "Two-factor forward model on log F(t, T) with deterministic\n"
        "delivery-month seasonality, anchored by daily-mean OMIE spot:\n"
        "\n"
        "    ln(S_t) = chi_t + xi_t + s(month(t))\n"
        "    ln F(t, T) = e^{-kappa·tau} chi_t + xi_t + A(tau) + s_delivery(T)\n"
        "\n"
        "Spec and decisions in `src/mibel_derivatives/models/forward.py`;\n"
        "diagnostic in `reports/diagnostics/forward_model_calibration.md`.\n"
        "\n"
        "**Wall-time note**: a full Kalman + L-BFGS-B fit on a weekly\n"
        "sub-sample (~200 trade dates × ~15 contracts) takes ~15 min.\n"
        "The notebook uses the 100-most-recent trade dates as default\n"
        "to keep the EDA interactive (~7 min)."
    ))

    cells.append(code(
        "from __future__ import annotations\n"
        "\n"
        "import time\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from mibel_derivatives.models import forward\n"
        "\n"
        "pd.set_option('display.float_format', '{:.4f}'.format)\n"
        "plt.rcParams['figure.dpi'] = 110\n"
    ))

    cells.append(md(
        "## 1. Load OMIP forward + OMIE daily-mean spot\n"
        "\n"
        "OMIP contract layout: 9 380 M rows (FTB M Mmm-YY) + 14 468 YR rows\n"
        "(FTB YR-YY) over 1 565 trade dates 2019-2024. The fit subsamples\n"
        "to the 100 most-recent trade dates for interactive use."
    ))
    cells.append(code(
        "omip = pd.read_parquet('data/curated/omip_forward_2019_2024.parquet')\n"
        "omip['trade_date_dt'] = pd.to_datetime(omip['trade_date'])\n"
        "last_dates = sorted(omip['trade_date_dt'].unique())[-100:]\n"
        "omip_sub = (omip[omip['trade_date_dt'].isin(last_dates)]\n"
        "            .drop(columns=['trade_date_dt']).reset_index(drop=True))\n"
        "print('OMIP rows:', len(omip_sub), '  trade_dates:', omip_sub['trade_date'].nunique())\n"
        "print('  M rows:', (omip_sub['maturity_bucket']=='M').sum(),\n"
        "      '  YR rows:', (omip_sub['maturity_bucket']=='YR').sum())\n"
    ))
    cells.append(code(
        "omie = pd.read_parquet('data/curated/omie_spot_es_2019_2024.parquet')\n"
        "omie = omie.set_index('datetime_utc').sort_index()\n"
        "hourly = omie[omie.index.minute == 0]['price_eur_mwh']\n"
        "daily = (hourly.tz_convert(None)\n"
        "         .groupby(hourly.tz_convert(None).index.normalize())\n"
        "         .mean().rename('price'))\n"
        "daily.index.name = 'trade_date'\n"
        "print('OMIE daily mean:', len(daily), 'days')\n"
    ))

    cells.append(md(
        "## 2. Calibration\n"
        "\n"
        "`forward.fit(omip, omie_daily_mean)`:\n"
        "  1. `prepare_observations` reshapes wide OMIP → long format.\n"
        "  2. `_prepare_kalman_arrays` precomputes per-date numpy arrays.\n"
        "  3. scipy L-BFGS-B over (kappa, sigma_chi, sigma_xi, rho, mu_xi,\n"
        "     mu_xi*, lambda_chi, eps_M, eps_YR, 11 seasonal dummies).\n"
        "\n"
        "Bounds: kappa ∈ [0.1, 5.0]/y, sigma_chi ∈ [0.05, 5.0],\n"
        "sigma_xi ∈ [0.05, 2.0], rho ∈ (-0.99, 0.99),\n"
        "mu_xi / mu_xi* / lambda_chi ∈ [-0.50, 0.50]/y, eps ∈ [1e-4, 0.50]."
    ))
    cells.append(code(
        "t0 = time.time()\n"
        "fit = forward.fit(omip_sub, daily, max_iter=100)\n"
        "wall = time.time() - t0\n"
        "p = fit.params\n"
        "print(f'fit wall: {wall:.1f}s  (n_obs={fit.n_obs}, n_dates={fit.n_dates})')\n"
        "print()\n"
        "print('--- Stochastic factors ---')\n"
        "print(f'  kappa            : {p.kappa:.4f} /y  (half-life {np.log(2)/p.kappa:.2f} y)')\n"
        "print(f'  sigma_chi        : {p.sigma_chi:.4f} / sqrt(y)')\n"
        "print(f'  sigma_xi         : {p.sigma_xi:.4f} / sqrt(y)')\n"
        "print(f'  rho              : {p.rho:.4f}')\n"
        "print()\n"
        "print('--- Drift / risk premia ---')\n"
        "print(f'  mu_xi (physical) : {p.mu_xi:+.4f} /y')\n"
        "print(f'  mu_xi*  (Q meas) : {p.mu_xi_star:+.4f} /y')\n"
        "print(f'  lambda_chi       : {p.lambda_chi:+.4f}')\n"
        "print(f'  lambda_xi impl   : {(p.mu_xi - p.mu_xi_star):+.4f}')\n"
        "print()\n"
        "print('--- Measurement noise + diagnostics ---')\n"
        "print(f'  epsilon_M        : {p.epsilon_m:.4f}')\n"
        "print(f'  epsilon_YR       : {p.epsilon_yr:.4f}')\n"
        "print(f'  rmse_log_M       : {fit.rmse_log_m:.4f}')\n"
        "print(f'  rmse_log_YR      : {fit.rmse_log_yr:.4f}')\n"
        "print(f'  log_likelihood   : {fit.log_likelihood:.1f}')\n"
    ))

    cells.append(md(
        "## 3. Filtered state trajectories (chi_t, xi_t)\n"
        "\n"
        "Plot the latent Schwartz-Smith state factors over the sample\n"
        "window. The chi component should mean-revert; xi should drift\n"
        "with mu_xi."
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(fit.state_chi.index, fit.state_chi.values, color='steelblue')\n"
        "axes[0].set_title(f'chi_t (short-term, kappa={p.kappa:.2f})')\n"
        "axes[0].set_xlabel('trade date'); axes[0].grid(alpha=0.3)\n"
        "axes[1].plot(fit.state_xi.index, fit.state_xi.values, color='darkorange')\n"
        "axes[1].set_title(f'xi_t (long-term, mu_xi={p.mu_xi:+.3f})')\n"
        "axes[1].set_xlabel('trade date'); axes[1].grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.bar(range(1, 12), p.seasonal_dummies, color='teal',\n"
        "       tick_label=['Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])\n"
        "ax.set_title('Delivery-month seasonal dummies (January reference)')\n"
        "ax.set_ylabel('log offset'); ax.grid(alpha=0.3, axis='y')\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "## 4. Forward-curve fit quality\n"
        "\n"
        "Per-bucket scatter of observed vs predicted log F."
    ))
    cells.append(code(
        "obs = forward.prepare_observations(omip_sub, daily)\n"
        "obs = obs.merge(\n"
        "    fit.state_chi.rename('chi'), left_on='trade_date', right_index=True,\n"
        ")\n"
        "obs = obs.merge(\n"
        "    fit.state_xi.rename('xi'), left_on='trade_date', right_index=True,\n"
        ")\n"
        "obs['model_log_F'] = obs.apply(lambda r: forward.futures_log_price(\n"
        "    p, float(r['chi']), float(r['xi']), float(r['tau']),\n"
        "    delivery_month=(int(r['delivery_month']) if not r['is_yearly'] else None),\n"
        "    is_yearly=bool(r['is_yearly']),\n"
        "), axis=1)\n"
        "obs['resid'] = obs['log_F'] - obs['model_log_F']\n"
        "for bucket in ('M', 'YR'):\n"
        "    bk = obs[obs['bucket'] == bucket]\n"
        "    print(f'{bucket}: n={len(bk)}  mean(|resid|)={bk[\"resid\"].abs().mean():.4f}  '\n"
        "          f'rmse={np.sqrt((bk[\"resid\"]**2).mean()):.4f}')\n"
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "for ax, bucket in zip(axes, ('M', 'YR')):\n"
        "    bk = obs[obs['bucket'] == bucket]\n"
        "    ax.scatter(bk['log_F'], bk['model_log_F'], s=4, alpha=0.4)\n"
        "    lo, hi = bk['log_F'].min(), bk['log_F'].max()\n"
        "    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)\n"
        "    ax.set_title(f'{bucket} bucket: observed vs model log F')\n"
        "    ax.set_xlabel('observed'); ax.set_ylabel('model')\n"
        "    ax.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(md(
        "## 5. Forward state-path simulation\n"
        "\n"
        "Simulate (chi, xi) paths for 1 year forward from end of sample\n"
        "under the physical measure."
    ))
    cells.append(code(
        "chi0 = float(fit.state_chi.iloc[-1])\n"
        "xi0 = float(fit.state_xi.iloc[-1])\n"
        "chi_paths, xi_paths = forward.simulate(\n"
        "    p, initial_chi=chi0, initial_xi=xi0,\n"
        "    start=fit.trade_dates[-1] + pd.Timedelta('1D'),\n"
        "    n_days=365, n_paths=1000, seed=2026,\n"
        ")\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "for arr, ax, name in ((chi_paths, axes[0], 'chi'), (xi_paths, axes[1], 'xi')):\n"
        "    ax.fill_between(range(arr.shape[1]),\n"
        "                    np.percentile(arr, 10, axis=0),\n"
        "                    np.percentile(arr, 90, axis=0), alpha=0.2)\n"
        "    ax.plot(arr.mean(axis=0), lw=0.6)\n"
        "    ax.set_title(f'{name} paths (P10/P50/P90, n=1000)')\n"
        "    ax.set_xlabel('days forward'); ax.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(md(
        "## 6. Pieza 1 / Pieza 2 integration: spot.fit_with_forward_anchor\n"
        "\n"
        "When Pieza 2 is fit, Pieza 1's slow factor theta_t can be\n"
        "replaced by (chi_t + xi_t + s_delivery(t)) from the SSFit.\n"
        "Pieza 1 then only fits the intraday seasonality + fast OU+Kou\n"
        "residual on top — the slow factor inherits OMIP-forward\n"
        "consistency for free.\n"
        "\n"
        "    from mibel_derivatives.models import spot\n"
        "    omie_hourly = (omie[omie.index.minute == 0]['price_eur_mwh']\n"
        "                   .rename('price_eur_mwh'))\n"
        "    omie_hourly.index.name = 'dt_utc'\n"
        "    spot_fit_anchored = spot.fit_with_forward_anchor(omie_hourly, fit)\n"
        "\n"
        "(Heavy — not run automatically; see the diagnostic for results)."
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
