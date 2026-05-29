"""Production calibration of the Schwartz-Smith forward model.

Fits Pieza 2 on the FULL 1565-day OMIP forward series + daily-mean
OMIE spot. Serialises the resulting :class:`forward.SSFit` to
``data/curated/forward_fit_production.pkl`` and writes a markdown
summary to ``reports/diagnostics/production_fit_summary.md``.

This script is intended for a cloud pod (see
``reports/pod_runbook.md``); on the user's laptop the wall time is
prohibitive (~3 hours). It is idempotent: if the output pickle exists
the script aborts unless ``--force`` is passed.

Usage:

    python scripts/run_production_fit.py            # run, refuse to overwrite
    python scripts/run_production_fit.py --force    # overwrite the pickle
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mibel_derivatives.models import forward


OMIP_PATH = Path("data/curated/omip_forward_2019_2024.parquet")
OMIE_PATH = Path("data/curated/omie_spot_es_2019_2024.parquet")
OUTPUT_PKL = Path("data/curated/forward_fit_production.pkl")
SUMMARY_PATH = Path("reports/diagnostics/production_fit_summary.md")

# Logging: tee to stderr + file so the pod log can be reviewed off-line.
LOG_PATH = Path("data/_production_fit.log")


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-5s %(name)s %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    if not OMIP_PATH.exists():
        raise FileNotFoundError(
            f"missing input: {OMIP_PATH} — copy the curated parquet "
            "from the laptop to the pod (see reports/pod_runbook.md § 3)"
        )
    if not OMIE_PATH.exists():
        raise FileNotFoundError(
            f"missing input: {OMIE_PATH} — copy the curated parquet "
            "from the laptop to the pod (see reports/pod_runbook.md § 3)"
        )
    omip = pd.read_parquet(OMIP_PATH)
    omie_raw = pd.read_parquet(OMIE_PATH)
    omie_raw = omie_raw.set_index("datetime_utc").sort_index()
    hourly = omie_raw[omie_raw.index.minute == 0]["price_eur_mwh"]
    daily_mean = (
        hourly.tz_convert(None)
        .groupby(hourly.tz_convert(None).index.normalize())
        .mean()
        .rename("price")
    )
    daily_mean.index.name = "trade_date"
    return omip, daily_mean


def _summarise(fit: forward.SSFit, wall_seconds: float) -> str:
    p = fit.params
    lines = [
        "# Production Schwartz-Smith fit — summary",
        "",
        f"- **Run wall**: {wall_seconds:.0f} s ({wall_seconds/60:.1f} min)",
        f"- **n_obs**: {fit.n_obs}",
        f"- **n_dates**: {fit.n_dates}",
        f"- **log_likelihood**: {fit.log_likelihood:.2f}",
        f"- **rmse_log_M**: {fit.rmse_log_m:.4f}",
        f"- **rmse_log_YR**: {fit.rmse_log_yr:.4f}",
        "",
        "## Stochastic factors",
        "",
        f"| Param | Estimate | Bound | At bound? |",
        f"|---|---|---|---|",
    ]
    bounds_table = [
        ("kappa", p.kappa, forward.KAPPA_BOUNDS),
        ("sigma_chi", p.sigma_chi, forward.SIGMA_CHI_BOUNDS),
        ("sigma_xi", p.sigma_xi, forward.SIGMA_XI_BOUNDS),
        ("rho", p.rho, forward.RHO_BOUNDS),
        ("mu_xi (physical)", p.mu_xi, forward.MU_XI_BOUNDS),
        ("mu_xi_star (Q)", p.mu_xi_star, forward.MU_XI_STAR_BOUNDS),
        ("lambda_chi", p.lambda_chi, forward.LAMBDA_CHI_BOUNDS),
        ("epsilon_m", p.epsilon_m, forward.EPSILON_BOUNDS),
        ("epsilon_yr", p.epsilon_yr, forward.EPSILON_BOUNDS),
        ("epsilon_spot", p.epsilon_spot, forward.EPSILON_SPOT_BOUNDS),
    ]
    tol = 1e-3
    for name, val, (lo, hi) in bounds_table:
        at_lo = abs(val - lo) < tol
        at_hi = abs(val - hi) < tol
        mark = "yes (lower)" if at_lo else ("yes (upper)" if at_hi else "no")
        lines.append(f"| {name} | {val:+.4f} | [{lo}, {hi}] | {mark} |")
    lines.extend([
        "",
        "## Seasonal dummies (Feb..Dec, January reference)",
        "",
        "| Month | s |",
        "|---|---|",
    ])
    for i, m in enumerate(("Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")):
        lines.append(f"| {m} | {p.seasonal_dummies[i]:+.4f} |")
    lines.extend([
        "",
        "## Filtered state at end of sample",
        "",
        f"- last trade date: `{fit.trade_dates[-1]}`",
        f"- chi_T = {float(fit.state_chi.iloc[-1]):+.4f}",
        f"- xi_T  = {float(fit.state_xi.iloc[-1]):+.4f}",
        "",
        "## Next steps",
        "",
        "1. Copy `data/curated/forward_fit_production.pkl` back to the laptop.",
        "2. Run `pytest tests/models/test_forward_validation.py -v` locally;",
        "   V5 (composition spot MAE) and V6 (composition forward) will now",
        "   execute against this fit (the `requires_full_fit` marker no",
        "   longer skips).",
        "3. Review the per-bucket RMSE above; if M-bucket RMSE exceeds 0.25",
        "   the calibration probably needs a richer model (see Limitations §",
        "   in forward_model_calibration.md).",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing forward_fit_production.pkl",
    )
    parser.add_argument(
        "--max-iter", type=int, default=300,
        help="L-BFGS-B max iterations (default 300)",
    )
    args = parser.parse_args(argv)
    _setup_logging()
    log = logging.getLogger("production_fit")

    if OUTPUT_PKL.exists() and not args.force:
        log.error("output pickle already exists: %s. Pass --force to overwrite.",
                  OUTPUT_PKL)
        return 2

    log.info("loading OMIP forward + OMIE daily-mean spot ...")
    omip, daily_mean = _load_inputs()
    n_dates = pd.to_datetime(omip["trade_date"]).nunique()
    log.info("OMIP rows: %d, trade_dates: %d, OMIE daily mean rows: %d",
             len(omip), n_dates, len(daily_mean))

    log.info("running forward.fit on the FULL series (no sub-sampling) "
             "with max_iter=%d", args.max_iter)
    t0 = time.time()
    fit = forward.fit(omip, daily_mean, max_iter=args.max_iter)
    wall = time.time() - t0
    log.info("fit wall: %.1f s (%.2f min)", wall, wall / 60.0)
    log.info("n_obs=%d  n_dates=%d  loglik=%.2f  rmse_M=%.4f  rmse_YR=%.4f",
             fit.n_obs, fit.n_dates, fit.log_likelihood,
             fit.rmse_log_m, fit.rmse_log_yr)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PKL.open("wb") as fh:
        pickle.dump(fit, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("serialised SSFit → %s (%d bytes)",
             OUTPUT_PKL, OUTPUT_PKL.stat().st_size)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(_summarise(fit, wall), encoding="utf-8")
    log.info("wrote summary → %s", SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
