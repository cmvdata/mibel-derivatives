"""
descriptive_premium.py — descriptive + model-free forward risk premium for the
Iberian power forward-premia paper (2019-2024). INDEPENDENT of the estado_arte
published papers; reuses ONLY the curated OMIP forward + OMIE spot data in this
repo (data complete, 0 gaps; see mibel_datasets_2019_2024/INFORME_RESUMEN_COBERTURA.md).

Headline (model-free, no model risk): the ex-post front-month forward premium
F_quoted / realized_delivery_spot - 1 roughly triples through the 2021-2023 gas
crisis and stays elevated in the recovery — the Bessembinder-Lemmon (2002)
hedging-pressure signature, identified WITHOUT the Schwartz-Smith model.

The structural counterpart is lambda_chi (short-term risk premium) from the
production Schwartz-Smith fit (+0.444/y); regime-split structural fits are a
pod-run extension (see reports/pod_runbook.md).

Run: python paper_derivatives/descriptive_premium.py
Outputs: paper_derivatives/
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OMIP = ROOT / "mibel_datasets_2019_2024" / "01_omip" / "processed" / "omip_forward_2019_2024.parquet"
OMIE = ROOT / "mibel_datasets_2019_2024" / "03_omie_esios_600" / "omie_spot_es_2019_2024.parquet"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Regime boundaries: gas crisis ramp (TTF surge autumn-2021) -> normalisation (2023 H2).
CRISIS_START = pd.Timestamp("2021-09-01")
CRISIS_END = pd.Timestamp("2023-07-01")
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def regime(d):
    d = pd.Timestamp(d)
    if d < CRISIS_START:
        return "1 pre-crisis (2019-01..2021-08)"
    if d < CRISIS_END:
        return "2 crisis (2021-09..2023-06)"
    return "3 recovery (2023-07..2024-12)"


def _delivery_period(contract):
    m = re.search(r"([A-Z][a-z]{2})-(\d{2})", str(contract))
    if not m:
        return None
    return pd.Period(year=2000 + int(m.group(2)), month=MONTHS[m.group(1)], freq="M")


def main():
    omip = pd.read_parquet(OMIP)
    omip["trade_date"] = pd.to_datetime(omip["trade_date"])
    omip["F"] = omip["reference_d_minus_1_eur_mwh"].fillna(omip["reference_d_eur_mwh"])
    omip = omip.dropna(subset=["F"])
    omip = omip[omip["F"] > 0]

    omie = pd.read_parquet(OMIE)
    omie["datetime_utc"] = pd.to_datetime(omie["datetime_utc"], utc=True)
    omie["d"] = omie["datetime_utc"].dt.tz_convert(None).dt.normalize()
    spot_daily = omie.groupby("d")["price_eur_mwh"].mean()
    spot_monthly = spot_daily.groupby(spot_daily.index.to_period("M")).mean()

    lines = []
    def p(*a):
        lines.append(" ".join(str(x) for x in a))

    p("=" * 74)
    p("DESCRIPTIVE — Iberian power forwards & spot, 2019-2024")
    p("=" * 74)
    p(f"OMIP: {omip.trade_date.min().date()} -> {omip.trade_date.max().date()}, "
      f"{len(omip):,} quotes (F>0); buckets {omip.maturity_bucket.value_counts().to_dict()}")
    p(f"OMIE spot daily: {spot_daily.index.min().date()} -> {spot_daily.index.max().date()}, "
      f"{len(spot_daily):,} days")
    p("")

    sd = spot_daily.to_frame("spot")
    sd["reg"] = sd.index.map(regime)
    p("--- OMIE daily-mean spot by regime (EUR/MWh) ---")
    p(sd.groupby("reg")["spot"].agg(["count", "mean", "std", "min", "max"]).round(1).to_string())
    p("")

    # --- model-free ex-post front-month forward premium ---
    M = omip[omip["maturity_bucket"] == "M"].copy()
    M["dp"] = M["contract"].map(_delivery_period)
    M = M.dropna(subset=["dp"])
    M["lead"] = (M["dp"].astype("datetime64[ns]") - M["trade_date"]).dt.days
    M = M[M["lead"] > 0]
    front = M.sort_values("lead").groupby("trade_date").first().reset_index()
    front["realized"] = front["dp"].map(lambda q: spot_monthly.get(q, np.nan))
    front = front.dropna(subset=["realized"])
    front["prem_pct"] = 100 * (front["F"] / front["realized"] - 1)
    front["reg"] = front["trade_date"].map(regime)

    tab = front.groupby("reg")["prem_pct"].agg(["count", "mean", "median", "std"]).round(1)
    p("--- EX-POST front-month forward premium (model-free) ---")
    p("premium% = 100*(F_quoted / realized_delivery_spot - 1); + = buyers pay above realized")
    p(f"(mean front-contract lead = {front['lead'].mean():.0f} days)")
    p(tab.to_string())
    p(f"pooled mean {front.prem_pct.mean():.1f}%  median {front.prem_pct.median():.1f}%  n={len(front)}")
    p("")
    p("Structural counterpart (production Schwartz-Smith fit, full sample):")
    p("  lambda_chi = +0.444 /y (short-term risk premium); rmse_log_YR 0.043, rmse_log_M 0.18.")
    p("  Regime-split structural fits: pod-run extension (reports/pod_runbook.md).")

    txt = "\n".join(lines)
    (OUT / "descriptive_premium.txt").write_text(txt, encoding="utf-8")
    tab.to_csv(OUT / "forward_premium_by_regime.csv")
    front[["trade_date", "contract", "dp", "lead", "F", "realized", "prem_pct", "reg"]] \
        .to_csv(OUT / "front_month_premium_panel.csv", index=False)
    print(txt)

    # --- Figure: 6-month rolling mean of front-month premium + regime shading ---
    fp = front.sort_values("trade_date").set_index("trade_date")
    roll = fp["prem_pct"].rolling(60, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0, color="grey", lw=.8, ls="--")
    ax.axvspan(CRISIS_START, CRISIS_END, color="#f2c14e", alpha=.25, label="gas crisis")
    ax.plot(roll.index, roll.values, color="#1f4e79", lw=1.6)
    ax.scatter(fp.index, fp["prem_pct"], s=4, color="#9bb8d3", alpha=.35, zorder=0)
    ax.set_ylabel("Front-month forward premium (%)")
    ax.set_xlabel("Trade date")
    ax.set_title("Ex-post forward risk premium, Iberian power (60-day rolling mean)")
    ax.set_ylim(-60, 80)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_forward_premium.pdf")
    print(f"\n[fig] {OUT/'fig_forward_premium.pdf'}")


if __name__ == "__main__":
    main()
