"""
inference_premium.py — inference on the forward risk premium (referee points 1-2):
  (1) regime means with 95% CI + a test that crisis != pre-crisis, using SE
      CLUSTERED by delivery month (overlapping contracts on the same delivery
      are not independent), cross-checked with a block bootstrap over delivery
      months.
  (2) TERM STRUCTURE: does the premium also appear at the YEAR-AHEAD (YR) horizon,
      not just front-month.

Run: python paper_derivatives/inference_premium.py
Outputs: paper_derivatives/inference_premium.txt + premium_inference.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OMIP = ROOT / "mibel_datasets_2019_2024" / "01_omip" / "processed" / "omip_forward_2019_2024.parquet"
OMIE = ROOT / "mibel_datasets_2019_2024" / "03_omie_esios_600" / "omie_spot_es_2019_2024.parquet"
OUT = Path(__file__).resolve().parent

CRISIS_START = pd.Timestamp("2021-09-01")
CRISIS_END = pd.Timestamp("2023-07-01")
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def regime(d):
    d = pd.Timestamp(d)
    if d < CRISIS_START:
        return "pre"
    if d < CRISIS_END:
        return "crisis"
    return "recovery"


def _load():
    omip = pd.read_parquet(OMIP)
    omip["trade_date"] = pd.to_datetime(omip["trade_date"])
    omip["F"] = omip["reference_d_minus_1_eur_mwh"].fillna(omip["reference_d_eur_mwh"])
    omip = omip.dropna(subset=["F"])
    omip = omip[omip["F"] > 0]
    omie = pd.read_parquet(OMIE)
    omie["datetime_utc"] = pd.to_datetime(omie["datetime_utc"], utc=True)
    omie["d"] = omie["datetime_utc"].dt.tz_convert(None).dt.normalize()
    spot_daily = omie.groupby("d")["price_eur_mwh"].mean()
    return omip, spot_daily


def _front_panel(omip, spot_daily):
    spot_monthly = spot_daily.groupby(spot_daily.index.to_period("M")).mean()
    M = omip[omip["maturity_bucket"] == "M"].copy()

    def dp(c):
        m = re.search(r"([A-Z][a-z]{2})-(\d{2})", str(c))
        return pd.Period(year=2000 + int(m.group(2)), month=MONTHS[m.group(1)], freq="M") if m else None
    M["dp"] = M["contract"].map(dp)
    M = M.dropna(subset=["dp"])
    M["lead"] = (M["dp"].astype("datetime64[ns]") - M["trade_date"]).dt.days
    M = M[M["lead"] > 0]
    front = M.sort_values("lead").groupby("trade_date").first().reset_index()
    front["realized"] = front["dp"].map(lambda q: spot_monthly.get(q, np.nan))
    front = front.dropna(subset=["realized"])
    front["prem"] = 100 * (front["F"] / front["realized"] - 1)
    front["regime"] = front["trade_date"].map(regime)
    front["dp_str"] = front["dp"].astype(str)
    return front


def _yr_panel(omip, spot_daily):
    spot_yearly = spot_daily.groupby(spot_daily.index.year).mean()
    Y = omip[omip["maturity_bucket"] == "YR"].copy()

    def yr(c):
        m = re.search(r"YR-(\d{2})", str(c))
        return 2000 + int(m.group(1)) if m else None
    Y["dyear"] = Y["contract"].map(yr)
    Y = Y.dropna(subset=["dyear"])
    Y["dyear"] = Y["dyear"].astype(int)
    Y = Y[Y["dyear"] <= 2024]                      # only deliveries we can realise
    Y["lead"] = (pd.to_datetime(Y["dyear"].astype(str) + "-07-01") - Y["trade_date"]).dt.days
    Y = Y[Y["lead"] > 0]
    Y["realized"] = Y["dyear"].map(lambda y: spot_yearly.get(y, np.nan))
    Y = Y.dropna(subset=["realized"])
    Y["prem"] = 100 * (Y["F"] / Y["realized"] - 1)
    Y["regime"] = Y["trade_date"].map(regime)
    return Y


def main():
    omip, spot_daily = _load()
    front = _front_panel(omip, spot_daily)
    yr = _yr_panel(omip, spot_daily)

    lines = []
    def p(*a):
        lines.append(" ".join(str(x) for x in a))

    p("=" * 74)
    p("INFERENCE ON THE FORWARD RISK PREMIUM (points 1-2)")
    p("=" * 74)

    # ---- (1a) regime MEANS with 95% CI, SE clustered by delivery month ----
    cat = "C(regime, levels=['pre','crisis','recovery'])"
    m_means = smf.ols(f"prem ~ 0 + {cat}", data=front).fit(
        cov_type="cluster", cov_kwds={"groups": front["dp_str"]})
    ci = m_means.conf_int()
    p(f"\n[1] FRONT-MONTH premium: regime means, SE clustered by delivery month "
      f"(n={len(front)}, {front['dp_str'].nunique()} clusters)")
    for name in m_means.params.index:
        reg = name.split("[")[-1].rstrip("]").split(".")[-1].strip("'")
        p(f"    {reg:<9} {m_means.params[name]:+6.2f}%  "
          f"95% CI [{ci.loc[name,0]:+.2f}, {ci.loc[name,1]:+.2f}]")

    # ---- (1b) TEST crisis != pre, recovery != pre (clustered) ----
    m_diff = smf.ols(f"prem ~ {cat}", data=front).fit(
        cov_type="cluster", cov_kwds={"groups": front["dp_str"]})
    cid = m_diff.conf_int()
    p("\n    Difference vs pre-crisis (clustered SE):")
    for name in m_diff.params.index:
        if name == "Intercept":
            continue
        reg = name.split(".")[-1].rstrip("]").strip("'")
        p(f"    {reg:<9} {m_diff.params[name]:+6.2f} pts  "
          f"[{cid.loc[name,0]:+.2f}, {cid.loc[name,1]:+.2f}]  p={m_diff.pvalues[name]:.3g}")

    # ---- (1c) block bootstrap over delivery months (robustness) ----
    rng = np.random.default_rng(2026)
    clusters = front["dp_str"].unique()
    g = {k: v["prem"].values for k, v in front.groupby("dp_str")}
    greg = {k: v["regime"].iloc[0] for k, v in front.groupby("dp_str")}
    boot = {r: [] for r in ["pre", "crisis", "recovery"]}
    for _ in range(3000):
        samp = rng.choice(clusters, size=len(clusters), replace=True)
        acc = {r: [] for r in boot}
        for c in samp:
            acc[greg[c]].append(g[c])
        for r in boot:
            if acc[r]:
                boot[r].append(np.concatenate(acc[r]).mean())
    p("\n    Block-bootstrap (resample delivery months, 3000 reps) 95% CI of mean:")
    for r in ["pre", "crisis", "recovery"]:
        lo, hi = np.percentile(boot[r], [2.5, 97.5])
        p(f"    {r:<9} mean {np.mean(boot[r]):+.2f}%  [{lo:+.2f}, {hi:+.2f}]")

    # ---- (2) TERM STRUCTURE: YR-ahead premium ----
    p(f"\n[2] YEAR-AHEAD (YR) premium term structure (n={len(yr)}, deliveries "
      f"{sorted(yr['dyear'].unique())}, mean lead {yr['lead'].mean():.0f} d)")
    p("    Caveat: YR contracts overlap heavily on few delivery years -> means are")
    p("    descriptive corroboration, not independent-sample inference.")
    yt = yr.groupby("regime")["prem"].agg(["count", "mean", "median"]).round(1)
    yt = yt.reindex(["pre", "crisis", "recovery"])
    p(yt.to_string())
    p(f"    pooled YR premium: mean {yr['prem'].mean():+.1f}%  median {yr['prem'].median():+.1f}%")
    # cluster by delivery year (few clusters -> report but flag)
    m_yr = smf.ols(f"prem ~ 0 + {cat}", data=yr).fit(
        cov_type="cluster", cov_kwds={"groups": yr["dyear"].astype(str)})
    ciy = m_yr.conf_int()
    p("    regime means, SE clustered by delivery year (few clusters; wide CIs):")
    for name in m_yr.params.index:
        reg = name.split("[")[-1].rstrip("]").split(".")[-1].strip("'")
        p(f"      {reg:<9} {m_yr.params[name]:+6.2f}%  [{ciy.loc[name,0]:+.2f}, {ciy.loc[name,1]:+.2f}]")

    txt = "\n".join(lines)
    (OUT / "inference_premium.txt").write_text(txt, encoding="utf-8")
    # tidy CSV of the headline front-month inference
    rows = []
    for name in m_means.params.index:
        reg = name.split("[")[-1].rstrip("]").split(".")[-1].strip("'")
        rows.append(dict(horizon="front_month", regime=reg, mean_pct=round(m_means.params[name], 2),
                         ci_lo=round(ci.loc[name, 0], 2), ci_hi=round(ci.loc[name, 1], 2)))
    pd.DataFrame(rows).to_csv(OUT / "premium_inference.csv", index=False)
    print(txt)


if __name__ == "__main__":
    main()
