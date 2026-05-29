# Pod runbook — Schwartz-Smith production fit

This document is the **complete step-by-step** for running
`scripts/run_production_fit.py` on a cloud pod. The fit calibrates
Pieza 2 (Schwartz-Smith two-factor forward model) on the full
1565-day OMIP series and emits the artefact
`data/curated/forward_fit_production.pkl` that unlocks V5 and V6 of
the validation suite.

The fit cannot run on the laptop in a reasonable wall — empirical
benchmark estimates ~3 h on a fast laptop core, longer with thermal
throttling. A 4-core cloud pod runs it in ~1.5-2 h.

## What to read first

- `reports/diagnostics/forward_model_calibration.md` § Structural
  limit — the reason a full fit is needed.
- `scripts/run_production_fit.py` docstring — what the script
  outputs and how to re-run it.
- The previous laptop-side blocker that this runbook exists to
  prevent: **the curated parquets in `data/curated/` are gitignored.**
  The pod gets the code via `git clone`, but the input data must be
  copied separately. Skipping § 3 below will land you in `setup_pod.sh`
  failing with "MISSING …".

## Step 1 — start the pod

Recommendation: a CPU-only instance with at least 4 vCPU and 8 GB RAM.
The job is CPU-bound; the per-Kalman call is ~7 ms on a recent x86
core, so 4 cores in parallel does not help (the L-BFGS-B inner loop
is sequential), but headroom keeps the OS responsive.

| Provider | Suggested instance | Approx cost |
|---|---|---|
| Lambda Labs | gpu_1x_a10 (CPU portion suffices) | ~$0.60/h |
| Vast.ai | any 4 vCPU CPU instance | $0.10-0.30/h |
| AWS EC2 | c6i.xlarge (4 vCPU, 8 GB) | ~$0.17/h on-demand |
| Hetzner Cloud | CPX31 (4 vCPU, 8 GB) | ~€0.02/h |

Ubuntu 22.04 LTS is fine. Confirm Python 3.11 is the default; if not,
install via `sudo apt install python3.11 python3.11-venv` before § 4.

Note the pod's public IP and SSH user — you'll need both for § 3.

## Step 2 — clone the repository

On the pod:

```bash
cd ~
git clone https://github.com/cmvdata/mibel-derivatives.git
cd mibel-derivatives
```

The repository has the calibration code, the scripts and the runbook
(this file). It does NOT have the curated parquets — those follow in
§ 3.

## Step 3 — copy the curated parquets from the laptop

**This step is where the previous mibel-forecasting cloud-fit run
stalled.** The parquets live only on the laptop; they are gitignored
because they are large and partially derived from third-party data
that we do not redistribute.

From the **laptop** (Windows PowerShell or Git Bash):

```bash
cd C:\Users\Carlo\Desktop\Projects\Mibel_derivatives

# Adjust USER and POD_IP to match the pod credentials from § 1.
scp data/curated/omip_forward_2019_2024.parquet  USER@POD_IP:~/mibel-derivatives/data/curated/
scp data/curated/omie_spot_es_2019_2024.parquet  USER@POD_IP:~/mibel-derivatives/data/curated/
```

Both files together are ~900 KB so the copy completes in seconds even
on a residential uplink.

If you also want the production fit to refresh from a different
OMIE / OMIP cut, the file sources are documented in
`reports/diagnostics/data_provenance.md` § Manus drop and § bulk
download runner; copy those parquets to the same paths.

**Do not** try to re-derive the parquets on the pod (no scraping
credentials, no Manus drop). The only supported flow is laptop → pod
via scp.

## Step 4 — verify the pod environment

```bash
cd ~/mibel-derivatives
bash scripts/setup_pod.sh
```

`setup_pod.sh` does three things and aborts with a clear message if
any fails:

1. Installs Python deps (`uv sync` if available, `pip install -e .`
   otherwise).
2. Checks the two required parquets are at
   `data/curated/{omip_forward_2019_2024,omie_spot_es_2019_2024}.parquet`.
   If anything is missing it lists what and points at § 3.
3. Runs `pytest -m "not slow"`. If the fast suite is red on the pod,
   something is wrong with the environment and the production fit
   should not run.

Exit codes: `0` = pod ready, `1` = data missing, `2` = deps install
failed, `3` = pytest red.

## Step 5 — launch the production fit

```bash
# Foreground (good for first run so you can watch progress in stderr).
python scripts/run_production_fit.py 2>&1 | tee data/_production_fit_stdout.log

# Or background (so the ssh disconnect doesn't kill it):
nohup python scripts/run_production_fit.py > data/_production_fit_stdout.log 2>&1 &
echo $! > data/_production_fit.pid
```

The script:

- Refuses to overwrite the output pickle unless `--force` is passed
  (idempotent re-runs do nothing).
- Logs to both stderr and `data/_production_fit.log` so you can
  inspect progress even after the ssh session drops.
- Emits a Kalman log-likelihood line per L-BFGS-B iteration when
  scipy's `disp` option is set (currently silent — switch in script
  if needed).

Expected duration: 1.5-3 h on a 4-vCPU pod.

## Step 6 — copy the artefact back to the laptop

When the fit finishes, two files matter:

- `data/curated/forward_fit_production.pkl` (~50-200 KB pickled SSFit)
- `reports/diagnostics/production_fit_summary.md` (text summary the
  diagnostic links to)

From the **laptop**:

```bash
cd C:\Users\Carlo\Desktop\Projects\Mibel_derivatives
scp USER@POD_IP:~/mibel-derivatives/data/curated/forward_fit_production.pkl  data/curated/
scp USER@POD_IP:~/mibel-derivatives/reports/diagnostics/production_fit_summary.md  reports/diagnostics/
```

Then locally:

```bash
# V5 + V6 of the validation suite now execute instead of skip.
pytest tests/models/test_forward_validation.py -v

# The composition diagnostic and notebook can also consume the pickle:
#   from mibel_derivatives.models import forward, spot
#   import pickle
#   ss_fit = pickle.load(open("data/curated/forward_fit_production.pkl", "rb"))
#   composed = spot.fit_with_forward_anchor(omie_hourly, ss_fit)
```

If V5 (composition spot MAE) or V6 (composition forward at end of
history) fail, **stop and review** — this is the case the FASE A
diagnostic flagged as the production gate.

## Step 7 — destroy the pod

Once the artefacts are on the laptop and validated:

```bash
# On the pod
exit  # close ssh

# In the provider console: terminate the instance.
```

The pod has no state worth preserving — the fit pickle is the only
output and it is now on the laptop.

## Troubleshooting

- **`setup_pod.sh` fails with "MISSING data/curated/…"**: § 3 was
  skipped or only one parquet was copied. Re-run `scp` for the
  missing files.
- **Fit pickle is ~200 MB**: shouldn't happen — the SSFit dataclass
  is small. If you see it, you probably re-ran with `--force` and
  appended; delete and re-run.
- **pytest fast suite fails on the pod but not on the laptop**: most
  often a numpy/scipy version skew. Re-run `uv sync --refresh` (or
  `pip install -e ".[dev]" --force-reinstall`).
- **The fit raises `RuntimeError: …` mid-run**: the bound-active
  warning was downgraded in commit `c042d7c`; only NaNs in the
  parameter vector should still raise. Capture the full traceback
  from `data/_production_fit.log` and open an issue.
