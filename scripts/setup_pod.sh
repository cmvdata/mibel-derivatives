#!/usr/bin/env bash
# Verify pod environment for the Schwartz-Smith production fit.
#
# What this script does (in order):
#   1. Installs Python dependencies via uv (preferred) or pip.
#   2. Verifies the curated parquets and any other inputs that
#      scripts/run_production_fit.py imports are on disk at the
#      expected paths. If any are missing, aborts with a clear message
#      pointing at the laptop → pod copy step.
#   3. Runs the fast pytest suite (`-m "not slow"`) so the pod is known
#      good before launching the multi-hour production fit.
#
# What this script does NOT do:
#   • Download data. The curated parquets are gitignored; they must be
#     copied from the laptop via scp (see reports/pod_runbook.md § 3).
#   • Regenerate the data. The scraping pipeline lives in src/.../data/
#     but is not exercised here.
#
# Usage:
#   bash scripts/setup_pod.sh
#
# Expected exit codes:
#   0  — pod ready; safe to launch `python scripts/run_production_fit.py`
#   1  — missing data; copy the listed parquets and re-run
#   2  — dependency install failed
#   3  — pytest fast suite failed

set -euo pipefail

RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RESET=$'\033[0m'

log_info()  { echo "${GREEN}[INFO]${RESET}  $*"; }
log_warn()  { echo "${YELLOW}[WARN]${RESET}  $*"; }
log_error() { echo "${RED}[ERROR]${RESET} $*" >&2; }

# ---- Step 1: dependencies ---------------------------------------------------

log_info "Step 1/3: installing Python dependencies"

if command -v uv >/dev/null 2>&1; then
    log_info "  using uv"
    uv sync || { log_error "uv sync failed"; exit 2; }
else
    log_warn "  uv not found, falling back to pip"
    python -m pip install --quiet -e ".[notebooks,dev]" \
        || { log_error "pip install failed"; exit 2; }
fi

# ---- Step 2: data presence verification ------------------------------------

log_info "Step 2/3: verifying curated inputs"

REQUIRED_INPUTS=(
    "data/curated/omip_forward_2019_2024.parquet"
    "data/curated/omie_spot_es_2019_2024.parquet"
)
# These are the only two inputs scripts/run_production_fit.py needs (it
# imports forward.fit which only consumes OMIP + the daily-mean OMIE
# series derived from omie_spot_es_2019_2024.parquet).

missing=0
for f in "${REQUIRED_INPUTS[@]}"; do
    if [[ -f "$f" ]]; then
        size_mb=$(du -m "$f" | cut -f1)
        log_info "  found  $f  (${size_mb} MB)"
    else
        log_error "  MISSING  $f"
        missing=$((missing + 1))
    fi
done

if (( missing > 0 )); then
    log_error ""
    log_error "$missing required input file(s) missing."
    log_error "These parquets are GITIGNORED — they live on the laptop only."
    log_error ""
    log_error "Copy them from the laptop to the pod, e.g.:"
    log_error "  scp data/curated/omip_forward_2019_2024.parquet  USER@POD:~/mibel-derivatives/data/curated/"
    log_error "  scp data/curated/omie_spot_es_2019_2024.parquet  USER@POD:~/mibel-derivatives/data/curated/"
    log_error ""
    log_error "Then re-run this script. See reports/pod_runbook.md § 3 for the"
    log_error "full copy procedure."
    exit 1
fi

# ---- Step 3: fast pytest sweep ---------------------------------------------

log_info "Step 3/3: running pytest fast suite ('-m \"not slow\"')"

if python -m pytest -m "not slow" -q; then
    log_info "fast suite passed"
else
    log_error "fast pytest suite failed — do not launch the production fit"
    log_error "until the environment is repaired."
    exit 3
fi

log_info ""
log_info "POD READY. Next step:"
log_info "  python scripts/run_production_fit.py"
log_info ""
log_info "Expected wall: ~2-3 hours on a 4-core instance."
log_info "Output: data/curated/forward_fit_production.pkl"
log_info "        reports/diagnostics/production_fit_summary.md"
