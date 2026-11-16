#!/usr/bin/env bash
# Fresh-clone reproduction check: follows README.md step by step in a throwaway
# directory and verifies the committed predictions.csv is reproduced exactly.
#
# Clones from this working copy because the branch is not pushed; everything
# after the clone is the documented path a reviewer would take.
#
#   bash scripts/verify_reproduction.sh [workdir]
#
# Set CWFC_RELEASE_CACHE to a local data-v1.tar.gz to skip the 112 MB download.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$(mktemp -d)}"
RELEASE_URL="https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mFAIL: %s\033[0m\n' "$1"; exit 1; }

step "0. Fresh clone into $WORKDIR"
rm -rf "$WORKDIR/checkout"
git clone --quiet --no-local "$REPO_ROOT" "$WORKDIR/checkout"
cd "$WORKDIR/checkout"
git checkout --quiet "$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)" 2>/dev/null || true
echo "cloned $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
[ -d challenge_data ] && [ -n "$(ls -A challenge_data 2>/dev/null | grep -v README.md || true)" ] \
  && fail "clone already contains release data; it must not be committed"
echo "clone does not carry the release data (correct)"

step "1. Install (README section 1)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[dev]" --quiet
python -m pytest -q

step "2. Get the data (README section 2)"
if [ -n "${CWFC_RELEASE_CACHE:-}" ] && [ -f "${CWFC_RELEASE_CACHE}" ]; then
  echo "using cached release: ${CWFC_RELEASE_CACHE}"
  cp "${CWFC_RELEASE_CACHE}" data-v1.tar.gz
else
  curl -sL -o data-v1.tar.gz "$RELEASE_URL"
fi
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
( cd challenge_data && sha256sum -c SHA256SUMS )

step "3. Generate predictions (README section 3)"
python -m cwfc.predict --data challenge_data --output predictions.csv

step "4. Compare against the committed submission"
if diff -q "$REPO_ROOT/predictions.csv" predictions.csv > /dev/null; then
  echo "predictions.csv reproduced byte-for-byte"
else
  diff "$REPO_ROOT/predictions.csv" predictions.csv | head -20
  fail "predictions.csv differs from the committed file"
fi

step "5. Score the holdout (README section 4)"
python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv | tail -12

step "6. Reproduce the analysis (README section 5)"
for script in analysis/01_data_and_workload.py analysis/02_cycle_attribution.py \
              analysis/03_validation_and_policy.py analysis/figures.py; do
  python "$script" > /dev/null
  echo "ok  $script"
done

printf '\n\033[32mALL README STEPS REPRODUCED\033[0m  (workdir: %s)\n' "$WORKDIR"
