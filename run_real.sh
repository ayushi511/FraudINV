#!/usr/bin/env bash
# Train and evaluate on a REAL dataset. Usage:
#   ./run_real.sh paysim        (optional 2nd arg: fraction of legit rows to keep, default 0.1)
#   ./run_real.sh ibm           (default 0.2)
# Results are copied to reports_paysim/ or reports_ibm/ so the two runs don't overwrite each other.
set -euo pipefail
export PYTHONPATH=.
RAW=datasets/data/raw
case "${1:-}" in
  paysim) python -m datasets.loaders paysim  "$RAW/PS_20174392719_1491204439457_log.csv" --frac "${2:-0.1}" ;;
  ibm)    python -m datasets.loaders ibm_aml "$RAW/HI-Small_Trans.csv" --frac "${2:-0.2}" ;;
  *) echo "usage: ./run_real.sh paysim|ibm [frac]"; exit 1 ;;
esac
rm -f fraud.db                      # old alerts belong to the previous dataset
python -m models.train
python -m evaluation.evaluate_reports --cases 60
python -m evaluation.latency --tiers 100,1000
python -m pytest -q
rm -rf "reports_$1" && cp -r reports "reports_$1"
echo "Done: results in reports_$1/. Start the app with: uvicorn backend.app:app"
