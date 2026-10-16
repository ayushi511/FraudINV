#!/usr/bin/env bash
# Rebuild everything from scratch: data -> models -> evaluation -> tests -> diagrams.
set -euo pipefail
export PYTHONPATH=.
python -m datasets.synthetic            # or: python -m datasets.loaders paysim <csv>
python -m models.train                  # Experiments 1-4 + fusion weights
python -m evaluation.evaluate_reports --cases 60   # Experiment 6 + ablation
python -m evaluation.latency --tiers 100,1000       # Experiment 7
python -m scripts.export_diagrams
python -m pytest -q
echo "Done. Start the app with: uvicorn backend.app:app  then open http://127.0.0.1:8000"
