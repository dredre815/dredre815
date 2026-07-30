#!/usr/bin/env bash
set -euo pipefail
python src/run_pilot.py --n-events "${N_EVENTS:-24}" --model "${MODEL:-gpt-5.4-mini-2026-03-17}" --max-cost-usd "${MAX_COST_USD:-8}" --output-dir results
python src/analyze.py --input results/decisions.jsonl --output-dir results/analysis
