#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0017/benchmark_controls_guard_r1.json --timeout 1800 -- python3 scripts/trr0017/benchmark_predict.py --group controls 2>&1 | tee experiments/TRR-0017/benchmark_controls_r1.log
python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0017/benchmark_continuous_guard.json --timeout 1800 -- python3 scripts/trr0017/benchmark_predict.py --group continuous 2>&1 | tee experiments/TRR-0017/benchmark_continuous.log
python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0017/benchmark_discrete_guard.json --timeout 1800 -- python3 scripts/trr0017/benchmark_predict.py --group discrete 2>&1 | tee experiments/TRR-0017/benchmark_discrete.log
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/TRR-0017/benchmark_scoring_guard.json --timeout 300 -- python3 scripts/trr0017/benchmark_score.py > experiments/TRR-0017/benchmark_score.log 2>&1
