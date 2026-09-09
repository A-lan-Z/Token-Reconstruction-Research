#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src
# Run after the active final_cpu_adam_r2 phase completes. Native comparison and
# static component are serial to avoid CPU timing contention.
python3 scripts/agent4/cpu_cache_qualification.py
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/agent4-prefix-only-inversion/evidence/final_cpu_a1a2_watchdog.json --timeout 1800 -- python3 scripts/agent4/predict.py --panel final --device cpu --run final_cpu_a1a2 --method a1a2
python3 scripts/agent4/score.py --panel final --runs final_cpu_adam_r2 final_cpu_a1a2 --output final_cpu_score.json
python3 scripts/agent4/first_error_diagnostic.py
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/agent4-prefix-only-inversion/evidence/static_capture_watchdog.json --timeout 600 -- python3 scripts/agent4/capture_static.py --device cpu
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/agent4-prefix-only-inversion/evidence/static_cpu_adam_watchdog.json --timeout 1200 -- python3 scripts/agent4/predict.py --panel static --device cpu --run static_cpu_adam --settings experiments/agent4-prefix-only-inversion/adam_settings.json
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/agent4-prefix-only-inversion/evidence/static_cpu_a1a2_watchdog.json --timeout 1200 -- python3 scripts/agent4/predict.py --panel static --device cpu --run static_cpu_a1a2 --method a1a2
python3 scripts/agent4/score.py --panel static --runs static_cpu_adam static_cpu_a1a2 --output static_cpu_score.json
