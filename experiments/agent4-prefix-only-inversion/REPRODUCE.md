# Reproduce the bounded pilot

Use the task branch and Python3.12.3 / torch2.10.0+cu128 / transformers5.3.0.
Detailed installed dependency versions and actual archive hashes are in
`evidence/dependency_backup.json`; a clean package import restore is recorded
in `evidence/clean_restore.json`. Python, OS libraries and GPU drivers remain
external runtime requirements. The global repository pyproject is unchanged.

All commands run from this worktree, with:

```sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src
```

Large actual backups are under `outputs/agent4-prefix-only-inversion/backup`
and `restore`; the original public checkpoint is the exact cached revision in
`evidence/backup.json`. The tracked source archive pins official SipIt. Its code
was read, not installed or invoked as an opaque attack package.

Create-only output rules mean reruns require a fresh checkout/output root or
new run names. Never overwrite existing predictions to repair a run. For a
fresh setup:

```sh
python3 scripts/agent4/backup.py
python3 scripts/agent4/archive_dependencies.py
python3 scripts/agent4/restore_check.py  # historical attempt omitted numpy.libs
python3 scripts/agent4/restore_fix.py    # companion library archive, clean imports
python3 scripts/agent4/smoke.py
python3 scripts/agent4/cpu_diagnostic.py
python3 scripts/agent4/cpu_search.py
python3 scripts/agent4/cpu_adam.py
python3 scripts/agent4/capture.py --panel development_cpu --device cpu
python3 scripts/agent4/predict.py --panel development_cpu --device cpu --run development_cpu_adam --settings experiments/agent4-prefix-only-inversion/adam_settings.json
python3 scripts/agent4/score.py --panel development_cpu --runs development_cpu_adam --output development_cpu_score.json
python3 scripts/agent4/comparator_check.py --device cpu
python3 scripts/agent4/cpu_largest.py
```

The selected final text is already retained in `final_sources.json`; use those
bytes for exact replay. Running `select_final.py` again consults the pinned
selection seed but public websites can change; its original byte hashes and
paragraph indices are retained. The solver configuration was frozen before
that script selected any final sources.

```sh
python3 scripts/agent4/capture.py --panel final --device cpu
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/agent4-prefix-only-inversion/evidence/final_cpu_adam_r2_watchdog.json --timeout 1800 -- python3 scripts/agent4/predict.py --panel final --device cpu --run final_cpu_adam_r2 --settings experiments/agent4-prefix-only-inversion/adam_settings.json
bash scripts/agent4/finish_cpu.sh
```

GPU diagnostics are provided in `diagnostics.py` but require a coordinated GPU
lease and >=8GiB free before load. Do not label unexecuted diagnostics as results.
Changing observation device/backend constitutes a separately reported run, not
an implicit replacement of saved observations. No batching workaround is used
in the solver. Native comparator batching/cache logic is preserved and directly
qualified against the historical decision implementation.

The final resource guards record exact child commands, wall timestamps, host
RSS, host available memory, device status and failures. Prefix snapshots stay
fixed. The static capture script is evaluator-only and creates actual target
backup copies; neither reconstructor receives its target asset path.
