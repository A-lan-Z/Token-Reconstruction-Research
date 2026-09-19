# Reproducing development29 and30

Development29 execution commit:572ab8f4cebd7825c5cfd462dc563070427db2c8.
Development30 execution commit:bbb247242df676b58a45072273756d27b807d2be.
Each source/asset/environment binding and exact worker command is in its freeze/phase records.
Use the pinned neighboring TRR-0014 observations and supplied public prefix declared by
support.py. The truth source is retrospective and is opened only after the complete matrix.

From this task worktree, using fresh output/receipt locations or verified exact resumes:
```sh
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev29_guard.json --timeout 1200 --wait-timeout 300 -- python3 scripts/trr0020_stage29/run.py
python3 scripts/trr0020_stage29/score.py
python3 scripts/trr0020_stage29_archive.py
python3 scripts/trr0020_stage29_summary.py
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev30_guard.json --timeout 1200 --wait-timeout 300 -- python3 scripts/trr0020_stage30/run.py
python3 scripts/trr0020_stage30/score.py
python3 scripts/trr0020_stage30_archive.py
python3 scripts/trr0020_stage30_summary.py
python3 scripts/trr0020_stage29_30_confidence.py
```

Do not overwrite archived runs. Recheck live resources; each guard is create-only and
each worker validates exact source and artifact hashes before resuming. Both studies
qualify largest warm64/tau1/128 first and use isolated configuration processes. Their
48-cell freezes each precede negative completeness/source/output gates and scoring.

Each raw archive has144logical members and108unique byte objects. Follow the object map
in dev29_archive.json or dev30_archive.json to restore logical paths. Both one-part ZIP
archives and every raw member were round-trip hash-verified. Preparation and all capture
events are separate from warmed per-record timings; actual first-input total includes
capture. Checkpoint times include the declared extra already-executed training call.
The confidence diagnostic runs after both freezes are scored and cannot alter outputs.
Neither study adds an active canonical method or establishes a replacement.
