# Reproduce the development27 decoder study

Execution commit:1d6a035b54d533d1c8752a59578c4c79cd320c16. All method, plan,
asset, observation and qualification hashes are stored in dev27_decoder_freeze.json.
The runtime environment and exact worker commands are in each phase receipt.
Use the pinned neighboring worktree assets and observations specified by support.py.
Run from this task worktree with the existing output locations absent, or verify and
resume exact matching receipts. Never overwrite existing raw results.

The original command was:
```sh
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev27_decoder_guard.json --timeout 1800 --wait-timeout 300 -- python3 scripts/trr0020_stage27_decoder/run.py
python3 scripts/trr0020_stage27_decoder/score.py
python3 scripts/trr0020_stage27_decoder_archive.py
python3 scripts/trr0020_stage27_decoder_summary.py
```

run.py launches eight isolated configuration workers, largest first. Each requires
the public CPU/GPU derivative qualifications, both whole-decoder public geometries,
exact warm and initial-mixture anchors, and resource limits. It freezes all64cells.
score.py validates the full matrix and three negative gates before opening labels.
These are retrospective development inputs, not a fresh or canonical comparison.
The raw output archive has176logical members,144objects and151971122bytes.
Concatenate the four listed parts in order; verify the whole SHA256 in
dev27_decoder_archive.json, then restore logical paths using its object map.
All raw members and part reassembly were byte-verified.

The movement diagnostic is computed only from frozen outputs: concatenate local_trace
across all eight records for each configuration, take per-update means of columns
2(direction norm) and1(cosine error), and count column3(clip factor) below1.
Embedding movement is mean norm(embedding_final-initial_embedding). This diagnostic
does not use source truth and is not a pooled accuracy score.
