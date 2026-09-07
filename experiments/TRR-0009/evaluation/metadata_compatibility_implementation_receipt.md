# TRR-0009 metadata compatibility implementation receipt

This receipt records the explicitly authorized implementation of the exception proposed in `metadata_compatibility_review.md`. The approved proposal remains byte-identical at its original path.

The new task-local adapter is `scripts/trr0009_eval_gate_compat.py`; focused tests are `tests/test_trr0009_eval_gate_compat.py`. The adapter keeps the generic gate source unchanged and scopes its state-semantic replacement to a single wrapped call. It accepts only the registered continued-fixed state with path `experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors`, 29,390,492 bytes, SHA-256 `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`, TRR7 schema, selected step 400, exact geometry, exact inference-contract string, exact decoder kind/role, and complete current-H/full-vocabulary loader binding. Missing redundant header booleans are the sole compatibility allowance; present false or malformed values fail.

Revalidation pins the reloaded registration to historical inference commit `55ef2d46413de28be70b4014f596b277c0e1582c`, compares adapter/gate/truth source paths plus current bytes/hashes, reruns the exact loader qualification receipt (`e45d4f4618aaedac7cc6eaf5c6bda1f028d6b62a87edaca6d91040f459f2c552`), and requires all compatibility truth/source/label/candidate flags to be explicit `false`. The separate maintenance commit is recorded in the receipt. Registered registration, selected states, predictions, timing, and the preserved generic gate failure remain unchanged.

Focused verification passed with 37 tests:

    PYTHONPATH=.:src:scripts python3 -m pytest -q tests/test_trr0009_eval_gate_compat.py

The suite includes synthetic negatives for state/schema/geometry/step/hash/loader/role/path changes, contradictory metadata, historical registration changes, current source and loader binding changes, non-false compatibility flags, and wrapper failures before truth/prediction reads. It also includes actual fixed-state metadata and qualification-receipt smoke checks plus truth-context keyword dispatch. No public freeze, truth preparation, scoring, training, inference, or timing rerun has been performed.

After root commits and reviews this implementation, run the public freeze and validation commands from the parent proposal only after confirming the maintenance commit differs from the historical inference commit. Stop for root review if either command fails; do not open truth on a failed or altered receipt.
