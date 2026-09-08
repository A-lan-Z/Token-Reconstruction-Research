# TRR-P10 execution adapter contract

This note records the CPU-only execution adapter prepared for the amended
TRR-0011 confirmation. No new source selection, public forward, GPU run, truth
opening, P03 holdout access, or model execution was performed here.

The implementation is in `scripts/trr_p10/execution.py`, the command wrapper is
`scripts/trr_p10/evaluate.py`, and the focused regression suite is
`tests/test_trr_p10_execution.py`.

`validate_frozen_package` consumes a hash-bound public freeze and registration.
It requires expanded-fixed, current-fixed, and frozen A1+A2 prediction rows on
all four paired cells, verifies every prediction file and tensor digest, checks
BOS/vocabulary/geometry, and rejects truth/source access. Primary methods use
the full per-cell panel count. A1+A2 may use a smaller count only when its
prediction binding carries the ordered parent-row indices, count, and canonical
`indices_sha256`; expanded/current rows cannot silently use a reduced
denominator. The validator stores method/cell counts separately from the full
truth parent count.

`load_truth_after_freeze` requires a descriptor prepared after the public
freeze and matching freeze/registration/sidecar bindings. It reads the full
truth sidecar at the primary panel geometry. `build_error_inventory` projects a
full-method correctness matrix onto the exact A1 parent indices before pairing,
so expanded/current-vs-A1 denominators can differ without positional zipping.
It reports per-method counts, subset hash/alignment, token categories, and exact
record categories for expanded-fixed vs current-fixed, expanded-fixed vs A1+A2,
and current-fixed vs A1+A2. Domains and target conditions remain separate.

Optional proposal/rank trace files can be bound as opaque, truth-free,
per-prediction diagnostics. Their file bytes/hash and truth flags are checked;
candidate materialization in a prediction is accepted only when that exact
prediction binding carries a validated trace. The adapter does not interpret an
opaque trace as a rank decomposition. The inherited artifacts contain only final
`predictions` tensors, declare `candidate_arrays_persisted=false`, and have no
candidate arrays or rank traces. Consequently current diagnostics report A1
proposal failures and decoder-rank failures as `UNAVAILABLE`; a final mismatch
is never reclassified as a candidate or decoder failure.

The validator replayed the existing TRR-0010 freeze at
`experiments/TRR-0010/evaluation/public_prediction_watchdog_r5/public_freeze.json`
read-only: 12 required prediction artifacts (three methods across four cells),
all at the inherited 128-row geometry, validated with the published freeze hash
`73e0fce672c27d7e8e1cb931f975029a9309dbd099f94f884661ccde95447d91`.

Focused CPU-only result: `PYTHONPATH=. pytest -q tests/test_trr_p10_execution.py`
passed 5 tests. The tests use synthetic tensors and exercise hash corruption,
pretruth flag rejection, registration/descriptor binding, unequal primary/A1
denominators with parent-row alignment, subset digest rejection, no-rank
decomposition, and create-only output behavior.
