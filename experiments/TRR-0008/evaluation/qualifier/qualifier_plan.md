# TRR-0008 evaluator-path qualifier

This bounded pre-truth check exercises the actual
`scripts/trr0008_eval_runner.predict_current_h` path against the archived
TRR-0007 public prediction IDs. It uses the registered TRR-0007 observation
manifest, frozen method states, normalized public embedding, and archived
prediction artifacts only. It does not open source text, target labels, a
truth sidecar, or a fresh selection ledger.

The fixed matrix is the four scientific methods in
`trr0008_eval_contract.METHOD_ORDER`, all four paired public cells, and 128
rows per cell. Every output row must match the archived ID tensor exactly.
The executable commit and qualifier/runner/contract/loader source bindings are captured before loading and rechecked before writing a success receipt; a mid-run change fails closed.
Only tensor digests and input/code/resource bindings are written to the
create-only receipt:

`experiments/TRR-0008/evaluation/qualifier/runner_qualifier.json`

Failures are retained separately at:

`experiments/TRR-0008/evaluation/qualifier/runner_qualifier.failure.json`

The first authorized execution command is:

```text
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts python3 scripts/trr0008_eval_qualifier.py --execute --device cuda --repository-root . --trr7-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007 --output /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0008/experiments/TRR-0008/evaluation/qualifier/runner_qualifier.json --failure-output /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0008/experiments/TRR-0008/evaluation/qualifier/runner_qualifier.failure.json
```

No retry with altered rows, methods, ordering, or device is part of this
qualifier. A mismatch or resource guard failure remains a failed-closed
receipt.
