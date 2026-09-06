# TRR-0005 v2 prediction-root compatibility review

Status: static review after the create-only freeze stopped before truth. This
note does not modify the v1 prediction root, prediction tensors, source HEAD,
or evaluator truth, and it does not authorize freeze/scoring retry.

## Preserved failed attempt

The exact small attempt evidence is copied from
`/tmp/trr5/trr0005_freeze_score_v1` into this directory. The binding copy
succeeded; the freeze command stopped with:

```text
TRR-0005 freeze error: prediction identity changed: finance__public_base/enriched__affine_causal_h_attention128
```

The first failing v1 run receipt has schema
`token-reconstruction.trr0005-prediction-receipt.v1`, while the contract's
stable prediction descriptor schema is
`token-reconstruction.trr0005-fresh-confirmation-prediction.v1`. A read-only
scan of all 32 v1 receipts found the same schema on every receipt, complete
cell/method keys, task ID `TRR-0005`, prediction shape `[128,128]`, and the
required timing tuple `(warmup=1, measured=1, warmup/measured IDs exact,
measured output selected)`. The prediction and timing manifests carry the same
metadata schema. Thus the observed defect is the producer's merged metadata
identifier; moving to a new root also requires the path rewrites below.

The binary prediction writer records the stable prediction schema in each
safetensors header before the timing metadata is merged into JSON. The v2
export should verify public headers only and leave all 32 tensor bytes and
header metadata unchanged; the strict gate will perform the authoritative
artifact read and byte/hash checks.

## New-root export requirements

Use a new repository-local root, for example:

```text
experiments/TRR-0005/fresh_confirmation_v1/predictions_v2
```

Copy the 32 existing `.safetensors` files byte-for-byte to the matching
`pile/finance` and `public_base/public_lora_2601` subdirectories. Copy the
label-free `v1/evaluator_binding.json` to `v2/evaluator_binding.json` unchanged
before the v2 freeze. Do not copy a freeze receipt, truth sidecar, or any
private label file into v2.

For every copied `*.run.json`, update only these metadata/path fields:

- Set the top-level `schema` to
  `token-reconstruction.trr0005-fresh-confirmation-prediction.v1`.
- Rewrite the exact prefix
  `experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/` to
  `experiments/TRR-0005/fresh_confirmation_v1/predictions_v2/` in
  `prediction_artifact.path` and `artifact_relative_to_root`.
- Preserve `timing_schema`, all method/cell IDs, tensor `bytes`/SHA, prediction
  tensor digest, panel/selection/observation hashes, candidate policy, and
  every numeric timing value exactly. In particular, do not round or recompute
  `warmup_seconds_sum`, `measured_seconds_sum`,
  `per_record_measured_seconds`, `timed_interval_total_seconds`,
  `measured_elapsed_seconds`, or `runtime_load_seconds`.
- Rehash the changed small run-receipt JSON when updating the corresponding
  `run_evidence.methods[*].cells[*].receipt` descriptor; the copied prediction
  artifact descriptors retain their original bytes and SHA.

For `predictions.json`, preserve the top-level manifest identity, method/cell
sets, public input descriptors, and all numeric values. For each of its 32
entries, apply the same stable `schema` correction and exact v1-to-v2 path
rewrite to the prediction artifact fields. For `timings.json`, apply the exact
path rewrite and retain every timing field and total; setting its entry
`schema` to the stable prediction schema keeps the two manifests consistent,
while its `timing_schema` remains unchanged.

For `run_evidence.json`, rewrite only exact v1 prediction-root prefixes in:
`prediction_manifest.path`, `timing_manifest.path`,
`methods[*].cells[*].artifact.path`, and
`methods[*].cells[*].receipt.path`. Recompute only the small JSON manifest and
receipt descriptor bytes/SHA fields that changed. Preserve method state/code
paths and hashes, registration/selection/panel descriptors, method IDs, cell
IDs, prediction/timing counts, all method-specific simulation counts, all
numeric timing values/totals, and the no-truth status flags. Do not rewrite
unrelated paths containing `v1`, such as `joint_fit_v1`.

## Required v2 gate checks before retry

Before asking root to run the v2 freeze, an export-only check must establish:

1. exactly 32 run receipts and 32 tensor files exist under v2, with the same
   cell/method set as v1;
2. every run receipt has the stable prediction schema, task ID, `[128,128]`
   shape, candidate policy, `warmup_runs_per_record=1`,
   `measured_runs_per_record=1`, `warmup_output_exact_match_measured=true`, and
   `measured_output_selected=true`;
3. every v2 artifact path in run receipts and both manifests resolves inside
   v2 and has the original v1 artifact bytes/SHA descriptors;
4. all 32 timing manifests preserve the original sums and per-record timing
   arrays exactly, with one warmup and one measured call per record;
5. the label-free binding is present only at `v2/evaluator_binding.json`, and
   its bytes/content are unchanged from v1;
6. public safetensors headers remain bound to the unchanged panel, selection
   plan, observation, cell/method, geometry, stable prediction schema, and
   method binding. Header verification must not alter the tensors;
7. the v2 freeze receipt is new, outside v2, and the v2 freeze command receives
   `--output-root v2`; the subsequent scorer receives the same v2 root and v2
   manifests/receipt, with its result path outside v2.

The strict freeze will hash every v2 bundle entry and make the root read-only.
The strict scorer will rehash/validate state, code, runtime assets, observations,
prediction files, and the frozen binding before its truth loader hashes or opens
the external sidecar. No check should weaken that ordering or reuse the failed
v1 receipt.

## Scientific and runtime scope preserved

The two original dot-causal fits were successful development fits superseded by
the preregistered qknorm repair; they are not failed attempts. The actual
failed development attempts were the V1 qualification forecast guard and the
capture output-root collision. Runtime summaries must use each cell's
`measured_seconds_sum / records` for the steady mean, count method load time
once across four cells, and report per-cell CUDA peaks separately from one
whole-process OS RSS high-water mark. A2 simulation counts remain per-cell
(reset at `begin_cell`): 256 calls per cell, 8,323,072 candidate simulations,
65,280 public-prefix calls, and 32,768 prefix commit tokens over four cells;
these are not extra method loads or an independent one-pass timing run.
