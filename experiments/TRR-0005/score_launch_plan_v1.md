# TRR-0005 freeze and truth-gated scoring launch plan

Status: prepared; do not execute this plan until the coordinator confirms that
coverage has completed the frozen panel and all 32 prediction/timing artifacts.
This document contains commands only. It does not select sources, open truth,
or run the scorer.

Run from the TRR-0005 worktree:

```text
REPO=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005
PANEL=$REPO/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json
OBSERVATIONS=$REPO/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations.json
REGISTRATION=$REPO/experiments/TRR-0005/fresh_confirmation_v1/method_registration.json
PLAN=$REPO/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json
PRED_ROOT=$REPO/experiments/TRR-0005/fresh_confirmation_v1/predictions_v1
PREDICTIONS=$PRED_ROOT/predictions.json
TIMINGS=$PRED_ROOT/timings.json
FREEZE_RECEIPT=$REPO/experiments/TRR-0005/fresh_confirmation_v1/freeze_receipt.json
TRUTH_MANIFEST=/tmp/trr5/fresh_confirmation_v1.truth.manifest.json
TRUTH=/tmp/trr5/fresh_confirmation_v1.truth.safetensors
BINDING=$PRED_ROOT/evaluator_binding.json
FREQUENCIES=$REPO/experiments/TRR-0005/frequency_references_v1.json
RESULT=$REPO/experiments/TRR-0005/fresh_confirmation_v1/result.json
```

## Copy the label-free binding before freeze

After the producer has written the complete prediction matrix and before the
freeze command, copy only the producer's label-free descriptor into the
prediction root. The descriptor must be create-only and must not be replaced
after freeze:

```text
mkdir -p "$PRED_ROOT"
test ! -e "$BINDING"
cp -- "$TRUTH_MANIFEST" "$BINDING"
```

The copied descriptor is the only truth-related file allowed in the frozen
prediction root. The private safetensors sidecar remains at `$TRUTH` outside
the reconstruction and frozen-public roots. Do not inspect, hash, or open that
sidecar before scoring has passed the public gate.

## Freeze the complete public matrix before truth

The freeze adapter discovers all per-cell `*.run.json` receipts below
`$PRED_ROOT`; it requires all 32 cell/method receipts and validates their
public bindings and one-warmup/one-measured timing contract. It writes the
create-only receipt outside the prediction root:

```text
env PYTHONPATH=.:src:scripts \
  "$REPO/.venv-trr0005/bin/python" "$REPO/scripts/trr0005_freeze_confirmation.py" \
  --repository-root "$REPO" \
  --panel "$PANEL" \
  --registration "$REGISTRATION" \
  --output-root "$PRED_ROOT" \
  --plan "$PLAN" \
  --receipt "$FREEZE_RECEIPT"
```

Do not proceed if this command fails. Its `--output-root` is the frozen
prediction root, and the freeze receipt must bind the copied
`evaluator_binding.json` before any truth loader is called.

## Score only after freeze and the complete public gate

The scorer receives the panel and observation descriptors from
`panel_capture_v2`, the frozen registration/selection plan, the two producer
manifests, the freeze receipt, both public frequency references, and the
external truth sidecar. Its `--output-root` is the same frozen prediction root;
the result is deliberately outside that root. The scorer validates the receipt,
panel, plan, observations, state/code/runtime assets, prediction bytes/tensors,
row order, and timing records before the truth loader hashes or opens `$TRUTH`.

```text
env PYTHONPATH=.:src:scripts \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$REPO/.venv-trr0005/bin/python" "$REPO/scripts/trr0005_score_confirmation.py" \
  --repository-root "$REPO" \
  --panel "$PANEL" \
  --registration "$REGISTRATION" \
  --selection-plan "$PLAN" \
  --observations "$OBSERVATIONS" \
  --predictions "$PREDICTIONS" \
  --timings "$TIMINGS" \
  --receipt "$FREEZE_RECEIPT" \
  --truth "$TRUTH" \
  --truth-binding "$BINDING" \
  --output-root "$PRED_ROOT" \
  --result "$RESULT" \
  --frequency-manifest "$FREQUENCIES" \
  --bootstrap-draws 10000 \
  --bootstrap-seed 5005
```

The scorer reports Pile and Finance, P0 and synthetic-LoRA, both original and
enriched frequency references, joint frequency × position × domain diagnostics,
paired gains/regressions, descriptive source bootstrap, and finite-sample
exact bounds. It must not be rerun with altered paths, timing attestations,
method choices, frequency bins, or truth inputs after a result is written.

## Static interface checks recorded

- `scripts/trr0005_freeze_confirmation.py` passes `args.output_root` as
  `frozen_root` to `freeze_public_matrix`; the receipt path is outside that
  root in this plan.
- `scripts/trr0005_score_confirmation.py` passes the same `--output-root` to
  `validate_before_truth`, requires every prediction artifact to resolve under
  it, and rejects a result path inside it.
- The score CLI requires the panel, registration, selection plan, observations,
  predictions, timings, freeze receipt, truth, binding descriptor, output root,
  and result path. The frequency manifest must provide both `original` and
  `enriched` references.
- The planned truth sidecar and binding descriptor are not read or hashed by
  this preparation step.
