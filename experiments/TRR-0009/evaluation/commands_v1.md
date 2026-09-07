# TRR-0009 production command/config receipt

**Status:** `TEMPLATE_ONLY_NO_EXECUTION`  
**Task:** `TRR-0009`  
**Scope:** public capture, registration, prediction, timing, public gate, and post-gate truth preparation/scoring.  
**Generated:** 2026-09-07 (Australia/Sydney)

This receipt binds the frozen Finance/Pile panel and four reported method IDs. It is a command/config template, not a run receipt. No command below has been executed from this file. The selected states, support export, method freeze, and loader paths are bound below; only the source-selection/capture receipts, forward-equivalence receipt, and fit/deployment cost records remain owner-produced before registration. The live timing command is included below; it consumes the same frozen plan as the runner and writes a separate create-only balanced cost receipt.

## Frozen interface

| Binding | Value |
| --- | --- |
| Task | `TRR-0009` |
| Domains | `pile=128`, `finance=256` unique source records |
| Cells | `pile__public_base`, `pile__public_lora_2601`, `finance__public_base`, `finance__public_lora_2601` |
| Capture geometry | batch 8, full prefix 192 tokens, retain 128 including BOS, hidden size 2048 |
| Prediction geometry | 128 IDs including BOS, 127 scored post-BOS positions, vocabulary 128256 |
| Methods | `unchanged_anchor`, `continued_fixed_readout`, `continued_adaptable_readout`, `published_reference` |
| Planning method spelling | `retained_reference` in `planning/method_freeze_schema.md` maps to evaluator row `published_reference` |
| Timing plan | `experiments/TRR-0009/evaluation/timing/plan.json`, SHA-256 `58c9af1ae2583003f87aeca612fad75d9d35a3ceb90eb181374d5f435ef52463` |
| Timing geometry | 40 blocks, 32 records/cell, five paths including fixed alias, one warmup + one measured call |
| Cost rule | every cell candidate/fixed upper 95% block-ratio CI <= 1.25 |
| Alias rule | every cell alias 95% CI fully contained in [0.95, 1.05]; failure invalidates cost evidence |
| Truth boundary | no truth/source-label materialization before the public gate writes the freeze receipt |

The exact support count is 17,126 of 128,256 vocabulary IDs. The capacity-owned support export is present: adaptive state SHA-256 `93661674c5e92144b84779737dee46a604afeb1ce1553966e464f0c84b2133c3`, support digest `8090b9231042d6c9f9c52f3a0a9d8733b8ec1b739a4f28b12ab1057a7126969d`, full frequency/support container `experiments/TRR-0009/evaluation/frequency_support_v1.safetensors` (1301040 bytes, SHA-256 `7d0999a6d96671dd537bd020703d1c057fb5b25585bb9a011bf7d63ac0d2ee81`), and single-tensor loader records in `loadable_support.json` (SHA-256 `a6bb03c9c4bad30fa34f06b6e5c513509f03033837b698cb87f4a64dc6750e11`). Registration binds the single-tensor `support_ids.safetensors` and `support_counts.safetensors` files, each with its container bytes/SHA; payload tensor digests remain provenance checks.

## Environment

Set paths only after the owner has verified the actual files and the method-freeze receipt:

```bash
export TRR9_ROOT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0009
export TRR7_ROOT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007
cd "$TRR9_ROOT"
export PYTHONPATH="$TRR9_ROOT:$TRR9_ROOT/src:$TRR9_ROOT/scripts:$TRR7_ROOT/src:$TRR7_ROOT/scripts"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

The frozen published positionwise/joint loader modules are resolved from `TRR7_ROOT/src`; TRR9 evaluator/model modules and selected continuation states are resolved from this worktree and remain first on the path. The fixed continuation state currently available for CPU header/loader inspection is:

```text
experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors
sha256 5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14
selected_step 400
schema token-reconstruction.trr0007-positionwise.v1
method_id trr0007_residual_mlp512
```

A CPU-only loader check passed for that file with `load_positionwise_model_state(method_id="trr0007_residual_mlp512", hidden_size=2048, vocabulary_size=128256, context_width=128)`. This does not qualify its forward output or release GPU work.

## Method-freeze and support preflight

Before selection/capture, the training owner creates `experiments/TRR-0009/training/method_freeze.json` exactly as specified by `experiments/TRR-0009/planning/method_freeze_schema.md`:

```text
schema token-reconstruction.trr0009-selected-method-freeze.v1
status FROZEN_TRR0009_METHODS_BEFORE_SOURCE_SELECTION
method order unchanged_anchor, continued_fixed_readout, continued_adaptable_readout, retained_reference
state_selection_frozen true
source_selection_started false
truth_opened false
private_or_truth_payload_read false
fresh_evaluation_started false
```

Every `state_bindings` row must contain an actual path/bytes/SHA-256 record, loader module/function, role, and selected validation step. At registration time, the `retained_reference` row is copied to the evaluator ID `published_reference`; no fourth research arm is introduced.

`experiments/TRR-0009/evaluation/support_binding.json` and `loadable_support.json` are the capacity receipts. The evaluator requires exact agreement among the two single-tensor loader files, the embedded adaptive support IDs/counts, the enriched frequency map, the support digest, and the adaptive state header. A missing or placeholder field must stop registration.

## 1. Public capture (after selection and explicit compute release)

The selection file must already contain the frozen Finance/Pile rows and public source descriptors. The command writes only public activations and receipts:

```bash
python3 scripts/trr0009_eval_capture.py capture --execute \
  --repository-root "$TRR9_ROOT" \
  --selection experiments/TRR-0009/selection/source_selection.json \
  --tokenizer "$TOKENIZER_SNAPSHOT" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --lora-config "$LORA_CONFIG" \
  --lora-update "$LORA_UPDATE" \
  --output-root experiments/TRR-0009/evaluation/public_observations \
  --device cuda
```

The producer must qualify the fixed batch-8 x 192 full forward and retain only the first 128 positions. A capture failure is retained in `failure.json`; do not rerun with changed source, model, batching, or selection bindings without a new owner decision.

Expected create-only outputs:

```text
experiments/TRR-0009/evaluation/public_observations/observations.json
experiments/TRR-0009/evaluation/public_observations/panel.json
experiments/TRR-0009/evaluation/public_observations/capture.json
```

## 2. Registration (before prediction and before truth)

Copy `experiments/TRR-0009/evaluation/registration_payload.template.json` to a temporary payload and replace every `<...>` marker with the owner-produced file record or receipt. Keep the method rows in `contract.METHOD_ORDER`; use the exact loader interface `trr0009.current_h.full_vocabulary.v1` and set `current_h_only=true`, `full_vocabulary=true`, `history_enabled=false`, and `a2_enabled=false` on every row.

```bash
cp experiments/TRR-0009/evaluation/registration_payload.template.json /tmp/trr0009_registration_payload.json
# Edit /tmp/trr0009_registration_payload.json only after method_freeze.json,
# support_binding.json, capture.json, and initialization-equivalence receipt exist.
python3 scripts/trr0009_eval_register.py \
  --registration experiments/TRR-0009/evaluation/registration_v1.json \
  --payload /tmp/trr0009_registration_payload.json
```

Registration is create-only and must fail closed on a placeholder, missing bytes, changed state, changed public inputs, or a missing adaptive support binding. The payload fixes `records_by_domain` to `{"pile":128,"finance":256}`; it does not accept TRR8's larger 384/1024 panel.

## 3. Public prediction runner

This is the actual four-method current-H inference entrypoint. It writes predictions/timing rows and a truth-free run manifest under the registered output root:

```bash
python3 scripts/trr0009_eval_runner.py \
  --repository-root "$TRR9_ROOT" \
  --registration experiments/TRR-0009/evaluation/registration_v1.json \
  --device cuda
```

The runner checks code/input/state hashes before model load and again before writing success. Each warmup and measured call must return exact equal IDs. It records startup, per-record measured time, preparation time, synchronization boundary, and peak-memory telemetry. Any guard or hash failure leaves `run_manifest.failure.json` and stops.

Expected output:

```text
experiments/TRR-0009/evaluation/predictions_v1/**/<method_id>.safetensors
experiments/TRR-0009/evaluation/predictions_v1/**/<method_id>.run.json
experiments/TRR-0009/evaluation/predictions_v1/run_manifest.json
```

## 4. Timing plan check and live timing boundary

The pre-registered arithmetic schedule can be checked without device work:

```bash
python3 scripts/trr0009_timing.py --print-plan \
  --generated-utc 2026-09-07T00:39:31Z \
  > /tmp/trr0009_timing_plan.generated.json
cmp --silent /tmp/trr0009_timing_plan.generated.json experiments/TRR-0009/evaluation/timing/plan.json
```

`--print-plan` checks the exact pre-registered schedule without device work. The live executor consumes the frozen registration and this plan, loads the four declared methods, aliases the fixed model object for the fifth path, verifies exact fixed/alias IDs on all timed rows, and then runs the synchronized one-warmup/one-measured boundary for every scheduled method/cell entry. It records raw 40-block entries, per-record variability, startup/preparation, peak memory, guard telemetry/overhead, state/input/code hashes, and the prospective per-cell Student-t decisions in a create-only receipt:

```bash
python3 scripts/trr0009_timing.py --execute \
  --repository-root "$TRR9_ROOT" \
  --registration experiments/TRR-0009/evaluation/registration_v1.json \
  --output experiments/TRR-0009/evaluation/timing/result.json \
  --device cuda \
  --records-per-cell 32 --blocks 40 --warmup-runs 1 --seed 8008 --maximum-seconds 600
```

The receipt is authoritative for the timing/cost gate only when every alias CI is contained in [0.95, 1.05]. An alias failure is recorded as invalid control and cannot be converted into a candidate cost failure; an inconclusive alias leaves cost unresolved. The synthetic CPU test exercises the same loop, while this production command remains separately release-gated and has not been run from this template.

## 5. Public gate and freeze

After the runner and the authoritative timing receipt exist, validate the complete matrix and write the create-only public freeze:

```bash
python3 scripts/trr0009_eval_gate.py \
  --repository-root "$TRR9_ROOT" \
  --registration experiments/TRR-0009/evaluation/registration_v1.json \
  --run-manifest experiments/TRR-0009/evaluation/predictions_v1/run_manifest.json \
  --balanced-timing experiments/TRR-0009/evaluation/timing/result.json \
  --freeze-output experiments/TRR-0009/evaluation/public_freeze_v1.json
```

The gate verifies all four methods in all four cells, capture/source/state/input/code bindings, registration bytes and SHA, prediction/timing file and tensor digests, warmup/measured identity, timing plan and receipt, truth-free flags, and the exact current-H/full-vocabulary loader semantics. It must reject before truth on any altered prediction root, missing record field, state drift, source/input drift, or timing-control failure.

## 6. Truth preparation (only after the gate succeeds and root authorizes)

Truth preparation materializes labels from the already frozen public selection into a sidecar outside the repository and prediction root. The command is recorded here but has not been authorized or run:

```bash
python3 scripts/trr0009_eval_truth.py prepare --execute \
  --repository-root "$TRR9_ROOT" \
  --freeze experiments/TRR-0009/evaluation/public_freeze_v1.json \
  --registration experiments/TRR-0009/evaluation/registration_v1.json \
  --selection experiments/TRR-0009/selection/source_selection.json \
  --tokenizer "$TOKENIZER_SNAPSHOT" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --truth-output /secure/private/TRR-0009/truth_sidecar_v1.safetensors \
  --truth-binding experiments/TRR-0009/evaluation/truth_binding_v1.json
```

The sidecar path must be outside `TRR9_ROOT` and the registered prediction root. The truth binding remains metadata-only until scoring.

## 7. Score (only after truth binding and explicit scoring authorization)

```bash
python3 scripts/trr0009_eval_truth.py score \
  --repository-root "$TRR9_ROOT" \
  --freeze experiments/TRR-0009/evaluation/public_freeze_v1.json \
  --truth-binding experiments/TRR-0009/evaluation/truth_binding_v1.json \
  --truth-sidecar /secure/private/TRR-0009/truth_sidecar_v1.safetensors \
  --frequency-reference experiments/TRR-0009/evaluation/frequency_reference_v1.json \
  --output experiments/TRR-0009/evaluation/score_v1.json
```

The score command rechecks the freeze, truth-binding header, sidecar metadata, prediction digests, and the frozen fitting-frequency reference before its sole private tensor open. It cannot refit, route, expand the panel, revise timing, or rewrite predictions.

## Read-only TRR8 loader fixture

`trr8_loader_equivalence_fixture.json` binds the existing TRR8 public observation manifest, stored prediction files, current-bank state (`2a44a91b...d8c`), and retained reference state (`696eb9fc...99a2`). It checks only the first eight records in each of the four already-opened TRR8 cells and has no truth/source-label binding. It is a qualification fixture, not a TRR9 score input and not a replacement for the TRR9 128/256 panel. The trained-state extension is intentionally pending until the adaptive state and support artifact exist. Any execution needs a separate root GPU release and must preserve the fixture's exact records, loaders, and archived prediction hashes.
