# TRR-0010 final-evaluation execution plan

This is an inert command sheet. It records the order, interfaces, bindings,
and resource caps; it does not select sources, run capture, load a GPU model,
open final truth, or score. The companion
`final_evaluation_design_template_v1.json` is deliberately not acceptable to
registration because its status is pending the six-state freeze and its
`code_commit` is null.

The signed Stage-3 contract is
`experiments/TRR-0010/planning/shared_stage3_contract_v1.json` (25,643 bytes,
SHA-256 `23e4bde4475082cc004e7ef787c22fed6301275ce2ff23a0396d975b439faf9d`).
The v4 cost-threshold source is
`experiments/TRR-0010/planning/shared_contract.proposal.json` (38,409 bytes,
SHA-256 `3b0a68a4a05304e285c1fedcf8a2982bf4bef831e2e5bc0cebb20b8b2dba08b3`).
Before execution, the parent must replace the inert status with a final design
that binds the current full code commit and all six actual state/readout files.

## Frozen panel and six loader contracts

The panel is 128 Finance and 128 Pile identities selected from the half-open
ranges `Finance[12000,20000)` and `Pile[7000,10000)`. The signed task inventory is
`experiments/TRR-0010/planning/count_scan/inventory_7000_trr9_p08_union.json`,
SHA-256 `d64eae6b5e0137f1fb76e89ba7b618455853977330f13058877c6a7c3ab9038b`;
the command binds this exact file. Each identity is paired
across `public_base` and `public_lora_2601`; the selector also binds the final
B1 and approved opaque exclusion ledgers. Selection emits identity metadata
and digests only. It must run only after all six method rows are frozen.

The capture contract is the inherited full forward `B8 x 192`, retaining the
first 128 positions. The scored tensor is `[128, 128, 2048]` per cell and
uses 127 post-BOS positions over vocabulary size 128,256. The four cells are,
in order, `pile__public_base`, `pile__public_lora_2601`,
`finance__public_base`, and `finance__public_lora_2601`.

The exact loader interfaces are:

| method | interface and loader | state/resource status |
| --- | --- | --- |
| `unchanged_shared_start` | `trr0010.current_h.full_vocabulary.v1`; `token_reconstruction.trr0007_positionwise.load_positionwise_model_state(method_id=trr0007_residual_mlp512)` | Existing TRR-0009 selected state, step 400, SHA `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`; immutable public E SHA `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1` |
| `current_fixed` | `trr0010.current_h.full_vocabulary.v1`; `scripts.trr0010_p09_fixed_loader.load_p09_fixed_state` with strict P09 schema/15-tensor and metadata checks | B0 step 8000, 29,390,260 bytes, SHA `d2477cdf11422cb2d028196d72775825246f65f7e07394952375b3929627475a`; audit `experiments/TRR-0010/setup/b0_fixed_fit_audit_v1.json`, SHA `d573d342f54aec2ed4b37882b7803d69a36d4033182e42c417663ce959837449` |
| `current_directional` | `trr0010.current_h.full_vocabulary.v1`; base-only decoder loader plus one materialized `effective_readout_w` | Pending selected base-state and W file/hash after the directional fit |
| `expanded_fixed` | `trr0010.current_h.full_vocabulary.v1`; same strict P09 loader | A B1 step-13000 checkpoint is now available (29,390,268 bytes, SHA `5757506003f0a4b82cb1cd6de2a82f11345f2dbcbce13c8cf18c4f515de3ec3a`), but it is intentionally not promoted into the inert design until its complete descriptor is revalidated and frozen |
| `expanded_directional` | `trr0010.current_h.full_vocabulary.v1`; base-only decoder loader plus one materialized `effective_readout_w` | Pending selected base-state and W file/hash after the directional fit |
| `frozen_a1_a2_k256` | `trr0010.a1_a2.reconstructed_prefix_k256.v1`; native A1+A2 wrapper, proposal budget 512, candidate K=256, retained lens and public P0 prefix | Public descriptor `experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json`, SHA `5a299574d9b69b8365c6a67c39d8669949c8037103e2f178ed97085ba3b34928`; opened wrapper equivalence fixture passed with result SHA `f3842b816a8894fa836be4d32d02955e02ef66884000cb58809421970d8babeb` |

The existing directional Stage-3 r1 binding files are not promoted into this template: the file labeled B0 currently binds the B1 schedule (`be0745...`, semantic `8acdb2...`) and additions manifest, and its resource qualification reports zero GPU/RSS because it is CPU assembly metadata. Timing r2 must bind the exact B0/B1 schedule, bank manifest, imported provider sources, and measured qualification before either directional row can be frozen.

All direct rows must set `current_h_only=true`, `full_vocabulary=true`,
`history_enabled=false`, and `a2_enabled=false`. The A1+A2 row is the one
exception and must retain the explicit reconstructed-prefix/K=256 fields.
Directional deployment is a base-only decoder plus one merged full-vocabulary
W; a trainable delta or sparse fallback cannot be registered. Unsupported
vocabulary rows remain the immutable public E anchor.

## Resource qualification and reuse evidence

The prior TRR-0009 producer has a native-equivalent B8x192/first-128
qualification. The task-local A1+A2 opened fixture is exact for one row per domain, and the
zero-delta effective-W/base-only export fixture is exact (max
absolute logit difference 0 and exact argmax); its result is
`experiments/TRR-0010/evaluation/opened_fixture_zero_delta_v2/result.json`,
SHA `cd18d0b83be2297b9a15e41e5bcf365b068856b4af867b38886b0df7cddc6ab2`.
These receipts support reusing the core geometry. A final opened, truth-free
smoke is still required for each newly fitted directional loader and the B1
P09 state if their source bytes or loader implementation changes; otherwise
the registration gate must revalidate their exact file and imported-source
hashes before predictions.

Use fail-closed live preflight and an exclusive lease. Proposed caps are
binary-byte limits, pending parent confirmation at launch:

- capture: 900 seconds including model load and I/O; CUDA reserved <=10 GiB;
  GPU free >=11 GiB before load and >=2 GiB at runtime; host available >=12
  GiB before load and >=8 GiB at runtime; process RSS <=12 GiB; disk free
  >=20 GiB; output <=5 GiB;
- prediction: 900 seconds per method, with fixed CUDA reserved <=8 GiB and
  directional <=10 GiB, and the same host/GPU/disk floors.

The measured zero-delta fixture used 2.871 GB peak CUDA reserved and 3.68 GB
host RSS. The P09 directional qualification measured 7,600,078,848 bytes
reserved and 2,165,067,776 bytes host RSS for its discarded probe. These are
planning evidence, not permission to run the final matrix. A guard must charge
preparation, loader, forward, serialization, and I/O and preserve failures.

## Execution order and commands

Set the paths only from the final signed design and approved source ledger;
the placeholders below are intentionally unresolved until then:

```bash
ROOT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

DESIGN="$ROOT/experiments/TRR-0010/evaluation/final_evaluation_design.json"
INVENTORY="$ROOT/experiments/TRR-0010/planning/count_scan/inventory_7000_trr9_p08_union.json"
SELECTION="$ROOT/experiments/TRR-0010/evaluation/source_selection.json"
EXCLUSIONS="$ROOT/experiments/TRR-0010/evaluation/source_exclusions.json"
SELECTION_BINDING="$ROOT/experiments/TRR-0010/evaluation/source_selection_binding.json"
PRODUCER_SELECTION="$ROOT/experiments/TRR-0010/evaluation/producer_selection_bridge.json"
PRODUCER_ROOT="$ROOT/experiments/TRR-0010/evaluation/producer_capture"
OBS_ROOT="$ROOT/experiments/TRR-0010/evaluation/observations_v1"
REG_PAYLOAD="$ROOT/experiments/TRR-0010/evaluation/registration_payload.json"
REGISTRATION="$ROOT/experiments/TRR-0010/evaluation/registration.json"
PRED_ROOT="$ROOT/experiments/TRR-0010/evaluation/predictions_v1"
RUN_MANIFEST="$PRED_ROOT/run_manifest.json"
FREEZE="$ROOT/experiments/TRR-0010/evaluation/public_freeze.json"
COST="$ROOT/experiments/TRR-0010/evaluation/cost_evidence.json"
TRUTH_SIDECAR=/tmp/trr0010_final_truth_v1.safetensors
TRUTH_DESCRIPTOR="$ROOT/experiments/TRR-0010/evaluation/truth_descriptor_v1.json"
SCORE="$ROOT/experiments/TRR-0010/evaluation/score_v1.json"
```

1. After the six method rows and source/code bindings are complete, write the
   final design from the inert template with `status=
   FROZEN_TRR0010_FINAL_EVALUATION_DESIGN`, a current full code commit, and
   no null state/resource records. Do not execute the selector before this
   check. Confirm the exact final-B1 and approved opaque ledger records, then:

```bash
python3 "$ROOT/scripts/trr0010_select_public.py" select \
  --repository-root "$ROOT" --design "$DESIGN" --inventory "$INVENTORY" \
  --tokenizer "$TOKENIZER" --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW" --p08-opaque "$P08_OPAQUE" \
  --output "$SELECTION" --exclusions-output "$EXCLUSIONS"
```

   The selector is identity-only and create-only. Any exclusion intersection,
   count change, source-range change, or target-pairing change is a failure.

2. Run the actual producer-to-TRR-0010 repackager. It creates the producer
   selection bridge, runs the inherited public B8x192 producer for both
   conditions, verifies its explicit first-128 marker and producer-bound H
   descriptors, and writes new truth-free TRR-0010 observation/panel/capture
   artifacts:

```bash
python3 "$ROOT/scripts/trr0010_eval_capture.py" capture --execute \
  --repository-root "$ROOT" --selection "$SELECTION" --design "$DESIGN" \
  --selection-binding "$SELECTION_BINDING" \
  --producer-selection "$PRODUCER_SELECTION" \
  --producer-output-root "$PRODUCER_ROOT" --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" --finance-arrow "$FINANCE_ARROW" \
  --model-snapshot "$MODEL_SNAPSHOT" --lora-config "$LORA_CONFIG" \
  --lora-update "$LORA_UPDATE" --output-root "$OBS_ROOT" --device cuda
```

   The live watchdog must enforce the capture caps above. This is a new panel
   capture; no published TRR-0009 H payload is silently reused.

3. Once the observation, panel, capture, timing-plan, B0/B1 frequency, and all
   six method descriptors exist, create the registration payload from the
   final design. The existing CLI is:

```bash
python3 "$ROOT/scripts/trr0010_eval_register.py" --payload "$REG_PAYLOAD"
```

   The payload must point to the new selection/panel/observation/capture files,
   both frequency references, the final code bindings, and an output root
   under `experiments/TRR-0010/evaluation`. Registration must reject any
   pending/null method row.

4. Run exactly the six-method by four-cell public prediction matrix. The runner
   writes 24 prediction/timing records and a create-only manifest; it must not
   load tokenizer text, labels, or truth:

```bash
python3 "$ROOT/scripts/trr0010_eval_runner.py" \
  --repository-root "$ROOT" --registration "$REGISTRATION" --device cuda
```

   Preserve the run manifest even on failure. Do not rerun a failed contender
   with changed batch geometry, schedule, or state binding.

5. Validate and freeze all public outputs before any curator or score action:

```bash
python3 "$ROOT/scripts/trr0010_eval_gate.py" --repository-root "$ROOT" validate \
  --registration "$REGISTRATION" --run-manifest "$RUN_MANIFEST" \
  --require-current-head
python3 "$ROOT/scripts/trr0010_eval_gate.py" --repository-root "$ROOT" freeze \
  --registration "$REGISTRATION" --run-manifest "$RUN_MANIFEST" \
  --output "$FREEZE" --require-current-head
```

6. Extract truth-free cost evidence from the frozen public timing matrix. This
   must use the final signed v4 cost contract and remain before the sealed
   payload is opened:

```bash
python3 "$ROOT/scripts/trr0010_cost_evidence.py" \
  --repository-root "$ROOT" --run-manifest "$RUN_MANIFEST" \
  --contract "$ROOT/experiments/TRR-0010/planning/shared_contract.proposal.json" \
  --output "$COST"
```

7. Only after the public freeze passes, invoke the real trusted curator. The
   default materializer is the task adapter around the published renderer; it
   must create the sidecar and descriptor without opening the payload before
   descriptor validation:

```bash
python3 "$ROOT/scripts/trr0010_prepare_truth.py" prepare \
  --repository-root "$ROOT" --freeze "$FREEZE" \
  --truth-sidecar "$TRUTH_SIDECAR" --descriptor "$TRUTH_DESCRIPTOR" \
  --materializer scripts.trr0010_prepare_truth:materialize_public_truth \
  --require-current-head
```

8. Score once, after the descriptor and sidecar have been independently
   validated. The scorer keeps B0 and B1 frequency references separate and
   binds the cost artifact:

```bash
python3 "$ROOT/scripts/trr0010_score_cli.py" \
  --repository-root "$ROOT" --freeze "$FREEZE" \
  --truth-descriptor "$TRUTH_DESCRIPTOR" \
  --frequency-reference-b0 "$FREQUENCY_B0" \
  --frequency-reference-b1 "$FREQUENCY_B1" \
  --cost-evidence "$COST" --output "$SCORE" \
  --bootstrap-seed 10010 --bootstrap-draws 10000 \
  --require-current-head --execute
```

The only permitted truth-opening point is the final command's authorized
scorer call. Any failure in selection, capture, registration, public gate,
cost binding, curator descriptor validation, or resource guard stops the
sequence and preserves the failed receipt.

## Baseline source bindings

The inert template records the current hashes for the task-local selector,
capture, registration, runner, public gate, truth curator, scorer CLI, P09
loader, and inherited positionwise loader. These hashes describe the reviewed
baseline and must be rehashed into the final design's `code_bindings` after the
last mechanical integration change. The B0 P09 loader differs from the
TRR-0007 loader only in accepting the P09 schema while enforcing the same 15
decoder tensors; the current-H/full-vocabulary inference contract is unchanged.
