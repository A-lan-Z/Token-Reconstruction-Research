# TRR-0010 final-evaluation runbook (pre-execution)

This runbook records the existing entrypoints and the small adapters still
needed before a production run. It is planning-only: no fresh selection,
capture, model load, GPU run, truth access, or scoring was performed while
writing it.

## Frozen matrix and ordering

The final design must first be written with status
`FROZEN_TRR0010_FINAL_EVALUATION_DESIGN`, a current full code commit, the
six exact method IDs, and the declared decision rules:

`unchanged_shared_start`,
`current_fixed`, `current_directional`, `expanded_fixed`,
`expanded_directional`, and `frozen_a1_a2_k256`.

The public panel is 128 Finance and 128 Pile records, paired across
`public_base` and `public_lora_2601`. Capture remains B8x192 and stores the
first 128 positions. All code/state/input bindings must be immutable before
the prediction run. The runner and gate enforce this matrix and the
truth-free flags.

## 1. Select the final natural panel after design freeze

There is no TRR-0010 selector CLI in this checkout. The existing
`scripts/trr0009_select_public.py select` is not directly executable for
this study: it hard-codes TRR-0009's Finance=256/Pile=128 proposal, restricts
outputs to TRR-0009 selection directories, and emits TRR-0009 selection
metadata.

Required small glue is a TRR-0010 identity-only selector (or an equivalent
owner-reviewed wrapper) with this command shape:

```bash
python3 scripts/trr0010_select_public.py select \
  --repository-root "$ROOT" \
  --design "$ROOT/experiments/TRR-0010/planning/final_evaluation_design.json" \
  --inventory "$ROOT/experiments/TRR-0009/planning/source_inventory.json" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW" \
  --p08-opaque "$P08_OPAQUE" \
  --output "$ROOT/experiments/TRR-0010/evaluation/source_selection.json" \
  --exclusions-output "$ROOT/experiments/TRR-0010/evaluation/source_exclusions.json"
```

The adapter should reuse TRR-0009's trusted renderer, deterministic ordering,
identity classifier, P04/P06/P08 opaque ledgers, and final-B1 exclusions, but
must freeze 128 records per domain and emit only identity metadata, hashes,
and record-order digests. It must fail closed on any ledger intersection.
The resulting selection and exclusion files are inputs to the panel and
capture stages.

## 2. Capture paired public observations

`scripts/trr0009_eval_capture.py capture --execute` is the existing public
producer. Its exact lower-level invocation is:

```bash
python3 scripts/trr0009_eval_capture.py capture --execute \
  --repository-root "$ROOT" \
  --selection "$ROOT/experiments/TRR-0010/evaluation/source_selection.json" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW" \
  --model-snapshot "$PUBLIC_MODEL_SNAPSHOT" \
  --lora-config "$LORA_CONFIG" \
  --lora-update "$LORA_UPDATE" \
  --output-root "$ROOT/experiments/TRR-0010/evaluation/producer_capture" \
  --device cuda
```

This producer creates both public-base and public-LoRA observations, but its
schemas/task identity are TRR-0009. A second small glue adapter is therefore
required to repackage its create-only `observations.json`, `panel.json`,
and `capture.json` under the TRR-0010 schemas, preserving the selection
record, cell order, 128/128 counts, source-order digests, and B8x192 geometry.
The adapter must not materialize source truth.

## 3. Register the frozen public run

After the TRR-0010 selection, panel, observation manifest, capture receipt,
timing plan, frequency references B0/B1, and all six method rows exist, build
the registration payload with the keyword fields accepted by
`trr0010_eval_register.build_registration`:

```bash
python3 scripts/trr0010_eval_register.py \
  --payload "$ROOT/experiments/TRR-0010/evaluation/registration_payload.json"
```

The payload must bind `design_path`, `selection_path`, `panel_path`,
`observation_manifest_path`, `capture_path`, `timing_plan_path`,
`method_rows`, `frequency_reference_paths` for both B0 and B1,
`output_root`, `output_path`, and `selection_binding_path`.
Directional rows bind a base-only decoder plus one merged effective W;
direct rows bind public E; the A1+A2 row binds public E, retained lens, and
public P0 prefix. The A1+A2 loader is the published 512-candidate then
first-256 adapter, represented by the fixed K=256 interface; it must not be
described as native top-256.

## 4. Run all public predictions before truth

```bash
PYTHONPATH=.:src:scripts python3 scripts/trr0010_eval_runner.py \
  --repository-root "$ROOT" \
  --registration "$ROOT/experiments/TRR-0010/evaluation/registration.json" \
  --device cuda
```

This writes the complete 6-method x 4-cell prediction/timing matrix and a
create-only `run_manifest.json`. The runner resets CUDA peaks per method and
cell, charges method preparation once across its four cells, checks warmup
versus measured IDs, and never loads a tokenizer, source text, labels, or
truth.

Validate and freeze the public matrix:

```bash
python3 scripts/trr0010_eval_gate.py --repository-root "$ROOT" validate \
  --registration "$REGISTRATION" --run-manifest "$RUN_MANIFEST" \
  --require-current-head

python3 scripts/trr0010_eval_gate.py --repository-root "$ROOT" freeze \
  --registration "$REGISTRATION" --run-manifest "$RUN_MANIFEST" \
  --output "$ROOT/experiments/TRR-0010/evaluation/public_freeze.json" \
  --require-current-head
```

## 5. Validate the curator's truth descriptor

`trr0010_eval_register.validate_truth_descriptor` exists as a library
validator, but there is no TRR-0010 truth-preparation/descriptor CLI. This is
the third required small glue item. A safe wrapper must first call that
validator against `public_freeze.json`, checking the frozen registration,
run manifest, panel/observation bindings, both frequency references, cell
order, paired target geometry, and sealed truth-payload hash while leaving the
private payload unopened. Only after that check may the separate curator
materialize the four paired truth tensors from the selected public rows and
write a create-only TRR-0010 truth descriptor with
`prepared_after_public_freeze=true` and `truth_opened=false`.

The TRR-0009 truth CLI cannot be reused unchanged because it validates a
TRR-0009 freeze/schema and emits TRR-0009 truth bindings.

## 6. Score once after the final gate

`trr0010_score.score_after_gate` is implemented and tested as a library
call, but `scripts/trr0010_score.py` has no production CLI. The final small
glue should expose a command equivalent to:

```bash
python3 scripts/trr0010_score_cli.py \
  --repository-root "$ROOT" \
  --freeze "$ROOT/experiments/TRR-0010/evaluation/public_freeze.json" \
  --truth-descriptor "$TRUTH_DESCRIPTOR" \
  --truth-sidecar "$TRUTH_SIDECAR" \
  --frequency-reference-B0 "$FREQUENCY_B0" \
  --frequency-reference-B1 "$FREQUENCY_B1" \
  --output "$ROOT/experiments/TRR-0010/evaluation/score.json"
```

The wrapper must validate the truth descriptor before opening the sidecar, then
call `score_after_gate` exactly once with both B0 and B1 references. The
scorer already requires all six methods in all four cells, keeps domains and
targets paired, and computes separate source-record bootstrap summaries. It
must not pool the two frequency banks or use truth for routing or method
selection.

## Concrete missing pieces

1. TRR-0010 128+128 identity-only selector/panel binding.
2. TRR-0009 producer-to-TRR-0010 observation/panel/capture repackager.
3. TRR-0010 truth descriptor/curator wrapper around
   `validate_truth_descriptor`.
4. A production scoring CLI/truth-loader wrapper around
   `trr0010_score.score_after_gate`.

The registration, truth-free runner, public gate/freeze, and scorer library
are present; no broader evaluation framework is needed.
