# TRR-0004 fresh confirmation reproduction

This file records the post-freeze command chain and resource envelope. It is
an execution plan only: no TRR-0004 records have been selected, no fresh
observations or truth have been generated, and no model has been loaded by
this preparation step.

Root the commands at the TRR-0004 worktree, and run them only after the five
method marker is committed and the source freeze is complete.

```bash
ROOT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004
PY=$ROOT/.venv-trr0004/bin/python
SNAP=/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
PILE=/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow
F0=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow
F1=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow
OUT=$ROOT/experiments/TRR-0004/fresh_confirmation_v1
export PYTHONPATH=.:src:scripts
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

The marker file
`experiments/TRR-0004/fresh_confirmation_method_freeze.json` must be
present and status `FROZEN_METHOD_STATES` before this first command. The
producer's default exclusions include the TRR-0003 panel and plan, the
TRR-0003 public fit128 and validation metadata, the TRR-0004 adapter-v2 fit,
validation, manifest, and Alpaca split declarations, plus the public TRR-0002
Pile and Finance ledgers. The selector reads only public identity, content
hash, and source-index metadata from these files, rejects duplicate truncated
sequences across both styles, and takes the first 16 eligible rows per style.

## 1. Select public records

```bash
$PY "$ROOT/scripts/trr0004_produce_confirmation.py" select \
  --repository-root "$ROOT" \
  --prospective-plan "$ROOT/experiments/TRR-0004/fresh_confirmation_plan.json" \
  --method-freeze "$ROOT/experiments/TRR-0004/fresh_confirmation_method_freeze.json" \
  --tokenizer "$SNAP" \
  --pile-arrow "$PILE" \
  --finance-arrow "$F0" \
  --finance-arrow "$F1" \
  --output "$OUT/selection_plan.json"
```

This is CPU/public metadata work. It must finish before any capture process
is launched, and it writes no token IDs or source text.

## 2. Capture the four paired public observation cells

```bash
timeout --signal=TERM 600s "$PY" "$ROOT/scripts/trr0004_produce_confirmation.py" capture \
  --repository-root "$ROOT" \
  --selection-plan "$OUT/selection_plan.json" \
  --tokenizer "$SNAP" \
  --pile-arrow "$PILE" \
  --finance-arrow "$F0" \
  --finance-arrow "$F1" \
  --model-snapshot "$SNAP" \
  --lora-config "$ROOT/experiments/TRR-0002/configuration-search/public-pile/generation.json" \
  --lora-update /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors \
  --output-root "$OUT/panel_capture" \
  --device cuda
```

The capture uses one fixed batch geometry, 8 records by 192 tokens, and
qualifies the largest Finance128 slice first. Its guard is at least 8 GiB
free CUDA memory, at most 8 GiB reserved, and at most 16 GiB host RSS. The
public model checkpoint and the LoRA update are used only in this preparation
role; the LoRA weights are unavailable to reconstruction.

## 3. Prepare the separate public truth sidecar

```bash
"$PY" "$ROOT/scripts/trr0004_produce_confirmation.py" truth \
  --repository-root "$ROOT" \
  --selection-plan "$OUT/selection_plan.json" \
  --panel "$OUT/panel_capture/panel.json" \
  --tokenizer "$SNAP" \
  --pile-arrow "$PILE" \
  --finance-arrow "$F0" \
  --finance-arrow "$F1" \
  --output-root "$OUT/panel_capture" \
  --truth-sidecar /tmp/trr4/fresh_confirmation_v1.truth.safetensors
```

This is a separate public-data preparation role. The sidecar is outside the
checkout and is not reopened by the producer. It must remain unopened by
prediction and only be passed to the later scorer after its complete matrix
gate succeeds.

## 4. Register the exact five frozen methods

```bash
"$PY" "$ROOT/scripts/trr0004_produce_confirmation.py" register \
  --repository-root "$ROOT" \
  --selection-plan "$OUT/selection_plan.json" \
  --panel "$OUT/panel_capture/panel.json" \
  --binding-spec "$ROOT/experiments/TRR-0004/fresh_confirmation_method_binding_spec.json" \
  --output "$OUT/registration.json"
```

`register` resolves the null `code_commit` in the binding spec to the actual
source-frozen HEAD and rehashes every local state/config/source and external
runtime asset. The A2 predictor must receive the byte-identical public
reference copy
`$ROOT/experiments/TRR-0004/evidence/comparators/round001_teacher.py` via
`--reference`; standalone and Track B predictor processes must not load the
public prefix or use A2 fallback.

The later predictor is invoked once per method in five isolated processes,
each with the same panel, selection plan, registration, and output root. Its
steady interval is one warmed CPU activation transfer through prediction and
predicted IDs back to CPU; one warmup and three measured repetitions are
required, with the first measured output retained only after exact repeat
agreement. The scorer's `freeze` command must validate all 20 artifacts before
its `score` command opens the sidecar.

## Fixed assets and guard calculation

The public normalized embedding table is 1,050,673,488 bytes. The public
checkpoint blob is 2,471,645,608 bytes; the public config blob is 877 bytes.
The capture guard uses 8 GiB free / 8 GiB reserved / 16 GiB RSS and the
predictor uses 8 GiB free / 6 GiB reserved / 16 GiB RSS. The predictor's
largest qualification is Finance128 with A2 K256 and one-record full-float32
logits; the conservative calculation remains the existing 6 GiB reservation
ceiling. No numerical batching substitution is permitted without a separate
bit-exact equivalence check.

The three new decoder states are the fresh affine step-1900 state
`09c5b852...`, attention step-25 state `e9f1b31...`, and MLP step-1200 state
`300d718...`; the retained A1 state is `33b825d...` and is shared by both
comparator rows.
