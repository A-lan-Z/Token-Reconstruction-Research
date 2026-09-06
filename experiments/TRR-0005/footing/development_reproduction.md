# TRR-0005 development reproduction and archive checklist

This document records completed public development evidence. It does not select or inspect the fresh holdout or open evaluator truth. The report and manifest builder reads compact JSON receipts and learning curves only.

The retained TRR-0004 fit-bank diagnostic measured 99.9935676% token accuracy and 1,192/1,200 exact records on the original-like bank, versus 90.4447178% and 3/1,200 on the enriched bank. Initial identity recovery was approximately 31% on both banks. Final fit streams reached 100% on all eight arms, but this does not imply that every earlier public-validation selected state is perfect. Public positionwise selection compares affine with trained diagonal within each distribution only; it never compares against causal. Both distributions selected trained diagonal, so causal-versus-best-positionwise duplicates causal-versus-trained-diagonal and is not independent evidence.

## Regenerate the task-local report and manifest

From the TRR-0005 worktree:

```bash
.venv-trr0005/bin/python experiments/TRR-0005/footing/build_development_evidence.py
```

The builder writes `coordination/results/TRR-0005.md` and `experiments/TRR-0005/manifest.json`. It does not read safetensors or model weights.

## Recorded development commands

These commands are copied from the corresponding execution receipts. They are historical evidence; rerun them only under the root coordinator's resource and data-access decisions.

Corpus preparation (`experiments/TRR-0005/corpus_run/run_receipt.json`):

```bash
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python scripts/trr0005_prepare_public_corpus.py --execute --design experiments/TRR-0005/corpus_design.json --trr4-lengths experiments/TRR-0004/alpaca_split_plan.json --tokenizer /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --legacy-fit-labels /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors --output-root experiments/TRR-0005/corpus
```

Public activation capture (`experiments/TRR-0005/public_activation_v1/launch.json`):

```bash
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python scripts/trr0005_prepare_public_activations.py --mode capture --corpus-plan experiments/TRR-0005/corpus/corpus_plan.json --original-artifact ../TRR-0004/outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors --original-records ../TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json --common-validation-artifact ../TRR-0004/experiments/TRR-0004/fit/adapter_v2/validation_mixed_cut4.safetensors --common-validation-records ../TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json --embedding-table ../../outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors --output-root experiments/TRR-0005/public_activation_v1 --enriched-activation-artifact outputs/TRR-0005/enriched_fit_cut4.safetensors --model /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --device cuda --batch-records 8 --cut-depth 4 --min-free-gpu-gib 8 --max-reserved-gpu-gib 8 --max-host-rss-gib 16
```

The capture receipt records the fixed batch-8 × 192 bit-exact path. The unpadded batch-1 diagnostic is retained as excluded numerical evidence and must not be substituted for the captured path.

Corpus coverage recorded 11.474 million extra public token occurrences scanned. All controlled placements occur at token positions greater than or equal to 128. The sampler recorded 1.536 million draws per arm, with one replacement step containing 200 repeated draws. The eight decoder fits totalled about 801.7 seconds wall time before other preparation and qualification costs. Keep corpus preparation/coverage, activation capture, fitting, diagnostics, qualification, and archived prediction qualification as separate cost buckets.

Joint-fit interface and qualifier templates are in `experiments/TRR-0005/joint_fit_interface.md`; the completed original six-fit receipt is `experiments/TRR-0005/joint_fit_v1/run_evidence.json`, and the completed qknorm repair receipt is `experiments/TRR-0005/joint_fit_qknorm_v1/run_evidence.json`. Their receipts and curves are the source of all development selection/timing values in the generated manifest. The qknorm repair uses the predeclared cosine-scale-4 score rule and supplies the two repaired causal contenders for the eventual method freeze.

Public H-only attention diagnostic (`experiments/TRR-0005/attention_diagnostic_execution.json`):

```bash
.venv-trr0005/bin/python scripts/trr0005_attention_diagnostic.py --validation-manifest experiments/TRR-0005/public_activation_v1/original_manifest.json --state original=experiments/TRR-0005/joint_fit_v1/original/affine_causal_h_attention128/selected.safetensors --state enriched=experiments/TRR-0005/joint_fit_v1/enriched/affine_causal_h_attention128/selected.safetensors --output experiments/TRR-0005/attention_diagnostic.json --hash-inputs
```

The qknorm public H-only diagnostic (`experiments/TRR-0005/attention_diagnostic_qknorm_v1_execution.json`) was run with:

```bash
.venv-trr0005/bin/python scripts/trr0005_attention_diagnostic.py --validation-manifest experiments/TRR-0005/public_activation_v1/original_manifest.json --state original=experiments/TRR-0005/joint_fit_qknorm_v1/original/affine_causal_h_attention128/selected.safetensors --state enriched=experiments/TRR-0005/joint_fit_qknorm_v1/enriched/affine_causal_h_attention128/selected.safetensors --output experiments/TRR-0005/attention_diagnostic_qknorm_v1.json --hash-inputs
```

## Compact evidence to retain

- Setup/preflight, corpus design/plan/run, source-pool reservations, and coverage diagnostics.
- Public capture raw/normalized receipts, launch/preflight, manifests, record metadata, source bindings, output bytes/SHA, and the excluded equivalence diagnostic.
- Joint-fit design, memory preflight, sampler receipts/schedules, pretraining diagnostics, learning curves, selected states, arm timing, and both original/qknorm `run_evidence.json` files.
- Preserved V1 failure, successful V2 qualification, qknorm amendment/qualification and fit, the archived Finance-128 prediction qualification, and both attention diagnostic/result/receipt sets.
- Frozen method selection, method registration, panel descriptor, observation/prediction/timing receipts, truth-binding descriptor, freeze receipt, and scorer output after fresh evaluation exists.

Retain the task-owned raw H and fit tensors locally when needed for reproduction. The compact Git handoff should carry receipts, metadata, selected states, predictions, and hashes, while leaving the largest raw tensors in the archive with their recorded paths and reproduction commands. Keep the evaluator truth sidecar outside reconstruction and frozen-public roots; copy only the label-free binding descriptor into the frozen output root, and require the freeze receipt to cover its exact path, bytes, and SHA before the scorer loads truth.

The four fresh-result slots are Pile/P0, Pile/synthetic-LoRA, Finance/P0, and Finance/synthetic-LoRA, each with 128 paired sources and eight method-cell artifacts. Their result fields remain empty until the frozen panel, all 32 prediction/timing artifacts, and truth-gated scoring complete. The practical margins are already frozen in the decision plan: extra-H 0.5 percentage points token and 5 percentage points exact; enrichment 2 percentage points token and 5 percentage points exact.

For replay, check out the exact phase commit recorded in the relevant receipt, then restore compact evidence from the archive. Do not treat public-panel curves or fit-stream metrics as fresh holdout outcomes.


## Fresh producer resume and archive/reproduction

The resumed fresh sequence ran at `da82f6cac45e09ae83452198344c547553cb4433`.
The source-free method freeze selected the best public positionwise contender
mechanically from the recorded fit evidence: trained diagonal for both
`original` and `enriched`. The method-freeze digest was
`96330c8b935ff33ab3f69600c4456e556f901084ad2958e49287d2d329caa422`; the
selection plan and panel use that digest throughout. The producer receipts
under `experiments/TRR-0005/producer_execution_v1/` preserve the expanded
absolute argv, environment, start/end times, return code, resource file, and
source hash for each phase.

Use the worktree root below when replaying the commands. The declared source
pool paths are the pinned local caches; replay must retain the half-open
reserved ranges and seed from the selection plan.

```bash
REPO=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005
PYTHONPATH=src:scripts
PYTHON="$REPO/.venv-trr0005/bin/python"
TOKENIZER=/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
PILE=/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow
FINANCE0=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow
FINANCE1=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow
```

The method preselection was source-free with the repaired causal root bound by
the public H-only amendment:

```bash
$PYTHON scripts/trr0005_bind_methods.py preselect \
  --repository-root "$REPO" \
  --decision-plan experiments/TRR-0005/decision_plan.json \
  --fit-root experiments/TRR-0005/joint_fit_v1 \
  --causal-fit-root experiments/TRR-0005/joint_fit_qknorm_v1 \
  --attention-amendment experiments/TRR-0005/qk_score_repair_amendment_v1.json \
  --code-commit da82f6cac45e09ae83452198344c547553cb4433 \
  --output-freeze experiments/TRR-0005/method_freeze.json \
  --output-selection experiments/TRR-0005/public_validation_selection.json
$PYTHON scripts/trr0005_produce_confirmation.py preflight \
  --repository-root "$REPO" \
  --method-freeze experiments/TRR-0005/method_freeze.json \
  --decision-plan experiments/TRR-0005/decision_plan.json
```

The exact fresh selection command was:

```bash
$PYTHON scripts/trr0005_produce_confirmation.py select \
  --repository-root "$REPO" \
  --method-freeze "$REPO/experiments/TRR-0005/method_freeze.json" \
  --decision-plan "$REPO/experiments/TRR-0005/decision_plan.json" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE" \
  --finance-arrow "$FINANCE0" "$FINANCE1" \
  --output "$REPO/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json"
```

This wrote exactly 128 records per domain with seed 5005 from Pile
`[7000,10000)` and Finance `[12000,20000)`. The selection receipt is
`producer_execution_v1/select.json`; it completed with return code 0 and
`truth_opened=false`.

The first capture attempt is preserved in
`producer_execution_v1/capture.json` with return code 1. It failed closed on
the create-only manifest-root collision:

```text
TRR-0005 producer error: capture manifest root is create-only and already exists: .../experiments/TRR-0005/fresh_confirmation_v1
```

The selection plan and method choices were retained. The successful retry is
`producer_execution_v1/capture_v2.json` and uses a separate create-only raw
root and `fresh_confirmation_v1/panel_capture_v2` manifest root:

```bash
$PYTHON scripts/trr0005_produce_confirmation.py capture \
  --repository-root "$REPO" \
  --method-freeze "$REPO/experiments/TRR-0005/method_freeze.json" \
  --decision-plan "$REPO/experiments/TRR-0005/decision_plan.json" \
  --selection-plan "$REPO/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE" \
  --finance-arrow "$FINANCE0" "$FINANCE1" \
  --model-snapshot "$TOKENIZER" \
  --lora-config "$REPO/outputs/TRR-0002/public-calibration/generation.json" \
  --lora-update "$REPO/outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors" \
  --raw-root "$REPO/outputs/TRR-0005/fresh_confirmation_capture_v2" \
  --manifest-root "$REPO/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2" \
  --device cuda
```

The retry preserved the accepted padded 8-by-192 forward geometry and stored
the first 128 positions in the compact observations. The four compact
observation files are approximately 67 MiB each and their recorded
path/size/SHA-256 descriptors are:

- `experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations/finance__public_base.safetensors` — 67,257,168 bytes — `da033e05362cf4731d7132c066777102cc2569d5ff2119785bd6086ce9fe8eb9`.
- `experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations/finance__public_lora_2601.safetensors` — 67,257,176 bytes — `d2fdda52f9276a8e3864334f05b36f3bc431e8828428deb0abcf273dfba67dc6`.
- `experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations/pile__public_base.safetensors` — 67,257,160 bytes — `8faea0d80e52fe1a56827f249aaddc8205cfcd88b24a5ca9a3d6030b7c8d7e64`.
- `experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations/pile__public_lora_2601.safetensors` — 67,257,168 bytes — `864ef938da0bd5991cf6b016f560a2320df6252beb09c455d17132ea144281b3`.

The external padded capture files used to regenerate those compact files
are under `outputs/TRR-0005/fresh_confirmation_capture_v2/`. Their recorded
path/size/SHA-256 descriptors are preserved in `panel_capture_v2/capture.json`:
finance/P0 `101032760` bytes (`5adebb05b5c1f7411f75a9e5d0ccb91fa46bf64799542950eca0f846aa714691`), finance/LoRA `101032760` bytes (`b6f6fe342505d6f92a3e7156e93617b6496b403e3444daf5c3549464ee4aeeac`), Pile/P0 `101032752` bytes (`c9925ac106ace70ca1608ef78ab652bfb64932001ae1a15d4a2efbae48629aeb`), and Pile/LoRA `101032760` bytes (`c32e1f4dab12a03d7d1773011bc0d2715e1586f180364a0f71644b42e7655341`). The larger public fit H artifact remains external at
`outputs/TRR-0005/enriched_fit_cut4.safetensors`, 947,176,760 bytes,
SHA-256 `191cb77dae8d002402bcf3f126a20c5d8d34111a6e6871d66507503ca6725a99`.

Panel-bound method registration ran only after the successful panel and plan:

```bash
$PYTHON scripts/trr0005_bind_methods.py register \
  --repository-root "$REPO" \
  --method-freeze "$REPO/experiments/TRR-0005/method_freeze.json" \
  --panel "$REPO/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json" \
  --selection-plan "$REPO/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json" \
  --output-registration "$REPO/experiments/TRR-0005/fresh_confirmation_v1/method_registration.json" \
  --decision-plan "$REPO/experiments/TRR-0005/decision_plan.json" \
  --public-validation-selection "$REPO/experiments/TRR-0005/public_validation_selection.json"
```

The frozen producer's truth path reuses each domain's labels, masks, and
positions for both conditions. The direct safetensors writer rejects this
shared storage; the narrow helper `scripts/trr0005_truth_alias_adapter.py`
patches only the producer module's save symbol while calling its `truth` CLI,
cloning tensors at serialization and restoring the symbol in `finally`. Its
provenance and code hash are recorded in
`fresh_confirmation_v1/producer_execution_v1.json`:
`88bcf1115240904471cbc9bb571b739c3f331744b5d7ef19bfca9eff50990d7e`.
The synthetic-only test receipt is
`fresh_confirmation_v1/truth_alias_adapter_test_receipt.json`; its exact
command was
`PYTHONPATH=.:src:scripts .venv-trr0005/bin/python -m pytest -q tests/test_trr0005_truth_alias_adapter.py`
and its result was `1 passed in 0.86s`. The test first demonstrates raw alias
rejection, then invokes the frozen producer CLI through the helper and checks
every serialized key, tensor value/dtype/shape, and metadata value.

Producer-only truth preparation then used this command. It writes the truth
sidecar and manifest outside the reconstruction root; neither is copied into
this checkout and evaluator opening remains gated on the complete prediction
freeze:

```bash
$PYTHON scripts/trr0005_truth_alias_adapter.py truth \
  --repository-root "$REPO" \
  --method-freeze "$REPO/experiments/TRR-0005/method_freeze.json" \
  --decision-plan "$REPO/experiments/TRR-0005/decision_plan.json" \
  --public-validation-selection "$REPO/experiments/TRR-0005/public_validation_selection.json" \
  --selection-plan "$REPO/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json" \
  --panel "$REPO/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE" \
  --finance-arrow "$FINANCE0" "$FINANCE1" \
  --truth-output /tmp/trr5/fresh_confirmation_v1.truth.safetensors \
  --truth-manifest /tmp/trr5/fresh_confirmation_v1.truth.manifest.json
```

The truth execution receipt is
`fresh_confirmation_v1/producer_execution_v1.json`: return code 0, CPU-only,
5.592 seconds, peak RSS 1,238,659,072 bytes, host available memory
22.361 GiB before and 22.354 GiB after, and no failed attempts. It records the
external truth sidecar as 1,139,104 bytes with SHA-256
`1375a957254314814fa892d1289481d7d20905d9805594f15fbf38f79994adbe` and the
binding manifest as 26,782 bytes with SHA-256
`dec7115b9b44a88cba47f27a84035b6e012be6a1670acba95f758bbd6bc4e57e`. These
are binding metadata only; do not open or copy the truth tensors during
reconstruction archival.

For reproduction, check out the exact phase commit recorded by the receipts
(`da82f6cac45e09ae83452198344c547553cb4433`) before replaying freeze,
prediction, or scorer code. The helper was intentionally kept as a small
producer-side adapter and must be committed in the subsequent source commit;
use its recorded hash to verify that commit. Restore compact evidence first:
method freeze and public selection, fresh selection plan, panel and observation
manifests, method registration, producer receipts, selected joint decoder
states under `joint_fit_v1/` and `joint_fit_qknorm_v1/`, and completed
prediction/timing manifests under
`experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/`. Restore the
external 947 MB H and four padded capture artifacts from their recorded paths
and hashes when raw regeneration is needed. Keep the evaluator truth files at
their external `/tmp/trr5` locations (or regenerate them with the producer-only
command after the same panel/plan freeze); copy only label-free binding
metadata required by the post-gate scorer.

The preserved narrow failures are `producer_execution_v1/capture.json`
(create-only capture-root collision, return code 1) and the frozen truth
serialization alias condition covered by the raw-writer rejection assertion.
The successful `capture_v2.json`, `register.json`, truth receipt, and synthetic
test receipt are retained alongside them. These failures are evidence of the
original execution path and must not be silently replaced by a different
sample rule, geometry, method choice, or evaluator access order.
