# TRR-0006 prediction runner contract

This task-local contract runs the two already frozen enriched methods after a
separate registration and producer observation manifest have been frozen. It
never selects sources, creates truth, loads source text/token IDs, scores
predictions, fits a model, or changes the shared TRR-0005 contract.

The registration schema is
`token-reconstruction.trr0006-frozen-pair-prediction-registration.v1` with
status `FROZEN_PREDICTION_REGISTRATION`. It must contain:

* one positive `records_per_domain` value (the current plan may choose 1024 or
  1536; tests use 1024/1536 and the runner does not silently choose either);
* exact cell order `[pile__public_base, pile__public_lora_2601,
  finance__public_base, finance__public_lora_2601]` and exact method order
  `[enriched__affine_causal_h_attention128,
  enriched__affine_trained_diagonal_attention128]`;
* geometry `capture_batch_records=8`, `capture_sequence_tokens=192`,
  `stored_sequence_tokens=128`, `scored_sequence_tokens=128` (stored width
  including BOS), `scored_post_bos_tokens=127` (metric denominator),
  `hidden_size=2048`, and `chunk_records=8`;
* the exact normalized public E binding (1050673488 bytes,
  SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`),
  both exact published selected state paths and hashes, the frozen code
  bindings, and the producer observation-manifest binding;
* one warmup and one measured call per record with exact warmup/measured ID
  equality; and
* the fail-closed GPU/host guard values and a task-owned output root.

The runner requires code bindings for exactly `scripts/trr0006_run_predictions.py`,
`scripts/trr0006_prediction_contract.py`, and
`src/token_reconstruction/trr0005_joint_decoder.py`; these cover the executed
runner, its contract, and the imported `load_decoder_state` numerical
implementation.  The retained TRR-0005 runner is reference evidence only.
The two state files are pinned to scientific source commit
`da82f6cac45e09ae83452198344c547553cb4433` in the reviewed publication tree
`3a7e8f579e713c3e41d02639237042ca26fd019b`.  The post-score maintenance
commit `1dba67a8dc75844727866cb4273da28a311df216` is recorded as lineage only;
the fixture remains the inference-equivalence evidence.

The producer manifest schema is
`token-reconstruction.trr0006-public-observation-manifest.v1` with status
`FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH`. Each cell carries one source-record
pairing digest and one observation file binding. The observation file must
contain exactly `activations`, `attention_mask`, and `position_ids`, with
shape `[records_per_domain,128,2048]` BF16 for H and `[records_per_domain,128]`
for the sidecars. Every stored row must have all 128 positions valid, a binary
mask, and position IDs exactly `0..127`; right-padding is a different
estimand and fails closed. Its metadata must retain the qualified `8×192`
public full-forward provenance and `scored_post_bos_tokens=127`, including
`producer_only_lora` for the LoRA cells.
The runner hashes the manifest and every observation before loading a decoder,
then uses `safe_open.get_slice` in eight-record chunks. It never materializes
all cells or the full panel.

The runner pins and records the fixture-matched numerical recipe:
CPU intra-op/inter-op threads `8/32`, float32 matmul precision `highest`, CUDA
matmul TF32 disabled, cuDNN TF32 enabled as in the qualified environment, and
no autocast.  For each row the numerical boundary is the validated TRR-0005 path: CPU BF16
H is transferred as CUDA FP32, the mask is CUDA bool, the full normalized F32 E
table is used, the frozen decoder is run in inference mode, and argmax IDs are
normalized with BOS `128000` and suffix padding `-1`. No autocast, reduced E,
public-prefix call, candidate array, or batch-level decoder substitution is
used. The runner loads one state at a time, separates process-start E/state
load timing from per-record warmup/measured timing, and synchronizes CUDA after
each call.

On success it writes eight create-only prediction artifacts, eight adjacent
`.run.json` receipts, `predictions.json`, `timings.json`, and
`run_manifest.json` under the registration output root. The task-local
`trr0006_freeze_pair` gate consumes these exact files and returns prediction
IDs keyed by `(cell_id, method_id)` in the same order used by
`trr0006_score_pair`; it rehashes the public matrix before truth. Receipts bind the
registration, observation, state, E, code, geometry, chunk count, predicted-ID
digest, and measured decisions. Any incomplete chunk, duplicate/reordered
cell, mismatched source-pair digest, invalid sidecar, resource guard failure,
state/E/code hash mismatch, warmup/measured mismatch, or existing output
causes a create-only `failure.json`; no partial score is opened.

The bounded fixture and its raw-192-to-trimmed-128 exact-ID result are retained
under `experiments/TRR-0006/fixture_equivalence/`. The final 1536-record observations, truth gate, and scorer
remain owned by the coordinating producer/scorer agents; this task-local builder
creates the registration after those observations are frozen.

After the producer writes the frozen observation manifest and root has committed
the complete executable, build the main registration with the exact frozen plan,
completed source selection, and current producer manifest. The builder verifies
the plan and source-selection SHA, all source-free observation files, the
published state/E bindings, and current prediction plus scoring code hashes
before creating the registration:

```text
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006
env PYTHONPATH=.:src:scripts python3 scripts/trr0006_build_prediction_registration.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006 \
  --plan experiments/TRR-0006/decision_plan.json \
  --source-selection experiments/TRR-0006/source_selection.json \
  --observation-manifest experiments/TRR-0006/panel_capture_v1/observations.json \
  --normalized-public-E /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors \
  --output experiments/TRR-0006/prediction_registration.json \
  --output-root experiments/TRR-0006/predictions
```

Run only after root has frozen the registration and resource window:

```text
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 scripts/trr0006_run_predictions.py \
  --registration experiments/TRR-0006/prediction_registration.json \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006
```

After all eight prediction outputs are complete, the task-local driver checks
the public matrix again and then performs the one private-label load. The
binding manifest and sidecar paths are supplied explicitly; the pre-gate phase
reads only the binding JSON metadata and records the opaque sidecar hash.
`--truth-binding` and `--truth-path` must point outside this checkout and the
prediction output root:

```text
env PYTHONPATH=.:src:scripts python3 scripts/trr0006_freeze_score.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006 \
  --plan experiments/TRR-0006/decision_plan.json \
  --source-selection experiments/TRR-0006/source_selection.json \
  --registration experiments/TRR-0006/prediction_registration.json \
  --observation-manifest experiments/TRR-0006/panel_capture_v1/observations.json \
  --freeze-receipt experiments/TRR-0006/predictions/freeze_receipt.json \
  --truth-binding /private/trr0006/truth.binding.json \
  --truth-path /private/trr0006/truth.safetensors \
  --result experiments/TRR-0006/result.json \
  --report experiments/TRR-0006/report.md \
  --manifest experiments/TRR-0006/manifest.json \
  --execution-receipt experiments/TRR-0006/execution_receipt.json
```

The driver fails closed before touching the sidecar when the public receipt,
registration, plan, source selection, observations, or any prediction/timing
artifact is incomplete or changed. It writes result, report, manifest, and
execution receipt files create-only after scoring all four cells.
