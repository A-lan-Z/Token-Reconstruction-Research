# P04 student prediction and timing run plan

The implementation-owned runner is `scripts/trr0004_p04_prediction_runner.py` at
source commit `b7e33b13c3f54aaf46934834a7bc8a712b4a8e87`.  It consumes the
setup-owned paired observation index and activation artifacts, the frozen
selection metadata, the selected-state manifest, and the immutable public
normalized embedding table.  It does not open source text, token IDs, target
weights, teacher artifacts, or evaluator truth.

The selected-state manifest is
`experiments/TRR-P04/runtime/training-r1/selected_state_manifest-r2.json`,
78,480 bytes, SHA256
`5d2c2467d16f991ed540adec72438e74192a19e452e494dc2d6d1d34b3bce95d`.  Its
`states` list is exactly the 8 method/seed checkpoints used by prediction.  Its
`excluded_final_states` list binds the corresponding 8 final checkpoints for
provenance and explicitly marks them as ineligible inputs.  It retains the
training aggregate, finalizer receipt, late-finalization failure evidence,
fit-time sum (1,350.994774457009 seconds), per-seed timing, and the finalizer
serialization time (0.20225112499610987 seconds) with separate labels.  The
training CLI did not capture outer process start/end, so whole-run training
wall time remains null.

The only numeric prediction geometry is fixed at 72 records, padded shape
`[72, 192, 2048]`, full vocabulary 128,256, record batch 8, projection chunk
512, one warmup, and three synchronized measured repeats.  The 12 predeclared
32-token anchor records (384 post-BOS positions) are indexed from the same
loaded full panel.  Their separately timed packed result must be exactly equal
to the corresponding full-panel slice; a mismatch fails closed before any
prediction artifact is accepted.  Prediction JSONL and tie diagnostics are
written create-only per method/seed/condition, followed by a student-only
freeze manifest.  The root joint freeze still requires the setup-owned native
A1+A2 anchor files.

After setup releases `runtime/evaluator-observations-r1` and root releases an
uncontended timing window, use a fresh output directory and this guarded
command (replace `PREDICT_ROOT` with a new runtime directory):

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src \
  python3 scripts/trr_p04/resource_watchdog.py \
  --output-root experiments/TRR-P04/runtime/watchdog/student-predictions-r1 \
  --timeout-seconds 1800 --poll-seconds 0.5 \
  --max-rss-bytes 17179869184 --min-available-bytes 10737418240 \
  --label p04-student-predictions-r1 -- \
  python3 scripts/trr0004_p04_prediction_runner.py \
  --observation-index experiments/TRR-P04/runtime/evaluator-observations-r1/observation_index.json \
  --observation-root experiments/TRR-P04/runtime/evaluator-observations-r1 \
  --selection experiments/TRR-P04/setup/public_selection-r2.json \
  --state-manifest experiments/TRR-P04/runtime/training-r1/selected_state_manifest-r2.json \
  --embedding-table /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors \
  --output-root PREDICT_ROOT \
  --device cuda --warmup-repeats 1 --measurement-repeats 3 \
  --record-batch-size 8 --projection-chunk 512 --threads 4 --interop-threads 1 \
  --implementation-commit b7e33b13c3f54aaf46934834a7bc8a712b4a8e87
```

The wrapper’s host RSS/available-memory receipt is retained beside the runner
receipt.  The runner records wall timing after synchronization, table/state
startup, per-cell full-panel and anchor timing, repeat digests, mask/order
hashes, state method/seed metadata, and all input/output hashes.  No prediction
run has been launched while setup capture or the root timing reservation is
pending.
