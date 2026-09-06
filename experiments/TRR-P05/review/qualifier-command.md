# TRR-P05 qualifier handoff

This command is for the first GPU diagnostic cell only. It must run from the
clean P05 source head `09b6fe17e4573ce6b18943d5a2811a163d1b978f`, with the
model-free sample ledger already frozen at
`/tmp/trr-p05/experiments/TRR-P05/runtime/public-sample-r1/sample_index.json`.
The command reads only P04 public observations, cached public candidates and
teacher rows, the public normalized table, the recorded public affine state,
and stored public P04 S/H/D checkpoints. It never opens evaluator truth or a
target update and has no optimizer path.

## Exact qualifier command

```bash
cd /tmp/trr-p05
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python3 scripts/trr0004_p05_diagnostic.py diagnose \
  --correction-observations /tmp/trr-p04/experiments/TRR-P04/runtime/public-pool-capture-r1/correction_cut4.safetensors \
  --correction-records /tmp/trr-p04/experiments/TRR-P04/runtime/public-pool-capture-r1/correction_records.json \
  --replay-observations /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors \
  --replay-records /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/affine_fit_records.json \
  --teacher-evidence /tmp/trr-p04/experiments/TRR-P04/runtime/teacher-qualification-r3/teacher_evidence.safetensors \
  --schedule 1737=/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/seed-1737/position_schedule.safetensors \
  --schedule 2711=/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/seed-2711/position_schedule.safetensors \
  --sample-index /tmp/trr-p05/experiments/TRR-P05/runtime/public-sample-r1/sample_index.json \
  --candidate-preparation /tmp/trr-p04/experiments/TRR-P04/runtime/candidate-preparation-r2/candidate_preparation.safetensors \
  --embedding-table /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors \
  --state-manifest /tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/selected_state_manifest-r3.json \
  --affine-initial /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/fit_large_v1/historical_affine_ce_no_vocab_bias.safetensors \
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/qualifier-r1 \
  --device cuda --mode qualify --threads 4 --interop-threads 1
```

The output root is create-only and must be fresh. Wrap the process in the
approved external fail-closed watchdog with a 900 second timeout, minimum
8 GiB free GPU memory, maximum 6 GiB PyTorch reserved GPU memory, and maximum
16 GiB whole-process RSS. Preserve any failure receipt and do not retry with
changed batching or geometry.

The qualifier chooses one batch from the frozen sample ledger before model
values are computed: maximum teacher-row count, then maximum selected-row
count, then lower seed, then lower step. The current schedule metadata predicts
seed `2711`, step `1999`, with 512 selected positions and four teacher-active
positions. It runs the affine initial-function forward reference and that
selected D state forward sample, followed by the selected D backward
components on that exact batch. The production projection chunk is 512 and
record batch size is 8. No truth scoring is part of this command.

## Geometry and resource estimate

The public normalized table is `[128256, 2048]` FP32: 1,050,673,728 bytes
(about 0.978 GiB) in the source artifact and one device copy. The largest
full-vocabulary logit chunk is `[512, 128256]` FP32 (about 0.245 GiB); the
student state is about 26 MB. P04's closest teacher qualification measured
5,368,709,120 bytes reserved GPU and 3,669,938,176 bytes host RSS. The P05
qualifier adds the fixed table/chunk and backward graph but remains guarded by
6 GiB reserved GPU, 8 GiB free GPU, and 16 GiB host RSS limits. Expected wall
cost is a few minutes, with actual phase timings and sampled peaks recorded in
the diagnostic receipt.

## External wrapper

For the whole-process guard, run the existing fail-closed wrapper with the
same child command above (the wrapper's child is the final `python3 ...`
argument after `--`):

```bash
cd /tmp/trr-p05
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/trr-p05/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python3 /tmp/trr-p05/scripts/trr_p04/resource_watchdog.py \
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/watchdog-qualifier-r1 \
  --timeout-seconds 900 --poll-seconds 0.5 --max-rss-bytes 17179869184 \
  --min-available-bytes 10737418240 --cwd /tmp/trr-p05 --label TRR-P05-qualifier -- \
  python3 /tmp/trr-p05/scripts/trr0004_p05_diagnostic.py diagnose \
  --correction-observations /tmp/trr-p04/experiments/TRR-P04/runtime/public-pool-capture-r1/correction_cut4.safetensors \
  --correction-records /tmp/trr-p04/experiments/TRR-P04/runtime/public-pool-capture-r1/correction_records.json \
  --replay-observations /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors \
  --replay-records /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/affine_fit_records.json \
  --teacher-evidence /tmp/trr-p04/experiments/TRR-P04/runtime/teacher-qualification-r3/teacher_evidence.safetensors \
  --schedule 1737=/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/seed-1737/position_schedule.safetensors \
  --schedule 2711=/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/seed-2711/position_schedule.safetensors \
  --sample-index /tmp/trr-p05/experiments/TRR-P05/runtime/public-sample-r1/sample_index.json \
  --candidate-preparation /tmp/trr-p04/experiments/TRR-P04/runtime/candidate-preparation-r2/candidate_preparation.safetensors \
  --embedding-table /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors \
  --state-manifest /tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/selected_state_manifest-r3.json \
  --affine-initial /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/fit_large_v1/historical_affine_ce_no_vocab_bias.safetensors \
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/qualifier-r1 \
  --device cuda --mode qualify --threads 4 --interop-threads 1
```

The wrapper samples whole-process RSS and host `MemAvailable`; the child
samples CUDA free and reserved memory at each diagnostic phase. These are
separate guards and are reported separately in their receipts.
