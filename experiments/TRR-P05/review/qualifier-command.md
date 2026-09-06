# TRR-P05 qualifier handoff

This command is for the first GPU diagnostic cell only. The successful
qualifier ran from executable P05 source commit
`e022b56ff92b4987b88a47418b4df76ccd296cea`, with the
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
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/qualifier-r2 \
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

The public normalized table is `[128256, 2048]` FP32: 1,050,673,152 bytes
(0.978516 GiB); the source artifact is 1,050,673,488 bytes. The largest
full-vocabulary logit chunk is `[512, 128256]` FP32: 262,668,288 bytes
(250.5 MiB). The GRU student has 6,493,697 trainable FP32 parameters
(25,974,788 bytes) and no optimizer state is allocated by this diagnostic.
The gradient cell retains six direct autograd probes (CE, hard, rank, actual
P04 total, hypothetical D total, negative gold margin) plus the weighted
component vector: seven flattened FP32 vectors in total, 181,823,516 bytes.
The static resident GPU lower bound including the table, batch tensors, model
parameters, logit chunk, and these vectors is 1,552,072,736 bytes (1.445 GiB);
saved autograd intermediates and CUDA workspace make the conservative
preflight estimate 3.25 GiB allocated and 4.0 GiB reserved.

The correct closest baseline is the P04 student backward measurement:
2.51 GB allocated and 2.75 GB reserved. The 5,368,709,120-byte figure is the
P04 teacher qualification and is not a student diagnostic baseline. The
successful r2 qualifier measured max allocated 2,868,006,912 bytes, max
reserved 3,005,218,816 bytes, and peak RSS 5,961,986,048 bytes, with guards
passing. The full run uses the same peak geometry sequentially and passed the
same guards; actual full values remain in its immutable receipt.

`load_public_pool` retains the 900 MiB replay activation, 192 MiB correction
activation, and 1,092 MiB concatenated BF16 activation on CPU. The CPU table
is 1,002 MiB and candidate IDs are 68.25 MiB; the K512 proposal payload is not
loaded. The retained input arithmetic is about 3.180 GiB before
interpreter/file-cache overhead; the 16 GiB host guard remains the fail-closed
limit.

## Source and validation bindings

The successful r2 command was run from executable source commit
`e022b56ff92b4987b88a47418b4df76ccd296cea`; the current task head may include
documentation-only descendants. Its runner/module/test hashes are in
`experiments/TRR-P05/runtime/qualifier-r2/diagnostic_receipt.json`. The
focused CPU suite is recorded as 7/7 PASS in
`experiments/TRR-P05/review/implementation-audit.md`. The mode-regression
evidence is the retained r1 failure
(`experiments/TRR-P05/runtime/qualifier-r1/failure.json`, cuDNN backward in
evaluation mode) followed by the unchanged-geometry r2 PASS
(`experiments/TRR-P05/runtime/qualifier-r2/diagnostic_receipt.json` and
`experiments/TRR-P05/runtime/watchdog-qualifier-r2/finish.json`) after
`e022b56`; original argv, timing, and guard records are immutable and are not
rewritten here.

## External wrapper

For the whole-process guard, run the existing fail-closed wrapper with the
same child command above (the wrapper's child is the final `python3 ...`
argument after `--`):

```bash
cd /tmp/trr-p05
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/trr-p05/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python3 /tmp/trr-p05/scripts/trr_p04/resource_watchdog.py \
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/watchdog-qualifier-r2 \
  --timeout-seconds 900 --poll-seconds 0.5 --max-rss-bytes 17179869184 \
  --min-available-bytes 10737418240 --cwd /tmp/trr-p05 --label TRR-P05-qualifier-r2 -- \
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
  --output-root /tmp/trr-p05/experiments/TRR-P05/runtime/qualifier-r2 \
  --device cuda --mode qualify --threads 4 --interop-threads 1
```

The wrapper samples whole-process RSS and host `MemAvailable`; the child
samples CUDA free and reserved memory at each diagnostic phase. These are
separate guards and are reported separately in their receipts.
