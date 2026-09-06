# TRR-0005 final 32-cell prediction launch plan

Status: prepared and waiting for root's explicit final-run grant. This file is a
command/resource handoff; it does not reserve sources, open truth, or launch a
process.

## Exact launch command

Run from the TRR-0005 worktree after `method_registration.json` is frozen and
its `code_commit` equals executable HEAD:

```text
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-trr0005/bin/python scripts/trr0005_run_predictions.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --panel experiments/TRR-0005/fresh_confirmation_v1/panel.json \
  --selection-plan experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json \
  --registration experiments/TRR-0005/fresh_confirmation_v1/method_registration.json \
  --fit-root experiments/TRR-0005/joint_fit_v1 \
  --causal-fit-root experiments/TRR-0005/joint_fit_qknorm_v1 \
  --output-root experiments/TRR-0005/fresh_confirmation_v1/predictions_v1 \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --reference experiments/TRR-0004/evidence/comparators/round001_teacher.py \
  --lens experiments/TRR-0004/evidence/comparators/public_a1_lens.pt \
  --device cuda \
  --minimum-free-gib 8 \
  --maximum-reserved-gib 6 \
  --maximum-rss-gib 16 \
  --minimum-host-available-gib 10 \
  --max-seconds 1500
```

The frozen registration supplies and rehashes the shared normalized public E
for all methods and the public P0 checkpoint/config for A2. The command's
`--model-snapshot`, `--reference`, and `--lens` arguments select the registered
A2/P0 and retained-A1 runtime; no truth or source text is accepted by this
driver.

The exact method set is the contract's eight IDs, in order:

```text
historical_alpaca_a1
frozen_a1_a2_k256
original__joint_full_affine
original__affine_causal_h_attention128
original__affine_trained_diagonal_attention128
enriched__joint_full_affine
enriched__affine_causal_h_attention128
enriched__affine_trained_diagonal_attention128
```

The driver runs four cells by eight methods (32 predictions), one 128-token
record at a time, with exactly one warmup and one measured call per record.
A2 keeps its fixed proposal K=512 and selection K=256 rules and persists no
candidate arrays.

## Resource and stop plan

The archived one-record A2 qualification measured 1.073431 seconds warmup and
0.865568 seconds measured, or 1.938998 seconds for both calls. The 512 records
in four 128-record cells therefore forecast 992.77 seconds (16.55 minutes) of
A2 timed adapter work. The 1500-second process deadline leaves about 51% wall
margin for method startup, synchronization, artifact I/O, and normal variation.

The largest qualification GPU peak was 3,185,573,888 reserved bytes (2.967
GiB), with process RSS 5,546,471,424 bytes (5.166 GiB). Keep the live guards at
8 GiB free GPU, 6 GiB maximum reserved GPU, 16 GiB maximum RSS, and 10 GiB
minimum host availability. Run exclusively with CPU8 environment variables;
no test, model, training, tensor-hash, or competing evaluator process may run
beside it. Stop on any guard, allocator, thermal, ID-repeat, registration, or
artifact error and retain the driver's create-only `failure.json`; do not alter
batching or retry with a changed method recipe.

## Freeze/scorer handoff filenames

The driver writes these descriptor manifests after all 32 cells complete:

```text
experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/predictions.json
experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/timings.json
experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/run_evidence.json
```

Freeze discovers the 32 per-cell receipts at:

```text
experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/<style>/<condition>/<method_id>.run.json
```

and the corresponding prediction artifacts at the same cell/method path with
`.safetensors` in place of `.run.json`. The scorer receives `predictions.json`
and `timings.json`; freeze and scorer also use the panel, selection plan, and
method registration paths above. Truth remains unopened until the complete
matrix, timing receipts, and freeze gate are validated.
