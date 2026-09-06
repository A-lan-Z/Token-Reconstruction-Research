# TRR-P05 bounded diagnostic plan

**Status:** `PROPOSED_NO_DIAGNOSTIC_RUN` (no P05 diagnostic or forward
sweep has run). This implementation plan is subordinate to the setup-owned
`experiments/TRR-P05/plan.json`; its sample seed and batch geometry are copied
below so the two ledgers cannot drift.

This task diagnoses the already completed P04 run. It does not change any
P04 state, prediction, score, or truth artifact. The diagnostic will read
only the public correction pool, its public labels, the cached public teacher
evidence, the public normalized embedding table, the recorded P04
initialization, and stored P04 state files.

## Question and fixed inputs

The primary question is whether the D arm acquired the additional relative
teacher ordering signal and, if so, whether that signal points in the same
local direction as token reconstruction. The code will use the P04 objective
functions and constants verbatim: CE is a mean over selected rows; hard
confusion is a mean of per-row means over rows with at least one non-gold
candidate; ranking is the global weighted sum of retained adjacent pair
terms divided by the global sum of pair weights. Ranking ties use the P04
`max(1e-6, .01*sigma_q)` tolerance and token-ID ascending order only to
serialize equal scores. The fixed values are `sigma_q=0.014523349702358246`,
`tie_tolerance=0.00014523349702358246`, hard weight `0.25`, rank weight
`0.25`, hard margin `1.0`, student temperature `1.0`, and gradient clip
norm `1.0`.

The immutable source and public assets are:

* P04 parent/source head: `c3aa40abdf33b5e794b139671fcddf6fd5a2c65e`.
  P04 objective/training provenance is recorded in its source receipt and
  the state manifest; the P05 CLI source hash will be recorded in every
  runtime receipt.
* Correction observations and public labels:
  `/tmp/trr-p04/experiments/TRR-P04/runtime/public-pool-capture-r1/correction_cut4.safetensors`
  and records at the adjacent `correction_records.json`. Geometry is
  `256 x 192 x 2048`, BF16 observations, 45,596 active post-BOS positions.
* Cached public teacher evidence:
  `/tmp/trr-p04/experiments/TRR-P04/runtime/teacher-qualification-r3/teacher_evidence.safetensors`.
  It has 384 rows, K=32 candidate IDs/scores, exactly 256
  `difficult_a1_error` rows and 128 `uniform_audit` rows, and no evaluator
  target data.
* Frozen public candidate table:
  `/tmp/trr-p04/experiments/TRR-P04/runtime/candidate-preparation-r2/candidate_preparation.safetensors`.
  It is used only to bind cached candidates to public correction rows; no
  candidate regeneration is performed.
* Public normalized table:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`.
* Recorded P04 affine initialization:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/fit_large_v1/historical_affine_ce_no_vocab_bias.safetensors`.
  For the initial diagnostic, one affine initial-function reference is built
  directly from this recorded `W,b,s` state; no GRU initial checkpoint is
  recreated. No training step or trajectory reconstruction is performed. The
  affine initialization is required; a missing or changed asset is a
  fail-closed input error.
* P04 state manifest:
  `/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1/selected_state_manifest-r3.json`.
  The diagnostic includes the twelve stored S/H/D checkpoints: selected and
  final for seeds 1737 and 2711 (six states of each kind). It also includes
  one exact affine initial-function reference. P04 affine selected/final
  states remain outside this bounded matrix.
  P04's selected-only evaluation rule and publication results remain
  unchanged. All six final S/H/D files are available in the inherited public
  artifact; any missing or unreadable file is a fail-closed input error.

Setup may replace these absolute paths with byte-identical task-local public
copies. The CLI requires and records every final path and SHA-256, including
the source pool, records, teacher evidence, candidate table, embedding table,
initial affine state, state manifest, and each checkpoint.

## Frozen diagnostic sample

The sample is materialized once before any model load and saved as a
create-only `sample_index.json`. Its source and selection hash are inputs to
the diagnostic receipt.

1. The teacher set is exactly the 384 rows and order in
   `teacher_evidence.safetensors` metadata `rows_json`. Each `(record_id,
   position)` must be unique, active, post-BOS, and in the correction pool.
   The receipt keeps the original `kind` partition: 256 difficult rows and
   128 uniform rows. All 384 rows are evaluated for full-vocabulary accuracy,
   gold margin, candidate agreement, and original ranking loss.
2. The unscored control set is exactly 384 active post-BOS correction
   positions that are absent from the teacher set. Candidates are selected
   by the setup-owned deterministic seed `20260909`, using the exact
   selection ledger and hash convention in `experiments/TRR-P05/plan.json`.
   This produces a
   public same-pool control set with no teacher scores. If setup has already
   materialized a selection ledger, the CLI verifies that ledger against this
   rule and refuses a mismatch.
3. The gradient phase uses the actual immutable P04 schedule batches,
   retaining their 6 replay + 2 correction record composition and sparse
   teacher mask. The predeclared zero-based schedule steps are `[0, 999, 1999, 2999]`
   for each seed; these original P04 checkpoints were fixed from the
   schedule ledger before any P05 diagnostic and are independent of
   diagnostic values.
   Every selected position in those four batches contributes to the CE/hard
   reductions, while only positions present in cached teacher evidence
   contribute to the rank reduction. The exact batch indices, record rows,
   masks, and teacher-row counts are written to the sample index. The
   gradient phase is run at the recorded selected H, selected D, and final D
   checkpoints for each seed (24 cells total). It does not silently substitute
   another batch or checkpoint.

This keeps the full-vocabulary forward sample at 768 positions and the
gradient diagnostic at four actual schedule batches for each of three stored
checkpoints per seed (24 cells total across selected H, selected D, and final D).
Those batches intentionally retain P04's six replay and two correction rows;
no validation, evaluator, target, or P03 holdout rows are silently substituted.

## Measurements

For the available S/H/D states (`selected` and `final`; seeds 1737 and
2711) plus one exact affine initial-function reference, the CLI first writes
immutable per-row prediction/margin outputs and only then writes summary
metrics. P04 affine selected/final states and any recreated GRU initial
checkpoints are excluded from this diagnostic. It records unrestricted
full-vocabulary top-1 with the P04 lowest-ID rule (using `argmax` for the
prediction decision and reporting exact top ties), exact tie counts, gold-vs-best-other margin, and token
correctness. It then evaluates the cached candidate rows for teacher-order
agreement and calls the P04 `pairwise_teacher_loss` implementation for the
original loss/tie behavior. Per-state outputs include row counts, pair counts,
omitted ties, total pair-weight denominator, and separate difficult/uniform
and control summaries. No logits are archived.

The gradient phase runs on the same four fixed P04 schedule batches at the
recorded selected H, selected D, and final D checkpoints for each seed. It computes, without
`optimizer.step`, raw gradients for CE,
hard, rank, and the gold-margin improvement loss, then reports norms and
pairwise cosine similarities over all trainable model parameters. It also
reports weighted norms for the P04 `.25` terms, the combined raw norm, the
norm after the existing clip-1 rule, active rows, hard-negative terms,
retained pairs, omitted ties, and the rank pair-weight sum. The local rank
versus gold-margin cosine uses the gradient of negative gold margin; positive
cosine means descent on that component reduces negative gold margin and thus
improves the desired margin. A control batch has no ranking
gradient and is recorded as such. These are measurements at stored or exact
initial checkpoints, not claims about all intermediate steps.

## Execution and resource gate

Before the diagnostic, run only model-free import, sample-index, synthetic
loss/tie, and create-only serialization tests. No GPU execution occurs until
root reviews this plan and the implementation source commit. The first
approved GPU cell is the largest representative backward cell:
one selected D state, one of the four actual schedule batches (up to 512
selected positions, with its sparse teacher rows), full-vocabulary projection
chunk 512, and separate CE/hard/rank gradients. It also writes the first
forward sample cell. Only after this backward qualification passes with
measured margin may the remaining state matrix and gradient batches run.

Numeric subprocess settings are explicit: `CUDA_VISIBLE_DEVICES` selects the
reserved GPU when CUDA is used, `torch.set_num_threads(4)`,
`torch.set_num_interop_threads(1)`, the recorded P04 numerical runtime settings (with no additional deterministic-kernel flag), and fixed
selection seed `20260909` (the schedule-step selection is immutable P04
metadata). The
guard fails closed if free GPU memory falls below 8 GiB, PyTorch reserved
memory exceeds 6 GiB, host RSS exceeds 16 GiB, or an allocator/driver/
temperature error occurs. It records preflight, periodic/final GPU and host
samples, phase timing, and the failure reason; a failed attempt is retained
and excluded. P05 will not use CPU batching as a semantic workaround.

The table is 128,256 x 2,048 FP32, about 1.051 GB (0.978 GiB); a 512-row
logit chunk is about 0.263 GB (0.245 GiB), and the largest diagnostic model
state is about 26 MB. The prior P04 K=32 teacher qualification peaked at
3.67 GB host RSS and 5.37 GB reserved GPU; the P05 forward cell adds only
the fixed table/chunk and is budgeted below the 6 GiB reserved limit with
the 8 GiB free floor. The forward sample is expected to take under a minute
per state on the reserved GPU; the full 13-state forward matrix plus 24
gradient cells is estimated at under 15 minutes, with a 30-minute
outer timeout. Actual phase timings and peak RSS replace these estimates.

Outputs are create-only and task-local under
`experiments/TRR-P05/runtime/<run-id>/`: sample index, predictions, gradient
diagnostics, compact summaries, phase receipts, and resource samples. A
truth-free failure writes its receipt before exiting. No plot or report
finalizer can discard numeric outputs.

## Disposition rule

The report will choose exactly one packet disposition from the measured
evidence. If D improves cached ranking agreement but loses reconstruction,
the ranking objective is retired. If weighted gradient conflict is large and
the loss clearly transfers, one discriminating future scaling/optimization
experiment is described without running it. If ranking is not learned or
the available checkpoints/sample cannot distinguish the alternatives, the
transfer question remains unresolved while P04's negative panel result is
preserved.
