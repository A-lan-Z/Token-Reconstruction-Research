# TRR-P08 bounded implementation plan

Status: source planning only; no public tensor load, GPU run, fitting, truth read, or
prediction run has started.

## Controlled matrix

Reuse the published P06 `VisibilityAffineAttentionDecoder` family and its public
fit/validation loader, H128 crop, full-vocabulary float32 logits, cosine-normalized
Q/K score rule at scale 4, AdamW (`lr=1e-3`, weight decay 0, clip norm 1), and
validation selection rule. P08 will add one task-local runner rather than changing
P06 source or registering a new permanent method.

The four registered arms are the crossed pair:

| visibility | schedule | method ID |
| --- | --- | --- |
| positionwise diagonal | joint 3,000 updates | `p08_positionwise_joint` |
| positionwise diagonal | staged 1,000 affine + 2,000 full updates | `p08_positionwise_staged` |
| past-only causal | joint 3,000 updates | `p08_past_only_joint` |
| past-only causal | staged 1,000 affine + 2,000 full updates | `p08_past_only_staged` |

Each arm is run with paired seeds `6106` and `6107`, for eight fits total. The
recommended 1,000/2,000 split is a single predeclared choice, not a schedule
sweep. If design changes the split before source freeze, the chosen values must be
recorded in the runner and every receipt; no post-fit choice is allowed.

All arms start from the same explicit P08 standard initialization:
`W=I`, `b=0`, `s=3.0`, zero output weight/bias, and deterministic Q/K/V
initialization from the replicate seed. The published competent affine state is
retained as provenance context only; it is not loaded into P08. The joint arms
train all parameters from update 1. The staged arms use one common 3,000-update
schedule and optimizer: updates 1--1,000 train only `W`, `b`, and `s`; at the
recorded transition the attention parameters are unfrozen and updates
1,001--3,000 train the complete model. The cosine schedule is defined over all
3,000 updates and is not reset at the transition. Thus both schedules consume the
same 3,000 batches and 1,536,000 post-BOS draws per seed/arm; staged fitting has
2,000 attention-update exposures and 3,000 affine exposures, which will be
reported explicitly rather than treated as equal parameter-update counts.

A schedule is built once per seed from the fixed public fit mask with 8-record
batches and 512 post-BOS draws per update, then reused byte-for-byte by all four
arms for that seed. Validation is public-only and common, every 100 updates
including step 0 and the 1,000-update transition. Earliest maximum full-vocabulary
validation token accuracy selects a state; no fresh panel answer participates.

## Learning diagnostics

Before any arm updates, the runner will freeze a public-fit diagnostic ledger of
256 unique initial affine errors (64 in each P06 position bin) from the explicit
`W=I,b=0,s=3` direct path. The ledger is diagnostic only and cannot select an arm.
It will report initial, transition, and final token correctness on those same rows,
plus token error counts by position bin. For staged arms, the transition point is
measured after the 1,000 affine-only updates and before the first full-model
update. Joint arms receive the same step-1,000 ledger readout for comparison.
Learning curves retain phase labels, training/validation timings, selected step,
state hashes, and the separate affine/full update costs. The runner will never
write source tokens or evaluator truth into these receipts.

## Small source interface

The authoritative task source will be `scripts/trr_p08/fit_staged.py`, with
focused tests in `tests/test_trr_p08_fit_staged.py`. It will import the published
P06 decoder/data/schedule primitives and expose create-only `preflight`, `qualify`,
and `main` modes. `preflight` is CPU metadata/geometry only. `qualify` is a
disposable two-update past-only full-vocabulary backward cell at the exact 8 x
128 x 512 geometry, using the P08 standard initialization and all trainable
parameters. `main` requires PASS preflight and qualification receipts, then runs
the eight fits sequentially. The external P06 fail-closed watchdog remains the
outer guard; no alternative batching, projection chunk, or precision workaround
is planned.

Focused tests will cover standard-initial-state equality across masks and
schedules, exact shared schedule reuse, staged freeze/unfreeze boundaries and
optimizer exposure counts, H128/mask geometry, transition ledger accounting, and
checkpoint round-trip metadata. Synthetic tensors will verify that positionwise
and past-only visibility semantics are inherited unchanged. No test will open
private truth or run the full model.

## Resource estimate and guard

P06 measured 91.5--92.2 seconds per 3,000-update arm at this geometry, with
2.93--3.18 GiB peak reserved GPU and 5.64 GiB process RSS. P08 has the same
architecture, dense mask workspace, embedding table, batch, sequence, draw
budget, and optimizer state. A conservative upper bound is 8 x 92.2 = 738
seconds for the main fits, plus approximately 30 seconds for data loading,
validation setup, and receipts. Expected wall time is therefore under 13
minutes; retain the 1,800-second fail-closed deadline. The two-update qualifier
should be about 12--20 seconds based on the P06 12.31-second qualification.

Use four Torch intra-op threads and one inter-op thread, one exclusive GPU, at
least 8 GiB free GPU, at most 6 GiB PyTorch reserved GPU, at most 16 GiB whole
process RSS, and at least 10 GiB host available memory. The first action after
source freeze is a CPU preflight; the largest-cell qualification must pass before
the eight-fit matrix is released. Each phase records exact command, source
commit, asset hashes, phase timings, peak memory, schedule digest, and failure
receipt. No GPU launch occurs under this plan until root grants a fresh resource
window after source review.

## Boundaries

P08 uses only the published public fit/validation bank and later a setup-owned
frozen evaluation panel with identical observations for all arms. It has no
source-token inputs, guessed-prefix feedback, future-token feedback, candidate
simulation, A2 student, teacher loss, full-record arm, new architecture, target
preparation, P03 holdout, or truth-dependent model selection. Any optional A1+A2
anchor remains a separate parent-compatible diagnostic and is not part of this
fitter or its student comparison.
