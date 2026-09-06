# TRR-P06 — Does later-position activation evidence improve search-free reconstruction?

## Objective

Test a previously unexamined observation-access hypothesis:

Can later-position activation vectors from the already-observed current
record help reconstruct token x_i beyond what a positionwise or past-only
decoder can recover?

This is independent of agent one's TRR-0007 fitting-support-versus-
positionwise-capacity study. Hold fitting conditions and model capacity
reasonably controlled; the primary intervention is activation visibility.

Do not revive P04's teacher-ranking objective, P03's projected prototypes,
or their stopped variants.

## Governing access distinction

The full current-record activation tensor H_c is observable before
reconstruction begins.

The following requirements remain unchanged:

- Source-token access is BOS-only.
- No source plaintext, other true token IDs, correctness feedback,
  unavailable target-prefix weights, or target oracle is available.
- Recovered tokens are committed immutably left-to-right.
- Later reconstruction results cannot revise earlier commitments.
- Adaptation from the current record can affect only later records.

Later-position activation vectors are observations, not later source-token
labels or later reconstructed answers.

This assignment permits using those already-observed activation vectors.
Earlier tasks' restrictions to H_0 through H_i are retained as experimental
controls, not treated as a permanent method ban.

The deployed decoder must not consume guessed-token feedback or perform
A2 candidate simulation. It may compute activation-only features for the
whole record before emitting tokens in order.

## Starting point and ownership

Repository: A-lan-Z/Token-Reconstruction-Research

Use the published learned-decoder infrastructure from:
- Parent branch: task/TRR-0006
- Reviewed head: f10f8ba438973b3cb260d41707fbb14293db9cd3
- New task/branch: TRR-P06 / task/TRR-P06

Verify current state and task-name availability before creating anything.
Use a separate worktree, environment as needed, and task-local outputs/state.

Do not merge PRs, modify global coordination state or the active registry,
or edit agent one's checkout. Do not depend on unpublished TRR-0007 results.

Continue approved exclusion coordination without exchanging hidden answers.
P03's unopened holdout remains out of scope.

## Small controlled experiment

Compare:

1. A competent positionwise decoder.
2. A past-only activation-context decoder.
3. A decoder with later-position activation access.

Prefer one compact architectural family with controlled visibility masks
or windows and comparable parameter counts. A small fixed look-ahead window
or full-record activation attention is a reasonable third arm. Choose one
informative design rather than a broad window/architecture sweep.

Preserve a competent direct path so all arms begin with a meaningful
reconstruction baseline. Train each arm under its actual observation mask;
do not evaluate a causal checkpoint with its mask simply removed and call
that a fair architecture comparison.

Use the same permitted public fitting examples, full-vocabulary
same-position supervision, and comparable optimization and selection
opportunities. Avoid an auxiliary teacher-ranking loss.

Demonstrate that the added paths can learn on public examples the base
initially gets wrong. An already-perfect subset is not a useful capacity
qualification.

## Evaluation

Use a modest, newly selected natural panel with varied token identities
and contexts, separate from fitting and selection data.

Keep all compared methods on identical observations and report domains
separately. Prioritize a matched-public condition, with a paired changed
target where practical.

Freeze choices and all predictions before scoring. Do not use another
task's sealed records or evaluation results for model selection.

Report unrestricted token accuracy, exact-clip recovery, paired gains and
regressions, source-record uncertainty, and inference cost. Include a
bounded A1+A2 quality anchor with its separate denominator stated clearly.

Measure whether any benefit is concentrated at particular positions.
Handle record ends and padding explicitly: there are fewer later vectors
near the end, and invalid/padded rows must not provide artificial evidence.

Keep observation boundaries fixed. Do not give only the look-ahead method
extra activation rows outside the declared record or scored clip.

Document that this is a full-record method, not token-streaming inference.
Account for the entire activation-only forward computation and retained
state. No A2 fallback or candidate search is allowed in student inference.

## Decision

Define a practical benefit threshold before fresh evaluation.

- A useful later-activation gain supports advancing this observation model.
- A benefit that fails under target changes is a transfer question, not a
  broad success claim.
- A well-trained negative result support