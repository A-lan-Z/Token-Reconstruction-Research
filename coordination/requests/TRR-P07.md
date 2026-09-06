# TRR-P07 — Reconcile P06's past-only candidate with the retained positionwise reference

## Objective

P06 is complete. Its full-record versus past-only comparison did not meet
the registered useful-gain threshold. Preserve that negative decision.

A separate question remains: P06's past-only decoder improved Finance
exact-clip recovery over its own diagonal control, but it has not been
compared on common inputs with the retained TRR-0006 positionwise reference.

Determine whether this is a genuinely better frozen search-free checkpoint,
an improvement over a weaker local control, or a panel-dependent result.

This is a bounded retrospective frozen-model comparison, not a new training
campaign or an automatic promotion.

## Starting point and ownership

Repository: A-lan-Z/Token-Reconstruction-Research

Parent branch: task/TRR-P06
Reviewed head: 02c861dfbfc63e3c0b7684a48323fd476a3b268a
New task/branch: TRR-P07 / task/TRR-P07

Verify live/local state and task-name availability. Use a separate worktree,
task-owned outputs, and task-local status. Read the charter and relevant
P06/TRR-0006 reports and manifests.

Do not merge PRs, change global coordination or active benchmark state,
or modify the other agent's workspace.

Agent one continues TRR-0008 unchanged. Do not import its unpublished
results or add these models to its registered comparison.

P03's unopened holdout remains out of scope.

## Frozen methods

Use the exact published states for:

- TRR-0006's retained enriched positionwise reference.
- TRR-0006's corresponding frozen causal decoder.
- P06's past-only decoder, retaining both fit replicates.
- P06's diagonal control, retaining both fit replicates.

Read actual state and source hashes from the published manifests.

Do not choose the better P06 seed using evaluation results. Report each
replicate and a clearly labelled replicate-averaged contrast; do not
average logits into an unregistered ensemble.

The stopped P06 full-record model need not be rerun. Its primary negative
result is not being reconsidered.

No fitting, optimizer updates, new checkpoint selection, compression,
teacher objectives, or architecture changes.

## Common-input comparison

Use two already-opened panels:

1. The published P06 natural evaluation panel.
2. A modest, deterministically selected portion of the already-opened
   TRR-0006 panel, chosen without reference to model correctness.

Choose and record the subset rule and comparison plan before new scoring.
There is no need to rerun the entire 3,072-source study.

Within each panel, every compared method must receive identical observed
activation tensors, validity masks, position IDs, and scored positions.
Preserve the 128-position clip including BOS and all 127 post-BOS targets.

Retain domain and target pairing. Use accessible published assets or
documented reproduction of opened material. Do not access fresh hidden
records or another task's sealed truth to resolve exclusions.

These are retrospective evaluations: the records have already been opened.
Freeze the new predictions before scoring, but do not relabel the result
as fresh confirmation.

## Numerical and provenance checks

Reproduce a small published fixture for each execution path before comparing.

If a common runner changes numerical behaviour, distinguish an exact native
replay from a benchmark-compatible port. Do not silently change precision,
normalization, record batching, masks, or tie handling.

Record differences in fitting support, crop, selection metric, seed, and
initialization that may explain why the published checkpoints differ.
Do not infer that one difference caused the result without an intervention.

Do not turn this task into a broad infrastructure audit.

## Analysis

The main comparison is P06 past-only versus the retained positionwise
reference on identical inputs. Also retain:

- P06 past-only versus P06 diagonal.
- P06 diagonal versus the retained reference.
- P06 past-only versus the older causal checkpoint.

Report each domain and target separately. Include token accuracy, exact-clip
recovery, paired gains/losses, and source-record uncertainty.

Keep fit replicates and paired target conditions inside their source-record
clusters; they are not additional independent source records.

Do not interpret the difference between percentages from different original
panels as a measured model improvement.

A same-record A1+A2 subset may be reused where outputs already exist.
Do not launch a new full-panel A1+A2 campaign for this reconciliation.

Measure comparable warmed execution with balanced method ordering if cost
affects the recommendation. Distinguish batch throughput from single-clip
latency and preserve prediction equivalence.

## Decision

Return a clear disposition:

- Local-control improvement only: retain the established reference.
- Both new models improve: prioritize understanding fitting/checkpoint
  differences rather than attributing the result solely to context.
- Past-only improves consistently over the retained reference: identify it
  as a candidate for a separately planned fresh confirmation.
- Panel-dependent or uncertain result: state the dependency and do not
  promote a new global default.

No automatic new confirmation, training run, or sample expansion is
authorized by this task. P06's full-record variant remains stopped.

## Deliverables

Coordinate shared compute, use resource preflight, and incur no paid charges.

Save this assignment at coordination/requests/TRR-P07.md.
Write the result to coordination/results/TRR-P07.md and structured evidence
to experiments/TRR-P07/manifest.json, with task-local status.

Retain exact method/input identities, replay checks, frozen predictions,
paired results, timing boundaries, limitations, and failures.
Open a follow-on PR against the actual parent without merging it.

Lead the handoff with whether P06 produced a better checkpoint than the
retained reference—not merely whether it beat its own diagonal control.