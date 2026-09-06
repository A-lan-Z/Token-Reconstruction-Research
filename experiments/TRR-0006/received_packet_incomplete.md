# TRR-0006 — Frozen-pair confirmation of the value of additional activation context

## Research decision

Determine whether the already-trained enriched causal decoder provides a practically meaningful advantage over its already-trained enriched positionwise control on new natural inputs. This is a bounded architecture-selection study, not a new learning campaign or a claim that standalone reconstruction matches A1+A2.

TRR-0005 provides strong evidence that changing the public fitting distribution at fixed fitting counts improves reconstruction. It excludes the registered +0.5 percentage-point token benefit from extra context for the tested enriched pair, but does not yet exclude the registered +5 percentage-point exact-record benefit. Preserve that distinction. H_i already contains context; the comparison tests the incremental value of earlier observed activations, not the usefulness of context in general.

## Starting point and ownership

Repository: `A-lan-Z/Token-Reconstruction-Research`

- Parent: `task/TRR-0005`, PR #9.
- Reviewed publication head: `3a7e8f579e713c3e41d02639237042ca26fd019b`.
- New task and branch: `TRR-0006` / `task/TRR-0006`.

Verify local and live state. Use a separate worktree, task-owned output directory, and task-local status. Do not merge any PR, change global coordination state or the active registry, or modify the parallel agent's workspace. Read the charter and relevant source plans. This assignment authorizes a new exploratory confirmation without accepting previous PRs.

Agent A2's TRR-P04 owns offline teacher-supervision experiments. Do not duplicate them, import unpublished outputs, or silently change either study's fitting data or models. Shared assets may be reused read-only; coordinate actual resource use and isolate comparative timing. No paid compute is authorized.

## Frozen methods and scope

Use exactly the published selected states and decision rules for:

- `enriched__affine_causal_h_attention128`
- `enriched__affine_trained_diagonal_attention128`

Read their actual state/source hashes from the TRR-0005 registry and evidence. Pin the final repaired cosine/QK-normalized causal implementation, not an earlier saturated dot-product run. A post-score maintenance commit is not proof of identical inference: reproduce a small already-opened fixture before claiming equivalence.

No new fitting, checkpoint selection, teacher training, threshold tuning, architecture changes, or compression. Do not replace the selected trained-diagonal control with plain affine. The physical self-only control is positionwise despite its method name.

Do not rerun the original-versus-enriched training matrix or every historical method. A1+A2 is not required across the full larger panel. Retain its previous same-panel comparisons as historical evidence; a new modest, preregistered subset anchor is optional only when it answers a concrete question. Label its separate denominator explicitly and freeze it with the rest before scoring.

## New evaluation panel

The proposed fixed design is 1,024 new unique natural source records per domain: Pile and Finance, 2,048 source records total. Pair the same records across the published public-base and synthetic-LoRA target states. This produces four source/target cells, not 4,096 independent sources.

Preserve the declared 128-token clip including BOS, post-BOS scoring, source-formatting rules, qualifying numerical execution, and natural selection distribution from TRR-0005. Do not insert wrappers or replace a natural sampling distribution with a frequency-balanced diagnostic panel. Exact recovery means all 127 scored post-BOS tokens in the declared clip, not the complete original document.

Exclude accessible known training, validation, opened evaluation, and duplicate sources/sequences. Coordinate opaque reservations with the parallel stream without exchanging hidden answers. Do not inspect another study's sealed holdout or private truth to create exclusions. If exclusions leave insufficient eligible records in the published source range, resolve and preregister a compatible extension before selection or truth access; report limitations rather than silently reusing records or changing the population.

## Precision planning and analysis

Before creating/opening new evaluation truth, verify sample-size adequacy with the actual registered decision procedure. TRR-0005 uses conservative upper bounds for exact-rate differences: U(p_gain) - L(p_loss), with one-sided Clopper-Pearson tails 0.05/32. Scaling Finance's 5 gains and 4 losses per 128 to 40 and 32 per 1,024 gives U = 4.2857 percentage points. That plug-in calculation is not guaranteed decision probability.

Check sensitivity to plausible beneficial/harmful discordance rates and preserve the source-paired dependence between target conditions in any joint power claim. You may revise the sample size once before new selection, with a documented practical/resource justification; do not adapt it repeatedl
