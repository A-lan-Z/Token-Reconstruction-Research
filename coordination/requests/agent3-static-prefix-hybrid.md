&#x20;&#x20;

# Agent 3 — Proceed with the static-prefix B1+A2 hybrid pilot

Your shortlist study is complete. Preserve its findings and limitations.

The requirement for a maintained recovered-model-prefix implementation is
relaxed for this next bounded experiment. Use the validated static public
prefix and label the study accordingly.

This tests whether a stronger proposer permits cheaper verification under
the observed target changes. It does not demonstrate active prefix recovery
or cold-start tracking.

## Comparison

Use the same public-prefix weights, candidate-scoring rule, numerical
settings, and source observations for:

- historical proposer + A2 at K=256;
- frozen B1 + A2 at K=256;
- frozen B1 + A2 at K=16;
- historical proposer + A2 at K=16;
- frozen B1 alone.

Keep the exact B1 checkpoint used in PR25. Do not replace it with the newer
ordinary-continuation model.

Do not add confidence shortcuts, adaptive budgets, centering changes,
retraining, or fallback routing.

## Execution

First qualify the integration on a small already-opened slice. Each method
must use its own committed reconstructed token prefix and cache; no true
prefix or another method's successful reconstruction.

Then freeze the implementation and evaluate on a modest new natural panel
using the existing public-base/64/128/256 target snapshots. Reuse qualified
trajectory assets rather than training another target.

The earlier shortlist panel informed K=16 selection, so it is development
evidence, not a fresh confirmation of this selected hybrid.

Keep evaluator-only target access and source truth separate from
reconstruction. Coordinate source exclusions and compute with Agent 2.
No PR merges, paid compute, or P03 holdout access.

## Required result

Measure actual token accuracy, exact-clip recovery, shortlist omissions,
wrong selections despite inclusion, and failures following earlier wrong
commitments.

Report each domain and snapshot separately. Measure total proposal,
candidate simulation, cache/commit work, memory, and warmed reconstruction
time on equivalent workloads.

Do not infer speedup from candidate counts or claim equivalence from a
nonsignificant difference. Predeclare useful acceleration and acceptable
quality-loss criteria.

If K=16 preserves quality at lower measured cost, retain it as a
static-prefix hybrid candidate within the tested scope.

If shortlist coverage remains good but selection fails under drift,
identify that as evidence for investigating verifier/prefix mismatch.
Do not automatically build a recovery framework within this task.

Preserve model bytes, frozen predictions, costs, and replay instructions.
Use one task owner and existing validated phase interfaces. Return the
result before launching further variants.
