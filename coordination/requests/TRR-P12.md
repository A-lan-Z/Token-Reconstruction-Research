# Parallel study — Does the frozen expanded decoder survive actual target training?

## Question

Keep the restored B1 decoder fixed and test whether its reconstruction
quality survives a meaningful, compatible public fine-tuning trajectory.

The previous artificial perturbations caused no prediction changes. Do not
repeat those unchanged or treat them as evidence of broad target robustness.

Own this bounded study end to end. It does not depend on Agent 1's new fits.

## Design

Prefer existing, verifiable target snapshots when suitable. Otherwise,
predeclare one modest public adaptation run with compatible architecture
and tokenizer, including a baseline and a few checkpoint stages.

Keep source text, cut, rendering, and numerical execution common across
snapshots. Separate target fitting sources from reconstruction evaluation.
Record what weights changed and whether the adaptation was meaningful.

The target prefix remains evaluator-only. Reconstruction receives the
frozen decoder and permitted observations—not target weights or true tokens.

## Analysis

Distinguish:

- errors already present under the unchanged public model;
- previously correct tokens or clips broken by the update;
- errors improved by the update.

Measure absolute reconstruction and the paired changes. Include a limited
common A1+A2 comparator where useful.

Analyze displacement relative to the actual decoder decision boundaries.
A large unsigned movement is not proof of crossing a boundary. Any proposed
failure predictor must be checked away from the updates used to choose it.

One trajectory is evidence about that trajectory, not all fine-tuning.
Do not add decoder adaptation or training to rescue a transfer failure.

## Execution and handoff

Reuse the restored model package and validated evaluation path. Use a
task-owned branch from the published TRR-P11 work and verify name availability.

Coordinate compute and source exclusions with Agent 1, but do not exchange
hidden outcomes or create reciprocal implementation dependencies.
No paid compute, PR merges, or P03 holdout access.

Preserve create-only outputs and immediately back up new model artifacts
when target training is required. Use an early end-to-end smoke check.

Report whether transfer held, where it broke, and whether the proposed
explanation predicted failures. Do not launch a broader target sweep
automatically.
