# TRR-P08 — Does staged learning unlock a better direct reconstruction model?

## Objective

P07 established consistent token-accuracy gains from the P06 past-only
checkpoints over the retained reference on common opened inputs. It did not
establish global promotion, and it did not isolate why the newer fits improved.

Test one concrete hypothesis:

Does learning a competent affine inverse first, then training the complete
decoder, improve reconstruction or the incremental benefit of earlier
activation vectors?

This is a bounded fitting-mechanism study, not another frozen-model
confirmation or a broad architecture/provenance sweep.

## Starting point and ownership

Repository: A-lan-Z/Token-Reconstruction-Research

Parent branch: task/TRR-P07
Reviewed head: cffb10c14d3b74f4bd0227804e3ac380aa475686
New task/branch: TRR-P08 / task/TRR-P08

Verify live/local state and task-name availability. Use a separate worktree,
task-owned artifacts, and task-local status. Read the charter and relevant
P06/P07 fitting provenance.

Do not merge PRs or alter global coordination, the active registry, or the
other agent's workspace.

Agent one owns TRR-0009's adaptable-readout experiment. Keep the public
vocabulary readout fixed here, do not duplicate that work, and do not use
unpublished results.

P03's holdout remains unopened. P06's full-record variant remains stopped.
Do not revive the teacher-ranking objective.

## Controlled comparison

Prefer a small crossed comparison:

- Positionwise versus past-only activation access.
- Joint fitting from the standard initialization versus staged fitting.

For staged fitting, first train the affine component, then train the complete
model. The affine phase must use permitted fitting data only and count toward
the total training exposure and cost.

Keep the architecture, vocabulary readout, public data, clip geometry,
selection rule, and numerical execution common across the appropriate
contrasts. Paired seeds are preferred.

Do not give the staged arm uncounted extra data or optimization and attribute
the resulting advantage solely to initialization. Reusing an existing fitted
state is acceptable as a reference, but incomplete fitting provenance cannot
support an equal-budget causal claim.

Use a common H128 fitting/evaluation geometry for the primary test so that
crop length does not vary with the training procedure. Keep checkpoint
selection consistent across domains and arms.

Choose sensible phase lengths and optimizer details autonomously, using
public development evidence. Avoid a large schedule sweep.

## Make the learning result informative

Measure base errors at the transition to the correction phase, training and
held-out curves, and whether the added path learns on examples the affine
component initially gets wrong.

An already-perfect tiny subset does not establish useful correction learning.

The main question is whether staged fitting changes the past-only minus
positionwise advantage, not merely whether one selected checkpoint has the
largest pooled score.

Preserve the competent direct path. Neither model may use source-token inputs,
guessed-token feedback, future activations, A2 fallback, or candidate simulation.
These are the access settings of this experiment, not a new permanent method ban.

## Evaluation and cost

Freeze the chosen small set before a modest new natural evaluation with
distinct fitting, selection, and evaluation roles.

Use identical observations for all contenders. Report Pile and Finance
separately, including exact 127-token post-BOS clip recovery, token errors,
paired gains/losses, and source-record uncertainty.

Prioritize matched-public quality. Add a paired changed target when practical,
without turning this task into a trajectory sweep.

Retain the established positionwise reference and P06 past-only checkpoint
identities as context. Do not select the better published P06 seed from the
new evaluation answers.

A limited same-record A1+A2 anchor is optional where it answers a concrete
quality question. No full historical matrix is required.

Measure fitting cost including pretraining, deployed state, and warmed
inference with common batch and timing boundaries. Do not compare batch
throughput with single-record latency as if they were interchangeable.

## Decision and stopping

- Similar gains for positionwise and past-only models indicate a general
  fitting-procedure benefit, not a context-specific breakthrough.

- A larger, reproducible contextual advantage under staged fitting supports
  investigating that interaction.

- No useful gain under a well-qualified fit deprioritizes this staging
  hypothesis; do not automatically add more phases or optimization steps.

- An uninformative fit or uncertain comparison remains inconclusive.

Do not make a global promotion claim from this pilot or reinterpret P07's
registered decision. No automatic sample expansion or confirmation is authorized.

## Deliverables

Use resource preflight and coordinated shared compute. No paid resources.

Save this assignment at coordination/requests/TRR-P08.md.
Write the result to coordination/results/TRR-P08.md and structured evidence
to experiments/TRR-P08/manifest.json, with task-local status.

Preserve reproducible source, fitting exposures, state/input identities,
learning curves, frozen predictions, costs, and failed attempts.
Open a follow-on PR against the actual parent without merging.

Lead with whether staged learning improves the inverse itself, specifically
helps contextual learning, or supplies no useful benefit.