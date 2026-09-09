# Agent 4 — Recovered-prefix-only token inversion pilot

## Objective

Test whether the public/recovered model prefix can both propose and verify
tokens, allowing us to remove the separately trained A1/B1 token decoder.

The central hypothesis is:

“A maintained approximation of the target prefix can guide token search
through its own input gradients, then verify proposed tokens through forward
execution. We may therefore need only one target approximation, rather than
a separately trained guesser plus a forward-model checker.”

This is a bounded pilot, not a request for a large optimizer sweep or a
guarantee of exact inversion.

Success means a practically useful reconstruction method with fewer learned
components—not merely a low activation-matching loss.

## What “prefix-only” means

Distinguish these three objects:

- Target model prefix: the unavailable layers that produced the observation.
- Recovered model prefix: our permitted approximation of those layers.
- Reconstructed token prefix: tokens already committed earlier in the record.

The primary method may use the public/recovered model prefix, its input
embedding table, the tokenizer, observed activations, permitted metadata,
and its own reconstructed token prefix.

It must not use a fitted A1/B0/B1 inverse, learned proposal head, teacher
shortlist, or external language model to initialize or rescue reconstruction.

Existing online model-prefix recovery is allowed under the charter.
This removes a separate inverse-fitting phase; it does not eliminate all
learning, online optimization, candidate checking, or preparation costs.

## Ownership and starting point

Repository: A-lan-Z/Token-Reconstruction-Research

Use the published implementation resources at:
- Branch: task/TRR-0012
- Commit: 5bbc3bf42a81c814404cf84cb46d55f0d3418667

Create a separate task branch, for example:
task/agent4-prefix-only-inversion

Verify current state and task-name availability. Read RESEARCH_CHARTER.md,
the public-prefix executor, and the relevant historical A1+A2 implementation.

You own this study end to end. Reuse verified infrastructure and make
implementation choices autonomously within this scope.

Agent 2 owns target-trajectory experiments.
Agent 3 owns B1 shortlists and small-budget A2.

Reuse their permitted target or recovered-prefix assets when available,
but do not duplicate their experiments or wait for their final conclusions.
Coordinate shared compute and source exclusions. Do not access another
study’s sealed answers or modify another agent’s workspace.

No PR merges, global-registry changes, paid compute, or P03 holdout access.

## Algorithmic starting point

Read and pin the SipIt paper and official implementation:

- https://arxiv.org/abs/2510.15511
- https://github.com/giorgosnikolaou/SIPIT

Use a bounded, gradient-guided sequential inversion procedure inspired by
that implementation, adapted to our access and cost constraints.

The basic operation at position i is:

1. Keep the already committed token prefix fixed.
2. Initialize a provisional input embedding using permitted public
   information only.
3. Run it through the public/recovered prefix.
4. Measure the difference between its predicted boundary activation and
   the observed activation at position i.
5. Differentiate that difference with respect to the provisional embedding.
6. Use the gradient/updated embedding to propose discrete vocabulary tokens.
7. Forward-check promising tokens, then commit one under a frozen rule.

Gradients must be computed through our own model approximation. They are
not target-provided gradients, target labels, or correctness feedback.

Preserve the actual public input-embedding convention. A normalized table
used for nearest-neighbour scoring is not automatically interchangeable with
the embeddings the forward model expects.

Choose one credible implementation. A small corrective variant is acceptable
when a diagnosed problem warrants it; do not launch a solver sweep.

Document departures from upstream SipIt. Its matched-model guarantees do not
automatically hold under finite precision, a bounded search budget, or an
imperfect recovered prefix.

## Access and state rules

Only BOS is a known source token.

Main reconstruction runs must use their own committed tokens, not true
preceding tokens or another method’s successful reconstruction.

Keep recovered model-prefix weights fixed during reconstruction of a record.
Any permitted recovery update using that record may first affect later
records. Optimizing a provisional input embedding within the current record
is allowed; silently updating model weights from current-record information
is not.

Candidate trials must not contaminate the committed prefix cache.
Commit the chosen token once; do not revise earlier outputs.

Use the current-position activation and already reconstructed prefix for
this first sequential solver. This is the scope of this experiment, not a
new permanent restriction on all reconstruction research.

Public development examples may guide algorithm settings. They must remain
separate from the eventual evaluation and do not constitute permission to
train a hidden inverse model.

## Stage 1 — Can the solver work when the forward model matches?

This is the first deliverable. It must not wait for recovered-prefix integration.

Use the established Llama-3.2-1B-Instruct cut after four transformer blocks.
Generate observations through the matching public prefix.

Begin with a small short-sequence feasibility panel. If useful, progress
to the established 128-position clips including BOS. Keep ordinary public
text and a small unusual-token/identifier stress panel separately reported.

Establish:
- meaningful input gradients;
- correct position, embedding, mask and cache semantics;
- agreement between the continuous-input and discrete-token forward paths;
- the numerical residual floor for genuine matches.

A higher-precision diagnostic is allowed, but report separately whether
the method also works on the observation precision used in our experiments.

Use a finite work budget. Count gradient steps, candidate checks, restarts,
and timeouts. Do not achieve “success” by silently enumerating the entire
vocabulary after the advertised budget is exhausted.

Compare with the existing A1+A2 reference on a small common subset using
the same public prefix. Its learned proposer is a comparator only, never
an input to the new method.

A true-prefix or exhaustive check may be used on a tiny public diagnostic
to isolate a failure, but must not be reported as deployed reconstruction.

## Stage 2 — Does a realistically recovered prefix make it useful under mismatch?

Proceed if Stage 1 establishes useful numerical feasibility.

On identical target observations, compare the frozen solver using:
- the untouched public prefix;
- a legitimately recovered approximation of the changed target prefix.

Use an inaccessible compatible fine-tuned descendant and, where practical,
a few snapshots from continued fine-tuning. Reuse Agent 2’s qualified assets
rather than training another target unnecessarily.

Keep target weights and source truth evaluator-only. Never create a
“recovered prefix” by copying target weights or supplying the target’s
private fine-tuning adapter.

Document how each recovered prefix was obtained, which prior records it
could use, and whether an existing A1+A2 system supplied its reconstructions.
That latter case is a component test, not proof of independent cold start.

Do not assume a validated recovery implementation already exists. Locate
it and verify its provenance. If it is unavailable, complete matched and
static-surrogate tests where possible and report the specific integration
gap. Do not invent recovered weights or build an unrelated recovery
framework within this pilot.

Compare against the existing proposal-plus-A2 method with the same
model-prefix snapshot, to distinguish solver quality from surrogate quality.

Under mismatch, an exact activation match may not exist. Use a
development-defined distance, acceptance rule, and budget-exhaustion policy.
A small residual must not be called proof that the token is correct.

## Stage 3 — Short closed-loop tracking test, conditional on feasibility

If a validated recovery path exists and Stage 2 is promising, run a short
sequence of records in which each method updates its model approximation
from its own permitted completed-record information.

Keep two claims separate:

- Warm-start/component result: inversion works with a supplied recovered
  prefix whose provenance and preparation costs are disclosed.
- Cold-start tracking result: the procedure begins from public resources
  and can bootstrap and maintain recovery without a fitted token guesser.

Do not silently warm-start with target weights, true token prefixes, or
historical A1/B1 predictions.

Each method must live with its own reconstruction mistakes. An early error
may affect later token decisions and subsequent model-prefix recovery;
those consequences belong in the result.

No broad trajectory campaign is required for this pilot.

## Measurements and fair stopping

For each completed stage, report:

- Token accuracy and complete-sequence/clip recovery.
- Correct-token discovery versus incorrect final selection.
- First-error positions and later failures after wrong commitments.
- Activation residual versus actual correctness, evaluated after freezing.
- Gradient evaluations, forward candidate checks, vocabulary-scoring work,
  restarts and budget exhaustion.
- Median and tail reconstruction time, memory, and retained artifacts.

Retain failed and timed-out records in the primary denominator.
Specify whether exhaustion emits the best tried token or abstains;
do not drop difficult cases or invoke a hidden fallback.

Count backward computation, candidate verification, cache construction,
and model-prefix maintenance. Fewer candidate checks do not automatically
mean faster reconstruction.

Measure quality and total cost on common inputs. Separate one-off preparation,
warmed inference, and ongoing recovery costs. Predeclare a realistic
quality/cost target before the final evaluation.

If Stage 1 fails, distinguish numerical/implementation failure from genuine
failure of the tested search procedure. If Stage 1 works but Stage 2 fails,
distinguish approximation mismatch from solver instability where the
diagnostics support it.

Do not infer information loss merely because this optimizer failed.
Conversely, do not claim a useful inverse merely because its loss decreased.

## Scope, preservation, and deliverables

Run an early end-to-end synthetic smoke test. Use modest resource preflight
and reuse existing validated loaders, evaluators and evidence conventions.

Freeze settings before evaluating unused sources. Prior opened material
is development or retrospective evidence, not fresh confirmation.

Back up actual model-prefix states and required dependencies immediately,
with a verified clean restore. Hashes alone are not backups. Preserve
create-only predictions and traces before scoring.

Publish a task-local report, manifest, code, exact upstream/source identities,
reproducible commands, costs, and failed attempts in an unmerged PR.

Lead the handoff with:

1. Can we reconstruct without a separately fitted token guesser?
2. Does it work only with matching weights, or also with a recovered prefix?
3. Is the total cost competitive enough to justify further work?
4. Was cold-start tracking demonstrated, or only a supplied-prefix component?
5. Should we advance this method, make one specifically justified correction,
   or stop the tested variant?

The goal is to remove a learned component without losing practical
reconstruction quality—not simply to replace fast offline fitting with
unbounded online search.