# Agent 3 — B1 shortlists for small-budget A2 under target drift

## Objective

Start by answering:

“As the target changes, how often is the correct token still within
B1’s first 8, 16, or 32 choices?”

Then, if the shortlist results justify it, test whether replacing the
historical proposer with B1 allows A2 to check substantially fewer
candidates without materially reducing reconstruction quality.

This is a practical acceleration study. It is not a claim that B1 removes
offline fitting, or that the hybrid eliminates A2.

You own this study end to end. Choose sensible implementation details
autonomously. Prefer a small, decisive experiment over a broad search.

## Starting point and parallel ownership

Repository: A-lan-Z/Token-Reconstruction-Research

Published implementation starting point:
- task/TRR-0012
- Commit: 5bbc3bf42a81c814404cf84cb46d55f0d3418667

Independent evaluation and model provenance:
- task/TRR-P11
- Commit: 81299e07d4213cd75f455496d623a8371516688f

Use the preserved rebuilt model:
- TRR-0012/expanded_fixed_replication_1
- Selected step: 13,000
- State SHA-256:
  088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706

Verify the actual state, public readout, loader and numerical settings
against the preserved package. Do not silently substitute an earlier B1,
a directional-readout model, or a newly trained checkpoint.

Create a separate task branch, for example:
task/agent3-b1-small-budget-a2

Check task-name availability and use task-local outputs/status.
Read the charter and relevant A1+A2 protocol.

Agent 1 owns public hard-example fitting.
Agent 2 owns target-trajectory transfer testing.
Do not duplicate their training or change their experiments.

Reuse Agent 2’s target snapshots or observation-generation assets when
available, with proper evaluator-only access. Coordinate that interface
directly rather than creating a chain of small handoffs. You do not need
Agent 2’s accuracy results to measure shortlist recall.

Do not merge PRs, alter global research state, or open P03’s sealed holdout.

## Hypothesis

B1 may remain a useful proposer even when its first-choice accuracy drops.

For example, the correct token might move from rank 1 to rank 5 as the
target changes. B1 alone would fail, but A2 checking eight candidates
could still recover it.

Conversely, if the correct token moves outside the first 32 candidates,
even an excellent A2 selector cannot recover it from that shortlist.

High top-1 accuracy on the unchanged public model is therefore not enough.
Measure the distribution of correct-token ranks as the target evolves.

Keep B1 frozen throughout this study. No new decoder fitting, calibration,
distillation, or readout adaptation.

## Stage 1 — Shortlist recall across a changing target

This is the first deliverable. Do not make it wait for A2 integration.

### Target conditions

Include:
- the unchanged public base as a matched control;
- a compatible fine-tuned descendant, treated as unavailable to the
  reconstruction process;
- a few predeclared snapshots as that target undergoes further fine-tuning.

The descendant tests initial mismatch; the snapshots test continued drift.
Preserve architecture, tokenizer and observation boundary.

Use actual prefix-changing adaptation. Fine-tuning only layers after the
observed boundary is a null control, not a meaningful drift test.

Prefer verified existing trajectories or coordination with Agent 2.
If suitable assets are unavailable, agree on one bounded task-owned
trajectory rather than duplicating an ongoing training run. Static
base-versus-LoRA results may be a preliminary diagnostic, but cannot
complete the moving-target question.

Target weights and source labels are evaluator-only. B1 receives observed
activations and permitted metadata, not the target checkpoint, true tokens,
or correctness feedback.

### Common inputs

Pair identical source clips across the target snapshots so changes in
token rank cannot be explained merely by different text.

Use a modest natural panel covering Pile-like text and Finance, with the
established 128-position clip including BOS and 127 scored positions.
Keep reconstruction evaluation sources separate from target fine-tuning
and decoder fitting/selection data.

Record the amount of actual boundary-activation change. Do not assume that
more fine-tuning steps must produce monotonically worse reconstruction.

### Measurements

Generate full-vocabulary scores and retain deterministic nested shortlists.
Report K = 1, 8, 16 and 32; retain 64 and 256 as diagnostic reference budgets
if inexpensive.

For each domain, target snapshot and K, report:

- Token recall@K: fraction of positions whose correct token is in the list.
- Number of omitted correct tokens.
- Clip coverage@K: fraction of clips where every correct token is included.
- Recall among positions where B1’s first choice is wrong.
- Paired changes from the matched base and the descendant’s initial state.

Clip coverage is an upper bound on exact recovery for a perfect selector
restricted to those fixed lists—not an achieved A2 reconstruction score.

Run the historical A1 proposer on the same observations and report its
recall at the same budgets, including its usual K=256 reference. This
distinguishes a stronger proposer from simply lowering everyone’s budget.

Report source-record uncertainty and individual snapshots. Repeated target
observations are not independent source records. Do not pool away a late
snapshot where recall collapses.

No ground-truth token may be inserted into a shortlist. Labels are used
only after the candidate outputs are frozen.

## Stage 1 decision

Answer plainly:

- Does B1 preserve near-reference candidate coverage at K=8, 16 or 32?
- Does the smallest viable shortlist grow as the target changes?
- Does a good top-1 average hide unacceptable whole-clip omissions?
- Does B1 actually improve on historical A1 at comparable budgets?

Define acceptable candidate-loss levels prospectively, informed by the
desired reconstruction quality. A nonsignificant difference is not proof
of equivalent recall.

If small lists clearly lose too many correct tokens, report that result.
Do not add confidence-based fallback or retrain B1 to rescue the pilot.

If recall is promising, proceed to a bounded hybrid test. If Stage 1
results determine its settings, treat Stage 1 as development and use
separately reserved sources for subsequent confirmation.

## Stage 2 — Does small-budget B1+A2 preserve quality and save time?

First identify what the A2 implementation actually uses.

Distinguish:
- untouched public-prefix weights;
- recovered/adapted model-prefix weights;
- the already reconstructed token prefix and its execution cache.

Maintaining a token-prefix cache is not the same as adapting the model
prefix to follow target-weight changes.

The intended moving-target hybrid uses the maintained recovered model
prefix. Do not silently replace that with untouched public weights and
claim to have tested tracking.

If a validated recovery/update path is unavailable, finish Stage 1 and
document that integration gap. A static-public-prefix hybrid may be run
as an explicitly limited diagnostic, not as the intended adaptive system.
Do not build an unrelated recovery framework within this task.

### Controlled comparison

For a small shared panel, compare:

- Historical proposer + A2 at K=256.
- B1 + the same A2 at K=256.
- B1 + A2 at the promising small budget or budgets.
- Historical proposer + A2 at the selected small budget as a control.

B1 replaces the proposer; it is not an extra stage beside it.

Initially hold recovered model-prefix snapshots and selector rules common
across methods to isolate the proposer/budget effects. Those recovered
snapshots must themselves be produced under the permitted access model.

Run each reconstruction causally from BOS using its own committed tokens.
Do not substitute true preceding tokens or a successful comparator’s
reconstructed prefix.

Separate:
- failure because the correct token was omitted;
- failure despite its inclusion;
- later failures following an earlier wrong commitment.

If the controlled comparison supports acceleration and the existing
integration permits it, test a short closed-loop sequence where prefix
recovery follows each method’s own completed reconstructions. Otherwise,
state that end-to-end tracking remains untested.

Current-record learning may affect only subsequent records, as required
by the charter. Repeated evaluation clips must not become privileged
adaptation examples or cross-snapshot answer memory.

### Quality and cost

Measure token and exact-clip recovery, actual candidate simulations,
proposal/scoring time, prefix-cache work, memory, and total reconstruction
time. Include prefix-maintenance cost in any end-to-end tracking claim.

Do not convert “32 times fewer candidates” into “32 times faster.”
Measure the implemented system with comparable, uncontended timing.

Preserve fixed budgets for this first hybrid test. Do not simultaneously
introduce confidence shortcuts, abstention, adaptive routing or ensembles.

Predeclare tolerable quality loss and a meaningful runtime improvement.
The desired outcome is comparable reconstruction throughout the tested
trajectory at lower measured total cost—not merely higher candidate recall.

## Execution and preservation

Use a task-owned end-to-end path with logical evaluator/reconstructor
separation. Reuse verified components and run an early synthetic smoke test.

Preserve exact states, configuration, numerical settings, ranked candidates
and predictions before scoring. Back up actual required model bytes;
hashes alone are not backups.

Coordinate compute and source exclusions with the other agents.
No paid compute or unbounded experiment expansion is authorized.

## Final handoff

Lead with the answer to the original question:

“At the tested target snapshots, the correct token remained in B1’s
top 8/16/32 at these rates; whole-clip coverage was these rates.”

Then state separately whether:
- the small-budget hybrid was tested;
- its A2 prefix was static or genuinely maintained;
- it matched the reference’s quality at lower measured cost;
- the result held through drift or failed at particular snapshots.

Commit a concise report, structured evidence, reproducible commands,
frozen artifacts, costs and limitations. Open a task PR without merging.

A negative shortlist result is a completed scientific answer. A positive
shortlist result alone is not yet proof of a faster, equally accurate hybrid.