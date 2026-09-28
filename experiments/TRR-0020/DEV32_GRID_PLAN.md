# Development32 cold full-vocabulary reconstruction grid

The36-case public diagnostic is complete and archived. Cold updates with factor2
and four scalar iterations reduce observed cosine error to .47941/.34305 of the
starting value at lengths128/40, close to the16-iteration .47267/.34179 result.
The warm64 rules worsen T128 mean error, so this grid evaluates cold iterative
rules specifically. This follows the public evidence and the speed objective;
it is not a smaller replacement for required canonical validation.

Freeze seven exploratory rules: factor.5/1/2 x scalar iterations4/8, plus factor2
with16iterations as the more precise reference. No warm optimization: preserve the
existing full-vocabulary initialization and apply64observed-error-budget mirror
updates. Budget_i=clamp(factor*current_cosine_error_i,0,1); finite logits for the
entire vocabulary, no shortlist, no separate candidate verifier, no model updates.
The initialization/readout use prefix-derived tensors and are not lookup-free.

Save fixed checkpoints0/1/2/4/8/16/32/64 and the existing best-objective and
per-position best-observed-error diagnostic selections. Token output is full-logit
argmax. The extra early checkpoints expose whether useful accuracy arrives before
64steps. Report checkpoint times as observed times, which include the update already
executed after that checkpoint's evaluation; do not call them optimal standalone
early-stop runtimes. Total includes warm reset/evaluation,65mirror evaluations and
64backwards, transfers/diagnostics/synchronization; capture and asset preparation
are separately charged. No new target labels choose runtime stopping.

Matrix: seven rules x the unchanged eight retrospective development records,
56cells. All56outputs must freeze before scoring; preserve source, environment,
weights, observation/metadata hashes, commands, work counts, times and memory.
Run one isolated process per rule, factor2/16iterations first at T128. Require:
- original initialization tokens/traces and mixture anchors on all56records;
- repeated public outputs/traces and exact eager-versus-replay equivalence for
  both native lengths for each rule;
- an independent BF16 surrogate-gradient formula check;
- first-step public token/error/confidence equality with development32's matching
  one-step diagnostic for each configuration and length;
- direct FP64 KL checks of every position at steps1/9/33/64 during the eager
  public qualification, including per-position budgets, with feasibility tolerance
 2e-5. These validation computations are excluded from timed reconstruction.
- fail-closed resource and source guards; output-preserving failed attempts;
- negative truth-opening gates for missing cells, source changes and edited output.

No active canonical method is selected in advance. If a rule meets the development
accuracy/speed requirement, freeze its decision/checkpoint and run both canonical
setups with paired controls, preserving the complete active matrix. This grid alone
cannot establish baseline equivalence or efficacy on a recovering-prefix trajectory.
