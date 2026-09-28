# Development44 prospective: preserve reductions while reducing memory traffic

Development43found a useful speed improvement but worse prose reconstruction.
The scalar change is mathematically equivalent and passed tight local numeric
checks; complete trajectories nonetheless differ. Do not call it an exact port.

Next test only elementwise fusion around the unchanged original torch
log_softmax, softmax, sum, min/max and logsumexp reductions. Candidate targets:
constructing the normalized gradient and its probability-weighted products;
forming tilted logits; and the final recentering/active-row selection.
Keep the original FP32operation order and rounding boundaries. Disable fused
multiply-add; use correctly rounded division where the original requires it.
Do not replace a reduction, change its input geometry or reuse an approximately
equal tensor. Retain all128256coordinates, constants and stopping rules.

Require byte-exact full update arrays and every diagnostic against the original
function on tiny edge cases, full-vocabulary frozen public fixtures, and warm/
near-converged public states. Check eager/replay equivalence and input immutability.
Measure isolated speed only after exactness passes. If it passes and is useful,
qualify complete decoder trajectories and then freeze a retrospective grid.
Any failed exactness attempt remains a preserved numerical variant and is
excluded from the exact optimization claim.

The expected ceiling is lower than changing the reductions themselves. This
is an execution experiment, not a new inverse or a claim that current quality
already meets the objective. Other scientifically justified directions remain
in scope under the charter. Canonical claims still require the complete active
matrix. No new active method is registered by this prospective note.
