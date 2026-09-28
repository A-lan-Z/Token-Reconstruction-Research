# Development26: conditioning of the full causal normalized inverse

The qualified development25 full causal operator improved the true linear residual at16 iterations
but gave a weaker nonlinear correction than the diagonal control. This diagnostic tests whether
input-column scaling improves fixed-budget global CGLS. It uses only archived public numerical
fixtures, supplied prefix weights and fixed random signs. There is no vocabulary shortlist,
source-label access, fitted auxiliary model or separate token verifier.

Four fixed right scalings:
- identity control;
- input_norm: each position's current input norm;
- position_probe4: inverse RMS Jacobian-column sensitivity per position, estimated from four
  fixed Rademacher output probes passed through the current normalized Jacobian transpose;
- coordinate_probe4: inverse sensitivity per coordinate, with equal-weight shrinkage toward
  the corresponding position mean of the same four squared-transpose probes.

The sensitivity floor is1e-6 times the median non-BOS position sensitivity. Every positive
scale is normalized to median1, and BOS directions stay zero. Solve in scaled coordinates
using the qualified global CGLS recurrence. Its regularizer acts in those scaled coordinates;
this changes the numerical method and is not claimed to preserve the unscaled solution.
The probes depend only on geometry and fixed seed200063+length, and are regenerated identically
for each repetition. Count four extra transpose products for probe-based scaling.

Qualification: CPU float64 independent dense scaling, scaled J/JT, regularized solve,
zero and BOS checks, seed200062. Then guarded FP32 actual-prefix diagnostic on the same
archived public128/40 fixtures. Configurations are four scaling rules times8/16/32 iterations
times ridge multipliers1e-4/1e-2 times two lengths:48 cells, three repetitions each.
Check the identity8/16 controls against archived development25 direction and residual arrays.
Measure scale setup and solve separately, plus their sum; preserve all raw directions,
scales, explicit residuals, and clipped half-step nonlinear errors.

Use full global linear residual and actual nonlinear mismatch, not the diagonal recurrence
as a proxy for true improvement. The half step uses the same public median-embedding-norm
direction cap as development25. These are public numerical comparisons, not token accuracy.
A promising result must then receive a fixed retrospective reconstruction grid and, if selected,
both canonical benchmark cells with complete active-method reporting.

Resource preflight precedes GPU execution. Largest128/32 configuration first; no dense actual-prefix
Jacobian, no overlapping GPU process, deterministic FP32 with TF32 disabled.
