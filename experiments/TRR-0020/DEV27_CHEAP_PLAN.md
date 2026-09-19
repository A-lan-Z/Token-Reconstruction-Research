# Development27 cheaper current-token directions

The qualified single-token CGLS4 solve alone takes about23ms per token in the uncaptured FP32
implementation, before current forward, full-vocabulary readout and cache commit. That motivates
a cheaper update before any large reconstruction sweep.

Fixed directions, with r=normalized_target-normalized_prediction and g=Jn^T r:
- polyak0.25, polyak0.5, polyak1.0: delta=gamma*||r||^2/||g||^2*g, with a zero-gradient gate.
  One transpose product; no JVP and no iterative linear solve.
- cg1: the exact first stable CGLS update, ridge multiplier1e-4, with the unused final transpose
  product omitted. One transpose plus one JVP. Require exact direction equality to CGLS1.

These are continuous-input updates. They do not select token candidates or verify discrete tokens.
Any later decoder must choose from the full vocabulary and count its committed-context forward.

CPU seed200066: dense formula checks in FP64 and FP32, zero RHS and exact CGLS1 equivalence.
Then use the same six public committed-context fixtures as dev27 qualification. Four rules,
three repeats each:24cases and72executions. Preserve full-step and half-step nonlinear residuals,
explicit linear residuals, and synchronized direction cost. The direction cap remains the public
median embedding norm. Current-prefix forward and cache commit are separately charged when a
decoder is built; these direction timings are not end-to-end inference.

Require exact archived current-forward anchors, unchanged qualified current-token J/JT sources,
and independent exact CGLS1 directions. Source labels remain unopened; this public numerical
diagnostic does not select a reconstruction method or claim accuracy. Resource preflight and
exclusive GPU guard precede execution. Largest public context first.
