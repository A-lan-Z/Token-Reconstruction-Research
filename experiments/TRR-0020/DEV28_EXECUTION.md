# Development28 execution details

The fixed36-cell public diagnostic follows DEV28_PROSPECTIVE.md. The two nonidentity
matrices use FP32 symmetrized R R^T, a1e-3 mean-diagonal ridge, cholesky_ex and
cholesky_inverse, with the inverse symmetrized before median-diagonal normalization.
The CPU reference passes four SPD construction checks and twelve dense direction checks
across FP32/FP64, including zero residual/gradient cases and exact original identity outputs.

Build the existing prefix metric, including its unused embedding table, and charge that
preparation rather than hiding it. Save R, regularized A, inverse A and both normalized
operators. Independently inspect eigenvalues in CPU FP64; reject nonpositive metrics.
Require max absolute |A inverse(A)-I| <=5e-3 on the actual2048-wide metric. This tolerance
is frozen before the public run. Preserve failure evidence and do not relax it after results.

Use the unchanged public prefix in FP32 and variable native K/V lengths from dev27.
For length128 then40, evaluate positions last/middle/1, all three metrics and both rules,
three repetitions each. These are108direction executions and no reconstruction cells.
Require six exact current-forward anchors and36exact identity executions against dev27,
including direction, linear residual, nonlinear residuals and clipping. Save each result
before validity/repeat gates. Timing separates prefix and metric setup, cache construction,
current forward, direction (including added metric product), validation and artifact I/O.

The largest context (length128, position127) must finish with resource margin before
shorter contexts proceed. One guarded GPU process; n.guard retains its existing6GiB
reserved cap. No candidate selection, source-truth access, active canonical registration,
or reconstruction effectiveness claim belongs to this diagnostic.
