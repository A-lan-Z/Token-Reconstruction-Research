# Development40: diagnose homotopy linear-solve failure

This public numerical experiment distinguishes three causes of the development39
residual: unstable normal equations, ridge bias, or an inadequate Krylov space.
It does not change token candidates, use hidden source truth or fit parameters.

Retain the20qualified dev39fixtures; execute the eight length128/40 x
activation/embedding x strength1/.5points. Reproduce the original16-step
GMRES answer exactly. Construct a32-step twice-reorthogonalized Arnoldi space.
At16and32steps compare the original FP32normal-equation ridge1e-6 solution,
a CPUFP64 SVD least-squares solution to the identical augmented ridge problem,
and an unregularized CPUFP64SVD least-squares solution. SVD cutoff is1e-12.
Report actual J*d-b residual, projected residual, condition estimates, basis
orthogonality, direction norms and phase timings. Repeat all arrays three times.
The first16Arnoldi columns are required to match a fresh16-step construction.
This is sharing deterministic diagnostic work, not changing decoder geometry.

Qualify the audit on eight independent tiny dense CPU systems: well-conditioned,
nonnormal, ill-conditioned and zeroRHS, eachFP32/64. Compare reduced coefficients
with explicit SVD pseudoinverses and original answers with the unchanged solver.
Preserve rank-deficient cases; use no hidden target token identities.

Largest128/activation/strength1/32steps runs first under the existing fail-closed
GPU guard. No extra GPU job runs concurrently. Each actual-prefix point records
the full basis, Hessenberg matrix, RHS, directions and actual residuals so the
diagnostic is independently reproducible. Fixture and source hashes are bound.
No reconstruction accuracy or baseline-speed claim is made from this diagnostic.
If normal equations or ridge do not explain the failure, do not silently label
this an improved inverse. Choose a different solve/path based on the evidence.
