# Adjoint least-squares inverse: prototype and qualification plan

The development21whole-prefix Krylov reconstruction failed to improve sufficiently, and development22factor inversion increased the public full-system residual in all12paired comparisons. Both results remain rejected. The next question is whether a least-squares direction can reduce the residual more efficiently by using the transpose response as well as the forward response.

The new prototype implements the analytic transpose of each own-position causal Jacobian, including RMS normalization, grouped query/key/value paths, rotary transforms, softmax attention, SwiGLU and residual additions. It uses no learned predictor, candidate set or model-parameter update. This is the transpose of the block diagonal causal derivative, not the full sequence Jacobian transpose.

A fixed CGLS recurrence approximately solves (J^T J+lambda I)delta=J^T b. Lambda is damping times ||J^T b||^2/||b||^2 per position, fixed throughout that solve. Count each forward and transpose product separately. The prototype compares damping0and1e-4. Tiny CPUfloat64 checks must independently match dense autograd blocks, verify the dot-product transpose identity, match a dense regularized solve, and preserve zero BOS right-hand sides.

Before any reconstruction grid, qualify actual-prefix transpose products against independent HF autograd on the archived public40/128-position fixtures. Then compare fixed iteration budgets4/8/16with the prior whole-prefix Krylov4/8/16direction at explicitly recorded work budgets. CGLS usesKforward andK+1transpose products, so equal iteration counts are not equal work. Record true residuals, nonlinear residuals if probed, wall time, all counts and memory. No favorable-result assumption, stopping threshold or reconstruction decision has been selected yet.

GPU execution requires a fresh resource preflight and no overlapping job. No development labels, A1 predictions or unavailable target-prefix calls may enter this public diagnostic. A later promising frozen reconstruction rule still requires the full dual-benchmark matrix before any replacement claim.
