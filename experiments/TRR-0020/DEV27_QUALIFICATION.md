# Development27 actual-prefix qualification

The revised CPU reference passed4cached-forward,24derivative and8dense-solve checks, with
maximum dense direction error1.50e-13. Preserve the initial failed CPU attempt and its
post-convergence diagnostic. The new stable solver has a latched32*dtype-epsilon relative
normal-residual gate. Fixed-budget products continue; this is numerical stabilization, not
an execution-equivalent change or a measured speedup.

On archived public development21 random-token fixtures, qualify the FP32 one-position map
at128/40 native lengths and positions last/middle/1, largest context first. Build committed
past from the known synthetic public token IDs. This is a numerical fixture, never a hidden
reconstruction benchmark. Cache outputs must match independent HF whole-prefix outputs within
rtol5e-4/atol5e-5. Preserve the exact archived HF target anchor.

For each selected position, compare the noisy current embedding map against independent HF
layers using the same immutable past K/V via a disposable read-only cache. Check raw and
normalized analytic JVP/VJP against autograd, seed200065+length+position. The current key/value
paths remain differentiable; no past token or hidden source token is optimized or read.

Fixed normalized linear matrix:6contexts *3iteration counts16/8/4 *2ridge multipliers1e-4/1e-2
=36cases,3repetitions each. Preserve direction, explicit residual, recurrence residual, clipping,
convergence iteration, and both half/full-step nonlinear errors. No token reconstruction
accuracy is calculated. Prefix commit work, preparation, solve and artifact I/O are recorded.

Require source hashes matching dev27_cpu_reference_r1, deterministic CUBLAS and TF32off.
A live resource preflight and the exclusive GPU guard are mandatory. Do not overlap with
development26; wait for its exact handle to finish. Maximum estimated peak3.5GiB under6GiB cap,
based on prior actual-prefix qualifications near2.3GiB and small current-token/cache tensors.
No reconstruction grid is selected yet.
