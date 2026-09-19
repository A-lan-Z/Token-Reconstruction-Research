# Supplied-prefix qualification for normalized local inversion

Use only the archived public synthetic40/128-position fixtures from development21. The frozen CPU reference including normalization-floor cases passed. Load supplied public BF16 parameter values as FP32, with native FP32 rotary constants and TF32 disabled. No unavailable target weights, development labels or A1 outputs are read.

At128positions first, then40, require the unchanged raw FP32forward to reproduce the archived tensor, and its normalized output to exactly match independent installed HF eager output. Compare normalized own-position JVP and VJP at positions1, midpoint and last to independent HF autograd with a tangent/output-gradient supported at only that position. Fixed tolerance:rtol5e-4/atol5e-5. Preserve all outputs before gates.

Then exercise CGLS4/8and ridge multipliers1e-4/1e-2on the normalized observed-difference RHS, three repetitions each. Require finite and repeated directions, exact zero BOS direction, and recurrence residual agreement with explicit J(delta)-rhs usingrtol2e-4/atol5e-6. Record actual residual size without imposing a favorable-result gate. This is eight public linear configurations and twelve derivative comparisons, not a reconstruction benchmark.

The largest prior FP32forward/autograd probe used2.254GiB. Estimate below3GiB, enforce the existing6GiB process cap, exclusive admission with at least9000MiB free,2GiB free throughout,8GiB host available,10GiB child RSS and below80C. Expected runtime below90seconds; external guard timeout900seconds. No dense real-prefix Jacobian is constructed. No reconstruction grid or success claim follows until the full method is separately frozen and evaluated.
