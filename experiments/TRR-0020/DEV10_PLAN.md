# Development10: direct local quadratic update over every token
Dev9 complete56cells; hard-context soft optimization worsens prose. This next family removes the expensive128256-dimensional optimizer state. It maintains one discrete sequence, evaluates the prefix once, and differentiates each position's own cosine error with respect to its current embedding. Past token gradients are detached; use the independently tested TRR0017 eager diagonal derivative. All positions update together.

For current embedding z and gradient g, score EVERY vocabulary embedding e by
score(e)=(lambda*M*z-g).e - .5*lambda*e.M.e,
equivalent up to a constant to minus a first-order objective plus a quadratic distance from z. M=I(raw) or R*R^T(white), R the normalized prefix-derived transform. The table e.M.e is a direct weight calculation, not fitted. lambda=beta*||g*R^-T||^2/(2*max(current cosine error,1e-8)), clipped1e-6..1e6; raw usesR=I. beta=.25/.5/1/2/4 across two metrics.64iterations. Every token gets an explicit approximate score, then one direct argmax updates the sequence. No shortlist is supplied to an A2 verifier, no top-K, no per-token candidate trials, no offline learned parameters.

This is a new approximate numerical method: BF16 query/embedding matrix operands withFP32output, explicit eager attention, raw or white norm termFP32. It is not claimed equal to A2's exact discrete scores. Best whole-sequence objective and per-position best-error outputs use only observed activations; preserve all cycles/failures.

Independent CPU test: direct quadratic displacement expression agrees with factorized all-vocabulary scores up to a constant; for an identity linear forward the exact quadratic update recovers a token in one step, including the final vocabulary entry. Largest128public qualification3repeats with losses/outputs saved before equality checks.80developmentcells (10configs*8inputs), complete freeze before labels; no active canonical method.

Resources:prefix~1GiB,metriccache~1GiB, scorematrix127x128256 FP32~62MiB, no full-logit/moment state. Transformed norm build chunks4096x2048~32MiB. Two graph pools estimated<1.5GiB, total<=5GiB; retained6GiB cap,>=2GiBfree,9000MiBadmission,host>=8GiB/RSS<=10GiB,temp<80C. Expected<5min;timeout1200. Reject allocator/nonfinite/repeatability problems; preserve attempts.
