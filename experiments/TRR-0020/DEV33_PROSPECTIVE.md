# Next direction: causal full-vocabulary optimization of one current token

Development32 completed56reconstruction cells and did not meet accuracy/speed.
Its faster factor2/4-iteration rule gets253/254matched prose but246/254shifted prose
at1.075s/128positions. The16-update checkpoint is much faster (.290s) but only
225/254and222/254prose tokens. Increasing scalar-solve precision does not reliably
improve reconstruction.

The post-freeze diagnostic changes the next action: all eight final shifted-prose
errors for factor2/4iterations had been correct at at least one saved earlier
checkpoint. Three are finally over.9confident. Whole-sequence loss rose on24/28
of64updates for those two rows. This is not evidence that only diffuse uncertain
mixtures need sharpening, and it does not prove causal interference is the cause.
No oracle checkpoint selection becomes a deployed rule.

Investigate causal full-vocabulary optimization: commit earlier *emitted* tokens,
hold their per-layer K/V fixed, and optimize all128256 probabilities for only the
current token. Emit one argmax directly, then commit its actual embedding through
the prefix for subsequent positions. No top-K proposal, vocabulary mask, separate
candidate verifier, learned offline inverse, or target-prefix oracle. Initialization
can use the existing prefix-derived full-vocabulary similarities, with every finite
logit retained. This is not lookup-free.

This differs from development27, which made cheap continuous embedding corrections
after a64-step joint warm start and then used a metric readout. It also differs from
development9's parallel hard-context optimization. A targeted search in tracked
scripts/research found no existing sequential full-vocabulary mirror method; inspect
relevant older code again before implementing the decoder.

Initial mathematical primitive:
x=pE; y=F_current(x | committed history);
loss=1-normalize(y) dot normalize(H_current).
Obtain the input cotangent with the already qualified single-position VJP, then
G_p=(d loss/dx) E^T. This is a gradient over every vocabulary token. The scalar
probability update never selects token candidates for model evaluation.

Phase1: independent CPU autograd check on tiny linear and residual nonlinear
maps, fixed history, zero/near-zero normalization cases, FP32/64, raw probability
gradient and chained logit gradient. These tiny dictionaries are numerical fixtures,
not reduced-vocabulary reconstruction runs.

Phase2, conditional on CPU checks: actual public prefix qualification using the
existing FP32 single-position forward/VJP (development27). Verify its unchanged
forward/derivative anchors, the new probability chain rule, and public one-token
updates at early/middle/late positions for lengths128/40. Public synthetic past
tokens may define this numerical fixture; this is not reconstruction accuracy.
All real token choices must subsequently use only emitted history.

Start with the already qualified observed-error KL updates at factors.5/1/2 and
four scalar iterations; compare fixed1/2/4/8current-token updates in public
diagnostics before freezing a decoder grid. Preserve actual loss/confidence,
full-vocabulary finite logits, exact repeatability, forward and VJP counts,
all prefix commits, preparation/capture/inference/IO cost, and memory.

Speed is uncertain: causal commitment removes joint-history changes but gives up
large batched matrix products. Measure current-token forward, VJP, two vocabulary
matrix products, probability update, and commit separately. Do not infer end-to-end
speed from arithmetic counts. CUDA replay requires exact eager/output qualification.
Native context lengths are preferred; numerical padding/batching is not assumed
neutral and may not be silently used as a memory workaround. Estimate and qualify
the full128-position capture/cache geometry before releasing a decoder matrix.
Use isolated, restart-safe, fail-closed resource guards.

No real-prefix GPU experiment or decoder has yet been implemented for this direction.
The complete no-shortlist goal, required dual benchmarks, and recovering-prefix
trajectory limitation remain unchanged. A trust-region acceptance rule on the joint
objective and Gini regularization remain alternatives, not prohibited methods.
