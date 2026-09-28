# Development44: exact elementwise fusion around native reductions

Implement a distinct execution path for the unchanged factor2/four-iteration
whole-vocabulary KL update. Keep every original torch log_softmax, softmax,
logsumexp, sum, min/max and their native input shapes/strides. Fuse only:
1. normalized-gradient and probability-product construction;
2. centered-gradient, first-moment and squared-moment products;
3. tilted logits;
4. scalar safeguarded-Newton bookkeeping; and
5. final recentering and active-row selection.
AllFP32operations retain order and rounding boundaries, with fused multiply-add
disabled and correctly rounded division. A separately discarded derivative at
the initial upper bound need not be evaluated; its reported KL remains unchanged.
All returned arrays and diagnostics must be byte-identical, including signed zero.

First qualify a torch-only refactoring against the original on24CPU cases:
FP32/64 x vocabulary17/1023/1031 x random/peaked/flat-gradient/zero-budget.
Then qualify each Triton primitive and complete update on the12FP32cases,
two frozen full-vocabulary public step16states, and new cold/late public states.
The latter fixtures must come from unchanged original decodes and be frozen
without benchmark labels. Preserve all failed exactness attempts.

Require exact immutable inputs, repeatability, original/full-array anchors,
and eager/replay outputs before timing. Use three groups of20paired warmed
replays with rotating method order. Report compilation, setup and capture
separately. Largest127x128256case first under exclusive resource guard;
no padding, batching or vocabulary truncation is introduced.

This is not yet a decoder evaluation or a new active canonical method. Only a
qualified execution path advances to whole-trajectory output equivalence and
a complete frozen reconstruction comparison. The speed ceiling may be lower
than reduction fusion; neither speed nor accuracy is assumed.
