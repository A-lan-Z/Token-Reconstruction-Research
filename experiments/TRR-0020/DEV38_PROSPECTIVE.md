# Development38 prospective question: decide the first token from discrete responses

Development36 showed misleading soft-mixture paths at the first unknown
position. Development37's direct BOS-response table identified all16public
first tokens, whereas its common-shift continuation failed. Do not promote the
common-shift rule or interpret the BOS sanity check as benchmark accuracy.

Next investigate a direct first-token decision followed by full-vocabulary joint
optimization of the remaining positions. BOS is the only supplied known token.
The first unknown token would be reconstructed by scanning every BOS response
and immediately freezing its argmax. It then counts as reconstructed history,
not an additional true prefix. No top-kset or A2candidate verification is used.

The inspected stage32/33rules do not implement this. This is a hypothesis, not
a general novelty claim. The direct decision may be wrong under prefix mismatch
and could harm later recovery. Evaluate matched and shifted conditions with no
truth-based fallback or selective application.

Before a decoder grid:
1. Check first-token direct scores on both saved public FP32 and BF16 fixtures
   and a fixed additional public-token panel.
2. Inspect the joint optimizer; declare exactly how initial logits, frozen
   positions, gradients, normalization and timing change.
3. Keep an unchanged control that reproduces archived outputs exactly. Charge
   lookup preparation, first-token scoring and changed-prefix rebuild costs.
4. Qualify the largest128-position geometry and scores. Do not assume slicing,
   sequence-length changes or batching preserves outputs.

Initially retain factor2/four-scalar-iteration joint updates. Freeze a finite
set of update counts and paired rules with/without the reconstructed first-token
constraint before running and scoring the complete retrospective development
matrix. A promising fixed rule must then be registered and run in both canonical
setups. No selected contender currently exists, and public-prefix checks do not
substitute for an actual recovering-prefix trajectory.

Retained alternative: token-dependent low-rank corrections to every cached
response. The common-shift failure does not answer that question. Extra probes
and full-table products must earn their cost; do not simply enlarge probe count
without evidence that the context correction can be predicted from permitted
prefix information.
