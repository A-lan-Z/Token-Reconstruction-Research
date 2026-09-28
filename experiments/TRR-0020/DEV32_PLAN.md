# Development32: observed-error budgets and shorter scalar solves

Development31's complete public matrix shows a meaningful initial-state effect:
at tau1, actual cosine-error ratios were .48754/.36807 for lengths128/40 versus
.77410/.76030 with the span4 rule. At warm64 the same large budget worsened errors
41x/281x. The16-iteration scalar solve costs about29ms per128-position update.
This is numerical diagnostic evidence, not reconstruction accuracy.

Test a per-position budget tau_i=clamp(factor*observed_cosine_error_i,0,1),
with factor .5,1,2. All observations come from the current mixture through the
available public prefix. No labels, shortlist or candidate verifier. Negative
floating-point errors map to zero; a zero budget leaves probabilities unchanged.
All128256 vocabulary logits remain eligible. This differs from development30,
which scales the local variance/span-limited *rate*, not the actual KL budget.

Use the qualified exponential path, but compare4/8/16fixed safeguarded iterations.
Fewer iterations are distinct approximate decision rules, not equivalent execution
optimizations. Return only a tested feasible lower bracket and report unused budget.
Do not silently replace the16-iteration rule or call an approximate solve exact.

Before GPU, verify unchanged scalar-budget16-iteration outputs against immutable
development31, then test variable budgets in FP32/64 with independent direct KL,
zero/negative errors, tiny positive errors, finite output, and root diagnostics.
An approximate solve can leave budget unused; this is measured, not grounds to
relax the feasibility check. Preserve every failed qualification.

Public matrix: lengths128/40 x warm0/64 x factor.5/1/2 x iterations4/8/16,
36cases, three repeats each. Same public seed200051; only H enters reconstruction.
Require original warm/initial-mixture anchors, all four baseline artifact hashes
from development31r1, independent BF16 surrogate-gradient checks, and unchanged
scalar-budget16-iteration full-logit controls. Record actual nonlinear cosine error,
per-position budget use, confidence, time and memory. All full logits and saved
outputs must repeat exactly. Save the same complete small diagnostics as31.
Reuse the corrected eager-loss lifetime; no retained graph across captures.

Qualify largest128geometry first under isolated fail-closed resource guards.
If this yields useful public progress, define and freeze a reconstruction matrix;
do not infer reconstruction quality from public synthetic activations.
Active canonical registry remains unchanged until a method is selected.
