# Development42: tiled whole-vocabulary KL scalar evaluation

Development41profiles the current optimizer, with unchanged controls and native
geometries. At128positions, isolated KL update work is63.81% of full-update GPU
work. Test a focused numerical implementation of that existing mathematical rule.

Preserve the original path_update preprocessing, constants, four safeguarded
scalar iterations, diagnostics, final update and all128256vocabulary coordinates.
Replace evaluate(t)'s materialized tilted logits and probability arrays with
two Triton reductions. Per1024-entry tile compute maximum tilted logit, stable
exponential sum and centered-gradient weighted sum. Merge alltiles with another
stable log-sum-exp reduction. Return the original KL and derivative formulas.
No candidate list, pruning, sparse support, learned predictor or parameter
fitting is introduced. This is a numerical variant, not presumed byte-equivalent.

First qualify12tiny independent CPUFP64cases: vocabulary sizes17/1023/1031 x
random/peaked/flat-gradient/zero-budget. Compare tiled scalar values/derivatives
to direct logsumexp plus autograd, and full four-iteration updates to the
unchanged original function. Include partial tiles and zero-rate behavior.

Then qualify GPUscalar and whole-update outputs on those cases and the two
frozen public full-vocabulary step16fixtures from the accepted profile.
Require unchanged original anchors, finite/exact repeats, active/budget
agreement, scalar agreementrtol5e-4/atol5e-5, final probability total-variation
<=5e-4, and independentFP64forward-KL feasibility within2e-5. Report every
logit/step/probability discrepancy rather than calling this exact.
Compare synchronized warmed CUDAgraph times on identical frozen inputs,
three groups of20calls each, rotating original/fused order. Record compilation,
capture and preparation separately. This isolated update timing is not decoder
latency or reconstruction accuracy. Largest127x128256cell runs first under the
exclusive guard. No model is loaded for the GPUkernel qualification.

Only after qualification should a full decoder be defined and frozen for
reconstruction, with all truth-opening and dual-benchmark requirements.
