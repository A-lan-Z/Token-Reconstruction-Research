# TRR-0017: quantify candidate-free inversion and reuse A2 context

The existing decoder repeats its committed context for every candidate. This change supplies a shared-context execution adapter, checks every candidate output for equivalence and measures the complete reconstruction cost. It also evaluates two new whole-sequence candidate-free inversion methods and preserves unsuccessful development variants.

Validation: nine numerical tests; full 1,360-cell freeze; both canonical setups;56-cell active-method matrix; byte-identical native/shared tokens, candidate IDs, cosine and MSE arrays for 272 observations; archive hashes verified.

Result: coordination/results/TRR-0017.md
Manifest: experiments/TRR-0017/manifest.json

Local draft only; publication approval remains unresolved.
