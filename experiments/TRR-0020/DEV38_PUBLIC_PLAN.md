# Development38 public first-token qualification

The prior turn made scientific progress:512public full-vocabulary decisions
completed, showing that a common context shift is inadequate. No replacement
was established. This study tests the separate BOS-only direct decision.

Use the unchanged byte-bound BF16 native256-row response table from TRR-0014.
Every128256row is scored by direct cosine; argmax is final, with no candidate
list or A2verification. Freeze this rule before execution. Reuse the16common
public fixtures in both saved observation precisions and add128public token IDs
drawn uniformly over the entire vocabulary with CPU seed38038. Keep duplicates
or special IDs; do not filter them after observing results.

Generate each additional observation separately as [BOS,public_id] using the
public prefix: BF16/nativeSDPA and FP32/eager are explicit separate numerical
conditions. Load those prefix instances sequentially. No hidden target model,
source labels or real benchmark observations are involved.

Compare the scoring formula with an independent direct vector-cosine reduction
over the full table on the first two common fixtures in each precision. Direct
reference blocks of256are only an independent numerical reference, not an
execution-equivalence or candidate-pruning claim. Require scores close within
rtol2e-5/atol2e-6 and the same full-vocabulary argmax.

Run all288public decisions three times and require identical full score vectors.
Preserve the full first-repeat scores, chosen IDs, observations and public IDs.
Only summarize correctness after the complete public matrix is frozen. Keep
preparation, target generation, scoring, synchronization, IO and memory separate.

If this sanity check is adequate, implement the planned first-token constraint
in the joint optimizer. Keep all remaining logits and the native full-sequence
geometry. The control must exactly reproduce stage32factor2/four-iteration
outputs. Qualify whole decoders, gradients, actual first-token embedding use,
fixed-token persistence and largest128-position resource geometry before running
the full frozen retrospective grid. Public checks are not benchmark accuracy.
