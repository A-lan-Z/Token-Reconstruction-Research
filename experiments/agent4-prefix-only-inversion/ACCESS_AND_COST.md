# Access, source and cost interpretation

Only `solver.reconstruct(prefix, observation, settings, bos)` makes the primary
reconstruction decisions. It imports no source loader, truth scorer, lens, A1,
B0/B1, teacher, or LM. Each record starts with BOS; all other prefix IDs are its
own commitments. Public model parameters have requires_grad=False; autograd
only returns the current provisional embedding gradient. Full recomputation
has no committed KV cache to contaminate. The final observation row is only
indexed after the prior token is committed. No online model update is used.

Capture, prediction and scoring are separate processes. This is code-path
separation on the same Unix account, not an OS-enforced sealed evaluator. The
sandbox launcher is unavailable (missing bwrap). Main prediction only opens
observations and shape/group metadata plus the public assets. All predictions
and detailed candidate traces use create-only files and are hashed in a final
freeze receipt. Scoring verifies every named receipt before opening source IDs.
The tiny public CPU diagnostics deliberately capture and score in one process,
with prediction files created before comparison; these are development only.

The final Gutenberg paragraphs and generated identifier records were selected
after the frozen optimizer/configuration. They are unused within this task.
Identity-only Agent1/2 source ledgers are hash-bound. No cross-dataset global
identity equivalence is established, so no canonical or repository-wide fresh
confirmation claim is made. P03 source material was never accessed.

The A1+A2 comparator uses the retained historical Alpaca lens, native top512
proposal with stable tie ordering, first256 candidates, and native cached
_candidate_hidden execution. It uses direct cosine, its own committed prefix,
and no early exit/abstention. Short unpadded rows and record batch1 adapt the
geometry. A synthetic public fixture produced identical ordered candidates
and outputs to native decode_policy. Model precision and SDPA are preserved.

Reconstruction wall includes raw vocabulary-scoring setup, candidate execution,
backward computation, discrete scoring, prefix reconstruction/cache commit,
and Python loop overhead. Per-process wall also includes public asset loading
and JSON I/O. No target-prefix maintenance is executed. Historical A1 fitting
cost is not remeasured; the comparator is a warm supplied-lens component. The
pilot does not claim an end-to-end cold-start cost win on that basis.

Complete records, every post-BOS token, timed-out suffixes and exhausted tokens
remain in their denominators. Exhaustion emits the best actually forward-tested
candidate. Timeouts do not invoke another method or enumerate the vocabulary.
Correct-token discovery and wrong final selection are determined only after
freeze. A small residual is not itself evidence of correctness under a wrong
reconstructed prefix or surrogate mismatch.

The global dual benchmark matrix is NOT RUN. This task is locally registered
and obeys the packet's explicit ban on global-registry changes. No comparison-
complete or state-of-the-art claim is made.
