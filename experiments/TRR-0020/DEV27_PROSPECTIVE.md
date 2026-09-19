# Development27: committed-context continuous inversion prototype

This is independent CPU work while the fixed development26 matrix runs.
Prior continuous methods optimized a whole sequence, or changed all positions in parallel.
This prototype instead fixes the already recovered past, solves only the current embedding,
selects its final token from the full vocabulary, and then commits that token's prefix K/V
for subsequent positions. The commit output must never be used to reject or rerank that token;
it is context construction, not a separate candidate verifier. Count every commit forward.

Current work qualifies only the one-position map and its analytic J/JT against full-sequence
autograd on a tiny public random prefix. Cache keys/values are immutable during the current
continuous solve. The current key and value remain variable, including their attention paths.
Tests cover empty past, later positions, raw and normalized outputs, and regularized dense solves.

CPU seed200064, FP64, four positions, width8, two layers, two query heads and one KV head.
No source labels, vocabulary shortlist, auxiliary fitted model or unavailable target is used.
The eventual GPU prototype would require an explicit resource preflight, independent actual-prefix
qualification, a fixed initialization/update/readout rule, and an honest cost comparison.
Single-token matrix-vector work may be bandwidth-bound and slower; no speed claim is made.
No GPU run or reconstruction grid is selected by this document.
