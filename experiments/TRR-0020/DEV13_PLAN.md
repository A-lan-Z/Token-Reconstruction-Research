# Development13: continuous inputs with a vocabulary-distance penalty

This is a prospective independent development experiment. No GPU work may overlap the live canonical comparison or development12. It is not an active canonical method.

Optimize an embedding-sized vector at every unknown position. Feed that continuous vector through the frozen public prefix and minimize observed cosine error. Add a small penalty for distance to the nearest token embedding, found by scoring every vocabulary entry. This nearest token supplies a regularizer and final output; it is never passed as a shortlist to a separate A2 verifier. The prefix forward always consumes the current continuous input vector.

Compared with the query parameterization, this needs only one vocabulary matrix product per optimization step. Compared with the earlier unconstrained embedding and periodic-snap attempts, it adds a continuous distance penalty rather than hard replacement. It may still become trapped near incorrect tokens; no quality claim is made in advance.

Initialization is the full-vocabulary scale80 metric soft mixture, with no top-K. Optimize128 steps, Adam moments(.9,.995),epsilon1e-12,no bias correction, no decay, reset moments every32steps. Rates .001/.003/.01 crossed with penalty strengths0/.003/.01/.03 give12rules and96development cells. Penalty ramps from0 to its full strength over steps32..64. Clip coordinates only to the min/max of the entire embedding dictionary, preserving every real vocabulary embedding. Select nearest-embedding tokens at steps0/16/32/64/128, minimum whole objective after64, and per-position minimum observed error after64. The zero-strength arms isolate the penalty's contribution.

Use the same8retrospective development inputs, freeze all96cells before opening current labels, record all outputs and loss/error/distance traces, and qualify largest128geometry with3exact repeated outputs and traces before releasing each configuration. Nearest-token matrix products use BF16 inputs/FP32 accumulation, with FP32 embedding norms: this is a declared approximate nearest decision, not exact FP32 Euclidean equivalence.

Resource estimate: prefix about1GiB, metric table1GiB, less than0.2GiB score/initialization work, embedding variables and moments about4MiB, prefix activation graphs; expected below5GiB reserved. Retain6GiBcap, admission9000MiBfree, mandatory2GiBfree,hostavailable8GiB,RSS10GiB,temp80C. Estimate under5min afterGPUavailable,timeout1200. Stop on resource or repeatability failures.

The independent CPUfloat64 test compares factored all-vocabulary distances and their selected-center gradients with brute-force pairwise distances. It also checks that coordinate bounds preserve the complete random test dictionary. These are mathematical tests, not reconstruction runs.

Related primary source: [Prompt Inversion Attack against Collaborative Inference of Large Language Models](https://arxiv.org/html/2503.09022v2), equation13, motivates a nearest-embedding penalty. Its complete method uses candidate lists and reranking, which do not satisfy this task. We borrow only the continuous regularizer, use a cosine activation objective, and retain direct full-vocabulary final selection. Published accuracy does not transfer to our access model, prefix or benchmarks.
