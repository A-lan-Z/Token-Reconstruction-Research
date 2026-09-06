# TRR-0006 paired context report

Decision: **positionwise_default**; harm status: **harm_excluded**.

Bounds are causal minus trained diagonal. Bootstrap endpoints have approximate coverage; CP endpoints have exact marginal one-sided coverage.

| Cell | Both | Causal only | Positionwise only | Neither | Causal exact % | Diagonal exact % | Token Δ pp [95%] | Token practical [L,U] pp | Exact Δ pp [95%] | Exact practical [L,U] pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pile__public_base | 30 | 4 | 12 | 1490 | 2.214 | 2.734 | -0.310 [-0.374,-0.246] | [-0.396,-0.222] | -0.521 [-1.042,0.000] | [-1.669,0.643] |
| pile__public_lora_2601 | 34 | 10 | 11 | 1481 | 2.865 | 2.930 | 0.076 [0.008,0.144] | [-0.020,0.174] | -0.065 [-0.651,0.521] | [-1.404,1.275] |
| finance__public_base | 312 | 49 | 38 | 1137 | 23.503 | 22.786 | -0.117 [-0.155,-0.080] | [-0.171,-0.067] | 0.716 [-0.456,1.888] | [-1.861,3.285] |
| finance__public_lora_2601 | 355 | 51 | 44 | 1086 | 26.432 | 25.977 | 0.042 [-0.002,0.084] | [-0.019,0.103] | 0.456 [-0.781,1.693] | [-2.231,3.138] |


## Qualification, limitations, and historical context

Main-matrix qualification failure: **not recorded in this result skeleton**. The retained fixture qualification is separate and does not establish the main truth result.

Exclusion limitations: P04 target-fit disjointness cannot be certified from the approved aggregate exchange because per-record IDs, source ranges, and replay sequence hashes are unavailable.; Available P04 source/sequence hashes are applied. Unavailable target-fit256 individual identities/ranges and replay sequence hashes prevent a complete cross-study disjointness certificate. No underlying private P04 ledgers were opened.

Historical A2 gap: retained as a separate historical denominator; TRR-0006 does not recompute it.

Inherited preparation cost: unavailable without new training; quality inference and cost qualification remain separate.
