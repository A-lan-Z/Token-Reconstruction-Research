# Track A result fragment — TRR-0003

## What the checkpoint-only pilot learned

The public-prefix arithmetic control was exact on 24 disjoint public validation records: the actual pinned prefix reproduced cached cut-4 activations with `max_abs=0`, `relative_l2=0`, and both allclose checks true. This makes a cache/model-forward mismatch an implausible explanation for the subsequent inverse behavior.

The reverse pre-norm fixed-point inverse was numerically finite but did not converge as a composed inverse. Across both styles and both target conditions, the continuous forward-cycle residual was non-monotonic at every budget ladder and remained above 1 at iteration 32. The nearest-embedding discrete cycle residual stayed roughly 0.92–0.99. The matched public-base cells therefore fail the matched inverse diagnostic itself; the shifted-target cells are not the primary explanation for the failure.

All four complete panel cells were emitted with `truth_opened=false` and passed the public prediction completeness/binding checks. Accuracy remains deliberately pending the unified footing truth gate; this fragment does not open or score panel truth.

| style / condition | i0 continuous | i1 | i4 | i8 | i16 | i32 | i32 discrete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pile / public_base | 1.453 | 1.147 | 2.035 | 1.355 | 2.710 | 1.777 | 0.975 |
| pile / public_lora_2601 | 1.423 | 1.123 | 2.827 | 1.456 | 2.311 | 1.634 | 0.922 |
| finance / public_base | 1.385 | 1.137 | 1.136 | 1.174 | 2.956 | 1.568 | 0.985 |
| finance / public_lora_2601 | 1.390 | 1.134 | 3.589 | 1.213 | 3.028 | 1.406 | 0.963 |

At the local branch level, lower branches often reach tiny residuals by 16–32 steps, while deeper branches intermittently remain unstable. At iteration 32 the worst mean branch is `L3_mlp` in all four cells:
- `pile__public_base` 0.017
- `pile__public_lora_2601` 0.013
- `finance__public_base` 0.045
- `finance__public_lora_2601` 0.023

At earlier budgets, `L1_mlp` is worst at iterations 2 and 4 for the full cells, and `L3_attention` is worst for some iteration-16 cells. This branch switching, together with large composed cycle error despite small local branch residuals, is evidence of a non-contractive or ill-conditioned composition rather than a simple failure to execute enough steps. The local residual is therefore a diagnostic, not a token-recovery success criterion.

| budget | branch calls / full cell | prefix layer evaluations / full cell | Pile base seconds | Pile shifted seconds | Finance base seconds | Finance shifted seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 64 | 1.478 | 1.592 | 2.282 | 2.120 |
| 1 | 128 | 192 | 0.832 | 0.853 | 1.494 | 1.478 |
| 2 | 256 | 320 | 0.895 | 0.922 | 1.586 | 1.568 |
| 4 | 512 | 576 | 1.075 | 1.091 | 1.820 | 1.845 |
| 8 | 1024 | 1088 | 1.398 | 1.401 | 2.096 | 2.095 |
| 16 | 2048 | 2112 | 2.066 | 2.102 | 2.777 | 2.781 |
| 32 | 4096 | 4160 | 3.404 | 3.458 | 4.136 | 4.113 |

The branch-call column follows the fixed two branch evaluations per step: it is 16 × budget for a full eight-record cell. Each cell also includes two cycle forwards per record; the prefix-layer column includes those cycle forwards. Candidate-prefix simulations are zero.

## Cost and limitation classification

Each process had zero fitting or adaptation steps. Cold startup for loading and hashing the pinned public model was about 1.8–2.0 seconds. The retained public snapshot is 2,482,971,357 bytes on disk; the loaded cut-4 prefix and embedding state is 1,011,908,864 bytes. Preparation peak allocation was about 2.47 GB and the post-reset method peak was about 3.91 GB allocated / 5.39–5.40 GB reserved, with roughly 3.40 GB process RSS. The complete seven-budget ladder took approximately:
- `pile__public_base` 11.148 seconds resident inference
- `pile__public_lora_2601` 11.419 seconds resident inference
- `finance__public_base` 16.191 seconds resident inference
- `finance__public_lora_2601` 15.999 seconds resident inference

These are raw sums over the fixed seven-budget ladder. Per-record times are retained in `runtime_analysis.json` rather than averaged across different Finance valid lengths.

The dominant demonstrated limitation is numerical optimization/conditioning: the public forward control is exact, every branch remained finite, but the composed reverse map does not settle to a useful boundary cycle. The same pattern in matched public-base inputs rules out target-weight mismatch as the primary cause, and the similar shifted-target behavior gives no evidence that input-distribution transfer alone explains it. There is no fitted decoder capacity in this arm to assess. Projection adds a further problem—the discrete cycle remains near unit relative error—but it cannot rescue the already poor continuous estimate. Runtime grows with the fixed ladder, yet the inverse fails its matched control before an accuracy/utility claim can be made.

## Status and next decision

This is an exploratory Track A diagnostic, not a replacement claim and not a complete dual-benchmark comparison. If Track A is retained, the next experiment should test a solver with explicit contraction, line-search, or Jacobian control on the matched public validation control and require simultaneous improvement in continuous cycle residual and token accuracy. Stable matched convergence with discrete recovery improvement would justify revisiting the path; continued branch oscillation or high discrete error should end further fixed-point refinement in favor of a standalone learned decoder or another structurally constrained inverse.

Raw evidence and hashes are listed in `experiments/TRR-0003/track_a/runtime_analysis.json`; per-cell evidence remains under `outputs/TRR-0003/track_a_diagnostics/`.
