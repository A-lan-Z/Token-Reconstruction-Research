# TRR-0008 independent score review

Status: **complete** (`INDEPENDENT_SCORE_REVIEW_COMPLETE`). This review used only the serialized score and per-cell record evidence. The reviewer did not open private truth, rerun scoring, or modify scientific outputs.

## Provenance

- Score: `experiments/TRR-0008/evaluation/score_v1.json`; SHA-256 `e30ba5e4a05a2bc8fbb0869dd3ecffd6b7d570cd8af434c87d125390efdba453`.
- Decision contract binding: `141a3a506e20e1eadd3be1beb53c2f01508e9b651c93d19b2f3d9ff72344bcf3`.
- Precision40 timing binding: `a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`.
- Public freeze and truth-binding header hashes matched their serialized bindings; the truth sidecar itself was not opened by this review.

## Frozen comparison scope

The accepted arithmetic covers exactly three per-cell families: candidate versus reference, candidate versus current residual, and candidate versus improved diagonal. No pooled cross-cell aggregate is used.

## Primary candidate versus reference

| Cell | Records | Gains | Losses | Exact delta | Exact CP LCB | Token delta | Token LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| pile__public_base | 384 | 8 | 4 | 0.010416667 | -0.021171278 | 0.006151575 | 0.003649937 |
| pile__public_lora_2601 | 384 | 7 | 8 | -0.002604167 | -0.037412248 | 0.005167323 | 0.002993768 |
| finance__public_base | 1024 | 90 | 40 | 0.048828125 | 0.014206508 | 0.001899299 | 0.001145729 |
| finance__public_lora_2601 | 1024 | 121 | 36 | 0.083007812 | 0.046148775 | 0.001999262 | 0.001222625 |

For the preregistered Finance public-base primary cell: 90 gains, 40 losses, and 894 ties. The independently recomputed exact CP LCB is `0.014206508`, matching the serialized value. The exact and token practical routes fail (`0.014206508 < 0.05`; `0.001145729 < 0.01`), while the primary positivity criterion passes.

## Direct contrasts

Each row below is a separate frozen cell; no cross-cell gain/loss pooling is reported.

### Candidate versus current residual

| Cell | Records | Gains | Losses | Exact delta | Exact CP LCB | Token delta |
|---|---:|---:|---:|---:|---:|---:|
| pile__public_base | 384 | 6 | 8 | -0.005208333 | -0.038881207 | 0.003280840 |
| pile__public_lora_2601 | 384 | 5 | 7 | -0.005208333 | -0.036697520 | 0.002768209 |
| finance__public_base | 1024 | 73 | 40 | 0.032226562 | -0.000528207 | -0.000261443 |
| finance__public_lora_2601 | 1024 | 73 | 40 | 0.032226562 | -0.000528207 | -0.000023068 |

### Candidate versus improved diagonal

| Cell | Records | Gains | Losses | Exact delta | Exact CP LCB | Token delta |
|---|---:|---:|---:|---:|---:|---:|
| pile__public_base | 384 | 10 | 4 | 0.015625000 | -0.017885161 | 0.002952757 |
| pile__public_lora_2601 | 384 | 10 | 6 | 0.010416667 | -0.025403205 | 0.001619916 |
| finance__public_base | 1024 | 88 | 41 | 0.045898438 | 0.011328336 | 0.004598302 |
| finance__public_lora_2601 | 1024 | 100 | 39 | 0.059570312 | 0.024112238 | 0.003391055 |

Exact count deltas and CP bounds match the serialized fields for all 12 accepted per-cell rows. The serialized token rate uses float32 record-level means; the independent integer token-count check differs by at most `2.65463599974e-09` (all within the diagnostic tolerance `3e-9`).

## Safeguards and timing

| Cell | Exact safeguard LCB | Token safeguard LCB | Result |
|---|---:|---:|---|
| pile__public_base | -0.017418475 (min -0.05) | 0.004080546 (min -0.01) | PASS |
| pile__public_lora_2601 | -0.033276136 (min -0.05) | 0.003260336 (min -0.01) | PASS |
| finance__public_base | 0.018450667 (min -0.05) | 0.001276450 (min -0.01) | PASS |
| finance__public_lora_2601 | 0.050683403 (min -0.05) | 0.001345656 (min -0.01) | PASS |

| Timing cell | Mean ratio | 95% CI | Result |
|---|---:|---:|---|
| finance__public_base | 1.041115697 | [1.027319138, 1.054912256] | PASS |
| finance__public_lora_2601 | 1.054514350 | [1.033895994, 1.075132706] | PASS |
| pile__public_base | 1.052826241 | [1.035009099, 1.070643383] | PASS |
| pile__public_lora_2601 | 1.048974957 | [1.027640562, 1.070309351] | PASS |

All four timing cells pass the 1.25 upper-bound cost gate, and all four candidate/reference safeguards pass.

## Decision

The serialized decision is `RELIABLE_BUT_PRACTICAL_MAGNITUDE_UNCERTAIN` with promotion `retain_reference`. The independent per-cell review agrees: the candidate is reliably positive on the primary route but clears neither practical quality margin, so the frozen reference is retained.

The superseded pooled review is preserved at `experiments/TRR-0008/evaluation/independent_score_review_excluded_pooling.json` with status `EXCLUDED_REVIEW_INVALID_POOLING`; it is excluded from all findings.

The structured receipt is `experiments/TRR-0008/evaluation/independent_score_review.json`.
