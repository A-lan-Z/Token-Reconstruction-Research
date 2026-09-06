# Excluded initial TRR-0008 score review

**Status: EXCLUDED_REVIEW_INVALID_POOLING.** This historical review contained pooled 2,816-record aggregates. It is retained for audit history only and must not be used for findings, gates, or the decision.

# TRR-0008 independent score review

Status: **complete** (`INDEPENDENT_SCORE_REVIEW_COMPLETE`). This review used only `score_v1.json` serialized cell counts and paired record-level gain/loss counts. The reviewer did not open private truth, rerun scoring, or modify scientific outputs.

## Provenance

- Score: `experiments/TRR-0008/evaluation/score_v1.json`; SHA-256 `e30ba5e4a05a2bc8fbb0869dd3ecffd6b7d570cd8af434c87d125390efdba453`.
- Decision contract binding matched: `141a3a506e20e1eadd3be1beb53c2f01508e9b651c93d19b2f3d9ff72344bcf3`.
- Precision40 timing binding matched: `a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`.
- Public freeze and truth-binding header hashes matched their serialized bindings; the truth sidecar itself was not opened by this review.

## Primary candidate versus reference

| Cell | Records | Gains | Losses | Exact delta | Exact LCB | Token delta | Token LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| pile__public_base | 384 | 8 | 4 | 0.010416667 | -0.021171278 | 0.006151575 | 0.003649937 |
| pile__public_lora_2601 | 384 | 7 | 8 | -0.002604167 | -0.037412248 | 0.005167323 | 0.002993768 |
| finance__public_base | 1024 | 90 | 40 | 0.048828125 | 0.014206508 | 0.001899299 | 0.001145729 |
| finance__public_lora_2601 | 1024 | 121 | 36 | 0.083007812 | 0.046148775 | 0.001999262 | 0.001222625 |

For the primary Finance public-base cell: 90 gains, 40 losses, 894 ties; exact LCB recomputed as `0.014206508` and matched the serialized value; token LCB is `0.001145729`. Both routes are positive, but exact `0.014206508 < 0.05` and token `0.001145729 < 0.01`, so both practical routes fail.

## Three candidate/reference contrasts

| Contrast | Descriptive aggregate gains-losses | Exact net | Exact LCB | Token net |
|---|---:|---:|---:|---:|
| improved_public_bank__residual_mlp512 vs reference | 226-88 | 0.049005682 | 0.029741728 | 0.002961144 |
| current_enriched__residual_mlp512 vs reference | 183-107 | 0.026988636 | 0.008094705 | 0.002239732 |
| improved_public_bank__trained_diagonal vs reference | 88-68 | 0.007102273 | -0.007122098 | -0.000567623 |

The aggregate rows are descriptive and were not used as pooled decision gates. Exact CP bounds, count deltas, and token deltas matched the serialized per-cell evidence for all three contrasts.

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

All four timing cells passed the 1.25 upper-bound cost gate; timing status was `COST_PASS`. All four safeguards passed.

## Decision

The serialized decision is `RELIABLE_BUT_PRACTICAL_MAGNITUDE_UNCERTAIN` with promotion `retain_reference`. The independent review agrees: the candidate is reliably positive but does not clear either practical quality margin, so the frozen reference is retained.

The structured receipt is `experiments/TRR-0008/evaluation/independent_score_review.json`.
