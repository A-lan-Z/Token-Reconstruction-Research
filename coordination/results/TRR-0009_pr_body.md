# TRR-0009 final PR body

**Base/task:** `task/TRR-0008`

## Result

TRR-0009 evaluated a bounded supported-token readout adaptation on 256 Finance and 128 Pile natural source records, paired across `public_base` and `public_lora_2601` (384 unique records, 768 target observations, 127 post-BOS positions per record). The primary Finance `public_base` adaptable-versus-equally-continued-fixed contrast was +0.7813 exact percentage points (3 gains, 1 loss; 79/256 vs 77/256) and +0.024606 token points (8/32,512 positions). Its 97.5% lower bounds were −2.2808 exact points and −0.003076 token points, below the preregistered +2 exact / +0.25 token useful floors. The frozen decision is `FAIL / NO_USEFUL_PILOT_SIGNAL`: retain the reference and report practical adaptation magnitude as unresolved.

This does not establish adaptation harm. All four exact-harm and token-harm safeguards pass, all eight rare/absent support gates pass, and all four alias-qualified cost gates pass. The exact primary upper bound remains +3.7497 points, so a +2-point exact benefit is not ruled out; the token upper bound is +0.05536 points, below the token useful floor. The retained reference comparison is descriptively stronger (Finance P0 +7.4219 exact points and +0.424459 token points, token LCB +0.298351), while most adaptable-versus-unchanged Finance P0 token improvement is shared with continued fixed (75 of 83 extra correct positions).

No automatic replacement, retraining, or sample expansion follows this result. A future experiment should first freeze a fresh independent panel and then test one regularized token-specific directional correction at the same public bank and schedule with a matched fixed control. This separates restricted-direction capacity from scalar calibration; it should not widen gain/bias bounds without evidence of effective-bound contact or use a decoder-absorbable global linear reparameterization.

## Exact result and evidence paths

- Human-readable result: `coordination/results/TRR-0009.md`
- Detailed reproducibility appendix: `coordination/results/TRR-0009_reproducibility_appendix.md`
- Structured manifest: `experiments/TRR-0009/manifest.json`
- Completed score: `experiments/TRR-0009/evaluation/score_v2.json` (133,163 bytes; SHA-256 `62b15bbf2f41b58010f25a74f765b7a13f0e3e05dc77762309de07306006e53d`)
- Deterministic decision: `experiments/TRR-0009/evaluation/aggregate_decision_v2.json` (19,858 bytes; SHA-256 `e11a4fd746d9c44187c6186f92ff9f28f209bf6ee17bb51821d071edb783842f`)
- Cost summary: `experiments/TRR-0009/evaluation/timing/cost_summary_v2.json` (15,960 bytes; SHA-256 `15bb335ca8ed02a7196675f740946cc1a129bc3bb50e09bb94a910feb0abf765`)
- Authoritative corrected numeric review: `experiments/TRR-0009/evaluation/score_numeric_review_v2.json` (40,841 bytes; SHA-256 `429bdbfc7f0ad05e2e266fb66d8355184c739030ccb9523247e8b092332ab1a1`); review addendum: `experiments/TRR-0009/evaluation/score_numeric_review_v2_addendum.json` (4,023 bytes; SHA-256 `d698c41224ccaa704830c2c02d8180d90c7550d3cdfb048902fecb923734d13e`)
- Truth/score execution receipt: `experiments/TRR-0009/evaluation/truth_prepare_score_compat_v2_execution_receipt.json`
- Final truth binding: `experiments/TRR-0009/evaluation/truth_binding_compat_v2.json`

## Selection and validation

- Final inventory: 4,123 eligible Finance and 364 eligible Pile records.
- Corrected selection: `experiments/TRR-0009/selection_v2/source_selection.json` — 256 Finance + 128 Pile, identity-only, zero approved-ledger overlaps after repair.
- Original one-overlap selection is preserved as `EXCLUDED_BEFORE_CAPTURE`; no observations or score used it.
- Loader qualification passed 128 fixture checks; root-reviewed final suite passed 106 tests at full code commit `16e86a5`.
- Public capture, prediction, and precision timing completed before truth: 16.958598 s, 145.870593 s, and 235.260320 s respectively.

The first post-gate truth-preparation attempt materialized trusted selected labels in RAM, then failed serializing the frozen tokenizer directory; it wrote no sidecar, binding, or score. The bounded source adapter corrected that serialization boundary, prepared a private sidecar, and completed the frozen score. Raw truth and the private sidecar are not committed. The exact failure and repair receipts remain in the reproducibility appendix and manifest.
