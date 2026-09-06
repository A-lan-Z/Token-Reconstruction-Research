TRR-0005 tests whether broader public fitting coverage reduces the standalone reconstruction gap and whether extra earlier activation vectors still help after equally good fitting. Original-like and enriched banks each contain 124,371 post-BOS fitting positions, with matched training opportunities and a trained positionwise control for the causal extension.

The enriched fitting mix improved the preselected positionwise decoder by 5.54–5.61 percentage points on fresh Pile records and 9.91–9.93 points on Finance. Its token accuracy reached 94.68–94.94% on Pile and 97.85–97.90% on Finance. The extra-context token differences were between −0.09 and +0.16 points; all declared token-benefit upper bounds exclude the useful +0.5-point margin. Exact-record bounds still exceed the +5-point margin, so the context decision is **inconclusive**. This is a combined distribution/context/coverage intervention, not isolated proof of a vocabulary-coverage mechanism.

The A1+A2 quality anchor remains substantially better, especially for complete 128-token records. Measured warmed standalone reconstruction was approximately 4.4–5.6 ms per record versus 1.02–1.04 s for A1+A2. Standalone decoding still retains the approximately 1.05 GB public embedding table plus its decoder state. Preparation, all eight development fits, startup, memory, and measured timing scopes are reported separately. The next proposed experiment holds the trained pair fixed and tests enough new independent natural records to resolve the remaining exact-record uncertainty.

Validation: **302 tests passed**. All 32 prediction artifacts and timing receipts passed the complete public gate before scoring opened truth. A preserved receipt-schema failure was repaired through a metadata-only export with byte-identical predictions and unchanged timing values; the permanent source fix was applied and tested after scoring. The executed science remains bound to commit `da82f6cac45e09ae83452198344c547553cb4433` and its original source hashes.

Handoff:

- Result: `coordination/results/TRR-0005.md`
- Structured evidence: `experiments/TRR-0005/manifest.json`
- Task-local status: `experiments/TRR-0005/status.json`
- Fresh results, paired uncertainty, and preserved prediction artifacts: `experiments/TRR-0005/fresh_confirmation_v1/`
- Replay: `experiments/TRR-0005/fresh_prediction_reproduce_v2.md`

This exploratory follow-on is based on the reviewed PR #7 head `6e8b683e404c0acb70cd59b7dd6d6868b2061f61`, with PR base `task/TRR-0004`. No prior PR was merged; global coordination state and the active benchmark registry remain unchanged. No paid compute was used. This is not a canonical comparison-complete or replacement claim.

Scientific artifact commit: `c88203883038a151ef70e1aba31fab06daf3b65f`; post-score maintenance and replay helpers: `1dba67a8dc75844727866cb4273da28a311df216`.

Reviewed handoff commit: `2aba3c90f9c4e13819213146bf8e61a3b6ecd068`.
