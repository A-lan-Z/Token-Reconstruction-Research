# TRR-P11 — final evidence handoff

Status: `FINAL_EVIDENCE_ASSEMBLED_CONFIRMATION_SCORE_PENDING_CURATOR_MATERIALIZATION`.

The public prediction freeze, transfer capture, and transfer analysis are complete. The confirmation score is still awaiting curator materialization, so this record is a bounded handoff and does not claim comparison completion.

The selector input remains the immutable 32,973-byte
`experiments/TRR-P11/manifest.json` snapshot (SHA-256
`22e3b51e4835e85ccaf84c52248bb53c02e702c586f6d36116422e0fd5670f3a`). The released canonical-sequence exclusion audit is `recovery_identity_audit_r17.json` (SHA-256 `43416d0d1945821812278df8dfc3ee7639f872e9c8407abdf245fb281bda9d65`), with identity union `identity_union_export_r14.json` (SHA-256 `125275eab38c66117d45f5e0df4085058dae8a5c939fff1554cab7e608eb1fe1`) and root release `root_selection_release_r1.json` (SHA-256 `0e1816711965b0601a2e00ce4c187bded9f0c0fca4493e71ef9478a772659f95`). P03 remains sealed and outside the audit inventory.

The restored fixed pair passed the smoke gate in
`experiments/TRR-P11/restore/restore_receipt_actual_r2.json` (SHA-256
`311ed83312c620eef81a8401901876cd55d5c93ea6860955aa4c6f44df4bd7f8`). B0 is `current_fixed_replication_1` at step 8000 and B1 is `expanded_fixed_replication_1` at step 13000. The replication provenance record is
`experiments/TRR-P11/replication-provenance-r1.json` (SHA-256
`280ff8fa5e59f3ac778033955f2c6f12abf19b0426468710587cf0cdc582d3a4`). The original exact-state confirmation remains blocked by unavailable selected artifacts; those states were not reused.

The prediction freeze passed before truth in
`outputs/TRR-P11/private-evaluation/evaluation/freeze-r1.json` (SHA-256
`1b27af8ca3c395187a68ce548cd60e3b812a5b3353c7c5d38795c36b318d481b`). B0/B1 primary integrity passed in
`outputs/TRR-P11/private-evaluation/primary/post-run-integrity-r2.json` (SHA-256
`469c2ae652cc1774fc05a37d6011d108c18bfb34cc61adf6bc5e836235d790aa`). A1/A2 qualification and predictions were completed without truth in the private r3 receipts; the comparator score is held for the pending confirmation-score materialization.

The transfer analysis is represented by the r3 successor
`experiments/TRR-P11/transfer/analysis_summary_r3.json` (SHA-256
`8ca868aa5665d9bef5f0b9aee21c2dd9b4a3c25357ba558f8b3eb669f7deadc2`). It supersedes, without modifying, r2 (SHA-256
`5068e26d1e789657336dbe0e08df2ad2bc43823951e0831d6d65dd0a8a3220d0`) and copies only aggregate geometry fields from the completed sanitized result
`outputs/TRR-P11/private-evaluation/transfer-analysis/analysis_result_r2.json` (SHA-256
`880c9b934292d285e656256fed92e8eb76d82eb4359715268ddbcd1c5db1d83e`). The transfer result covers the clean public baseline, five artificial variants consisting of four controlled prefix perturbations and the after-cut null, and the separate historical public-LoRA benchmark. The five artificial variants have zero broken rows in both domains; the after-cut null passed exact activation and prediction equality. The historical benchmark has one broken Finance token and four broken Pile tokens. Those are token counts, not exact-record inventories. The fixed AUCs are descriptive bounded outputs; no AUC reliability, threshold, deployment, or overall canonical claim is made.

The first transfer-analysis callback failure is preserved in
`outputs/TRR-P11/private-evaluation/transfer-analysis/analysis_failure_r1.json`
(SHA-256 `2f20ea799669f6af23c8cc403cc51bc00692f2b9c14ad9ec1aeaab1d5112b2d9`) and records the incompatible `[rows, 2, hidden]` geometry shape. It was superseded by the completed sanitized result without substituting a P11-side scorer. The A1/A2 attempts remain distinct: r1 failed CUDA-device discovery before a model run (`a1_a2-execution-r1/failure.json`, SHA-256 `9de1783be5f3e095bf7d74df4909c96f92b346669bfcbdc43d6e6e67fb8e6b9d`); r2 was excluded by the mistaken post-load 8 GiB free-memory floor (`a1_a2-execution-r2/resource_watchdog_qualification.json`, SHA-256 `728c0b749568baf84407a3e1a88b31a3479f78edf173657e43e729ba4851dc67`); and r3 completed the first successful Finance/base qualification and saved its inference, cost, and trace artifacts before its top-level summary failed with `NameError: _git_head` (`a1_a2-execution-r3/failure.json`, SHA-256 `2398f51079ca0c9933afde9ab4943b098cea2175a3dbd1a857946affd9418b6f`). The exact r3 GPU peak was not persisted: the watchdog passed, wrapper RSS was 4,808,160 KiB over 492.054 seconds, and the 7,478 MiB device reading was a sample rather than a peak. The subsequent three cells persisted their peaks. Freeze-validation failures before output creation and the earlier capture guard failure remain preserved as excluded attempts.

The fixed paired-fit recipe uses seed 4010. The statistical bootstrap uses seed 9009 with the registered 10,000 draws, and source selection uses seed 5011; `paired_exact_cp` remains the exact-record interval implementation. No seed-reliability claim is made. PR24 remains open, draft, and unmerged; no result is presented as a merged or published canonical benchmark. The P03 holdout remains unopened.

The hash-only packaging audit is
`experiments/TRR-P11/evidence-packaging-inventory-r1.json` (SHA-256
`e1f290597e3ff9decdb77b6c1448d874834dd72abe0800c9e1b41618c53d784a`). It records 129 artifacts without copying or publishing raw source, prediction, tensor, or truth values. Its optional private backup estimate is 2.756 GiB against a 19.306 GiB Windows-free-space snapshot; no backup copy was performed. Bulky evaluator arrays and truth payloads remain task-local, while sanitized aggregate results and opaque hash bindings are suitable for the later publication review.

The remaining action is curator materialization and review of the confirmation score against this frozen evidence set. This handoff does not start another experiment, reopen P03, change the selector-bound manifest, or merge PR24.
