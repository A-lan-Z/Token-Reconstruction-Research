# TRR-0009 reproducibility appendix

This appendix preserves the complete detailed handoff behind the compact result report. It contains exact commands, artifact paths and hashes, training and timing receipts, planning estimates, excluded attempts, selector repair chronology, and the pre-truth gate failure. The main report is decision-led; this file is the reproducibility and audit record.

> **Historical chronology notice.** The pre-score chronology below is superseded by the [Final score and decision addendum](#final-score-and-decision-addendum), which records truth opening, completed scoring, numeric review v2, and the final action. Earlier statements that truth or quality were pending describe their historical checkpoint only.

Current authorization evidence is recorded in [`metadata_gate_authorization.json`](../../experiments/TRR-0009/coordination/metadata_gate_authorization.json), and the publication preflight is [`publication_preflight.json`](../../experiments/TRR-0009/coordination/publication_preflight.json).

---

# Historical pre-truth checkpoint — completed continuation, prediction, and timing pilot

TRR-0009 has completed its truth-free continuation diagnostic, aggregate eligibility inventory, identity-only natural source selection, public capture, the four-method prediction matrix, and precision timing. The frozen panel contains 256 Finance and 128 Pile records, paired across `public_base` and `public_lora_2601` (384 unique source records and 768 target observations). Truth remains unopened and fresh natural quality scoring and the prospective decision remain unavailable pending the authorized maintenance repair and gate rerun. The existing TRR-0008 retained-reference decision remains in force while the adaptation question is evaluated; no replacement or promotion claim is available. The training curves below are public fitting-bank and fixed development diagnostics, not fresh natural quality results.

The experiment starts from the published TRR-0007 current-bank residual MLP checkpoint selected at step 2100 (29,390,628 bytes; SHA-256 `2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`). The improved-public-bank residual had already reached 100% on its own 124,371-position fitting bank, so this single current-bank prefit choice supersedes the earlier proposal; the superseded design record is retained and is not a scientific run. It registers three reported arms: the unchanged checkpoint, continued fitting with the public embedding readout fixed, and the same continuation with a supported-token post-logit gain and bias. The retained TRR-0006 trained-diagonal state is a historical comparison and receives no new fitting.

The adaptable readout applies `logit_v = base_logit_v * gain_v + bias_v` to the 17,126 token IDs supported by the improved fitting bank, starts at gain 1 and bias 0, and leaves 111,130 absent IDs fixed. It has 34,252 scalar parameters and does not change embedding directions. The initialization is separate from the exact penalty `1e-4 * mean((1 + 4/sqrt(count_v)) * (raw_gain_v^2 + raw_bias_v^2))`; absent IDs have no correction parameter and no count=0 branch is evaluated. Both continued arms use the same 3,000-step, seed-4005, batch-8 schedule and decoder learning rate 2e-4; the readout group uses 1e-3, zero weight decay, gradient clipping 1, gain/bias limit 0.25, and the predeclared frequency-weighted raw-parameter anchor penalty.

The frozen panel has 256 Finance and 128 Pile unique natural source records, paired across `public_base` and `public_lora_2601`: 384 source records and 768 target observations, with 127 scored post-BOS positions per record. Finance uses `[12000,20000)` and Pile `[7000,10000)`. The completed identity-only scan found 4,123 eligible Finance records and 364 eligible Pile records after the inherited known ledgers, TRR-0008, P04/P06, and approved P08 hash-only exclusions; both requested counts were sufficient. The exact selected identities are recorded only in the selection artifacts, without source text or target labels.

The primary cell is Finance `public_base`, adaptable versus continued fixed. A useful pilot signal requires either an exact point difference of at least +2 percentage points with a conservative one-sided 97.5% CP lower bound above zero, or a token point difference of at least +0.25 percentage points with a one-sided 97.5% source-record bootstrap lower bound above zero. A positive lower bound below the point floor remains positive evidence only. All four cells must meet exact harm ≥−5 points and token harm ≥−1 point at one-sided 95% bounds. The two rare/absent bins (`0`, `1-4`) have eight cell-by-bin gates, each requiring at least 32 exposed source records and a one-sided 95% source-record bootstrap lower bound no worse than −1 token point; insufficient support is UNKNOWN.

The shared improved-public-bank continuation frequency strata are fixed at `0`, `1-4`, `5-9`, `10-49`, and `50+`, with distinct-ID counts 111,130, 14,257, 1,587, 1,061, and 221 respectively; total supported IDs are 17,126 across 124,371 post-BOS positions. Here `0` means absent from the continuation bank; it is not a lifetime-unseen claim for the inherited checkpoint or retained reference. The same bin definitions and exposed-token denominators apply to every reported method. Natural token accuracy will be computed as summed correct positions divided by summed exposed positions, with source-record bootstrap resampling of both numerator and denominator; per-record stratum accuracies are not unweighted when exposure counts differ.

The warm candidate/fixed runtime ratio must be at most 1.25 in each cell. Preparation, training wall time, adapter/optimizer state, peak memory, and deployed footprint will be reported separately. A resource-guard failure stops the pilot and leaves the adaptation question unresolved. No automatic replacement, retraining, sample expansion, or new A2 run follows an outcome.

## Completed truth-free continuation diagnostic

The completed run receipt records 770.069974 seconds for the unchanged anchor and both continued arms. The same public fitting bank and fixed 48-record development set supplied the selection metric; this is not an independent test or a fresh natural panel.
The style-balanced accuracy and pooled correct-token count are separate diagnostics; the pooled count is shown only as its own denominator-bearing column.

| arm | selected step | selected fit errors | selected development style-balanced accuracy | pooled development correct / 3,133 tokens | selected challenge recovery | arm wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unchanged anchor | 0 | 621 / 124,371 | 0.956996283 | 3,037 / 3,133 | 0 / 621 | 8.389 |
| continued fixed readout | 400 | 77 / 124,371 | 0.962714307 | 3,050 / 3,133 | 550 / 621 | 355.823 |
| continued adaptable readout | 400 | 84 / 124,371 | 0.962714307 | 3,050 / 3,133 | 548 / 621 | 393.803 |

At step 3000, both continued arms reached 0 fit errors and recovered 621 / 621 initially wrong challenge rows; this describes optimization on the public continuation bank. The fixed arm used 92.328 seconds of gradient updates and 260.988 seconds of evaluation; the adaptable arm used 107.982 and 283.770 seconds respectively, with optimizer states of 58,776,644 and 59,050,668 bytes. The selected states and run receipt are bound in `experiments/TRR-0009/training/method_freeze.json`. The adaptive loader now binds the capacity-owned single-tensor `support_ids` and `support_counts` files; their container file hashes/bytes and embedded tensor payload digests are recorded separately, alongside `loadable_support.json` and the full three-tensor source container. The reproducible static diagnostic is [diagnostic_curves_v1.png](../../experiments/TRR-0009/training/diagnostic_curves_v1.png), with [diagnostic_curves_v1.svg](../../experiments/TRR-0009/training/diagnostic_curves_v1.svg) and `scripts/trr0009_plot_curves.py` as the vector/script sources.

## Cost accounting boundary

The continuation run receipt is now bound. Its arm-specific fit and optimizer costs are incremental TRR-0009 costs; inherited checkpoint and public-bank/E preparation remain historical inputs and are not charged again. Capture, prediction, and precision timing receipts are complete; retained footprints are bound below. Disk-cold startup remains unmeasured.

| scope | exact prior evidence or TRR-0009 status | accounting treatment |
| --- | --- | --- |
| inherited current-bank residual start | TRR-0007 current-enriched residual fit wall `123.139566510 s`; selected state `29,390,628` bytes, SHA-256 `2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8` | Published starting state; do not count as TRR-0009 continuation fit wall. Evidence: `coordination/results/TRR-0007.md` and `experiments/TRR-0007/manifest.json`. |
| inherited public-bank preparation | TRR-0007 accepted CPU preparation `23.666454504 s`; improved activation capture `8.224661700 s`; normalized E file `1,050,673,488` bytes | Historical public-bank/E preparation. It is not a TRR-0009 incremental charge. Evidence: `coordination/results/TRR-0007.md`. |
| earlier shared preparation context | TRR-0005 full corpus preparation `20.013 s`, eight historical fits `801.7 s` total, and fresh observation capture `16.961 s` | Historical context only; these receipts predate the TRR-0009 continuation and are not added to its run total. Evidence: `coordination/results/TRR-0005.md`. |
| TRR-0009 continuation arms | fixed arm wall `355.822723317 s`, adaptable arm wall `393.802617079 s`; shared preparation `9.148822392 s` per arm receipt; optimizer states `58,776,644` and `59,050,668` bytes; whole run `770.069974422 s` | Incremental training receipt, deduplicated by arm. Evidence: `experiments/TRR-0009/training/run_v1/run_receipt.json`. |
| TRR-0009 qualifier/challenge/shared observation preparation | training largest-cell qualification `10.092721462 s` and initial-wrong challenge `3.878089745 s` are recorded; fresh public capture completed in `16.958598 s` | Training qualification/challenge and capture preparation are bound separately. No fresh natural quality result is claimed. |
| TRR-0009 timing and deployment | four-cell warmed timing, alias qualification, deployed state footprint, and disk-cold startup | Timing and retained-footprint receipts are complete; disk-cold startup remains unmeasured. The inherited E byte count above is reported separately from any new adapter/state footprint. |

The training receipt records 3,915,382,784 reserved CUDA bytes for the fixed arm, 3,917,479,936 for the adaptable arm, and 5,620,727,808 shared peak host-RSS bytes. Qualification took 10.092721462 seconds and the initial-wrong challenge took 3.878089745 seconds. Arm wall sum is 758.014065665 seconds versus the authoritative whole-run wall of 770.069974422 seconds; the shared 9.148822392-second preparation receipt is charged once, and lifecycle components are reported separately rather than arithmetically added to whole-run wall.

Preparation entries repeated in per-cell receipts are deduplicated per method. Shared E/public-bank preparation is counted once, and repeated arm or cell load entries are not summed. Warmups, measured calls, and whole-run wall time remain separate; any residual whole-run difference is guard, I/O, or other overhead rather than decoder latency.

### Completed capture, prediction, and timing costs

Fresh public capture completed in 16.958598 seconds for the v2 panel at B8×192 with 128 retained positions. The prediction matrix completed before truth in 145.870593 seconds across 16 method-cell runs. The precision timing receipt completed in 235.260320 seconds using 40 blocks, 32 records per cell, one warmup and one measured call per record; its all-cell alias and ≤1.25 candidate/fixed cost gates passed. These are separate phase walls, not a combined decoder latency.

The measured prediction interval is synchronized BF16 current-H staging → FP32 decoder → full-vocabulary argmax → CPU IDs. Warmups matched measured outputs exactly. The table reports both warmup and measured latency; per-cell run receipts repeat method preparation, so preparation is charged once per method below.

| method | cell | records | warmup ms/record | measured ms/record |
| --- | --- | ---: | ---: | ---: |
| unchanged_anchor | finance__public_base | 256 | 6.693779 | 6.015883 |
| unchanged_anchor | finance__public_lora_2601 | 256 | 6.699701 | 6.034440 |
| unchanged_anchor | pile__public_base | 128 | 8.882122 | 4.295500 |
| unchanged_anchor | pile__public_lora_2601 | 128 | 6.627816 | 5.944459 |
| published_reference | finance__public_base | 256 | 6.450987 | 5.809866 |
| published_reference | finance__public_lora_2601 | 256 | 6.436928 | 5.792628 |
| published_reference | pile__public_base | 128 | 6.467211 | 5.744660 |
| published_reference | pile__public_lora_2601 | 128 | 6.467418 | 5.789367 |
| continued_fixed_readout | finance__public_base | 256 | 6.661556 | 6.005353 |
| continued_fixed_readout | finance__public_lora_2601 | 256 | 6.676740 | 6.078479 |
| continued_fixed_readout | pile__public_base | 128 | 6.611873 | 5.997085 |
| continued_fixed_readout | pile__public_lora_2601 | 128 | 6.642810 | 6.031722 |
| continued_adaptable_readout | finance__public_base | 256 | 7.397013 | 6.698216 |
| continued_adaptable_readout | finance__public_lora_2601 | 256 | 7.327265 | 6.766928 |
| continued_adaptable_readout | pile__public_base | 128 | 7.453800 | 6.688600 |
| continued_adaptable_readout | pile__public_lora_2601 | 128 | 7.340709 | 6.654128 |

The adaptable/fixed mean runtime ratios were 1.095871 (Finance public_base), 1.087951 (Finance public_lora_2601), 1.076975 (Pile public_base), and 1.080318 (Pile public_lora_2601); every cell passed the 1.25 cost limit. Prediction model preparation, deduplicated once per method, was 0.046400 s for unchanged, 0.018163 s for the retained reference, 0.029251 s for continued fixed, and 0.066710 s for continued adaptable. The prediction receipt reports 0.125458 s to load the shared E runtime embedding; timing reports 0.133498 s for its own loaded-process boundary. Disk-cold startup was not measured.

The pooled prediction process reached 1,352,663,040 reserved GPU bytes and 1,738,211,328 process-RSS bytes. The five-model precision timing process reached 1,440,743,424 reserved GPU bytes and 1,740,296,192 process-RSS bytes. These are benchmark-process peaks. Retained single-method state sizes are 29,390,628 bytes for unchanged, 20,990,652 bytes for the historical reference, 29,390,492 bytes for continued fixed, and 29,870,540 bytes for continued adaptable. The shared E file is 1,050,673,488 bytes: 1,050,673,152 FP32 tensor-payload bytes plus a 336-byte header. A deployed method retains one selected state plus E; the pooled timing peak is not its deployed footprint. Receipts are `experiments/TRR-0009/evaluation/public_observations_v2/capture.json`, `experiments/TRR-0009/evaluation/predictions_v2/run_manifest.json`, `experiments/TRR-0009/evaluation/timing/result_v2.json`, and `experiments/TRR-0009/evaluation/timing/cost_summary_v2.json`.

The exact prospective CP sensitivity table is in [power_analysis.json](../../experiments/TRR-0009/planning/power_analysis.json); the root-approved frozen decision contract, completed identity inventory, selector adapter, decision aggregator, and safety tests are bound in [manifest.json](../../experiments/TRR-0009/manifest.json). The completed source-selection artifacts bind the frozen contract, finalized count-only inventory, and approved opaque exclusion export; capture, prediction, and timing are complete, while truth remains gated by the public metadata compatibility check. Root review of the separate preselection selected-method freeze at `experiments/TRR-0009/training/method_freeze.json` is complete; that freeze binds the actual method states, decision rules, loader/source-code hashes, and training receipt. The later timing registration is an integrity receipt after observations and does not satisfy this preselection gate.

## Completed identity inventory and source selection

The approved P08 hash-only export was copied byte-identically to `experiments/TRR-0009/coordination/approved_opaque/p08_opaque_hash_exchange_sanitized.json` (80,045 bytes; SHA-256 `3383356794d3498830f5cd69ed2c552f4ed527eceed7e05f61cc3c49eba2eb38`). It contains 512 public-record hashes and 512 H128 final-sequence hashes, with no source identities, text, token IDs, labels, answers, model weights, or truth. Its producer recipe is bound to the previously approved generic P06 hash-construction receipt at `/tmp/trr-p06/coordination/parallel/TRR-P06-hash-construction-receipt.json` (2,339 bytes; SHA-256 `b06ca9ccae7b831318604351ce76f183a8c2745780494e20b263279f146ba92c`); the P08 export is used only for identity exclusion. The earlier no-reservation reply remains retained as historical coordination metadata and is superseded for this completed scan.

The one-thread count-only scan completed with status `IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH`. It scanned only the declared public ranges and wrote aggregate counts and commitment digests; source rows were transiently rendered for hashing, while no source text or token IDs were written and no model, predictions, or truth were opened.

| domain | scanned rows | valid rows | invalid rows | eligible unique | requested | surplus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Finance | 8,000 | 5,182 | 2 | 4,123 | 256 | 3,867 |
| Pile | 3,000 | 425 | 366 | 364 | 128 | 236 |

The aggregate inventory is `experiments/TRR-0009/planning/source_inventory_final.json` (25,126 bytes; SHA-256 `8821c8e365fd9e2cff386c7a1f5ad379e15f0da63ef280f94e970d04d16ef718`). It records H128 duplicate handling and the P04/P06/TRR-0008/P08 exclusion chain without exposing source payloads.

Both domains cleared the frozen minimums. The first identity-only selector output is preserved as an excluded pre-capture attempt: `experiments/TRR-0009/selection/source_selection.json` (283,874 bytes; SHA-256 `4ef38b31041fd4989d0f3ecb364c9afb7b4a6728dfa9971f324153b5c75e24ab`) and its exclusions ledger (18,980 bytes; SHA-256 `d4c2f860c52cebf7c48672df38c9d3f8c99152c29f261a695bb7264436c84b88`). It used the same frozen seed, ranges, natural ordering, counts, and method bindings, but the selector did not pass the approved P08 source/H128 sets. The count-only repair audit found zero public-record overlap and one Finance H128 overlap with P08. No capture, prediction, observation, or truth used this panel.

The repaired selector ran with the unchanged frozen design and the approved P08 export, writing the canonical corrected artifacts under `experiments/TRR-0009/selection_v2/`: `source_selection.json` (287,719 bytes; SHA-256 `c2e996514f7f45e55d7bfadc27fc8048bdb07a2c979b208de8716972d8d1def2`) and `source_exclusions.json` (20,664 bytes; SHA-256 `bba988428e6e1d13c2916b5e9c001acd86aa37d838bda553744eecd0f88d806f`). The corrected status is `FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH`, with 256 Finance and 128 Pile records, P08 postselection intersections of zero source hashes and zero H128 hashes, and zero intersections with the approved P06, TRR-0007, and TRR-0008 selected ledgers (P04 source overlap zero; its H129 comparison is unavailable from selected ledgers alone). The corrected reservation contains 384 unique source and 384 unique H128 hashes: `opaque_source_sequence_reservation.json` is 60,745 bytes with SHA-256 `73e07e1fd2c964eface6af387229bac738b96ecc5b7e1ee3059f57f9a8c2b4f8`. A2 verified the replacement hash-only reservation independently; the original and replacement reservations have a 385-hash union in each field, so the correction changes one reserved source/sequence entry. The exchange receipt is `experiments/TRR-0009/coordination/p08_replacement_exchange_ack.json` (2,054 bytes; SHA-256 `4dc7f1c5fee3c19199a793d73d6f22f9136a42457348566c7ac32656fa51bcce`).

The exact count-only inventory invocation was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src:scripts \
python3 scripts/trr0009_plan.py inventory-final --repository-root . \
  --decision-contract experiments/TRR-0009/planning/decision_contract.json \
  --method-freeze experiments/TRR-0009/training/method_freeze.json \
  --trr7-method-freeze experiments/TRR-0007/method_freeze.json \
  --tokenizer "$TOKENIZER_SNAPSHOT" --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --p08-opaque experiments/TRR-0009/coordination/approved_opaque/p08_opaque_hash_exchange_sanitized.json \
  --output experiments/TRR-0009/planning/source_inventory_final.json
```

The historical original selection invocation (now excluded before capture) was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src:scripts \
python3 scripts/trr0009_select_public.py select --repository-root . \
  --decision-contract experiments/TRR-0009/planning/decision_contract.json \
  --planning-status experiments/TRR-0009/planning/source_inventory_final.json \
  --inventory experiments/TRR-0009/planning/source_inventory_final.json \
  --method-freeze experiments/TRR-0007/method_freeze.json \
  --task-method-freeze experiments/TRR-0009/training/method_freeze.json \
  --tokenizer "$TOKENIZER_SNAPSHOT" --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --output experiments/TRR-0009/selection/source_selection.json \
  --exclusions-output experiments/TRR-0009/selection/source_exclusions.json
```

The corrected identity-only selection invocation was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src:scripts \
python3 scripts/trr0009_select_public.py select --repository-root . \
  --decision-contract experiments/TRR-0009/planning/decision_contract.json \
  --planning-status experiments/TRR-0009/planning/source_inventory_final.json \
  --inventory experiments/TRR-0009/planning/source_inventory_final.json \
  --method-freeze experiments/TRR-0007/method_freeze.json \
  --task-method-freeze experiments/TRR-0009/training/method_freeze.json \
  --tokenizer "$TOKENIZER_SNAPSHOT" --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --p08-opaque experiments/TRR-0009/coordination/approved_opaque/p08_opaque_hash_exchange_sanitized.json \
  --output experiments/TRR-0009/selection_v2/source_selection.json \
  --exclusions-output experiments/TRR-0009/selection_v2/source_exclusions.json
```

The selector and exclusion regression file passed all 15 tests after the corrected artifacts were produced. Selection v2 completes the pre-truth source gate. Capture, prediction, and timing are complete before truth; fresh quality scoring remains blocked by the metadata compatibility check. No result from this handoff changes the frozen criteria, sample sizes, method states, or retained-reference decision.

The corrected capture-facing opaque reservation was generated from the v2 identity ledger with the existing `scripts/trr0009_select_public.py reserve` helper and the approved P06 generic hash recipe. `experiments/TRR-0009/selection_v2/opaque_source_sequence_reservation.json` has status `READY_FOR_TRR0009_CAPTURE_HASH_ONLY`, 60,745 bytes, and SHA-256 `73e07e1fd2c964eface6af387229bac738b96ecc5b7e1ee3059f57f9a8c2b4f8`. It contains exactly 384 unique public-record hashes and 384 unique H128 final-sequence hashes, plus fixed privacy and construction metadata; it contains no source IDs, row indices, source text, token IDs, labels, answers, or truth. Root transmitted this replacement reservation to A2 before capture; the original reservation remains retained as an excluded attempt.

### Provenance and bounded deviations

The first attempted final inventory stopped before scanning any source row because the original planner reader did not recognize the producer-authored P08 sanitized hash schema (`p08 opaque reservation schema is not hash-only`). It wrote no inventory or selection output. The preserved planner deviation is a 128-line reader-only compatibility change to `scripts/trr0009_plan.py` (92 insertions, 36 deletions): it accepts the exact P08 sanitized schema while retaining the legacy P06-shaped path, and checks the exact top-level fields, status, privacy boundary, recipe, hash conventions, count, uniqueness, and lowercase-hex constraints. It does not change source ranges, natural ordering, exclusion rules, sample counts, decision criteria, model states, or selection logic.

A second, separate selector coverage deviation was required after the original pre-capture panel exposed one P08 H128 overlap. The selector now loads and passes the approved P08 source/H128 sets through the existing classifier, asserts zero postselection overlap, and permits create-only outputs in the isolated `selection_v2/` namespace. This changes no population, range, natural-order, sample-size, decision, or model choice. The original selector and reservation remain preserved as excluded evidence. The v2 repair audit is `experiments/TRR-0009/selection_v2/exclusion_repair_audit_v2.json` (7,174 bytes; SHA-256 `29fb75a982dacb54e66f31b10e35ac7b2a91685b50525728288fe29af317f428`). The v2 selector ran from checkout `64aa610` with the seven-line output-directory guard change and was published as `55ef2d4`; the selection execution records selector-source SHA-256 `9b88520d553328ee74691c20dd99d2b28df7aa152bdc610572b92a50169104fd`, and no v2 artifact was rerun after publication.

The pre-capture loader qualification is independently bound at `experiments/TRR-0009/evaluation/loader_qualification_v2/qualification.json` (110,493 bytes; SHA-256 `e45d4f4618aaedac7cc6eaf5c6bda1f028d6b62a87edaca6d91040f459f2c552`), status `TRR8_PUBLIC_FIXTURE_LOADER_EQUIVALENCE_PASS`, 14.438396249 seconds, and 128 fixture row checks across the four methods and four cells. It loaded no source text or target labels and opened no truth; this is loader equivalence evidence, not a fresh natural quality result. Capture, prediction, and timing receipts are now complete; fresh quality scoring and the prospective decision remain pending metadata-gate compatibility review.

## Receipt-backed execution-window estimate (not a TRR-0009 run)

The existing TRR-0008 receipts provide a bounded planning reference at the same capture geometry and GPU class. TRR-0008 capture took 22.740515 seconds for 1,408 source records paired across two targets (2,816 target observations); proportional work for the 384-record TRR-0009 panel is 6.201959 seconds, an estimate that excludes startup and I/O. TRR-0008 prediction took 475.091281 seconds for 2,816 records per method across four methods; proportional work for the planned 768 records per method is 129.570349 seconds, again only a workload estimate. The fixed 40-block timing design has 25,600 measured calls plus the same number of warmups, matching the prior TRR-0008 precision receipt at 322.198460 seconds.

| phase | receipt-backed reference | TRR-0009 planning estimate | boundary |
| --- | ---: | ---: | --- |
| public capture | 22.740515 s for 2,816 target observations | 6.201959 s row-scaled | same 8×192 qualification; startup/I/O unmeasured |
| four-method prediction | 475.091281 s for 2,816 records per method | 129.570349 s row-scaled | includes prior startup/guard/I/O; state-loading difference unmeasured |
| precision timing | 322.198460 s for 25,600 measured + 25,600 warmup calls | 322.198460 s prior baseline | TRR-0009 measured receipt is 235.260320 s; detailed actuals below |
| combined decoder work | — | 457.970767 s before new startup/guard/gate overhead | arithmetic sum only, not a measured wall time |

The existing fail-closed phase limit is 600 seconds. Prior peak telemetry was 1,352,663,040 reserved GPU bytes and 1,999,114,240 process-RSS bytes. The opaque export is now bound and the measured capture, prediction, and timing receipts are recorded above; a 1,800-second lease remains a planning envelope for independently guarded phases and gate boundaries, not a single measured wall time. References are `experiments/TRR-0008/evaluation/public_observations_v1/capture.json` (SHA-256 `00c28a7350b7454184ea49423de6a99360665da43986d8077b413e6886573dc1`), `experiments/TRR-0008/evaluation/predictions_v1/run_manifest.metadata_completed.json` (SHA-256 `3d0b1737d8388232882c5f7066f46d00b55594c23f34b6ac3ba0b42fce0e67ca`), `experiments/TRR-0008/timing/precision40_result.json` (SHA-256 `a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`), and the resource preflight (SHA-256 `9712b1101c77321c78e6ec074f1c1a46cf134ba2b3a9f762754d41aa91df3914`).

## Truth-gate status

Capture, predictions, and precision timing completed with truth unopened. The public gate then failed before truth at `_validate_state_semantics` with `state inference semantics changed: continued_fixed_readout`; its `freeze_output` was not written. The receipt is `experiments/TRR-0009/evaluation/public_gate_v2_failure.json` (2,158 bytes; SHA-256 `75274ae75b9198bfe1d64d709c029fd1ad5f332547ba09b045141339490347d2`). This is a metadata compatibility blocker, not a quality finding. User authorization is recorded for the narrow metadata repair; implementation and the gate rerun remain pending, and no gate override or truth access has occurred.

## Outcome interpretation template (quality result not yet available)

If the metadata gate is repaired and a truth score exists, apply one of these pre-registered interpretations without pooling domains, target conditions, or source records:

- **Useful primary signal:** Finance `public_base`, adaptable minus fixed, meets either the exact +2-point or token +0.25-point useful floor with its primary one-sided lower bound above zero; all four-cell harm bounds, eight rare/absent safeguards, and alias-qualified cost pass. Describe this as mechanism evidence only; do not promote or expand automatically.
- **Positive below useful floor:** the primary lower bound is above zero but the observed point difference misses both useful floors, while safeguards and cost pass. Report positive evidence with practical magnitude unresolved and retain the reference decision.
- **Inconclusive:** a primary interval, safeguard, rare-bin support floor, timing alias, or cost result is missing, crosses its decision boundary, or is underpowered. Report `UNKNOWN`/inconclusive for that component and retain the reference.
- **Harm or cost failure:** any required cell violates the exact −5-point or token −1-point harm safeguard, or the alias-qualified runtime cost fails. Retain the reference and report the failing cell and denominator; do not reinterpret another cell as a rescue.

The Pile route, `public_lora_2601` route, frequency-stratum rows, and retained reference remain descriptive or historical-comparison rows unless the frozen contract explicitly promotes a primary status. No outcome changes training, selection, truth handling, sample size, or the frozen criteria.


## Coordination-only CPU release and acknowledgement history

A2 explicitly released CPU at approximately 03:47 UTC after completing its work; no new GPU or timing work was requested. A root acknowledgement was rejected twice by automatic review because it would send nonpublic compute-scheduling metadata to an unverified or unauthorized destination, despite destination metadata verification and prior scheduling approval. The acknowledgement was not delivered and no workaround was attempted. This is coordination history, not a scientific failure. The task-local receipt is [`compute_release_followup.json`](../../experiments/TRR-0009/coordination/compute_release_followup.json) (1025 bytes; SHA-256 `fe47eed4b8f05b5e2b106de169f93df54748ef73dd08b74d5dcd7248c95e5db6`).

## Final score and decision addendum

This addendum supersedes the earlier pre-score status statements in the detailed chronology. The completed score is `experiments/TRR-0009/evaluation/score_v2.json` (133,163 bytes; SHA-256 `62b15bbf2f41b58010f25a74f765b7a13f0e3e05dc77762309de07306006e53d`), status `SCORE_COMPLETE_AFTER_PUBLIC_FREEZE`, with `truth_opened=true`. The frozen deterministic aggregation is `experiments/TRR-0009/evaluation/aggregate_decision_v2.json` (19,858 bytes; SHA-256 `e11a4fd746d9c44187c6186f92ff9f28f209bf6ee17bb51821d071edb783842f`), status `FAIL`, outcome `NO_USEFUL_PILOT_SIGNAL`, and action `RETAIN_REFERENCE_AND_REPORT_FAILED_GATE`. The bound cost summary is `experiments/TRR-0009/evaluation/timing/cost_summary_v2.json` (15,960 bytes; SHA-256 `15bb335ca8ed02a7196675f740946cc1a129bc3bb50e09bb94a910feb0abf765`).

The authoritative corrected numeric review is `experiments/TRR-0009/evaluation/score_numeric_review_v2.json` (40,841 bytes; SHA-256 `429bdbfc7f0ad05e2e266fb66d8355184c739030ccb9523247e8b092332ab1a1); it supersedes numeric review v1 because pooled arithmetic is descriptive only and exact practical benefit remains unresolved. The final review addendum is `experiments/TRR-0009/evaluation/score_numeric_review_v2_addendum.json` (4,023 bytes; SHA-256 `d698c41224ccaa704830c2c02d8180d90c7550d3cdfb048902fecb923734d13e`). It records the same next-experiment constraint: the observed raw gain/bias values do not establish effective-bound contact, so a larger bound is not justified.

The primary Finance `public_base` adaptable-minus-fixed result is 3 exact gains versus 1 loss, +0.78125 percentage points, with a 97.5% paired CP lower bound of −2.2807643 points; token accuracy is +8/32,512, +0.0246063 points, with a 97.5% source-record bootstrap lower bound of −0.003075879 points. The useful floors are +2 exact points and +0.25 token points, so both registered primary routes fail the useful-pilot criterion. The exact upper bound remains +3.7497033 points, so a +2-point exact benefit is not ruled out; the token upper bound is +0.0553641 points, below the token useful floor. All four exact-harm and token-harm gates, all eight rare/absent gates, and all four cost gates pass.

The rare/absent score uses 10,000 source-record bootstrap draws and seed 9009, with exposed-source counts ranging from 127 to 256; every lower bound is at least −0.1773 token points against the −1-point gate. The cost receipt reports mean adaptable/fixed ratios 1.095871, 1.087951, 1.076975, and 1.080318 across Finance P0, Finance LoRA, Pile P0, and Pile LoRA, respectively, with all 95% upper bounds below 1.25.

The first post-gate truth-preparation attempt is retained at `experiments/TRR-0009/evaluation/truth_prepare_compat_v2_failure.json` (2,369 bytes; SHA-256 `5c7b0ba2b40bf09e413e8932ac180d136b74dcc4391ce4e94fb22878080ff811`). It materialized selected labels in RAM after public-gate validation, then failed while serializing the frozen tokenizer directory; no sidecar, binding, or score was written and scoring did not start. The later bounded source adapter prepared the private sidecar and completed the score. Its execution receipt is `truth_prepare_score_compat_v2_execution_receipt.json` (14,312 bytes; SHA-256 `43768e63f6417564b41c7c7b2e87eb53412536e08667acd62dbba56523c99dcd`), which records the private sidecar at `/tmp/trr0009_truth_sidecar_compat_v2.safetensors` (787,736 bytes; SHA-256 `57b1fbd5f378a9bfc72c36804a8de0a437be7b9327da7a9c4a709f603d74f989`). Raw truth and the private sidecar are not committed. The successful run used full code commit `16e86a5fab3c693c6942620aabfdbdb865b5108d`; all 106 root-reviewed tests passed at that commit.

The final truth binding is `truth_binding_compat_v2.json` (6,207 bytes; SHA-256 `1a90e98092116ddcd0fae6876e8609f987cdb72f1d0a386c6b3375879cb423aa`). The public freeze is `public_freeze_compat_v2.json` (210,986 bytes; SHA-256 `acbb51fb8872dba1658cec708af10043e66b710b714dbaa53b1cf7545f0dd0c1`). The metadata compatibility implementation receipt is `metadata_compatibility_implementation_receipt.md` (2,545 bytes; SHA-256 `5683ec89d2542a43dec3c049c1539c0c18b685e3774715949dd0ebf5bc32c4ac`); the truth-source compatibility receipt is `truth_source_compatibility_implementation_receipt.md` (2,750 bytes; SHA-256 `120a361ab0ab2f84e9f4bf9ae3a443dc0b1bd1dd1d6bd75481407c736a5685cc`).
