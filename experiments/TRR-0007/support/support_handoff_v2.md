# TRR-0007 support handoff v3

Date: 2026-09-06
Task code commit for the frozen bank: `1cc665c47d343a5eab8af3dc6599ca18f401b652`
Scope: CPU bank materialization, provenance/compatibility accounting, public fitting-prefix exclusions, an opaque future reservation helper, and a read-only evaluator-interface review. No fresh evaluation source was selected and no evaluator truth was opened.

## Final broader-bank state

The frozen broader recipe is materialized at `experiments/TRR-0007/support/broader_bank_v5/`. The bank is ready for the registered TRR-0005 producer and its real P0/public-prefix forward; the construction receipt status is `METADATA_BANK_READY_FOR_REAL_P0_CAPTURE`.

- `constructed_public_tokens.safetensors`: 1,200 rows by 192 columns, SHA-256 `6d5b33e3eff1ce82a13f85153d85ecfe4da42977dcb227b2f301c141679c7d87`.
- The bank has 124,371 post-BOS positions, 1,080 retained natural rows, and 120 controlled rows: 60 new Pile parents and 60 new Finance parents.
- The controlled component has **3,600 distinct controlled IDs total** and 3,600 replacement occurrences. Of the IDs, **2,000 are preserved from the current enriched bank and only 1,600 are current-enriched-unseen additions**. Each selected ID is consumed once; every controlled row receives 30 replacements.
- Replacement positions are one-based and cover positions 1–127. Aggregate strata are 360 occurrences at 1–15, 720 at 16–39, 1,080 at 40–79, and 1,440 at 80–127.
- The retained-natural CPU fixture passed exact token-ID and attention-mask equality against the current enriched bank. The existing capacity receipt records `PASS_CAPTURED_H_EXACT_ON_1080_NATURAL_ROWS` with captured H shape `[1200,192,2048]`, `torch.equal` comparison, zero mismatches, and maximum absolute difference 0.0. This handoff does not rerun or reinterpret that capture or the improved fits.

The source-bound candidate map is `experiments/TRR-0007/support/candidate_pool_frequency_v2.json` (SHA-256 `aab98863c5a60b1dc087f477ee0583e54d6c375e43b6a3ee2f5969ca8eedc240`). It contains 73,772 public post-BOS IDs and 7,180,155 occurrences; the frozen selection has 58,535 eligible current-enriched-unseen candidates. The scan retained 4,495 eligible Pile rows and 9,760 eligible Finance rows after the declared exclusions. It used public fit partitions only and did not use prior-evaluation error identities.

The final parent exclusion manifest records 2,449 record IDs, 1,848 source-row keys, and 4,073 opaque sequence/reservation digests. The exact final ledger inputs for evaluator inventory and selector are:

| role | path | SHA-256 |
|---|---|---|
| corpus plan | `experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json` | `35d74424df60a1d62f7962850eaa3132fa57dd7001024f915cc4fc470a6d0e76` |
| selected parents | `experiments/TRR-0007/support/broader_bank_v5/selected_parent_rows.json` | `7de2ce7fca1d9489f652e0229b7cbfeacd625ca6ebd37c56d4405962da79f547` |
| parent/source exclusions | `experiments/TRR-0007/support/broader_bank_v5/public_parent_exclusion_manifest.json` | `bd1359f1184091570023e22a7682d1f97c08f8f05e47f69b3e6be089cd0181` |

These are the v5 paths; the older `broader_bank_v3` manifest must not be used as the final bank binding.

## CPU preparation accounting

The accepted preparation scope is exactly the source-bound candidate-v2 scan plus final constructor-v5 materialization:

- candidate-v2 scan: 12.335812667995924 s wall time, 1,374,408,704-byte maximum RSS;
- constructor-v5: 11.330641836000723 s wall time, 1,395,294,208-byte maximum RSS;
- **accepted preparation subtotal: 23.666454503996647 s (23.67 s).**

The excluded/repeated preparation scope is retained for audit and is not part of the accepted subtotal:

- candidate-v1 scan: 12.342694004997611 s;
- bank-v1: 11.115587488005986 s;
- bank-v2: 11.098388849000912 s;
- bank-v3: 11.342081937997136 s;
- bank-v4: 12.010443061997648 s;
- **excluded/repeated subtotal: 57.909195341999293 s.**

Prior light diagnostic receipts are a separate scope: broader-recipe v1/v2, label-only v1/v2 (including the pre-correction receipt), and public-development projections v1/v2. Their top-level receipt runtimes sum to **19.876863287019661 s**; nested projection runtimes are already included and were not added again. The recorded CPU campaign preparation total is therefore **101.45251313301560 s**, equal to accepted + repeated + prior diagnostic scopes. The untimed TRR6 boundary/equality receipt is excluded from this numeric total. Candidate scanning and bank materialization were model-free; the separate public-development projection diagnostics used the existing frozen decoder on CPU. The campaign used no network or reserved holdout rows and opened no private truth.

## Preserved construction attempts and corrections

The five bank directories are retained. Their `constructed_public_tokens.safetensors` files are byte-identical and all have SHA-256 `6d5b33e3eff1ce82a13f85153d85ecfe4da42977dcb227b2f301c141679c7d87`. The attempts are metadata/provenance corrections, not alternative scientific recipes:

1. v1 materialized the frozen rows and positions but labeled selected parent domains with inherited natural-source labels.
2. v2 corrected those parent-domain labels to `controlled_finance_context` and `controlled_pile_context` without changing any token row.
3. v3 rebound the receipt to the source-bound candidate-frequency rerun v2. The selected-ID digest and token bytes stayed unchanged.
4. v4 added exact record-ID, source-row-key, and opaque sequence/reservation exclusion sets to the parent ledger. The selected rows and token bytes stayed unchanged.
5. v5 cleaned source-scan identity digests and final output bindings. The selected parent rows, masks, token bytes, and exclusion sets stayed unchanged.

The historical `controlled_token_selection.selected_token_ids` field remains a 2,000-ID field because the registered TRR-0005 producer expects that legacy contract. The explicit `trr0007_support.selected_token_ids` field carries all 3,600 TRR-0007 controlled IDs, with the 1,600 current-enriched-unseen additions separately identified. Existing producer checks passed for 1,200 rows, width 192, 124,371 post-BOS positions, 120 controlled rows, and 3,600 replacement occurrences.

The separate first qualifier failure remains at `experiments/TRR-0007/qualification_enriched_v1/failure.json`. It stopped before tensor materialization because a TRR-0005 manifest relative activation path resolved to an absent TRR-0007-local `outputs/TRR-0005` path; no fit or gradient started. The corrected qualifier used the actual sibling TRR-0005 manifest and passed. None of the broader-bank v1–v5 attempts was a failed model run.

## Public fitting-prefix exclusion ledger

The final create-only ledger is `experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json` (259,979 bytes, SHA-256 `c4993d24d838ce2635b28b6736b85ffa849045f37e3a1e38b3904f3cbdb709e1`). It was generated by `scripts/trr0007_support_prefix_ledger.py` (SHA-256 `a83054813ba461548fdf1659b6f2769dbb7cad0b468a70d1fb9e55d1f242975e`) from only these existing public fit tensors and metadata:

- current enriched: `../TRR-0005/experiments/TRR-0005/corpus/coverage_mix_v1/constructed_public_tokens.safetensors` and its `public_activation_v1/enriched_fit_records.json`;
- improved public bank: `experiments/TRR-0007/support/broader_capture_v2/enriched_fit_cut4.safetensors` and its `enriched_fit_records.json`.

Each artifact contributes 350 rows with at least 128 active tokens, 700 row observations total. The descriptive union has 213 Pile-origin hashes, 179 Finance-origin hashes, and 78 other-origin (Alpaca) hashes, for **470 distinct hashes**. To close cross-style duplicate-prefix gaps, all 470 hashes are repeated under both fresh-style collector buckets; the trusted collector test reports 470 Pile hashes and 470 Finance hashes. No row IDs, indices, source text, token IDs, target labels, activations, weights, or truth are written.

The hash is the trusted scalar `final_sequence_sha256` key, computed over the first 128 active token IDs including BOS with the exact `scripts/trr0005_produce_confirmation.py:_sequence_digest` convention (`torch.int32` native C-order bytes followed by SHA-256). The attention masks were verified binary contiguous prefixes, and metadata active counts matched the masks. The earlier v1/v2 ledgers used lists of strings that the trusted recursive scanner would silently ignore; they are preserved as superseded preparation attempts. v3 uses one scalar keyed object per hash and was verified before handoff.

Both the count-only inventory and the selector must consume this v3 ledger through `--exclude-source`, in addition to the three explicit final-v5 bank-ledger arguments above. The same public prefix path must reach both collectors. A fresh Pile sequence must be checked against the full 470-hash set, and a fresh Finance sequence must be checked against the same full set; source-style provenance is retained only descriptively in this ledger.

## Evaluator exclusion-plumbing review

The selector now calls `scripts/trr0007_bank_ledger.py` to verify the final v5 exclusion manifest, selected-parent ledger, and corpus plan before loading public tokenizer/Arrow inputs. Its trusted collector also extends those three verified final-bank files into the selector exclusion paths. The final selection receipt should retain the v5 descriptors and counts above and the v3 prefix-ledger descriptor.

The inventory path must satisfy the same contract. Its dedicated final-bank arguments validate the v5 ledgers, but the inventory collector must also receive the three ledger paths (or an equivalent verified bundle) through its exclusion paths; otherwise `selected_parent_rows.json` and its variable-length `constructed_sequence_sha256` values are validated but not applied as exclusions. The evaluator interface also needs to replace any stale `broader_bank_v3` `--exclude-source` value with v5 and add `public_fit_prefix_exclusions_v3.json` to both inventory and selector commands. No inventory or selector scan was run by this handoff.

Before fresh selection, verify regular-file status and exact SHA-256 for all four exclusion inputs, fail closed on an omitted/stale/changed ledger, and include their descriptors in both create-only receipts. This preserves the final parent/source/sequence ledger and the separate first-128 public prefix ledger without opening fresh rows early or opening truth.

## Future opaque reservation helper

`scripts/trr0007_opaque_reservation.py` (SHA-256 `d85649b4e27a58bac81fc924e9b582e34ecf6b8814ae7014f76549f1318da482`) is a create-only helper for a future completed identity-only source selection. Root can invoke it after selection with:

```bash
PYTHONPATH=.:src:scripts python3 scripts/trr0007_opaque_reservation.py \
  --selection <completed-identity-only-selection.json> \
  --output experiments/TRR-0007/coordination/p06_opaque_source_sequence_reservation.json
```

It reads no fresh source rows and exports only sorted `public_record_sha256` and `final_sequence_sha256` values. The output contains no source text, record IDs, source indices, domain labels, target labels, token IDs, weights, or truth. It was intentionally not run because no fresh selection ledger exists yet; root should invoke it after the selector's identity-only result is frozen.

## Task-owned evidence

- final bank: `experiments/TRR-0007/support/broader_bank_v5/`;
- candidate frequency receipt: `experiments/TRR-0007/support/candidate_pool_frequency_v2.json`;
- capacity capture verification: `experiments/TRR-0007/support/broader_capture_v2/capture_verification_receipt.json`;
- public fitting-prefix ledger: `experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json`;
- prefix-ledger builder: `scripts/trr0007_support_prefix_ledger.py`;
- future opaque reservation helper: `scripts/trr0007_opaque_reservation.py`.

Focused TRR-0007 support tests passed before this handoff; the new prefix helper was compiled, run against both existing tensors, and checked through the trusted collector with 470 hashes in each fresh style. No fresh-data selection, GPU run, or denied truth-sidecar access was attempted.
