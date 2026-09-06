# TRR-0007 public fitting-support diagnostic

This artifact is a descriptive, label/mask-only analysis of the existing public TRR-0005 enriched fitting bank. It does not select fitting rows from prediction errors, access evaluator-private truth, or create fresh evaluation truth. The optional frozen public-development projection path is implemented in `scripts/trr0007_support_diagnostics.py` but was not run here.

The earlier output is preserved at `experiments/TRR-0007/support/label_only_v1/` for audit. Its recipe and label-only correctness fields are superseded by this corrected v2 output. The corrected v2 joint cells use `null` plus `correctness_status: not_computed` because this run loaded labels for support counts but no predictions.

## Inputs and command

The run used these existing public assets:

| asset | path | SHA-256 |
|---|---|---|
| enriched fit labels/mask and tensor container | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/outputs/TRR-0005/enriched_fit_cut4.safetensors` | `191cb77dae8d002402bcf3f126a20c5d8d34111a6e6871d66507503ca6725a99` |
| enriched fit records | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_fit_records.json` | `7f197b077aa0aa66edfdd8d92c8daa5b4cb2ae36bdbb80938e4ab8f1de117943` |
| public corpus plan | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/corpus/corpus_plan.json` | `ef8b44bf786b2f7c81e078c771072de278c6fd42b5cd0543d977a2d1ad0b5d84` |
| public development labels/mask | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/adapter_v2/validation_mixed_cut4.safetensors` | `a8e7633ffb369864af33754c5ebb2d9a4ca9d6e7d4550731e8ff26e20c8200cf` |
| public development records | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json` | `30b422b681bef5e7af4c26d339e57dfb3571ecef8077bdc4be5d960ef05c9777` |

Exact command:

```text
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 scripts/trr0007_support_diagnostics.py --fit-tensor /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/outputs/TRR-0005/enriched_fit_cut4.safetensors --fit-records /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_fit_records.json --corpus-plan /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/corpus/corpus_plan.json --validation-tensor /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/adapter_v2/validation_mixed_cut4.safetensors --validation-records /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json --output-root experiments/TRR-0007/support/label_only_v2
```

The run used CPU only, no network, and no private truth. Runtime evidence in `support_summary.json` is approximately 0.65 seconds elapsed and 557,916,160 bytes maximum resident set size; the 947 MB fit container was hashed/read for its small label and mask tensors, while activations were not loaded.

## Exact bank geometry

The enriched fit bank has 1,200 records, stored width 192, 125,571 active BOS-inclusive labels, and exactly 124,371 post-BOS opportunities. It contains 1,200 BOS positions and 104,829 padding positions; BOS and padding are excluded from supervision. Of post-BOS opportunities, 112,825 are in one-based positions 1–127 and 11,546 are in positions 128–191.

| input style | records | post-BOS positions | positions 1–127 | positions 128–191 |
|---|---:|---:|---:|---:|
| natural_alpaca | 600 | 55,206 | 52,397 | 2,809 |
| natural_pile | 300 | 31,047 | 28,264 | 2,783 |
| natural_finance | 180 | 18,719 | 16,924 | 1,795 |
| controlled_pile | 60 | 9,443 | 7,620 | 1,823 |
| controlled_finance | 60 | 9,956 | 7,620 | 2,336 |
| **total** | **1,200** | **124,371** | **112,825** | **11,546** |

The fit bank contains 15,602 distinct post-BOS token IDs. Global post-BOS frequency bins are: seen once, 7,156 IDs/7,156 occurrences; seen 2–4, 5,507 IDs/14,336 occurrences; seen 5–16, 2,246 IDs/18,090 occurrences; seen 17–64, 525 IDs/15,254 occurrences; seen at least 65, 168 IDs/69,535 occurrences. The median ID frequency is 2, the 90th percentile is 8, the 99th percentile is 68, and the maximum is 3,899. `unseen_0` is zero by construction for a support table computed from the fit bank itself; public-development rows can be unseen relative to this fit map.

## Joint support finding

The table reports examples in the rare support bins `seen_1 + seen_2_4` for each input style and position range. The complete style × position × frequency table, including distinct IDs, exact examples, and explicit zero cells, is in `support_summary.json`.

| input style | 1–15 | 16–39 | 40–79 | 80–127 | 128–191 | rare total |
|---|---:|---:|---:|---:|---:|---:|
| natural_alpaca | 0 | 738 | 3,495 | 1,913 | 577 | 6,723 |
| natural_pile | 1,074 | 1,798 | 2,407 | 1,582 | 722 | 7,583 |
| natural_finance | 0 | 17 | 1,046 | 601 | 246 | 1,910 |
| controlled_pile | 264 | 488 | 803 | 1,004 | 586 | 3,145 |
| controlled_finance | 111 | 181 | 613 | 727 | 499 | 2,131 |

Across all styles, rare-support share is 8.05% at positions 1–15, 11.19% at 16–39, 20.55% at 40–79, 23.00% at 80–127, and 22.78% at 128–191. The current controlled rows therefore contribute rare IDs throughout the evaluation range, but the controlled supplement is not balanced by position: its recorded replacement placement is concentrated according to the legacy full-prefix spacing, while the natural rows have domain-specific gaps (for example, Alpaca has no rare examples at positions 1–15 in this fit bank because those positions are dominated by repeated template tokens).

The support table is descriptive. It does not claim that a frequency/position cell causes an error, and it does not use any error identity to choose fit rows.

## Replacement correction and verification

The current corpus plan has 120 controlled rows, 3,600 recorded replacements, and 2,000 distinct replacement IDs. Every controlled target record has post-BOS length at least 128 (minimum 128, maximum 191), but that is a statement about target record length, not replacement offset.

The corpus metadata stores a zero-based offset `o` after BOS, and `apply_replacements` writes it to `token_ids[o + 1]`. The v2 verifier compared every recorded `(slot, o, replacement_token_id)` against the captured fit `token_ids[slot, o + 1]`:

- checked controlled records: 120/120;
- checked replacement occurrences: 3,600/3,600;
- mismatches: 0;
- result: `PASS`.

The actual recorded offsets range from zero-based 2 through 187. In one-based post-BOS coordinates they range from positions 3 through 188: 2,877 occurrences are in positions 1–127 and 723 are in positions 128–191. The earlier premise that all replacements were at offsets at least 128 was a conflation of long controlled target records with their replacement offsets. The exact raw/one-based counts and the mismatch receipt are in `support_summary.json` under `controlled_replacement_summary` and `recorded_replacement_verification`.

## Improved public sampling recipe

The proposed matched-support arm keeps the current public natural source identities, the current 120 controlled slots, and the ordered target length/mask vector. It changes the controlled replacement placement and requires materializing each constructed sequence with one registered real P0 forward at cut 4. It does not splice activations, inspect target weights, or use fresh/hidden answers.

- Keep 1,200 records, 124,371 post-BOS opportunities, width 192, and the 3,000-step × 512-draw full-vocabulary CE schedule (1,536,000 draws).
- Keep 600 Alpaca, 300 Pile, and 180 Finance natural rows, plus 60 controlled Pile and 60 controlled Finance rows.
- Keep the current controlled slot indices exactly, so the natural complement and natural rendered identities are unchanged. The recipe requires each controlled target length to cover positions 1–127.
- Keep the same 2,000 ordinary public-frequency-selected replacement IDs and 30 replacements per controlled row for a matched 3,600-occurrence arm. Any future comparison that changes the ID set should be registered as a separate support intervention, because shifting positions alone does not test broader identity/context coverage.
- For each controlled row, choose ordinary positions in four one-based bins with quotas `(3, 6, 9, 12)` for `1–15`, `16–39`, `40–79`, and `80–127`. Store each selected position as zero-based offset `position - 1`. Exclude structural BOS/PAD values, rotate the evenly spaced choices using the SHA-256 task/seed/record/bin digest, and fail closed if a bin lacks its quota.
- Use the real registered P0 `ContiguousPublicPrefix.forward_full` for every complete constructed sequence. Recompute activations after replacement; activation splicing is prohibited.
- Preserve public source partitions: existing registered Alpaca fit rows, Pile train rows `[2000,7000)`, and Finance train rows `[2000,12000)`. Reserve Pile `[7000,10000)` and Finance `[12000,20000)` for the evaluator without inspecting them here.

The machine-readable recipe is `improved_public_sampling_recipe.json`. It is a proposal pending real P0 materialization and should be compared against a capacity-only arm under the same frozen evaluation protocol. A future broader-identity/context arm should retain the same record/opportunity budget while adding public natural contexts or a separately registered public replacement-ID set; it must preserve the exclusion manifest and remain independent of prediction errors.

## Exclusions and status

`public_exclusion_manifest.json` contains exact fit record IDs, source record IDs, rendered hashes, and active-sequence hashes for all 1,200 current fit rows, plus all 48 registered TRR-0004 public development validation rows when supplied. It also declares the untouched future ranges above. The manifest states `private_truth_accessed: false`, `fresh_evaluation_truth_accessed: false`, `target_weights_accessed: false`, and `source_text_retained: false`.

The support outputs are descriptive evidence only. The public-development validation section reports support and coverage counts; correctness/error/accuracy remain `null` with `not_computed` status until an explicitly authorized frozen projection is run.
