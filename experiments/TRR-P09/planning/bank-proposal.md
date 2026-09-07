# TRR-P09 Agent 2 public-bank proposal

**Status: PROPOSED, count/tokenization-only metadata review.** A permitted public Alpaca rendering/tokenization count audit has occurred; no P09 source selection, public-model forward, fitting, final-panel selection, or truth access has occurred. Inputs are published TRR-0004/0005/0007/0009 metadata at parent commit `4602f98eeb03ac121d8bbc230d7e2b219551b914`.

## Published B0 construction and identity check

`improved_public_bank` contains 1,200 complete constructed rows:

| stratum | rows | source-parent interpretation |
| --- | ---: | --- |
| natural Alpaca | 600 | 600 natural public parents |
| natural Pile | 300 | 300 natural public parents |
| natural Finance | 180 | 180 natural public parents |
| controlled Pile | 60 | 60 public parent contexts, derived replacement rows |
| controlled Finance | 60 | 60 public parent contexts, derived replacement rows |
| **total** | **1,200** | **1,080 natural + 120 controlled parents** |

Published `enriched_fit_records.json` metadata (`TRR-0007/support/broader_capture_v2`, SHA-256 `808cfd0f95ea3ee66c1d4094f3c10f1e346ba986a6d77138a61b8f8f13c4a738`) confirms 1,200 rows, 1,080 `synthetic=false` and 120 `synthetic=true`; `record_id`, `source_record_id`, and `rendered_sha256` are each unique across all 1,200. The natural and controlled `source_record_id` sets are each unique and have zero cross-overlap. Thus the published count-level `unique_public_parent_rows=1200` is supported: controlled rows have distinct parent identities, but remain derived variants rather than independent natural examples.

The bank has 124,371 valid post-BOS fit positions, width 192 and hidden size 2,048, 112,825 positions in the H128 range, and 17,126 distinct target IDs. These remain the baseline geometry/support counts.

## Two candidate 12,000-row expansions

The user goal is roughly tenfold **independent natural support**, so row totals and parent totals must be reported separately.

| design | final natural rows | final controlled rows | final constructed rows | new natural parents | independent-natural multiplier |
| --- | ---: | ---: | ---: | ---: | ---: |
| **proportional full strata (recommended if capacity passes)** | 10,800: 6,000 Alpaca, 3,000 Pile, 1,800 Finance | 1,200: 600 Pile, 600 Finance | 12,000 | 9,720 (5,400/2,700/1,620) | **10× B0 natural support** |
| all-natural additions | 11,880: 6,600 Alpaca, 3,300 Pile, 1,980 Finance | 120: 60 Pile, 60 Finance retained from B0 | 12,000 | 10,800 (6,000/3,000/1,800) | **11× B0 natural support** |

The proportional target preserves B0's exact construction proportions (natural 600:300:180 and controlled 60/60) while still delivering 10× natural-parent support; its additions are +5,400 Alpaca, +2,700 Pile, +1,620 Finance natural parents and +540 each controlled Pile/Finance contexts. The all-natural option gives more independent examples but drops the controlled fraction from 10% to 1%, so a quality difference would mix bank size with a construction-mixture change. I recommend the proportional target for the single bounded comparison, with all natural and controlled parent IDs counted separately. Do not run both as an unregistered extension.

## Public source/cache capacity

The published Alpaca source is `tatsu-lab/alpaca`, train revision `dce01c9b08f87459cf36a430d809084718273017`, 52,002 rows. The registered cache is `/home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow` (46,230,248 bytes, SHA-256 `f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794`). TRR-0004's public fit split touched 1,200 rows selected from that dataset (seed 7, zero rejected) and has a separate 24-row validation continuation.

The permitted count-only rendering audit (`experiments/TRR-P09/planning/alpaca-capacity-audit.json`, SHA-256 `d1f11364cabe6db876bb83f09ce1d8d7db1cdc3f76ac2e8bafaed149ba33a5c9`) rendered all 52,002 rows with the pinned tokenizer and historical recipe: all 52,002 meet the minimum full-token threshold of 32. After excluding 1,224 keyed fit/validation row IDs from the published ledgers, 50,778 rows remain; 39,765 have at least 64 post-BOS tokens, 24,401 at least 96, 14,317 at least 128, and 6,824 at least 160. This is ample public capacity for either +5,400 or +6,000 Alpaca parents as a count-only upper bound. The exclusion metadata contains a union of 4,193 digest values (4,073 opaque sequence/reservation digests plus rendered identity values); the audit found zero rendered-text SHA-256 hits against that union. Those opaque values use a different or unresolved canonicalization, so zero hits are not evidence of disjointness. The 50,778 row count is therefore an upper bound until a namespace-correct identity ledger is frozen; no source rows were selected or written.

For Pile and Finance, the approved fit partitions are Pile `[2000,7000)` and Finance `[2000,12000)`. Two published counts must be distinguished:

- `improved_public_sampling_recipe.json` reports Pile 4,640 and Finance 9,760 after current-fit exclusion only.
- The final `broader_bank_v5/bank_construction_receipt.json` reports the conservative post-filter counts Pile **4,495** and Finance **9,760**. For Pile, 5,000 partition rows were checked: 56 were too short, 375 were metadata-excluded, and 74 were hash-excluded, leaving 4,495. This is the count to use for planning; 4,640 is an intermediate pre-full-exclusion figure.

Agent 1's uninspected reservations, Finance `[12000,20000)` and Pile `[7000,10000)`, remain excluded. The final conservative Pile/Finance capacity is therefore 14,255 rows. It covers the proportional target's new Pile/Finance natural parents plus new controlled contexts (3,240 Pile and 2,160 Finance parent contexts), subject to the same opaque/length audit. No new P09 validation pool is proposed here: the natural `public_base` validation binding already opened by Agent 1 remains a separate root-owned contract. The Alpaca count-only audit above covers its required 5,400 parents; final freezing still requires the same identity/hash ledger.

## Selection and forward boundary

Before selection, setup must provide a count-only ledger covering the exact Alpaca cache, Pile/Finance partitions, B0 IDs, all registered public-development IDs, approved opaque ledgers, and Agent 1 reservations. The bounded recipe is: preserve B0 as the byte-equivalent prefix; use the inherited post-BOS bins 40–63, 64–95, 96–127, and 128–191; allocate the proportional target by the published domain/bin counts; restrict controlled replacements to the inherited 128–191 range and four inherited position bins; and order eligible public rows by the inherited stable SHA-256 source/revision/row key. Apply exact row IDs, source-row keys, and namespace-matched sequence digests plus Pile `[7000,10000)` and Finance `[12000,20000)` reservations before any candidate can qualify. This recipe is documented only and has not executed selection. Preserve B0 exactly as a nested subset. If proportional Alpaca capacity or the full source/length quotas fail, stop and revise the contract; do not fill the gap with repeated rows or controlled variants.

All selected natural parents and controlled parent contexts must be counted separately from complete constructed sequences, valid position examples, token support, and repeated exposures. New activation pairs must be generated later by the registered public model's contiguous cut-4 forward using the inherited tokenizer, BOS/PAD, width-192 masks, and public numerical settings; no activation splicing, private/final source, or truth payload is in this proposal.

The Alpaca count-only audit is complete. The next permitted action is the joint identity-ledger check and storage preflight; source selection, public forwards, and fitting remain pending root/Agent 1 common-contract approval. The existing Agent 1-owned natural `public_base` validation binding is reused if the common contract approves it; this proposal creates no validation panel.
