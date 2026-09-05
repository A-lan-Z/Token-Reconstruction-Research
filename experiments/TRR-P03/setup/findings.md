# TRR-P03 setup findings

This is the task-local, pre-observation preparation record. It is input for root review; it is not a final attack index or a scored result.

## Frozen candidate panel

The canonical preparation is `experiments/TRR-P03/setup/panel-20260906-frozen`, generated from selector commit `ebb814aa04049140b3a1dc68e59272b3aff48a88` (selector SHA256 `db91435a1d3077b70c5d463e31ae0fee94f2edbc29f149679606ab1f45027afe`). It uses cached `HuggingFaceH4/no_robots` train Arrow revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` (Arrow SHA256 `5a9193e927d899d167fd40553d0b403499f5f9cf9a9254db19399a4d0b3550fb`). The selector reads the existing `prompt` field as-is, ignores `messages`, prepends BOS 128000, and crops the first requested post-BOS tokens. No instruction or Unicode wrapper is added.

Stage 1 has 24 records, six each at 16, 39, 64, and 128 scored post-BOS tokens, for 1,482 scored tokens per target condition. Stage 2 is a separate disjoint 24-record holdout with the same quotas. Metadata styles are `coding`, `question_answer`, and `creative_generation`, with two records per style per length and eight records per style per stage. The predeclared A1/A2 anchor is the length-39 Stage-1 within-stratum selection `[0, 2, 4, 5]`, with opaque IDs `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and `p03-s1-r0012`; it covers all three styles.

The frozen replay and r5 have identical Stage-1/Stage-2 selection IDs, source text hashes, metadata hashes, and separate truth-file hashes. The token-free comparison receipt is `experiments/TRR-P03/setup/frozen-vs-r5-comparison.json`; it emits no token values. Stage-1 and Stage-2 truth are in separate evaluator-only directories and remain unopened.

## Exclusion audit

The selector checks all 51 P02 tuples at every scored prefix and skips an entire candidate record when a tuple occurs. It does not globally exclude any token ID. It also rejects exact source-text hashes from the known opened P01/Pile manifests and exact source rows only when the dataset identity matches. The selected no_robots Stage-1 and Stage-2 source text hashes have zero overlap with the 368 known opened hashes. The frozen audit recorded zero P02 prefix collisions and 90 too-short candidate rejections. Two older blind selection commitments disclose no source mapping; their overlap remains explicitly unaudited rather than being claimed absent.

## Runtime interface

The shared truth-free contract is `experiments/TRR-P03/setup/interface.json`. The evaluator should materialize one runtime index per opaque bundle using schema `token-reconstruction.trr-p03-observation-index.v1`, with grouped safetensors observations containing `activations`, `attention_mask`, and `position_ids`. Each activation has shape `[records, sequence_length, 2048]`, `bfloat16`, and cut depth 4; `sequence_length = scored_tokens + 1`, positions start at 0, and BOS is not scored. Runtime index records expose only opaque IDs, geometry, artifact bytes/hashes, and mask/position digests. Source IDs, text hashes, category/style metadata, target labels, token IDs, and truth are forbidden. The two evaluator bundle IDs have identical record order, masks, and positions; the bundle-to-target map remains evaluator-only.

The source-side evaluator panels and stage-specific JSONL truth sidecars are under `stage1/` and `stage2_holdout/` in the generated panel directory. They contain token IDs and must stay outside reconstruction. Stage 1 truth is eligible for opening only after both target observations and prediction artifacts are frozen. Stage 2 truth remains sealed until the Stage-1 gate disposition and compactness constants are frozen.

## Assets and compute boundary

The matched public base is `meta-llama/Llama-3.2-1B-Instruct` revision `9213176726f574b556790deb65791e0c5aa438b6`. The required paired shifted target is the full public `Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct` snapshot revision `7fa9d06a59246629244cdd3b6b92e4fc756baa0f`; P01 historically called the condition `shifted_target_lora`, but the cached resource itself is full SFT. The read-only boundary table SHA256 is `51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3`; the historical Alpaca lens SHA256 is `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`.

Preparation used only cached dataset/tokenizer work. No model load, forward pass, full table read, GPU use, or truth scoring occurred. The exact frozen command was:

```text
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/trr_p03/prepare_panel.py --output-root experiments/TRR-P03/setup/panel-20260906-frozen
```

It ran from `2026-09-05T14:40:57.093488150Z` through `2026-09-05T14:41:06.012840223Z`; `/usr/bin/time` recorded elapsed 8.91 s, user 10.04 s, sys 2.15 s, and maximum RSS 1,213,016 KiB. The complete receipt is `experiments/TRR-P03/setup/panel-20260906-frozen/preparation-run-receipt.json`. Heavy observation generation remains root-controlled; this setup task did not launch it during the released window.

## Reproducibility and retained attempts

The selector is `scripts/trr_p03/prepare_panel.py`. Preparations r1 through r5 remain retained as unscored development evidence with exact receipt hashes and supersession reasons in `experiments/TRR-P03/setup/preparation-attempts.json`; r4 completed successfully and no failed-r4 receipt exists. Only `panel-20260906-frozen` is bound to a committed source and eligible as the canonical preparation. Task-local state is `coordination/parallel/TRR-P03.json`; root owns the final plan, manifest, result, commit, and publication metadata.
