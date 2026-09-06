# TRR-P03 implementation interface

This checklist is for the setup, observation, ranking, and scoring runners.
It keeps the natural panel and all target/method arms comparable without
allowing truth-bearing convenience fields into the readout process.

## Panel and observations

- Select 24 `p03-s1` Stage 1 records and 24 disjoint `p03-s2` holdout records
  from cached `HuggingFaceH4/no_robots` train revision
  `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`.
- Use seeds `20260906` and `20260907`, lengths 16/39/64/128, six records per
  length, and the exact coding/question-answer/creative-generation mapping
  and style-major slots in `plan.json`.
- Store source tokens, dataset indices, and text hashes only in evaluator
  private manifests/truth. Public observation metadata contains opaque IDs,
  shapes, masks, positions, and asset/config identities only.
- Reject exact prior overlap and whole records with any BOS-anchored P02
  prefix tuple collision. Token IDs remain permitted.
- Build matched and shifted observations from identical selected sequences,
  ordered IDs, masks, and position IDs. Only target weights change.
- Keep all `p03-s2` truth unopened until Stage 1 disposition and compact
  method freeze.

The native A1+A2 anchor is `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`,
and `p03-s1-r0012`, the length-39 stratum indices `[0,2,4,5]`. Preserve its
published 40-slot and K=256 policy.

## Static readout API

Each static method receives the same observation bundle and returns one full
vocabulary top-1 prediction and diagnostics per active post-BOS slot:

```text
predict(observation, attention_mask, position_ids, public_assets, config)
    -> prediction_ids, top1_scores, runner_up_scores, truth_free_metadata
```

The API must not accept source sequences, target condition labels, target
model identity, text hashes, style labels, correctness, or prior results. The
truth-free metadata may include top-1 ties, runner-up margins, and method/tie
policy IDs, but no true rank.

Raw scores `normalize_fp32(h)` against the read-only boundary table. Projected
scores `normalize_fp32(g(h))` against one shared full table of
`normalize_fp32(g(b_v))`. Native A1 uses the transformed query against
`normalize_fp32(E_v)` and makes decisions from native FP32 `exp(s)`-scaled
logits; divide by `exp(s)` only for post-decision cosine-equivalent reporting.
Standalone arms use descending score then ascending candidate ID on exact
finite-precision ties. Strict rank is `1 + count(score > true_score)`.

The A1+A2 anchor retains the published `torch.topk` candidate order and
first-argmax tie behavior. Report it separately rather than silently applying
the standalone tie rule.

## Freeze and access protocol

The aggregate pre-score validator must require freeze receipts for both target
arms and every method, including the anchor, with exact ordered IDs,
observation hashes, mask/position digests, geometry, source/config/asset
hashes, and immutable prediction rows. Validate every receipt and hash before
opening any Stage 1 truth. Open truth once after validation; never let a score,
route, timing, or correctness result trigger a rerun.

## Stage 1 scoring

For every condition and static arm, calculate token accuracy, exact-record
count/rate, per-length and per-style tables, first errors, strict ranks,
margins, and ties. For projected versus A1, pass one row per common record to
`paired_record_statistics` with `length`, correct counts, exact flags, and
optional paired correctness vectors. Use 10,000 draws with seed `20260905`,
length-stratified paired record-cluster resampling, micro token delta as the
primary statistic, and macro/exact deltas as secondary statistics. Report
correct-count changes separately from position-level token gains/losses when
vectors are unavailable. The gate uses only the criteria in `plan.json`.

## Stage 2 scoring

Only a passing Stage 1 gate unlocks `p03-s2`. Freeze exactly ranks 128 and 256
and their factors before holdout truth. Compare the uncompressed projected
parent, compact projected factors, and same-rank compact A1 controls using the
same query set, factor precision, row normalization, score kernel, and tie
rules. Report construction and deployed footprints separately. Stop after the
two predeclared ranks; do not add a rescue rank after a failure.
