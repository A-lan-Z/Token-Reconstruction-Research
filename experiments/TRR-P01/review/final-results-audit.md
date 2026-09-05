# Final numerical results audit

**Scope:** independent light audit of the completed TRR-P01 matrix, score joins, and pre-truth freeze ordering. Raw scores, code, reports, and manifests were not modified; private truth tensors were not reopened by this audit.

## Join, coverage, and freeze checks

The two score artifacts are:

- `experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000-score.json` (`matched_public`)
- `experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001-score.json` (shifted full-SFT; legacy condition ID `shifted_target_lora`)

Each has status `SCORED_AFTER_VERIFIED_FREEZE`, eight declared method/metric arms, the same ordered IDs `p01-r0001` through `p01-r0016`, 39 post-BOS positions, and 624 scored tokens per method. Each corresponding prediction artifact has eight `[16, 40]` int32 tensors, 128 JSONL rows (16 records × 8 arms), fixed BOS column 0, finite values, and one row per record/arm. Per-position sums and per-style sums agree with every aggregate count; each style contributes four records and 156 scored tokens. All 16 records therefore participate in every condition/method cell, with no missing or duplicate join.

The private condition map pairs arm-000 with `matched_public` and arm-001 with the raw condition ID `shifted_target_lora` (the shifted full-SFT checkpoint) while pointing both to the same private truth artifact and panel manifest. Public IDs, record order, mask digests, position digests, geometry, model identity, and execution constants match across arms. The two public activation tensors differ for all 16 rows, as expected for the two target-weight conditions, and each row digest matches its own public tensor. The public interface contains no condition label or target callable.

`joint-freeze-sidecar.json` reports `JOINT_MATRIX_FROZEN_BEFORE_TRUTH_OPEN`; `joint-freeze-validation-pre-score.json` reports `JOINT_HASH_VALIDATION_PASS_BEFORE_TRUTH_OPEN` for both arms, all eight methods, and 28 file records. Both regular arm freeze receipts report `FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN`, and both finish receipts report `OUTPUTS_HASHED_AFTER_PRETRUTH_FREEZE`. The score files explicitly report truth opened only after freeze verification. The final reconstruction evidence and receipts use implementation commit `e43a595d0f4300d5db8f93c86881b455dfa30ea4`; the evaluator public source certification records the committed public-generation source at `6e05feeade57593cdadea2d4db4ce40085a51f59`.

## Accuracy results

| method | matched public | shifted full-SFT | shift |
|---|---:|---:|---:|
| boundary cosine | 255/624 = 40.865% | 243/624 = 38.942% | −12 tokens (−1.923 pp) |
| boundary L2 | 243/624 = 38.942% | 236/624 = 37.821% | −7 (−1.122 pp) |
| raw embedding cosine | 231/624 = 37.019% | 231/624 = 37.019% | 0 |
| raw embedding L2 | 175/624 = 28.045% | 180/624 = 28.846% | +5 (+0.801 pp) |
| reference corrected cosine | 81/624 = 12.981% | 80/624 = 12.821% | −1 (−0.160 pp) |
| reference corrected L2 | 74/624 = 11.859% | 80/624 = 12.821% | +6 (+0.962 pp) |
| historical A1 cosine | 513/624 = 82.212% | 510/624 = 81.731% | −3 (−0.481 pp) |
| historical A1+A2 port cosine | 613/624 = 98.237% | 613/624 = 98.237% | 0 |

The A1+A2 port is the only arm with complete records: 15/16 records are entirely correct in both conditions (93.75% exact-record rate). Every other arm has zero fully correct records. The shifted full-SFT condition changes the weaker methods by only 0–12 tokens on this paired panel and does so inconsistently; A1+A2 is identical at aggregate, style, and position counts.

## Position and context effect

The reference-corrected methods are correct for all 16 records at position 1 in both conditions, but only 3/16 at position 2 for matched public and 4/16 for shifted full-SFT. Their first error is at position 2 for 13/16 and 12/16 records respectively; position-2 failures therefore occur before any prior reconstructed-token error can propagate. Accuracy then remains very low (position 4 is 1/16 and position 39 is 2/16 in both conditions). This is evidence for an immediate reference-offset/context correction failure, rather than evidence that the whole effect is caused by accumulated early reconstruction mistakes.

Boundary cosine/L2 also get position 1 correct for all 16 records, followed by 6/16 and 4/16 position-2 accuracy in matched public and 4/16 and 3/16 under shifted full-SFT. In contrast, the historical A1+A2 port remains 16/16 through positions 1–4 and 15/16 at position 5 in both conditions. The high control performance and the correction collapse make the current fixed reference correction the least credible variant for another unchanged run.

## Cost and recommendation

Per arm, static full-vocabulary lookup took 6.94–7.12 s, historical A1+A2 took 32.57–34.60 s, and reference correction took 528.91 s (`cosine` 243.30 s plus `L2` 285.60 s) for arm-000 and 595.39 s (`287.82 + 307.55` s) for arm-001. Each arm executed 1,248 correction probes, 1,280 persistent correction commits, and 159,744 historical candidate simulations; peak RSS was 5,239,828–5,244,048 KiB (about 5.00 GiB, or 5.37 GB), and the CPU resource guards passed. The correction phase dominates the run cost by an order of magnitude.

Recommendation: **neither current static/correction variant merits another unchanged scientific round**. Keep boundary and raw embedding as baselines, and retain the correction outputs as a negative diagnostic. A follow-up is justified only for a different deterministic context model or a narrowly verified correction rule after investigating the fixed reference-220 offset. The key uncertainty is statistical and causal: the panel has only 16 records (four styles) with strongly correlated positions, so small condition deltas are not reliable effect estimates, and position-2 evidence localizes the correction failure but does not by itself distinguish an implementation sign/offset error from a fundamentally unsuitable fixed-reference assumption. The historical A1+A2 result is a strong geometry-port control, but it should not be treated as an exact native A1+A2 benchmark.
