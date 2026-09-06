# TRR-P06 result report

**Status: scored; the registered public-base gate is a qualified negative for `p06_full_record` versus `p06_past_only`, and the task-local decision is to retain `p06_past_only`.** The result is exploratory evidence for this H128 natural panel and is not a canonical dual-benchmark replacement or a universal claim about later activations.

## Question and frozen scope

P06 asked whether later-position activations improve reconstruction beyond a past-only activation-context decoder within one fixed H128 decoder family. The primary estimand was `p06_full_record - p06_past_only`; `p06_positionwise_diagonal` was a competent positionwise control. The public changed target was evaluated as a paired transfer condition but was excluded from the registered gate.

All three arms used the same inherited public affine direct path, normalized FP32 public embedding readout, H128 crop, optimizer, fit records, labels, validation records, and two replicate seeds (6106 and 6107). The masks and ordered position schedules were shared within each replicate. The diagonal arm has the same nominal parameterization but structurally inactive Q/K paths; its results are a control, not an equal effective-capacity claim.

The source-only H128 preflight, disposable full-record backward qualifier, public-fit-error capacity probe, and six main fits passed. The capacity probe was diagnostic only: it used 256 frozen public-fit error positions, 512 with-replacement query draws per update, 300 updates, and discarded states. The six main fits ran 3,000 updates, selected the earliest maximum of micro token accuracy including step 0, and retained separate state and schedule receipts. The earlier missing-observation qualification attempt remains an excluded engineering failure; corrected r2 qualification supplied the release evidence.

## Freeze, scoring, and uncertainty

The complete student matrix and the separate 64-record-per-domain A1+A2 anchor were frozen before truth materialization. The no-truth freeze validated all four domain/target cells, both fit replicates, all three student methods, identical source/mask/position bindings, and both anchor domains. The scored output is `experiments/TRR-P06/runtime/scored-r1/results.json`, SHA-256 `496100a7d00e74fcea71f9392df35b45fb57806bb147ee0b57758932cb3fe224`. Its provenance binds the prediction manifest `6bb3cfcb85b0a704e38d6116af67ba2eeebbacb8926b0e73f540aea60bffbcbb`, joint freeze receipt `d3f7d69a153d045d4b32dfd833125146e611b675b51615fd7ac8c7bc25aa26f7`, source selection `d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a`, and truth manifest `21c07d64c489e16bd9a220f4175c23d225515c786e659613ee05ec1f01770e48`.

Each student cell has 256 source records and 127 scored tokens per record, or 32,512 scored tokens. The paired bootstrap used 10,000 draws with seed 6306, domain-only strata, and source-record clusters. The two training replicates were averaged within each source record before resampling; seeds were not treated as independent records. The same source-index schedule was reused across the two target conditions within each domain. Exact-record contrasts use denominator 256.

The `p06_full_record` arm is explicitly offline full-record inference, not a token-streaming decoder: it may precompute activation-only features for the complete valid H128 record and then emits immutable predictions left to right, without revising earlier outputs. During fitting and inference it receives only the stored activation observations, validity masks, position IDs, and common direct affine path; it receives no source plaintext or token IDs, target prefix, guessed-token feedback, A2 procedure, or later reconstructed answers. The six selected main-fit states (one per arm and replicate) are retained only for frozen inference and provenance; the two-update qualifier and 300-update probe states are discarded and cannot initialize or select a main fit. The source capture is 192 tokens, but every arm crops to stored H0..H127 before masks, schedules, fitting, validation, or scoring; positions 128..191 are never keys, queries, labels, or metric denominators. The four raw observation payloads are approximately 134.5 MB each, remain local-only, and are represented in the freeze manifest by byte counts and SHA-256 hashes rather than hosted in the PR.

### Absolute student results

The entries are mean percentages over the two fit replicates. Exact percentages use 256 records; token percentages use 32,512 scored tokens per seed.

| Domain | Target | Diagonal token / exact % | Past-only token / exact % | Full-record token / exact % |
|---|---|---:|---:|---:|
| finance | `public_base` | 99.0373 / 44.3359 | 99.2557 / 53.1250 | 99.2249 / 51.7578 |
| finance | `public_lora_2601` | 99.0757 / 46.6797 | 99.2864 / 54.8828 | 99.2664 / 53.9063 |
| pile | `public_base` | 93.7085 / 3.9063 | 94.4190 / 5.2734 | 94.3544 / 4.4922 |
| pile | `public_lora_2601` | 93.8669 / 3.7109 | 94.7096 / 4.1016 | 94.5989 / 4.8828 |

### Registered primary contrast

The gain/loss column reports paired token positions where full-record was correct while past-only was wrong, and the reverse, respectively. These are descriptive rates in percentage points from the replicate-averaged source-record pairs.

| Domain | Target | Full − past token delta pp (95% CI) | Full − past exact delta pp (95% CI) | Gains / losses pp |
|---|---|---:|---:|---:|
| pile | `public_base` | −0.0646 [−0.1246, 0.0000] | −0.7813 [−1.5625, −0.1953] | 0.2045 / 0.2691 |
| finance | `public_base` | −0.0308 [−0.0584, −0.0031] | −1.3672 [−2.9297, 0.0000] | 0.0338 / 0.0646 |
| pile | `public_lora_2601` | −0.1107 [−0.1707, −0.0508] | +0.7813 [+0.1953, +1.5625] | 0.1907 / 0.3014 |
| finance | `public_lora_2601` | −0.0200 [−0.0415, +0.0015] | −0.9766 [−2.5391, +0.5859] | 0.0246 / 0.0446 |

The registered public-base thresholds were a 1 percentage-point token benefit and a 5 percentage-point exact-record benefit, with corresponding −1 and −5 percentage-point harm bounds. Neither public-base arm supported the benefit threshold. Both public-base token intervals had upper bounds at or below zero, and both exact intervals had upper bounds at zero or below. The gate therefore returned `QUALIFIED_NEGATIVE_RETAIN_PAST_ONLY`; its harm bound was cleared, so this is a qualified failure to show the predeclared benefit rather than evidence of a large harmful effect.

Full-record did beat the diagonal control on token accuracy in all four cells. The full-minus-diagonal token deltas were +0.1876 pp (finance base), +0.1907 pp (finance changed target), +0.6459 pp (pile base), and +0.7320 pp (pile changed target), with all four 95% intervals above zero. Exact-record improvements were less stable in the pile cells. This shows that the full path is functioning as a contextual control, while the primary comparison shows no added value over the past-only path on this panel.

### Position bins

The following are descriptive mean token deltas for full-record minus past-only across the two seeds. Bins are positions 1–15, 16–39, 40–79, and 80–127; they are not separately bootstrapped gate criteria.

| Domain | Target | Early | Early-middle | Late-middle | Near-end |
|---|---|---:|---:|---:|---:|
| finance | `public_base` | 0.0000 | −0.0570 | −0.0635 | 0.0000 |
| finance | `public_lora_2601` | 0.0000 | −0.0163 | −0.0098 | −0.0366 |
| pile | `public_base` | −0.0911 | −0.0244 | +0.0244 | −0.1506 |
| pile | `public_lora_2601` | −0.1953 | −0.1058 | −0.0732 | −0.1180 |

There is no consistent late-position recovery. The natural Pile panel shows the largest full-record losses, including the near-end bin; the changed target does not reverse that pattern.

## A1+A2 anchor and cost

The anchor is a benchmark-compatible CPU-embedding port of the retained A1+A2 K=256 decision rule. It uses the first 64 public-base records per domain and has a separate denominator from the student matrix. Finance scored 8,119/8,128 tokens and 61/64 exact records (99.8893% and 95.3125%); Pile scored 8,104/8,128 and 58/64 (99.7047% and 90.625%). The port must not be described as a native published-parent rerun.

The student prediction receipts report batch-8 throughput-derived measured costs of approximately 3.95–4.03 ms per record, with maximum CUDA reserved memory 2,661,285,888 bytes and process RSS 3,606,114,304 bytes. These are batch-throughput figures, not single-record latency. The anchor measured 53.9515 seconds for Pile and 53.9755 seconds for Finance, about 843 ms per record. The measured portion executed 2,080,768 candidate simulations per 64-record domain, or 4,161,536 across both domains; including warmup, the total was 4,161,536 per domain, or 8,323,072 across both domains. Maximum CUDA reserved memory was 5,385,486,336 bytes. The six public fits summed to 551.192 seconds; these fit costs are separate from prediction and anchor costs.

The retained capacity replay repeated 900 optimizer updates across the three probe arms, with 28.103 seconds summed arm wall time and a passing 35.917-second watchdog. It made no additional fit choice or truth/source access. Together with 18,000 main-fit updates, 900 original-probe updates, and the 2-update qualifier, the recorded total is 19,802 optimizer updates. The detailed timing and replay accounting are in `experiments/TRR-P06/runtime/report-cost-summary.json` and `experiments/TRR-P06/runtime/capacity-retention-replay-r1/retention_equivalence.json`.

The public capture child returned successfully and all four inner forward qualifications passed, but its outer watchdog was retained as a `FAIL_CLOSED` post-exit `/proc` race. The report makes no clean-watchdog claim for that capture. This exception does not alter the frozen observation hashes accepted before scoring.

## Decision and limits

Later-position observations did not provide a useful promotion signal for this fixed H128 full-record variant: on the registered public-base comparison, full-record was slightly below past-only in both domains, and the paired intervals excluded the predeclared practical benefit. The changed-target results were supporting transfer evidence only and were not used to select a method or trigger the gate.

The next decision is to stop promotion of `p06_full_record` and retain `p06_past_only` as the task-local arm for this family; do not spend another run collecting the same later-position observations under the same mask and objective. The result does not establish that every architecture or objective should ignore later positions. A future attempt would need a new, predeclared mechanism or hypothesis that explains why later observations should add information beyond the past-only path, rather than a repeat of this static full-record variant.

This remains an exploratory P06 natural-panel result. It does not replace the canonical dual-benchmark matrix, establish an overall best reconstruction method, or authorize an active-registry update. The complete public-fit and resource history remains in `experiments/TRR-P06/review/training-summary.json`; the frozen design and gates remain in `experiments/TRR-P06/plan.json`.
