# TRR-P04 result and decision record

**Status:** complete bounded exploratory comparison; the specific D ranking variant is stopped.

## Decision

On the predeclared 72-record broader stratified public natural diagnostic, D (GRU plus the fixed weighted non-gold adjacent-pair objective) is below H in both target conditions. The primary paired token estimates are D−H = −0.7407 pp (95% CI [−1.2153, −0.2778]) for `public_base` and −0.6366 pp (95% CI [−1.0880, −0.1968]) for `p04_evaluator_target_update_v1`. D−S is also negative: −0.7523 pp [−1.1806, −0.3472] and −0.8449 pp [−1.1690, −0.5324]. These are clear regressions on this natural-text panel, although the magnitudes are below the 1 pp practical promotion threshold.

S and H do not establish a gain over one another or over the same-data affine reference. Their paired intervals span zero in both conditions. The study therefore stops this D objective and does not promote a teacher/ranking student, run Stage 2, repeat the same compression, or claim that all GRU or teacher methods fail. It retains the affine control and native A1+A2 accuracy anchor for context. P03's static/projected variant remains stopped and unmerged.

## Design and truth boundary

The panel contains 3 styles (`pile_plain`, `finance_chat`, `alpaca_instruction`) × 4 post-BOS lengths (16, 32, 64, 128) × 6 source records = 72 records. Every target/seed student result scores 4,320 post-BOS tokens; the two seeds are 1737 and 2711. The student arms are `affine_same_data`, S (full-vocabulary CE), H (S plus the fixed label-derived hard-confusion term), and D (H plus one weighted non-gold adjacent-pair ranking term). The panel is a broader stratified public natural diagnostic, not a general public-source representativeness claim.

The separate anchor is 12 predeclared length-32 records (4 per style, 384 tokens per condition) and uses the native PR7 A1+A2 procedure. Both target arms, all 16 student cells, and both native anchors were frozen before truth. The joint freeze was created at `2026-09-06T04:13:14Z`, has SHA-256 `3c8ff3ea0e3435ded45b3e2a75b0498fbd2eee9fce39ab409ec5a942b69417d7`, and records `truth_accessed=false`. Scoring completed at `2026-09-06T04:18:15Z` from score artifact SHA-256 `f9a604727eba9cd0f7e7bdb2a2c1e1ed828e15d9c9ae551b375a18c640277853`; prediction files were read before truth and not rewritten, and truth was opened only after the verified freeze.

Primary uncertainty uses 10,000 seeded draws (`20260908`), source-record clusters, 12 style×length strata, and paired seeds within each source record. Target conditions are paired observations rather than independent clusters. The estimates are descriptive task decisions; no pooled universal inference is claimed.

## Full-panel outcomes

Each cell is correct/scored tokens, token accuracy, and exact records. Exact records use the same six-record stratum groups; all denominators are shown.

| target | seed | affine | S | H | D |
| --- | ---: | --- | --- | --- | --- |
| target-update | 1737 | 4123/4320 (95.4398%; 38/72 exact) | 4133/4320 (95.6713%; 38/72 exact) | 4118/4320 (95.3241%; 39/72 exact) | 4095/4320 (94.7917%; 37/72 exact) |
| target-update | 2711 | 4125/4320 (95.4861%; 38/72 exact) | 4118/4320 (95.3241%; 39/72 exact) | 4115/4320 (95.2546%; 38/72 exact) | 4083/4320 (94.5139%; 36/72 exact) |
| public | 1737 | 4107/4320 (95.0694%; 38/72 exact) | 4114/4320 (95.2315%; 37/72 exact) | 4110/4320 (95.1389%; 38/72 exact) | 4080/4320 (94.4444%; 36/72 exact) |
| public | 2711 | 4109/4320 (95.1157%; 38/72 exact) | 4104/4320 (95.0000%; 38/72 exact) | 4107/4320 (95.0694%; 37/72 exact) | 4073/4320 (94.2824%; 35/72 exact) |

The primary paired results use 8,640 tokens and 144 record comparisons per target (72 records paired across both seeds). Gains/losses/ties are token-level paired correctness counts; exact counts are left/right record totals.

| target | comparison | token delta | 95% bootstrap CI | token gains/losses/ties | record gains/losses/ties | exact left/right |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| target-update | D-H | -0.6366 pp | [-1.0880 pp, -0.1968 pp] | 60/115/8465 | 1/5/138 | 73/77 |
| target-update | D-S | -0.8449 pp | [-1.1690 pp, -0.5324 pp] | 49/122/8469 | 1/5/138 | 73/77 |
| target-update | H-S | -0.2083 pp | [-0.4282 pp, +0.0347 pp] | 26/44/8570 | 1/1/142 | 77/77 |
| target-update | S-affine | +0.0347 pp | [-0.2083 pp, +0.2662 pp] | 34/31/8575 | 2/1/141 | 77/76 |
| public | D-H | -0.7407 pp | [-1.2153 pp, -0.2778 pp] | 64/128/8448 | 0/4/140 | 71/75 |
| public | D-S | -0.7523 pp | [-1.1806 pp, -0.3472 pp] | 65/130/8445 | 0/4/140 | 71/75 |
| public | H-S | -0.0116 pp | [-0.2431 pp, +0.2199 pp] | 29/30/8581 | 1/1/142 | 75/75 |
| public | S-affine | +0.0231 pp | [-0.2315 pp, +0.2546 pp] | 31/29/8580 | 0/1/143 | 75/76 |

The 24 six-record cells per target and comparison were also bootstrapped separately (2,000 draws, seed 2711). The following are ranges across cells, not pooled intervals; they show where the aggregate signal is concentrated without replacing the primary analysis.

| target | comparison | point range (pp) | CI lower-end range (pp) | CI upper-end range (pp) | cells excluding 0 |
| --- | --- | --- | --- | --- | ---: |
| target-update | D-H | [-3.125, +0.781] | [-10.417, +0.260] | [-1.042, +2.083] | 5/24 |
| target-update | D-S | [-4.167, +0.651] | [-8.333, +0.000] | [-1.042, +1.953] | 5/24 |
| target-update | H-S | [-1.693, +0.911] | [-3.385, +0.000] | [-0.391, +3.125] | 3/24 |
| target-update | S-affine | [-1.042, +1.562] | [-3.125, +0.651] | [-0.130, +3.646] | 2/24 |
| public | D-H | [-3.125, +0.000] | [-10.417, +0.000] | [-1.042, +2.083] | 4/24 |
| public | D-S | [-4.688, +0.000] | [-9.896, +0.000] | [-1.042, +2.604] | 2/24 |
| public | H-S | [-1.562, +1.042] | [-4.688, +0.000] | [+0.000, +3.125] | 0/24 |
| public | S-affine | [-2.083, +2.083] | [-4.167, +0.521] | [+0.000, +5.208] | 1/24 |

## Style and length diagnostics

The next table reports each style across its four lengths, separately by target and seed. It uses 24 records and 1,440 tokens per row.

| target | seed | style | affine | S | H | D |
| --- | ---: | --- | --- | --- | --- | --- |
| target-update | 1737 | `alpaca_instruction` | 1416/1440 (98.3333%; 18/24 exact) | 1410/1440 (97.9167%; 18/24 exact) | 1417/1440 (98.4028%; 18/24 exact) | 1409/1440 (97.8472%; 17/24 exact) |
| target-update | 1737 | `finance_chat` | 1407/1440 (97.7083%; 16/24 exact) | 1411/1440 (97.9861%; 17/24 exact) | 1407/1440 (97.7083%; 17/24 exact) | 1405/1440 (97.5694%; 16/24 exact) |
| target-update | 1737 | `pile_plain` | 1300/1440 (90.2778%; 4/24 exact) | 1312/1440 (91.1111%; 3/24 exact) | 1294/1440 (89.8611%; 4/24 exact) | 1281/1440 (88.9583%; 4/24 exact) |
| target-update | 2711 | `alpaca_instruction` | 1417/1440 (98.4028%; 18/24 exact) | 1415/1440 (98.2639%; 18/24 exact) | 1418/1440 (98.4722%; 18/24 exact) | 1405/1440 (97.5694%; 17/24 exact) |
| target-update | 2711 | `finance_chat` | 1408/1440 (97.7778%; 16/24 exact) | 1405/1440 (97.5694%; 17/24 exact) | 1405/1440 (97.5694%; 17/24 exact) | 1403/1440 (97.4306%; 16/24 exact) |
| target-update | 2711 | `pile_plain` | 1300/1440 (90.2778%; 4/24 exact) | 1298/1440 (90.1389%; 4/24 exact) | 1292/1440 (89.7222%; 3/24 exact) | 1275/1440 (88.5417%; 3/24 exact) |
| public | 1737 | `alpaca_instruction` | 1412/1440 (98.0556%; 18/24 exact) | 1406/1440 (97.6389%; 18/24 exact) | 1411/1440 (97.9861%; 18/24 exact) | 1404/1440 (97.5000%; 17/24 exact) |
| public | 1737 | `finance_chat` | 1405/1440 (97.5694%; 16/24 exact) | 1408/1440 (97.7778%; 16/24 exact) | 1404/1440 (97.5000%; 16/24 exact) | 1400/1440 (97.2222%; 16/24 exact) |
| public | 1737 | `pile_plain` | 1290/1440 (89.5833%; 4/24 exact) | 1300/1440 (90.2778%; 3/24 exact) | 1295/1440 (89.9306%; 4/24 exact) | 1276/1440 (88.6111%; 3/24 exact) |
| public | 2711 | `alpaca_instruction` | 1413/1440 (98.1250%; 18/24 exact) | 1412/1440 (98.0556%; 18/24 exact) | 1412/1440 (98.0556%; 18/24 exact) | 1405/1440 (97.5694%; 17/24 exact) |
| public | 2711 | `finance_chat` | 1406/1440 (97.6389%; 16/24 exact) | 1403/1440 (97.4306%; 16/24 exact) | 1406/1440 (97.6389%; 16/24 exact) | 1395/1440 (96.8750%; 15/24 exact) |
| public | 2711 | `pile_plain` | 1290/1440 (89.5833%; 4/24 exact) | 1289/1440 (89.5139%; 4/24 exact) | 1289/1440 (89.5139%; 3/24 exact) | 1273/1440 (88.4028%; 3/24 exact) |

For a compact length view, each entry is `D−H / D−S / H−S / S−affine` in percentage points, aggregated over both seeds (12 records per style×length cell; order is 16, 32, 64, 128).

| target | style | 16 | 32 | 64 | 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| target-update | `alpaca_instruction` | +0.000/+0.000/+0.000/+0.000 | +0.000/+0.000/+0.000/+0.000 | +0.000/+0.000/+0.000/+0.000 | -1.367/-0.716/+0.651/-0.521 |
| target-update | `finance_chat` | +0.000/+0.000/+0.000/+0.000 | -0.260/-0.260/+0.000/+0.521 | -1.823/-1.562/+0.260/-0.130 | +0.716/+0.326/-0.391/+0.000 |
| target-update | `pile_plain` | -3.125/-3.646/-0.521/-0.521 | -2.083/-2.604/-0.521/+0.521 | -0.781/-1.823/-1.042/+0.130 | -0.651/-1.497/-0.846/+0.521 |
| public | `alpaca_instruction` | +0.000/+0.000/+0.000/+0.000 | +0.000/+0.000/+0.000/+0.000 | +0.000/+0.000/+0.000/+0.000 | -0.911/-0.586/+0.326/-0.456 |
| public | `finance_chat` | +0.000/+0.000/+0.000/+0.000 | -0.260/-0.260/+0.000/+0.000 | -1.302/-1.302/+0.000/+0.130 | -0.260/-0.326/-0.065/-0.065 |
| public | `pile_plain` | -3.125/-2.604/+0.521/-1.042 | -3.125/-4.167/-1.042/+1.042 | -1.562/-1.042/+0.521/-0.260 | -0.326/-0.716/-0.391/+0.586 |

The negative D signal is largest in the short `pile_plain` cells and remains negative across all target-update `pile_plain` lengths. The `finance_chat` and `alpaca_instruction` cells contain many near-ceiling ties, so their small differences should not be read as evidence for a positive method effect.

## Native A1+A2 anchor

The native anchor has a separate 384-token denominator and is not pooled with the 4,320-token student panel. Student rows use the selected seed checkpoints; the native row has no seed.

| target | method | correct/scored (accuracy; exact) |
| --- | --- | --- |
| target-update | native A1+A2 | 383/384 (99.7396%; 11/12 exact) |
| target-update | affine, 1737 | 374/384 (97.3958%; 8/12 exact) |
| target-update | S, 1737 | 375/384 (97.6562%; 8/12 exact) |
| target-update | H, 1737 | 375/384 (97.6562%; 8/12 exact) |
| target-update | D, 1737 | 371/384 (96.6146%; 9/12 exact) |
| target-update | affine, 2711 | 375/384 (97.6562%; 8/12 exact) |
| target-update | S, 2711 | 373/384 (97.1354%; 8/12 exact) |
| target-update | H, 2711 | 374/384 (97.3958%; 8/12 exact) |
| target-update | D, 2711 | 369/384 (96.0938%; 8/12 exact) |
| public | native A1+A2 | 384/384 (100.0000%; 12/12 exact) |
| public | affine, 1737 | 374/384 (97.3958%; 8/12 exact) |
| public | S, 1737 | 375/384 (97.6562%; 8/12 exact) |
| public | H, 1737 | 375/384 (97.6562%; 8/12 exact) |
| public | D, 1737 | 371/384 (96.6146%; 8/12 exact) |
| public | affine, 2711 | 376/384 (97.9167%; 8/12 exact) |
| public | S, 2711 | 375/384 (97.6562%; 8/12 exact) |
| public | H, 2711 | 375/384 (97.6562%; 8/12 exact) |
| public | D, 2711 | 369/384 (96.0938%; 8/12 exact) |

Native A1+A2 remains very strong: 384/384 tokens and 12/12 records on `public_base`, and 383/384 and 11/12 under the target update. The native procedure uses PR7 published proposal order and first-argmax behavior. The three static student readouts use lowest-token-ID tie handling on exact finite-precision ties; this is a deliberate tie-rule exception and does not imply that native A1+A2 used the student rule.

The subset diagnostic compares native corrections with the same-data affine control and D, separately by target and seed. It is diagnostic only and did not select a model.

| target | seed | native−affine: token Δ; gains/losses; exact gains/losses | native−D: token Δ; gains/losses; exact gains/losses | native-fixed affine errors: tokens/records | D recovers: tokens/records |
| --- | ---: | --- | --- | ---: | ---: |
| target-update | 1737 | +2.3438 pp; +10/-1; exact +4/-1 | +3.1250 pp; +13/-1; exact +3/-1 | 10/4 | 1/1 |
| target-update | 2711 | +2.0833 pp; +9/-1; exact +4/-1 | +3.6458 pp; +15/-1; exact +4/-1 | 9/4 | 0/0 |
| public | 1737 | +2.6042 pp; +10/-0; exact +4/-0 | +3.3854 pp; +13/-0; exact +4/-0 | 10/4 | 0/0 |
| public | 2711 | +2.0833 pp; +8/-0; exact +4/-0 | +3.9062 pp; +15/-0; exact +4/-0 | 8/4 | 0/0 |

D recovers none of the native-fixed affine errors in three of four rows and 1 token/1 record in the remaining target seed. This anchor gap is consistent with retaining the native A1+A2 accuracy anchor; it is not a claim that the native algorithm and student architecture have equal compute or inputs.

## Public teacher and fair training setup

D's teacher is a privileged public-prefix training scorer. It is used only to create a fixed training signal from public correction labels; it is not a deployed selector or a native BOS-only accuracy result. The qualification covered 256 difficult proposer-error positions and 128 uniform audit positions:

| audit subset | positions | affine proposer correct | teacher correct | gold in K=32 / K=512 | fixes / introduced | retained / omitted pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| difficult proposer-error | 256 | 0 | 250 | 250 / 256 | 250 / 0 | 7,626 / 60 |
| uniform audit | 128 | 115 | 125 | 125 / 127 | 10 / 0 | 3,817 / 26 |
| **all** | **384** | **115** | **375** | **375 / 383** | **260 / 0** | **11,443 / 86** |

The audit had finite scores throughout, 9/384 K=32 proposal misses and 1/384 K=512 misses, score scale `sigma_q=0.01452335`, tie tolerance `0.0001452335`, and 86 omitted near-tie pairs. The raw JSON field `a1_accuracy` means the frozen P04 affine proposer here, not historical A1. All three public styles were present, but the error-driven audit was dominated by `pile_plain` (313/384 positions), so it is qualification evidence rather than a fresh-panel balance claim.

Each seed used the same schedule for all four arms: 3,000 updates with six replay and two correction records per update (75/25 by records). Unequal lengths yielded measured token exposure 74.5509% replay and 25.4491% correction. Teacher positions were exposed 9,033 times for seed 1737 and 9,006 times for seed 2711. Validation was performed every 100 updates rather than the proposed 200, uniformly for all arms and seeds; this is a recorded pre-execution grid deviation. D's rank term was active at 20/30 post-update checkpoints; its raw rank reduction ranged 0.2343–1.3170 (median 0.4509), with the rank term averaged over teacher rows rather than all selected positions. No objective sweep or refit was performed.

## Costs and resource evidence

Student full-panel timings are batch-8 throughput-derived per-record values, not single-record latency: affine 2.2176–2.2516 ms/record, S 2.7859–2.8235 ms/record, H 2.8077–2.8561 ms/record, and D 2.7447–2.7878 ms/record. Native A1+A2 measured 205.613 ms/record on `public_base` and 206.517 ms/record on the target arm over 3 repeats × 12 records. The native and student paths have different algorithms and resource footprints, so these figures do not support an amortized speed-quality claim.

The student state is approximately 16.79 MB for affine and 25.98 MB for S/H/D. The shared FP32 embedding table is 1,050,673,488 bytes (1.047 s load plus 0.203 s device transfer). The student runner's peak GPU allocation was not recorded. Its external watchdog attached late and ended after child exit, with fail-closed status; it sampled whole-device use rather than PyTorch reserved memory (maximum sampled use 7,808,745,472 bytes, minimum free 8,945,401,856 bytes, host high-water 5,882,933,248 bytes, maximum temperature 76°C). These are limitations on cost characterization, not reasons to reinterpret the accuracy result.

The eight public fits took 1,350.994774457 seconds in aggregate. The `wall_seconds=0.202251` value in `training_result.json` is finalizer time, not fitting time; a late aggregate `NameError` did not invalidate or rerun the serialized fits. Thirty-four focused validation tests passed, including freeze, prediction, scoring, objective, and report-support paths.

## Historical qualification record

These entries explain engineering deviations and are not fresh performance outcomes:

- Capacity qualifier r1 failed closed in evaluation-mode cuDNN backward; r2 retained an erroneous 256-error requirement on the eight-row probe; r3 demonstrated probe learning but lacked complete resource/source/time evidence; corrected r5 supplied full-pool feasibility, probe improvement, and resource evidence.
- Teacher preflight r1 failed on a helper import; r2 qualified K=32 at active position 191. Teacher qualification r1 failed on an embedding-path invocation, and r2 OOMed while materializing a 45,596×128×256 FP32 logit matrix (about 21.79 GiB) before the cached proposer path was used; r3 then passed the bounded audit above.
- The old teacher OOM was an engineering allocation failure, not a negative teacher or student result. No qualifier failure changed the fresh truth-gated comparison.

## Reproducible evidence and scope

The compact derived tables are in `coordination/results/TRR-P04-derived-tables.json`, SHA-256 `1821be64925e21fa70dc470a7505f3737eee129cdb507e1ff22b65090614cf56`. They contain aggregate counts and intervals only; correctness was computed in memory from frozen predictions and post-gate truth, and no truth rows or token IDs are persisted.

Key inputs are:

- panel `experiments/TRR-P04/setup/public_selection-r2.json`, SHA-256 `05f941e0dbcf29ea3efc47c7bc8abb3a7146a266eeea770f05052bb7728cde6a`;
- freeze `experiments/TRR-P04/runtime/freeze.json`, SHA-256 `3c8ff3ea0e3435ded45b3e2a75b0498fbd2eee9fce39ab409ec5a942b69417d7`;
- score `experiments/TRR-P04/runtime/score/evaluator_score_r1.json`, SHA-256 `f9a604727eba9cd0f7e7bdb2a2c1e1ed828e15d9c9ae551b375a18c640277853`;
- cost table `experiments/TRR-P04/runtime/student-predictions-r3/student_cost_table.json`, SHA-256 `c6fa0e43ffac0e03f08d3088ed80fb0bfea815b88b9ec24910455805824ed8d9`;
- native anchor receipts `experiments/TRR-P04/runtime/native-anchor-public-r2/native_anchor_receipt.json` and `experiments/TRR-P04/runtime/native-anchor-target-r2/native_anchor_receipt.json`;
- teacher receipt `experiments/TRR-P04/runtime/teacher-qualification-r3/teacher_receipt.json`, SHA-256 `8267fe11bf25db4a667bf0b6e21556d9eb6e0ca74549d3808c5e080d408d9505`;
- training result `experiments/TRR-P04/runtime/training-r1/training_result.json`, SHA-256 `2e0f9566c33fa19c1351885d4027cd00b1f22f95c4464aa1d8be4f3a55f23d19`.

This is an objective-specific exploratory decision. The observed D regression supports stopping this fixed ranking construction and preserving simpler controls; it does not support a family-wide negative conclusion or a new canonical benchmark claim.
