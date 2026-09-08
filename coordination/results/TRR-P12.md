# TRR-P12 — Frozen B1 under actual target training

Status: `COMPLETE_BOUNDED_TRAJECTORY`.

Frozen B1 retained nearly the same aggregate token accuracy after the declared target adaptation, while individual correct tokens did break. At 256 updates Finance had 21 errors versus 22 at baseline; Pile had 171 versus 170. This is evidence of transfer on one modest prefix-LoRA trajectory, not a formal noninferiority result or broad fine-tuning robustness. B1 was never adapted.

## Target and access contract

The target was meta-llama/Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`. A single 256-update public Alpaca adaptation trained q_proj/v_proj LoRA in layers 0–3 (rank 8, alpha 16; 212,992 trainable parameters), with AdamW, learning rate 5e-4, zero weight decay, gradient clip 1, batch 2, length 128 and seed 6012. The 256 fitting and 64 validation records were disjoint from the reconstruction panel and checked against the shared exclusions. All 146 base parameter tensors stayed unchanged; only the 16 adapter tensors were trainable. Every fixed stage was retained without checkpoint selection.

Validation next-token NLL fell from **3.404798** at baseline to **1.406234 / 1.385361 / 1.376923** at updates 64 / 128 / 256. Effective prefix update L2 norms were **2.254110 / 2.756073 / 4.326658**. This meets the declared narrow meaningful-adaptation criterion: lower held-out NLL and a nonzero effective weight update. It does not establish general task-quality improvement. All three new adapter artifacts were immediately copied to the independent Windows backup volume and hash-verified before advancing.

The restored P11 expanded B1 state was fixed at SHA256 `088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706`; its public normalized readout was `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`. The paired panel contains 128 Pile and 128 Finance clips, each with 127 scored positions after BOS. Source text, rendering, cut 4, masks, positions and numerical execution stayed common across stages. The capture path used the actual P11 B8×192 padded public-prefix forward, retaining 128 BF16 activations. Its public-fixture outputs were independently bitwise equal to full-model hidden_states[4] for both baseline and adapted smoke conditions. The earlier manifest wording “full-target forward” is corrected by the additive capture-profile clarification.

The evaluator alone loaded adapted target weights and source tokens. B1 received sanitized observations and its frozen public package. A1+A2 used the unchanged public reference prefix and its own reconstructed prefix, never the adapted target prefix. All predictions, costs, early forecasts and actual boundary artifacts were frozen before the evaluator opened fresh reconstruction truth. Freeze SHA256: `3436de3394c0113d91303a0271d637371362cbe95aafa8638c0746e5920edf7a`. P03 was never accessed.

## Absolute and paired B1 outcomes

Every row below scores 16,256 tokens in 128 clips. “Broken” means correct at baseline and wrong at that stage; “improved” is the reverse. Clip changes use exact reconstruction of all 127 scored tokens.

| Domain | Updates | Token errors | Token accuracy | Exact clips | Broken / improved tokens | Broken / improved exact clips |
|---|---:|---:|---:|---:|---:|---:|
| Finance | 0 | 22 | 99.8647% | 121/128 | 0 / 0 | 0 / 0 |
| Finance | 64 | 21 | 99.8708% | 123/128 | 1 / 2 | 0 / 2 |
| Finance | 128 | 22 | 99.8647% | 122/128 | 1 / 1 | 0 / 1 |
| Finance | 256 | 21 | 99.8708% | 123/128 | 1 / 2 | 0 / 2 |
| Pile | 0 | 170 | 98.9542% | 78/128 | 0 / 0 | 0 / 0 |
| Pile | 64 | 142 | 99.1265% | 86/128 | 18 / 46 | 3 / 11 |
| Pile | 128 | 159 | 99.0219% | 82/128 | 24 / 35 | 5 / 9 |
| Pile | 256 | 171 | 98.9481% | 81/128 | 35 / 34 | 6 / 9 |

At the final Pile stage, 35 previously correct tokens broke and 34 previous errors improved; 29 additional changes replaced one wrong prediction with another. Thus 98 changed predictions are not 98 newly introduced errors. Six formerly exact Pile clips broke and nine became exact. Finance had one broken token and two improvements, with no formerly exact clip lost.

The source-paired 95% bootstrap intervals below describe stage-minus-baseline token accuracy in percentage points (10,000 draws, seed 9012). They are not an equivalence test; no noninferiority margin was declared.

| Domain | Updates | Accuracy change, pp | 95% paired interval, pp |
|---|---:|---:|---:|
| Finance | 64 | +0.0062 | [-0.0123, +0.0246] |
| Finance | 128 | +0.0000 | [-0.0185, +0.0185] |
| Finance | 256 | +0.0062 | [-0.0123, +0.0308] |
| Pile | 64 | +0.1722 | [+0.0738, +0.2891] |
| Pile | 128 | +0.0677 | [-0.0062, +0.1476] |
| Pile | 256 | -0.0062 | [-0.0923, +0.0861] |

Exact-clip changes use the declared paired-discordance Clopper–Pearson method. The final intervals were −3.23 to +6.20 percentage points for Finance and −7.85 to +12.35 for Pile. Conditional improvement intervals with zero denominators in any bootstrap draw remain `UNKNOWN`; no draws were silently removed. In particular, Finance B1 and both A1+A2 improvement-given-baseline-error intervals are unknown. Full interval evidence is in `experiments/TRR-P12/score-r1.json`.

## Boundary explanation and later-update forecast

All full-vocabulary argmax checks matched the frozen B1 predictions. Baseline-to-stage changes at 64 / 128 / 256 were **3 / 2 / 3** in Finance and **88 / 84 / 98** in Pile. Every changed prediction had a signed score-margin crossing and a matching readout-normalized geometric crossing; no tie transition occurred. This is a retrospective explanation of changed decisions in the decoder’s projected space. An unsigned movement norm alone does not establish crossing, and a runner-up score gap is not a nearest Euclidean boundary distance.

The separate forecast linearly extrapolated the baseline-to-64 score direction to stages 128 and 256 using the fixed zero-margin threshold. Each forecast was written before later-stage features were loaded. Primary scoring uses the second 64 records per domain, conditional on baseline-correct tokens; the forecast never used correctness labels.

| Domain | Forecast stage | True positives | False positives | Missed failures | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Finance | 128 | 1 | 3 | 0 | 25.00% | 100.00% |
| Finance | 256 | 1 | 488 | 0 | 0.20% | 100.00% |
| Pile | 128 | 10 | 27 | 4 | 27.03% | 71.43% |
| Pile | 256 | 20 | 175 | 3 | 10.26% | 86.96% |

The forecast identified many later failures, but it overpredicted them, especially at the longer horizon. Finance final-stage recall rests on just one failure and 488 false alarms. Pile final-stage recall was 20/23 with 175 false alarms. This supports some within-trajectory association, not a reliable general failure predictor. The full-panel descriptive tables remain separate from this primary table. No threshold, predictor or decoder was changed after scoring.

## Limited common A1+A2 control

The native fixed direct-cosine K256 control proposed 512 candidates and executed the first 256, using one warmup plus three measured calls with exact repeated predictions. Both methods were compared on the same first 32 clips per domain (4,064 scored tokens). This auxiliary benchmark-compatible comparison is not the canonical dual-benchmark matrix.

| Domain | Updates | B1 errors / exact clips | A1+A2 errors / exact clips |
|---|---:|---:|---:|
| Finance | 0 | 6 / 30/32 | 3 / 31/32 |
| Finance | 256 | 6 / 30/32 | 36 / 0/32 |
| Pile | 0 | 30 / 19/32 | 6 / 31/32 |
| Pile | 256 | 31 / 20/32 | 8 / 29/32 |

A1+A2 lost 31 formerly exact Finance clips at the final stage (34 broken tokens, one improvement), whereas its Pile loss was two formerly exact clips (four broken tokens, two improvements). B1’s common-subset Finance errors stayed at six. These outcomes show that a high token score can hide a large change in exact-clip success. They do not establish an overall method ranking across benchmarks.

## Where errors occurred

This is a post hoc descriptive breakdown, not a new fitted explanation or confirmatory test. At the final Pile stage, the 35 newly broken B1 tokens were spread across 21 clips; 34 improvements occurred across 20 clips. Broken counts by fixed position bins 1–32 / 33–64 / 65–96 / 97–127 were **6 / 8 / 6 / 15**; final error counts were **36 / 38 / 48 / 49**. The last bin contained 15/35 new failures, but this alone does not establish a causal position effect. Finance's one newly broken B1 token lay in positions 65–96; most final Finance errors were already present at baseline.

For A1+A2 on Finance, every one of the 32 final clips had its first error at position **29**. The final bin error counts were **32 / 0 / 1 / 3**. Compared with its own baseline, 34 tokens broke and one improved; the separate same-stage comparison with B1 is a different contrast. This localized failure explains why exact-clip success fell to zero while token accuracy stayed above 99%. The report emits no token identities or source text. It compares frozen predictions with authorized evaluator truth, with no new adapted-target query or rescue run. Evidence: `experiments/TRR-P12/postfreeze-breakage-r1.json`.

## Execution, failures and reproducibility

Training took 31.94 seconds including the recorded outer execution; capture took 82.92 seconds across eight cells; B1 process walls summed to 129.82 seconds. A1+A2’s retained qualification cell took 158.72 seconds and the remaining three cells took 475.90 seconds. Finance and Pile geometry took 433.13 and 417.13 seconds; aggregate scoring took 9.05 seconds. These scopes differ and must not be treated as equal per-inference costs. Exact commands, per-record timing and stage artifacts are preserved in `experiments/TRR-P12/execution-audit-r1.json` and `experiments/TRR-P12/evidence/runtime-r1/index.json`.

Measured GPU reserved-memory peaks were 3.295 GB for target fitting, 1.353 GB for B1, 3.546 GB for A1+A2 and 1.208 GB for geometry. The public B1 CPU/GPU fixture preserved predictions, not bitwise projected features. Geometry qualification preserved argmax predictions with maximum CPU/GPU score difference 8.39e-5. Production geometry used the exact CUDA FP32 exp(base.s), 71.9100112915039. The maximum score-margin versus geometric-dot-product residual was about 1.34e-4; score and geometric crossing classifications agreed.

Preserved setup failures include the initial Pile range yielding 82 eligible clips rather than 128; the target-source manifest tuple-index error after tensor writing; B1 smoke CUDA initialization and BF16 hashing issues; an A1 policy-wrapper mismatch before a completed cell; and direct-script import failure resolved by module invocation. These did not cause a scientific checkpoint or panel selection based on outcomes. The BF16 failure and A1 failure lack dedicated persisted stderr, and qualification capture lacks an outer watchdog receipt because it finished before the first poll. These evidence gaps are explicitly recorded; internal capture guards passed. The padded qualification fixture and baseline-only API smoke are excluded from transfer evidence.

The authoritative immutable plan remains `experiments/TRR-P12/manifest.json`, with additive source-range and capture-profile amendments. The human result is this file; final structured evidence is `experiments/TRR-P12/final-evidence.json`; reproduction instructions are `coordination/results/TRR-P12-reproduction.md`. Raw task-owned outputs and independent model backups remain hash-bound. Existing PRs remain unmerged, work stays separate from Agent 1, and no paid compute, decoder rescue, P03 access or broader target sweep was used. Canonical comparison remains incomplete.
