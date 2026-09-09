# Agent 4 — Prefix-only inversion pilot

1. **Can reconstruction remove the fitted token guesser?** It can recover tokens, but the frozen pilot recovered 80/90 post-BOS tokens and 1/6 complete matched-model clips. No A1/B0/B1 or external language model entered the primary solver.
2. **Matching or recovered weights?** The original CPU results use matching rounded-RoPE ports; a corrected GPU subset is reported separately below. Static Vikhr mismatch recovered 29/30 tokens. **No legitimately recovered prefix was available or tested.** Static mismatch is not a recovery result.
3. **Is cost competitive?** Matched CPU reconstruction took 79.82× the native-policy A1+A2 comparator on the same inputs. These are two-thread CPU measurements. A later corrected GPU subset took 5.82× the comparator wall. The quality/cost target was >=95% tokens, >=50% complete clips, and <=2× comparator wall.
4. **Was cold-start tracking demonstrated?** No. This is independent token search using a supplied public prefix, with no model-prefix recovery updates and no tracking sequence. Historical A1 fitting is a comparator preparation cost not remeasured here.
5. **Recommendation:** **Stop the tested bounded variant.** One justified optimizer correction was already tested. These results do not show information loss or rule out other inversion algorithms.

## Method and scope

Started from `5bbc3bf42a81c814404cf84cb46d55f0d3418667` on a separate task branch.
Read and pinned [SipIt v4](https://arxiv.org/abs/2510.15511v4) and its [official implementation](https://github.com/giorgosnikolaou/SIPIT/tree/820683156b7257313046a4fb3c492e52519525b7).
Our budget of 128 trials does not inherit the paper's vocabulary-spanning recovery guarantee.
The initial SGD/snap procedure failed a one-token public CPU diagnostic after 128 checks.
The one corrective variant uses Adam (learning rate 0.01), a float32 provisional embedding cast to BF16 at the input, raw Euclidean vocabulary projection, no snaps/restarts, and at most 128 checks/gradient steps per token. It found that diagnostic token in 17 checks. Model weights stay frozen; only BOS and its own earlier commitments are known. It uses only the current observed activation, commits the best checked MSE candidate on exhaustion, and preserves timed-out suffixes as abstentions.

The source model is public Llama-3.2-1B-Instruct revision `9213176726f574b556790deb65791e0c5aa438b6`; boundary is after blocks 0–3. Forward input embeddings retain their original magnitudes. Full own-prefix recomputation avoids candidate-cache mutation but adds repeated computation. A1+A2 uses its native cached candidate helper and stable top 512 proposal, first 256 candidates, direct cosine and own prior commitments. Its CPU geometry port passed ordered-candidate and prediction equivalence against native `decode_policy` on a public fixture.

## Original CPU measurements — rounded-RoPE port

Four public-domain ordinary-text clips and two generated identifier clips, each 16 positions including BOS, were selected after settings freeze. They are unused within this task; this is not a canonical or repository-wide fresh benchmark. All 90 post-BOS tokens remain in the matched denominator. BF16 capture and reconstruction ran on AMD Ryzen 9 9950X3D with two CPU threads. During this original phase, the GPU remained leased to Agent 2. The later GPU supplement is separate.

| Method / observations | Tokens | Complete clips | Total reconstruction s | Median / p95 clip s | Candidate checks | Gradients | Vocabulary scans |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefix-only / matched | 80/90 (88.9%) | 1/6 | 1180.69 | 198.92 / 205.49 | 11411 | 11410 | 11410 |
| A1+A2 K256 / matched | 90/90 (100.0%) | 6/6 | 14.79 | 2.46 / 2.56 | 23040 | 0 | 90 |
| Prefix-only / static Vikhr | 29/30 (96.7%) | 1/2 | 438.51 | 219.26 / 220.97 | 3840 | 3840 | 3840 |
| A1+A2 K256 / static Vikhr | 30/30 (100.0%) | 2/2 | 5.16 | 2.58 / 2.59 | 7680 | 0 | 30 |

The two static records are a fixed subset of the six matched records. On that identical subset, the matched primary result is 29/30 tokens and 1/2 clips, versus 30/30 and 2/2 for A1+A2; static accuracy is therefore unchanged for both methods on the common subset. Do not compare 96.7% static with 88.9% full-panel matched as an improvement. Captured boundary drift is measured in `static_comparability.json`.

Tiny empirical p95 values describe this panel only. No records were dropped for exhaustion or timeout.

| Group | Prefix-only matched | A1+A2 matched | Prefix-only static | A1+A2 static |
|---|---:|---:|---:|---:|
| ordinary | 52/60 (86.7%) | 60/60 (100.0%) | 15/15 (100.0%) | 15/15 (100.0%) |
| stress | 28/30 (93.3%) | 30/30 (100.0%) | 14/15 (93.3%) | 15/15 (100.0%) |

## Failure analysis

| Method / record | Correct tokens | First error position | Later wrong tokens | Reconstruction s |
|---|---:|---:|---:|---:|
| prefix matched / final_book11_0 | 15/15 | none | 0 | 181.62 |
| prefix matched / final_book11_1 | 12/15 | 1 | 2 | 192.55 |
| prefix matched / final_book84_0 | 13/15 | 4 | 1 | 206.63 |
| prefix matched / final_book84_1 | 12/15 | 6 | 2 | 197.15 |
| prefix matched / final_stress_0 | 14/15 | 15 | 0 | 200.68 |
| prefix matched / final_stress_1 | 14/15 | 2 | 0 | 202.07 |
| A1+A2 matched / final_book11_0 | 15/15 | none | 0 | 2.41 |
| A1+A2 matched / final_book11_1 | 15/15 | none | 0 | 2.39 |
| A1+A2 matched / final_book84_0 | 15/15 | none | 0 | 2.58 |
| A1+A2 matched / final_book84_1 | 15/15 | none | 0 | 2.47 |
| A1+A2 matched / final_stress_0 | 15/15 | none | 0 | 2.46 |
| A1+A2 matched / final_stress_1 | 15/15 | none | 0 | 2.47 |
| prefix static / final_book11_0 | 15/15 | none | 0 | 221.16 |
| prefix static / final_stress_0 | 14/15 | 15 | 0 | 217.35 |
| A1+A2 static / final_book11_0 | 15/15 | none | 0 | 2.59 |
| A1+A2 static / final_stress_0 | 15/15 | none | 0 | 2.58 |

First-error positions are zero-based with BOS at 0. Later errors are reported after wrong commitments; that ordering alone does not establish their individual causes.

On matched, the correct token was tried at 80/90 positions; 0 final selections were wrong despite discovery. Budget exhaustion occurred at 89 positions, token timeout at 0, and suffix abstention at 0. Correct/wrong residual summaries are `{"correct": {"max": 0.002719162032008171, "median": 1.9846846953441855e-05, "min": 0.0, "n": 80}, "wrong": {"max": 0.0069391969591379166, "median": 0.004366067470982671, "min": 0.001683465437963605, "n": 10}}`. Residual magnitude is not a correctness certificate.

On static mismatch, the correct token was tried at 29/30 positions; 0 final selections were wrong despite discovery. Budget exhaustion occurred at 30 positions, token timeout at 0, and suffix abstention at 0. Correct/wrong residual summaries are `{"correct": {"max": 0.000300707237329334, "median": 0.00014792785805184394, "min": 0.00010229412146145478, "n": 29}, "wrong": {"max": 0.007497979328036308, "median": 0.007497979328036308, "min": 0.007497979328036308, "n": 1}}`. Residual magnitude is not a correctness certificate.

## Numerical checks, resources and costs

The early synthetic end-to-end test recovered its sequence and verified nonzero gradients, raw continuous/discrete equality and cache isolation. Actual public-prefix CPU diagnostics passed raw-input equality and finite nonzero gradients in FP32 and BF16. Genuine-match residuals, including sequential versus full and cached execution, are retained in `cpu_public_diagnostic.json` and `cpu_cache_qualification.json`; a high-precision diagnostic is not a claim of FP32 reconstruction performance.

The 16-position backward and full-vocabulary scoring qualification measured 2,939,879,424 bytes peak CPU RSS with a 10 GiB cap and >=8 GiB required available host memory. Each final phase has a fail-closed process-group watchdog; primary lifetime kernel VmHWM is separately recorded. Exact commands, start/end times, peak memory and input/output hashes are in the manifest and guard receipts. Vocabulary setup, backward passes, candidate checking and prefix/cache construction are included in reconstruction wall. Per-process wall includes loading and prediction JSON I/O. No prefix-maintenance computation was performed. One-off backup/restore costs and the old comparator-fitting provenance must not be confused with warmed reconstruction.

Actual public prefix, tokenizer, comparator lens and Python dependency archives were backed up, not merely hashed. Independent prefix copies reproduce outputs exactly, and restored dependency imports passed after adding NumPy's sibling shared-library directory. A complete required-distribution archive was additionally restored with Python `-S`, without global site-packages, and reproduced an actual prefix output; see `required_dependency_restore.json`. External interpreter/stdlib, system libraries and OS driver remain runtime requirements. Large assets remain in this task's `outputs` directory; the manifest hashes and locates them.

The stopping rule has a material numerical limitation: 37 of 38 correct decisions with an entirely correct preceding prefix exhausted the budget. Their genuine residuals reached MSE 1.31e-6 because BF16 execution changed with sequence length; elementwise allclose(atol=rtol=1e-5) was too strict. At the five first-error positions, evaluator-only genuine-token checks had MSE 0 to 8.28e-7, while the chosen wrong tokens had MSE 0.0033 to 0.00694. Those genuine tokens had never been tried. Thus both an overly strict verification gate and genuine bounded candidate-discovery failures are present. The observed 80× CPU gap describes this implementation; it is not a lower bound on a corrected solver. Any follow-up should first qualify sequence-length-consistent verification or a numerical tolerance on separate public development material, then use new confirmation inputs. No threshold was tuned on these opened answers.

The common warm-up was one public forward pass. Backward/optimizer initialization was not independently warmed and remains charged to the first primary record. Accordingly, these are post-load reconstruction measurements, not a controlled steady-state throughput benchmark. Retained historical lens fitting cost is unavailable as a directly remeasured preparation phase.


| Method / observations | Load/setup s | Whole process s | Peak host RSS GiB |
|---|---:|---:|---:|
| Prefix-only / matched | 0.06 | 1181.05 | 3.769 |
| A1+A2 / matched | 0.64 | 15.56 | 3.287 |
| Prefix-only / static | 0.05 | 438.73 | 3.762 |
| A1+A2 / static | 0.59 | 5.85 | 3.286 |

Whole-process wall includes loading and JSON I/O. GPU allocations were zero for these original CPU phases; measured GPU peaks for the supplement appear below. No online prefix-maintenance cost was incurred.

## Corrected GPU supplement — retrospective

After Agent 2 released the GPU, a public synthetic loader check found that the original whole-module BF16 conversion had also rounded the nonpersistent rotary `inv_freq` buffer. Learned weights were exact, but the 16-token output differed from an independently loaded public checkpoint by MSE 8.4363e-7. **The CPU tables above are matching rounded-RoPE ports, not exact native public-prefix executions.** The static comparison also includes this source-loader deviation. Original observations, traces and results are preserved.

Commit `ae5275561a6dd74c7e07ecbe93f26648c887c693` recreates rotary frequencies in native FP32 after moving learned parameters. The corrected CPU fixture exactly matches the independent public checkpoint (zero boundary MSE); GPU full discrete/continuous execution and independent asset restore checks pass. These are bounded fixture checks, not proof of equivalence for every input. The Adam search, random seed, 128-step budget and allclose tolerances are unchanged.

The GPU component check reuses the first ordinary and first stress clips, the same pair already used for static mismatch. Their truth had been opened in the CPU phase, so this is **retrospective**, despite freezing both new prediction sets before the new scorer ran. Both GPU methods use identical newly captured observations. Hardware is one RTX 5080 with 16 GiB VRAM.

| Method | Tokens | Complete clips | Reconstruction s | Median / p95 clip s | Checks | Gradients / scans | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefix-only Adam | 29/30 | 1/2 | 9.824 | 4.912 / 6.453 | 1028 | 1001 / 1001 | 2.928 |
| A1+A2 K256 | 30/30 | 2/2 | 1.688 | 0.844 / 0.927 | 7680 | 0 / 30 | 2.932 |

The ordinary clip is 15/15; stress is 14/15 for the primary method. It discovers 29/30 genuine tokens and never selects incorrectly after discovery. Three tokens exhaust the budget, versus 29/30 on the original CPU matching subset; there are no timeouts or abstentions. Quality meets the predeclared thresholds on this two-clip subset, but the 5.82× runtime ratio misses the <=2× cost target. This is too small and already opened to establish a general performance claim. The GPU/backend and loader both changed, so these measurements do not isolate a hardware speedup or the effect of rotary precision.

Primary/comparator load times are 0.415/0.441s and whole-process times are 10.657/2.559s. Peak allocated GPU memory is 2.917/1.994 GiB. Guards passed with a 6 GiB reserved-memory cap, >=2 GiB free GPU margin and >=8 GiB available host memory. Prior 128-position FP32/BF16 forward/backward qualification passed with <=2.01 GiB reserved; the native comparator fixture passed ordered-candidate and prediction equivalence. The GPU was explicitly released at 16:31:34 UTC on September 8, 2026. Exact commands, timings, peaks and hashes are in `gpu_supplement_summary.json`, prediction freezes and watchdog receipts.

The recommendation remains to stop this tested bounded variant. The GPU result narrows the CPU-only limitation but establishes neither recovered-prefix effectiveness nor cold-start tracking, and does not repair the absent canonical matrix.

## Stages 2 and 3 limitations

A two-record static fallback uses the existing [Vikhr descendant](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct/tree/7fa9d06a59246629244cdd3b6b92e4fc756baa0f), cast from its published fp16 weights to BF16 for capture. Tokenizer vocabulary identity and exact target restore were checked by the evaluator. Only the evaluator loads these target weights. Both reconstruction methods use the public Llama learned weights with the legacy rounded-RoPE loader. No target adapter was supplied as a recovery model, and no target was newly trained. The target is a public checkpoint intentionally withheld from the reconstruction code path; it is not cryptographically inaccessible to the shared Unix account.

No validated online model-prefix recovery implementation/state was located in the pinned resources. Agent 2 was asked for qualified assets and provenance; no such recovery state was integrated. Consequently, the required public-versus-recovered comparison and own-history closed-loop Stage3 were not run. No cold-start tracking claim is made.

## Evidence, deviations and handoff

Read `experiments/agent4-prefix-only-inversion/manifest.json` for exact evidence paths and hashes; replay commands are in `experiments/agent4-prefix-only-inversion/REPRODUCE.md`. The request was saved verbatim at `coordination/requests/agent4-prefix-only-inversion.md`. Access/cost interpretation is in `ACCESS_AND_COST.md`.

Predictions and traces are create-only and frozen before scoring. Capture, prediction and scoring use separate code paths/processes on one account; missing bwrap prevented OS-level isolation. This is explicitly not a sealed canonical benchmark. The single-token public diagnostics capture and score in one process after writing predictions and are development only.

Retained failures: initial missing-bwrap read attempts; dependency restore missing numpy.libs, then corrected; the stronger isolated restore exposed missing idna in OS package dependency metadata, repaired with an actual-module supplement; test import path failure, then 21 reused tests passed; initial SGD search failure; and a prematurely launched final prediction process terminated before scoring because its explicit 16-position qualification was not yet recorded. The aborted attempt is excluded as an orchestration error; the identical six-record panel was rerun after qualification with no setting changes or dropped records. Its approximately one-minute elapsed cost was not separately instrumented. Four new bounded-search/timeout invariant tests passed. The initial tiny smoke/asset-copy checks occurred before their first implementation commit; later scientific runs record exact full commits in their receipts.

The canonical dual-benchmark matrix is **NOT RUN / COMPARISON INCOMPLETE**. Method registration is task-local under the packet's ban on global-registry changes. No P03 holdout data was accessed, no PR was merged, and no other workspace was modified.

## Publication status

Published as [draft PR #29](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/29) on `task/agent4-prefix-only-inversion`, targeting `task/TRR-0012`. The PR remains unmerged. Research commit: `76c30972aec91aa4aa9c22b2b68135e033fc3642`.

The earlier two automatic approval rejections are preserved in `publication_status.json`. The user explicitly approved public publication on September 9, 2026; the branch push and PR creation then succeeded. WSL had no configured Git credential helper, so publication used the existing Windows Git credential manager without exposing credentials. Large model assets and evaluator truth remain local.
