# TRR-P03 result — Stage 1 gate outcome

**Disposition:** `STOP_VARIANT`  
**Stage 2:** `NOT_RUN_GATE_FAILED`  
**Status:** `STAGE1_COMPLETE_STOP_VARIANT_STAGE2_NOT_RUN_GATE_FAILED`

The static projected readout failed the predeclared Stage 1 natural-text panel gate on the frozen 24-record natural panel in both target arms. Stage 2 was not run, its holdout truth remains unopened, and no compactness or rescue claim is made. This is a bounded exploratory result for this fitted-origin projected variant; it is not a universal claim about all projections, no-fit methods, or compression mechanisms. The structured evidence manifest is `experiments/TRR-P03/manifest.json`.

## Disposition

Both full target bundles passed generation, reconstruction, prediction freeze, strict joint validation, and guarded Stage 1 scoring. The gate failed on all five predeclared checks: matched projected minus A1 accuracy was below the `+1.0` percentage-point threshold, its paired confidence interval was wholly below zero, shifted projected minus A1 accuracy was negative, and projected exact-record counts were lower in both arms. The independent machine-readable audit is `experiments/TRR-P03/review/gate.json`; the narrative review is `experiments/TRR-P03/review/stage1-gate-review.md`.

## Frozen panel and comparison scope

The canonical natural source is `HuggingFaceH4/no_robots` train, revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, cached Arrow SHA-256 `5a9193e927d899d167fd40553d0b403499f5f9cf9a9254db19399a4d0b3550fb`. Prompts are used as supplied: prepend BOS `128000` and crop the first requested post-BOS tokens. No instruction or Unicode wrapper is added.

Stage 1 contains 24 opaque records: six at each post-BOS length 16, 39, 64, and 128, with two coding, two question-answer, and two creative-generation records per length. Each target condition therefore has 1,482 scored positions and 792 distinct scored truth token IDs. The separately sealed Stage-2 holdout contains another 24 disjoint records with the same geometry and quotas. The predeclared A1+A2 anchor is `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and `p03-s1-r0012`, the length-39 stratum at zero-based indices `[0,2,4,5]`, retaining exact 40-slot geometry.

The required paired conditions are `matched_public` using `meta-llama/Llama-3.2-1B-Instruct` revision `9213176726f574b556790deb65791e0c5aa438b6` and `shifted_full_sft` using `Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct` revision `7fa9d06a59246629244cdd3b6b92e4fc756baa0f`. The latter is a full public SFT resource; P01’s historical condition label `shifted_target_lora` is retained only as provenance.

Stage-1 readouts are raw boundary full vocabulary, projected frozen-lens full vocabulary, historical native A1 full vocabulary, and the fixed A1+A2 anchor. Reconstruction receives only opaque IDs, activations, masks, positions, cut depth, public asset identities, and fixed method configuration. It receives no teacher-prefix input, source text, source token IDs, dataset indices, source hashes, style labels, target labels, correctness signal, or opened prior results.

## Completed preparation and runtime evidence

The frozen panel was generated from selector commit `ebb814aa04049140b3a1dc68e59272b3aff48a88`, selector SHA-256 `db91435a1d3077b70c5d463e31ae0fee94f2edbc29f149679606ab1f45027afe`, with the exact offline command recorded in its immutable receipt. It ran from `2026-09-05T14:40:57.093488150Z` to `2026-09-05T14:41:06.012840223Z`, elapsed `8.91 s`, maximum RSS `1,213,016 KiB`, exit `0`, with no model load, full-table load, forward pass, GPU use, or truth opening.

The reusable projected table was prepared once from committed source `23933e65868dcdfa58a44ce47b87a8f5a7455c51` under the guarded CPU window. The output is `experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors`, 1,050,673,728 bytes, SHA-256 `8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7`. Internal preparation timing was table/lens load `0.243698 s`, projection `2.170561 s`, and total `6.317762 s`; process maximum RSS was `4,026,428 KiB`. The watchdog elapsed `8.61958 s`, returned `0`, observed peak group RSS `3,335,843,840` bytes, minimum sampled host availability `20,452,384,768` bytes, and passed the 8 GiB RSS / 10 GiB host-availability limits. CUDA allocation was false and truth remained unopened.

The full Stage 1 matrix used source commit `6edb276a3a536988a1d2cc9f3aa4c29e90e1a6b1`, CPU-only deterministic Torch settings, eight intra-op and one inter-op thread, batch size 4 for observation generation, and query/prototype chunks 256/8192 for reconstruction. Both observation bundles, both reconstruction roots, and scoring passed their watchdogs. Internal reconstruction peak RSS was 7,818,420 KiB for matched and 7,817,944 KiB for shifted; the larger value is approximately 7.456 GiB. The strict joint receipt reports `VALIDATED` / `STAGE1_JOINT_VALIDATION_PASS` before truth was opened. The authorized scorer opened Stage 1 truth after both prediction roots were frozen and hash-validated; the final report-only supplement later read that already-scored Stage 1 truth for diversity and style summaries. All Stage 1 truth accesses were post-freeze.

## Asset and storage accounting

| Asset or workspace | Identity / size | Publication treatment |
| --- | --- | --- |
| Base model | Llama 3.2 1B Instruct, revision `9213176726f574b556790deb65791e0c5aa438b6`; run-plan estimate about 2.47 GB for decoder weights | Identity retained; snapshot excluded |
| Raw boundary table | `525,337,024` bytes on disk; BF16 tensor payload `525,336,576` bytes; SHA-256 `51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3` | Large construction/readout asset excluded |
| Historical affine lens | `16,787,653` bytes; SHA-256 `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742` | Large/fitted-origin asset excluded |
| Projected candidate table | `1,050,673,728` bytes on disk; float32 payload `1,050,673,152` bytes; SHA-256 `8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7` | Large inference lookup dictionary; binary excluded, identity retained |
| Full lookup scratch | `256 * 128256 * 4 = 131,334,144` bytes at query chunk 256 | Account in runtime receipts |
| A1+A2 anchor candidates | Native `4 × 40 × 512` int32 proposal payload is `327,680` bytes; only 256 proposals are simulated per anchor window, with a native `4 × 40 × 256` float32 selection-score payload of `163,840` bytes. Persisted full-matrix candidate tensors are padded `24 × 129 × 512` int32 (`6,340,608` byte payload; `6,341,272` bytes on disk) | Task-sized final artifacts included after freeze review |

The projected and historical readouts retain the historical public Alpaca fitting provenance. This task performed no fitting and does not support a fitting-free claim.

## Overlap, provenance, and audit limits

The selector rejects exact P02 `(full preceding prefix, endpoint position, endpoint ID)` tuples at each declared truncation and skips the whole candidate record. The audit contains 51 P02 keys and 51 unique sequences; no candidate in the frozen output triggered a prefix-collision rejection, 90 candidates were rejected for being too short, and no token ID was globally banned. The audit found 464 previously opened records and 368 unique prior text hashes; selected text-hash overlap was zero. Two accessible blind commitments expose neither source identity nor recoverable source hashes, so no non-overlap claim is made against those hidden rows. P01/P02 and other previously opened rows remain separate from fresh natural-panel claims.

The earlier P02 diagnostic covered three endpoint IDs across four short contexts, whereas this natural Stage 1 panel contains 792 distinct scored truth token IDs. The projected readout still gains over raw on this broader panel, but its A1 advantage vanishes in both target arms. These comparisons do not identify the mechanism behind the discrepancy.

During setup, a planning message was accidentally routed to the TRR-0004 task. It contained only P03 planning/interface text and no records, source truth, scores, or private data. The receiver reported its methods and records were already frozen and would not use the message. P03 uses no scientific output from that route. An early watchdog development receipt attempted to persist the full environment; it was sanitized before publication and the active receipt schema persists only the explicit safe environment allowlist. Frozen raw receipts remain immutable.

The current reviewed run-plan records the full prototype SHA-256 above, the full lens SHA-256 above, and projected-table SHA-256 `8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7`; these agree with the completed preparation and runtime evidence. The global state and active registry remain untouched by this task; the task-local manifest and parallel record carry the current runtime status.

## Stage 1 result

The score files report 1,482 post-BOS positions per target, deterministic top-1 decisions, and zero standalone top-1 tie positions. Native A1 decisions retain finite-precision FP32 scaled logits with `exp(s)=35.0647507` and their native decision units; raw and projected full-vocabulary methods use float32 cosine scores with descending-score then ascending-token-ID tie resolution. The native A1+A2 anchor retains published proposal ordering and first-argmax tie behavior.

| target arm | readout | correct / scored | token accuracy | exact records |
| --- | --- | ---: | ---: | ---: |
| matched public | historical A1 | 1332 / 1482 | 89.879% | 5 / 24 |
| matched public | projected | 928 / 1482 | 62.618% | 0 / 24 |
| matched public | raw boundary | 498 / 1482 | 33.603% | 0 / 24 |
| shifted full SFT | historical A1 | 1333 / 1482 | 89.946% | 5 / 24 |
| shifted full SFT | projected | 922 / 1482 | 62.213% | 0 / 24 |
| shifted full SFT | raw boundary | 479 / 1482 | 32.321% | 0 / 24 |

Relative to historical A1, the matched projected readout changed token correctness at 91 gain positions and 495 regression positions (837 both correct, 59 both wrong), for a net loss of 404 correct tokens and a `-27.2605` percentage-point delta. Its length-stratified paired record-cluster 95% CI is `[-32.7935, -20.1754]` percentage points. The shifted arm had 90 gains and 501 regressions (832 both correct, 59 both wrong), a net loss of 411 correct tokens and a `-27.7328` percentage-point delta, with CI `[-33.0634, -20.9852]` percentage points. Projected record accuracy was lower than A1 on 22 of 24 records in each arm; the macro deltas were `-30.2192` and `-30.5864` percentage points, respectively.

The exact-record delta was `-5` records in each arm (0 projected versus 5 A1), or `-20.8333` percentage points; its descriptive CI was `[-33.3333, -8.3333]` percentage points. The predeclared A1+A2 anchor reconstructed all 156 positions and all four records exactly in both arms. This anchor is an accuracy comparator with 10.526% coverage of the full Stage 1 positions and is not a promotion gate. The projected readout still exceeded the raw boundary readout by 430 matched tokens and 443 shifted tokens, but remained materially below A1.

The paired uncertainty used 10,000 record-cluster bootstrap draws, seed `20260905`, six records sampled within each of the four length strata. The final stratified supplement passed from source commit `090ae0446a30989c058f10ae0240feffadd891db`; it opened Stage 1 truth only after the frozen score outputs and adds the natural-panel diversity and style summaries: matched projected/A1 accuracy was 62.955%/83.401% for coding, 62.348%/93.725% for question-answer, and 62.551%/92.510% for creative-generation; shifted projected/A1 accuracy was 62.753%/82.996%, 61.538%/94.130%, and 62.348%/92.713%, respectively. Each style has eight records and 494 scored positions across the four lengths. Per-record outcomes and correctness vectors are retained in the scored `per_record.jsonl` artifact, while length, position, and style summaries are in `runtime/stage1-score-supplement-final/stratified_summary.{json,csv}`.

## Stage 2 disposition

Stage 2 is `NOT_RUN_GATE_FAILED`. No holdout observations, compact rank-128 or rank-256 decomposition, holdout truth opening, reconstruction, scoring, runtime comparison, or adaptive rescue was performed. The separate 24-record holdout remains sealed and its truth remains unopened.

## Publication boundary and next decision

The next decision is to deprioritize this static projected variant, retain the A1+A2 accuracy anchor as a separately labeled comparator, and make no automatic repeat of the same compression. The publication boundary retains the canonical metadata, reviewed source, frozen prediction and numeric score artifacts, strict validation receipt, independent gate audit, and safe watchdog receipts while excluding evaluator truth, holdout source rows and private indexes, large observation/model/prototype/construction assets, old backups, patches, and initial full-environment receipts. The complete path inventory is `experiments/TRR-P03/publication-files.json`.


Published as [PR #8](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/8) on `task/TRR-P03`, against `task/TRR-P02` at `7956b4357d076abce3ccfc407d3fcac832fd34f6`. The published evidence commit is `3d19786ed1db69a30f7d1673f842e67f52dddbac`; no PR was merged.
