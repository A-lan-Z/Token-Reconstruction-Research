# TRR-P03 runtime status

**Disposition:** `STOP_VARIANT`; **Stage 2:** `NOT_RUN_GATE_FAILED`.

The source is frozen at commit `6edb276a3a536988a1d2cc9f3aa4c29e90e1a6b1`. The full Stage 1 matrix completed with both target observation bundles and both four-method reconstruction roots under CPU-only deterministic settings. The strict joint validator passed before truth opening. The root-authorized scorer opened Stage 1 truth after prediction freeze, and the final report-only supplement later read the same already-scored Stage 1 truth for diversity and style summaries; all accesses were post-freeze. The separate Stage 2 holdout remains unopened.

## Setup and qualification

The canonical panel is `HuggingFaceH4/no_robots` at revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, with six records at each post-BOS length 16, 39, 64, and 128, two each of coding, question-answer, and creative-generation styles per length. Stage 1 has 24 records and 1,482 scored positions per target; the sealed holdout has 24 disjoint records. The A1+A2 anchor is `p03-s1-r0007,p03-s1-r0009,p03-s1-r0011,p03-s1-r0012`.

The selector was frozen at commit `ebb814aa04049140b3a1dc68e59272b3aff48a88` with source SHA-256 `db91435a1d3077b70c5d463e31ae0fee94f2edbc29f149679606ab1f45027afe`. The projected candidate table was prepared once from source commit `23933e65868dcdfa58a44ce47b87a8f5a7455c51`, output SHA-256 `8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7`, with no truth access or GPU use. The projected and historical readouts retain public Alpaca fitted provenance.

The four sequential qualifier commands passed with truth unopened. They qualified both target observation paths and the four reconstruction methods, including the native A1+A2 anchor, before the full matrix.

## Full Stage 1 runtime

- `stage1-generation-a`: PASS, matched public target, 24 records / 1,482 scored positions, watchdog elapsed 9.615 s, peak group RSS 3,399,958,528 B, minimum sampled host availability 22,356,934,656 B.
- `stage1-generation-b`: PASS, shifted full-SFT target, 24 records / 1,482 scored positions, watchdog elapsed 6.606 s, peak group RSS 4,806,803,456 B, minimum sampled host availability 19,824,836,608 B.
- `stage1-reconstruction-a`: PASS, four methods, watchdog elapsed 27.353 s, peak group RSS 7,450,578,944 B; internal peak 7,818,420 KiB.
- `stage1-reconstruction-b`: PASS, four methods, watchdog elapsed 27.372 s, peak group RSS 6,484,697,088 B; internal peak 7,817,944 KiB.
- `stage1-scoring`: PASS, watchdog elapsed 1.528 s, peak group RSS 537,247,744 B.
- Strict joint receipt: `VALIDATED` / `STAGE1_JOINT_VALIDATION_PASS`, SHA-256 `7e4d848fd33fb4218c0ffbfc1c41a5c0806f746d5107f5c02b430118e45b534d`.

All numeric processes used CPU, hid CUDA, enabled deterministic algorithms, used eight intra-op and one inter-op thread, and used the declared batch/query/prototype chunk sizes. All runtime watchdogs passed with no termination action.

## Stage 1 score and gate

The machine-readable gate is `experiments/TRR-P03/review/gate.json`; the narrative is `experiments/TRR-P03/review/stage1-gate-review.md`. Matched projected accuracy was 928/1,482 (62.618%) versus historical A1 1,332/1,482 (89.879%), a `-27.2605` percentage-point delta with 95% CI `[-32.7935,-20.1754]`. Shifted projected accuracy was 922/1,482 (62.213%) versus A1 1,333/1,482 (89.946%), a `-27.7328` percentage-point delta with 95% CI `[-33.0634,-20.9852]`. Projected exact-record counts were 0 versus A1 5 in each arm.

The matched paired token contingency was 91 gains, 495 regressions, 837 both-correct, and 59 both-wrong; the shifted contingency was 90 gains, 501 regressions, 832 both-correct, and 59 both-wrong. The predeclared A1+A2 anchor was 156/156 over four records in both arms and was not a promotion gate. Stage 2 was stopped before any holdout observation or truth opening.

## Evidence and publication

Final task-sized predictions, diagnostics, candidate tensors, score files, gate audit, strict validation, and safe watchdog receipts are enumerated in `experiments/TRR-P03/publication-files.json`. Large model, prototype, projected-table, and observation tensors, evaluator panels, truth files, holdout source rows/private indexes, development payloads, old backups/patches, and initial full-environment receipts remain excluded. The global `coordination/STATE.json` and active method registry were not modified by this task; root owns final publication metadata and commit.



## Development supplement disposition

An earlier Stage-1 supplement attempt is retained as excluded development metadata under `runtime/stage1-score-supplement`; the final report-only supplement below is the canonical stratified result.

## 2026-09-06 final report-only supplement — ALL SCIENCE COMPLETE / STOP_VARIANT

- Final replay source was frozen at `090ae0446a30989c058f10ae0240feffadd891db`; the exact report-only command and environment are recorded in `experiments/TRR-P03/runtime/stage1-score-supplement-final/supplement_evidence.json`. It read the same frozen Stage-1 score/prediction/panel/truth inputs, did no inference or score rerun, and reports `stage2_opened=false`.
- Final artifacts: `experiments/TRR-P03/runtime/stage1-score-supplement-final/stratified_summary.json` (SHA-256 `27bd5cd5175b2c4af35f9876a259ba5f3a22fe1fe8bcbb68d086d27938b1af74`), `stratified_summary.csv` (SHA-256 `355480ddfecb0c3c9ae77d5d7db65956493c697c0b5fb66a5da37f75faf12d913`), `accuracy_by_bundle.png` (SHA-256 `1add2417fbd0e00c972286f45dfef7c20e93f037beb1fc3c116b21825ea1a634`), and `supplement_evidence.json` (SHA-256 `f71d613e6d9e1724dc0d814c9b315a93ae0a1e1c2d3edf02c04862444022870c`). All final output files are read-only.
- Final CSV bytes and all numeric JSON fields match development supplement r1; only the generated timestamp and provenance differ. The final report has 1,482 scored positions per target, 792 distinct Stage-1 truth token IDs, 8 records each for coding/question-answer/creative-generation, and zero top-1 ties for every bundle/method. The A1+A2 anchor remains separate at 4 records/156 tokens and 156/156 correct in both arms.
- Final report source file is `scripts/trr_p03/supplement_scores.py` at the frozen report commit; source SHA-256 `e7a1e63e9855470397e7337403f5eba3faa816a35cec7e0c6afa2d7ad4075c1a`. No further implementation or experiment changes are planned in this task.
