# TRR-P06 pending result report

**Status: public fitting complete; fresh-panel capture, prediction freeze, truth scoring, and the registered decision gate are pending.** This skeleton records only the completed public-fit, capacity, qualification, and resource evidence. No fresh-panel truth or prediction payload was opened while preparing it.

## Question and frozen scope

P06 asks whether already-observed later-position activations improve reconstruction beyond a past-only activation-context decoder within the fixed H128 decoder family. The primary estimand is `p06_full_record - p06_past_only`; the positionwise diagonal arm is a competent positionwise control. The public changed target is a paired transfer condition. The study remains an exploratory task-local natural-panel result, not a canonical dual-benchmark replacement, overall-best claim, or active-registry update.

All arms use the same public trained affine direct path from TRR-0004 (`historical_affine_ce_no_vocab_bias`, selected public step 1900; state SHA-256 `09c5b852373d8555b06508a79bb00c94041202702b61b121b35fa2b6f9f64e65`), the same normalized F32 public embedding readout, H128 crop, optimizer, fit records, labels, and validation records. The six planned fits use replicate seeds 6106 and 6107. Within each replicate, all three masks consume the same ordered position schedule.

## Completed preconditions and public fit

The source-only preflight passed at commit `b1bbc75cc9c58f33b96e1e278ad70c41b75a399a`. It bound the 1,200 x 192 x 2,048 source artifact to the H128 crop; positions 0..127 were used and positions 128..191 were ignored before schedule construction, fitting, validation, and metrics. The cropped fit has 112,825 valid post-BOS positions and the common 48-record validation has 2,982.

The disposable two-update full-record backward qualifier passed at commit `925759bfbf57f4167ec6feabb1512fad47bd28d0`. It used the actual 8-record x 128-position full mask, 512 query draws, full-vocabulary readout, and all parameters trainable. Parameters and gradients remained finite, the residual output path became active, and the Q path became active. Its peak receipt reported 2,936,012,800 reserved CUDA bytes, 12,655,263,744 free CUDA bytes, and 5,641,351,168 process RSS bytes. Its state was discarded.

The earlier qualification attempt failed before model computation because the resolved public fit observation file was unavailable (`fit observations must be a regular file: /tmp/trr-p06/outputs/TRR-0005/enriched_fit_cut4.safetensors`). It is retained as an excluded engineering failure; the corrected r2 qualification is the release evidence.

The frozen-direct public-fit-error capacity probe passed at commit `59abe29be6adb044926b80feac21bc4a9e90e048`. It used 256 public-fit affine-error positions (64 in each declared position bin), initial correctness 0/256, seed 6106, 300 updates, 8-record batches, 512 with-replacement query draws per update, and one shared ordered schedule for all three masks. The final corrected counts were `p06_positionwise_diagonal` 256/256, `p06_past_only` 256/256, `p06_full_record` 255/256; each exceeded the 52/256 threshold. Probe states were diagnostic only and were discarded. The receipt retains hashes and aggregate metrics but not initial/after prediction arrays, so exact row-level replay remains pending; no probe state was used to initialize or select a main fit.

The six main fits passed at commit `f1b35756fc535b5e3350e4edd4feff9e46f80321`. Each ran 3,000 updates with 8-record batches and 512 post-BOS draws per update, with validation every 100 updates and earliest-maximum micro token accuracy selected including step 0. The fit geometry was 1,200 x 128 x 2,048; all six fits used the common H128 crop and full-vocabulary public CE only. The runtime receipt records no target truth access, source-token access, guessed-token feedback, candidate simulations, or A2 student fallback.

### Replicate schedules and initialization

- Seed 6106 used schedule SHA-256 `938e392344a701d961a8fd8709a4fd4da478602da198da0bf358aa0a375b280a`, 1,536,000 draws, and 382 repeated draws across two replacement steps; the schedule was shared by all three arms.
- Seed 6107 used schedule SHA-256 `ab61d30cea3519e37d1a47b3c3d59ce97f7b6abf653cf6d772c6c1b67a41e555`, 1,536,000 draws, and no replacement repeats; it was likewise shared by all three arms.
- The three seed-6106 arms share initial state SHA-256 `286033e364f0f21740a6548a6c63e7a35ae01bd0b34b411380b8cb5e5c927551`; the three seed-6107 arms share `62694a2036d0bc04555dff33a60268072eb2bec1d40549a14a008cd6b22435b6`.
- The direct affine state and inherited logit scale were loaded identically into every arm. The diagonal arm has the same nominal parameter count but structurally inactive Q/K gradients; its effective trainable count is reported separately.

### Selected public-validation checkpoints

| Seed | Arm | Selected step | Best micro token accuracy | Style-balanced token accuracy at selected step | Exact records / 48 | Arm wall time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 6106 | `p06_positionwise_diagonal` | 1700 | 0.967807 | 0.955673 | 23 / 48 | 92.202 |
| 6106 | `p06_past_only` | 1700 | 0.962106 | 0.953837 | 7 / 48 | 91.507 |
| 6106 | `p06_full_record` | 1700 | 0.962106 | 0.953837 | 7 / 48 | 91.797 |
| 6107 | `p06_positionwise_diagonal` | 900 | 0.969819 | 0.959748 | 18 / 48 | 91.851 |
| 6107 | `p06_past_only` | 2000 | 0.971496 | 0.960680 | 20 / 48 | 92.172 |
| 6107 | `p06_full_record` | 1500 | 0.970490 | 0.959077 | 21 / 48 | 91.662 |

The six arm wall times sum to 551.192 seconds; measured optimizer update time sums to 520.605 seconds and validation time to 18.316 seconds. The main watchdog completed in 555.367 seconds with no termination action. The child receipts report a maximum process RSS of 5,641,273,344 bytes and maximum CUDA reserved memory of 3,183,476,736 bytes; the watchdog's sampled group-RSS maximum was 4,629,921,792 bytes. These are fit costs and guards, not fresh-panel prediction latency.

Learning-curve receipts are retained per arm under `experiments/TRR-P06/runtime/main-r1/seed-*/p06_*/learning_curve.json`; the compact hashes and selected/step-zero/final summaries are in `experiments/TRR-P06/review/training-summary.json`. The capacity curves and their hashes are recorded there as well.

## Native and benchmark limits

The planned A1+A2 K=256 quality anchor is a benchmark-compatible CPU embedding port of the published parent decision rule. Its exact adaptations and separate per-domain denominator must be reported with the eventual anchor results; it must not be described as a native rerun. The P06 visibility arms are new task-local fits and are not a replacement for the canonical dual-benchmark matrix.

## Pending evidence and decision

The following remain deliberately pending: frozen natural-panel source and observation receipts, all student and anchor prediction artifacts, the create-only joint-freeze receipt, truth manifest with source-order and final-sequence binding, post-freeze metrics, paired source-record bootstrap, position-bin diagnostics, and the registered public-base Full-Past gate. The final report must keep public-base and changed-target cells separate, average the two fit replicates within each source record before bootstrap, and retain the anchor’s separate denominator.

Until those receipts exist, the scientific outcome is **PENDING**. No statement about later-activation benefit, harm, qualified negative, transfer, or inconclusive scoring is made from the public-fit learning curves or capacity probe.

Receipt inventory and hashes are in `experiments/TRR-P06/review/training-summary.json`. The complete planned scope remains in `experiments/TRR-P06/plan.json` and the design rationale in `experiments/TRR-P06/review/design-review.md`.
