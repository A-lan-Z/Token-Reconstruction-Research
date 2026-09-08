# TRR-P12 — frozen B1 under actual target training

Status: `IN_PROGRESS_NO_TRANSFER_RESULT`.

This bounded study retains the restored expanded B1 decoder and plans one public prefix-LoRA target trajectory with snapshots at 0, 64, 128 and 256 updates. No transfer outcome is available yet. The governing design is `experiments/TRR-P12/manifest.json`; `plan-amendment-r1.json` expands the Pile candidate range after the initial range yielded only 82 eligible clips, while retaining the planned 128 records per domain and every exclusion.

The inherited exclusion union and all 576 newly opened P11 identities were verified and shared for coordination. Target fitting uses separate public Alpaca sources. Evaluation sources and all target snapshots remain paired. Target weights and source tokens remain evaluator-only; reconstruction receives sanitized observations and the frozen B1 package.

The CPU implementation tests passed, and the preserved four-record public fixture has exact CPU/GPU B1 prediction equality. The GPU qualification took 7.948 seconds, with 1,338,564,096 bytes peak allocated GPU memory, 1,352,663,040 bytes peak reserved, and 4,663,066,624 bytes peak process RSS. Projected features differed by at most 8.94e-8 between CPU and GPU; this is a prediction-equivalence check, not a bitwise feature-equivalence claim or transfer evidence. Receipts are `experiments/TRR-P12/cpu-fixture-test-r2.receipt.json`, `gpu-equivalence-r1.json`, and `gpu-equivalence-execution-r1.json`.

Pending work is source preparation, a short target-training/capture/B1 smoke, the preregistered trajectory, frozen reconstruction and boundary forecasts, post-freeze scoring, and the final evidence package. The later-update predictor uses baseline/early observations only; exact boundary crossings are reported separately as explanatory diagnostics.

P03 remains unopened. Existing PRs remain unmerged. No decoder adaptation, broader target sweep, or paid compute is part of this task. The result will be limited to this trajectory; canonical comparison remains incomplete.
