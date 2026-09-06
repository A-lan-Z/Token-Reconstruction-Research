# TRR-P03 canonical publication whitelist

This list is the root-owned staging boundary for the final task publication. It keeps evaluator truth, source-selection metadata, large observation/model/construction assets, and unreviewed development payloads out of publication while allowing hash-validated, task-sized final prediction and numeric score artifacts. The frozen preparation receipt remains a metadata pointer; it does not contain source tokens. The final result and exact path inventory now record the completed Stage 1 stop disposition; root owns final staging and state reconciliation.

## Include

- `coordination/requests/TRR-P03.md` — byte-identical incoming control packet.
- `coordination/parallel/TRR-P03.json` — compact task-local coordination state.
- `experiments/TRR-P03/plan.json` — root-frozen plan and method/condition metadata.
- `experiments/TRR-P03/manifest.json` — final structured evidence manifest, when root publishes it.
- `coordination/results/TRR-P03.md` — final human-readable result.
- `experiments/TRR-P03/publication-files.json` — exact curated publication path inventory.
- `experiments/TRR-P03/runtime-status.md` — task-local runtime status and final Stage 1 disposition.
- `experiments/TRR-P03/setup/interface.json` — truth-free preparation/runtime interface.
- `experiments/TRR-P03/setup/findings.md` — concise setup findings and limits.
- `experiments/TRR-P03/setup/preparation-attempts.json` — unscored attempt retention and precise supersession reasons.
- `experiments/TRR-P03/setup/publication-whitelist.md` — this boundary record.
- `experiments/TRR-P03/setup/panel-20260906-frozen/preparation-run-receipt.json` — frozen selector commit, exact command, outer time/RSS, and no-model-use receipt.
- `experiments/TRR-P03/setup/frozen-vs-r5-comparison.json` — token-free certification that the frozen selection, metadata, and separate truth-file hashes match r5.
- `experiments/TRR-P03/setup/resource-watchdog-interface.md` — safe invocation contract and bounded CPU smoke status.
- `experiments/TRR-P03/setup/integration-review.md` — static, no-model source-panel/CLI contract review and issues for root resolution.
- `experiments/TRR-P03/setup/panel-20260906-frozen/observation_index.json` — pre-generation truth-free geometry template; root must publish final hashed per-arm indexes separately if required.
- `experiments/TRR-P03/runtime/projected-preparation-r1/preparation_evidence.json` — compact projected-table preparation identity, timing, resource, and asset-hash metadata; the referenced binary remains excluded.
- `experiments/TRR-P03/runtime/projected-preparation-r1/preparation_finish.json` — compact projected-preparation completion receipt. The referenced projected table is a large inferred lookup dictionary and remains excluded.
- `experiments/TRR-P03/runtime/watchdog/projected-preparation-r1/{command.json,resource_guard.json,resource_samples.jsonl,time.json,finish.json}` — safe allowlisted command, guard, timing, and memory evidence for the completed projected preparation.
- `scripts/trr_p03/prepare_panel.py` — frozen public selector.
- `scripts/trr_p03/prepare_projected.py` — committed projected-table preparation source.
- `scripts/trr_p03/resource_watchdog.py` — fail-closed process-group resource wrapper.
- `experiments/TRR-P03/review/run-plan.md` and `experiments/TRR-P03/review/preflight-audit.md` — reviewed execution and preflight contracts.
- Final reviewed TRR-P03 implementation sources under `scripts/trr_p03/` and `src/token_reconstruction/trr_p03/`, plus focused tests under `tests/`, selected by the root publication commit.
- Task-sized final prediction bundles for each reviewed arm, when present and hash-validated: under `runtime/qualifier-reconstruction-bundle-{a,b}/` and `runtime/stage1-reconstruction-bundle-{a,b}/`, include `predictions.safetensors`, `lookup_diagnostics.safetensors`, `candidate_sets.safetensors` when the anchor is active, and `predictions.jsonl`. These are deterministic prediction artifacts and contain no evaluator truth.
- For each included prediction bundle, include `preflight.json`, `reconstructor_evidence.json`, `phase_progress.jsonl`, `freeze_receipt.json`, and `finish_receipt.json` after root verifies their immutable hashes.
- Include the strict `runtime/stage1-joint-validation/joint_validation_receipt.json`, independent `review/gate.json` and `review/stage1-gate-review.md`, and the reviewed Stage-1 numeric score files `per_record.jsonl`, `paired_statistics.json`, `metrics.json`, `scoring_evidence.json`, and `pre_score_gate.json` under the root-selected score directory.
- Include the final post-score supplement `runtime/stage1-score-supplement-final/{stratified_summary.json,stratified_summary.csv,accuracy_by_bundle.png,supplement_evidence.json}`; it records aggregate length/position/style summaries and contains no source text or truth token values.
- Include safe allowlisted watchdog `command.json`, `resource_guard.json`, `resource_samples.jsonl`, `time.json`, and `finish.json` for completed qualifier/Stage-1 generation and reconstruction runs, or a compact equivalent with exact argv, timing, and memory.

## Exclude

- Every `stage1/private_truth.jsonl`, `stage2_holdout/private_truth.jsonl`, `truth_index.json`, and `evaluator_panel.json` under every preparation attempt.
- `panel_manifest.json`, `selection-audit.json`, and `prior-exclusion-audit.json` from the evaluator setup; these contain source row identities, source hashes, or exact prior exclusion lists. Preserve them task-locally and reference their hashes from the private evidence manifest.
- All holdout source rows and private indexes, including any Stage-2 source/evaluator rows or truth indexes.
- `plan-input.json`, `resource-discovery.json`, `environment.json`, and target-condition maps as raw setup drafts when they expose evaluator paths or condition routing; copy only the required sanitized identities into the final plan/manifest.
- Large evaluator observation payloads under `runtime/qualifier-observations-bundle-{a,b}/` and `runtime/stage1-observations-bundle-{a,b}/`, including their observation tensors and source-side evaluator payloads.
- The projected preparation table `runtime/projected-preparation-r1/projected_prototypes.safetensors` (a large inferred lookup dictionary), all external model snapshots, the raw boundary prototype table, the historical lens, and any other large construction-only asset. Retain immutable identities and hashes in final metadata.
- Private source/truth artifacts and holdout source rows, including `private_truth.jsonl`, `evaluator_panel.json`, `truth_index.json`, `panel_manifest.json`, `selection-audit.json`, and `prior-exclusion-audit.json`.
- Projected-preparation `stdout.txt` and `stderr.txt` remain optional local diagnostics; publication may retain only if root confirms they contain no unrelated data.
- All `panel-20260906-r1` through `panel-20260906-r5` payload directories. Their retention/status/hash pointers belong in `preparation-attempts.json`; no development truth or evaluator panel is publication input.
- `experiments/TRR-P03/setup/watchdog-smoke-*` — utility smoke receipts retained as task-local development evidence; publication may retain only their aggregate status/pointers.
- Initial watchdog environment receipts that captured the full process environment, specifically `setup/watchdog-smoke-positive/command.json` and `setup/watchdog-smoke-limit/command.json`, are excluded; only the sanitized safe-allowlist metadata and aggregate v2 smoke status are retained.

## Retained attempt pointers

`experiments/TRR-P03/setup/preparation-attempts.json` records r1, r2, r3, r4, and r5 as unscored development preparations with exact timestamps, receipt hashes, differences, and dispositions. No failed-r4 receipt exists in the task tree; r4 completed successfully and was superseded. The only canonical selector preparation is `panel-20260906-frozen`, produced from commit `ebb814aa04049140b3a1dc68e59272b3aff48a88` with selector SHA256 `db91435a1d3077b70c5d463e31ae0fee94f2edbc29f149679606ab1f45027afe`.
