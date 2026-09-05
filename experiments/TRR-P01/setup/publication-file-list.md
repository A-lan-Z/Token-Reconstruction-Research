# TRR-P01 publication file list

Curation status: complete task-local evidence, pending root commit/push/PR.
Machine-readable newline staging list: experiments/TRR-P01/setup/publication-staging-list.txt
The final paired matrix is scored after verified joint pre-truth freeze. This
list contains no source plaintext or private truth tensor. The authoritative
newline-only staging set is experiments/TRR-P01/setup/publication-staging-list.txt;
it is generated from present task files and excludes the full prototype table,
private truth, evaluator-private directories, and bytecode caches.

## Task records and committed source

coordination/requests/TRR-P01.md
coordination/results/TRR-P01.md
coordination/parallel/TRR-P01.json
experiments/TRR-P01/manifest.json
experiments/TRR-P01/pilot_plan.json
experiments/TRR-P01/setup/resource-preflight.md
experiments/TRR-P01/setup/publication-file-list.md
experiments/TRR-P01/review/design.md
experiments/TRR-P01/review/cpu-test-evidence.md
experiments/TRR-P01/review/final-runner-audit.md
experiments/TRR-P01/review/final-results-audit.md
experiments/TRR-P01/review/qualifier-cpu-20260905.command.txt
experiments/TRR-P01/review/qualifier-cpu-20260905.log
experiments/TRR-P01/runtime/runner-logging-handoff.md
experiments/TRR-P01/runtime/public-model-provenance-20260905.json

Committed executable source at e43a595:
scripts/trr_p01/build_table.py
scripts/trr_p01/check_post_bos.py
scripts/trr_p01/common.py
scripts/trr_p01/evaluate.py
scripts/trr_p01/freeze_score.py
scripts/trr_p01/qualify_methods.py
scripts/trr_p01/reconstruct.py
scripts/trr_p01/select_panel.py
scripts/trr_p01/identity_check.py (provisional provenance helper)
experiments/TRR-P01/runtime/build_joint_freeze_sidecar.py
src/token_reconstruction/trr_p01/__init__.py
src/token_reconstruction/trr_p01/boundary_prototype.py
src/token_reconstruction/trr_p01/historical_comparators.py

## Compact evidence to force-add

Preparation and qualification:
experiments/TRR-P01/runtime/cpu-table-20260905/build_evidence.json
experiments/TRR-P01/runtime/cpu-table-20260905/preflight.json
experiments/TRR-P01/runtime/cpu-table-20260905/qualification.json
experiments/TRR-P01/runtime/cpu-table-20260905/qualification_chosen.safetensors
experiments/TRR-P01/runtime/cpu-table-20260905/qualification_alternate.safetensors
experiments/TRR-P01/runtime/cpu-table-20260905/bos_identity.json
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2/build_evidence.json
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2/preflight.json
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2/qualification.json
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2/qualification_chosen.safetensors
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2/qualification_alternate.safetensors
experiments/TRR-P01/runtime/method-qualification-cpu-20260905/qualification_evidence.json
experiments/TRR-P01/runtime/method-qualification-cpu-20260905/preflight.json
experiments/TRR-P01/runtime/method-qualification-cpu-20260905/method_qualification.safetensors
experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_identity.json
experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_freeze.json
experiments/TRR-P01/runtime/post-bos-verified-20260905/preflight.json
experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_predictions.safetensors
experiments/TRR-P01/runtime/panel-20260905/panel_manifest.json
experiments/TRR-P01/runtime/panel-20260905/selection_evidence.json

Final public observation and freeze provenance:
experiments/TRR-P01/runtime/evaluator-final-20260905/command.txt
experiments/TRR-P01/runtime/evaluator-final-20260905/progress.md
experiments/TRR-P01/runtime/evaluator-final-20260905/source_certification.json
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-000/sanitized_config.json
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-000/observation_index.json
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-000/observations.safetensors
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-001/sanitized_config.json
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-001/observation_index.json
experiments/TRR-P01/runtime/evaluator-final-20260905/public/arm-001/observations.safetensors
experiments/TRR-P01/runtime/joint-freeze-sidecar.json
experiments/TRR-P01/runtime/joint-freeze-validation.json
experiments/TRR-P01/runtime/build_joint_freeze_sidecar.py
experiments/TRR-P01/runtime/joint-freeze-sidecar-command.sh
experiments/TRR-P01/runtime/joint-freeze-sidecar.log
experiments/TRR-P01/runtime/joint-freeze-pre-score-validation-command.sh
experiments/TRR-P01/runtime/joint-freeze-pre-score-validation.log
experiments/TRR-P01/runtime/joint-freeze-validation-pre-score.json

Final arm-000 and arm-001 public reconstruction bundles:
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/reconstructor_evidence.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/finish_receipt.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/freeze_receipt.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/phase_progress.jsonl
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/preflight.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/route.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/predictions.jsonl
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/predictions.safetensors
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000/lookup_diagnostics.safetensors
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000-score.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/reconstructor_evidence.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/finish_receipt.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/freeze_receipt.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/phase_progress.jsonl
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/preflight.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/route.json
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/predictions.jsonl
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/predictions.safetensors
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001/lookup_diagnostics.safetensors
experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001-score.json
experiments/TRR-P01/runtime/post_score_development_exclusion.json

Scores and final handoff identities:
arm-000 score SHA-256: 83c0adee9a247bd0e54c86e6527c6d7e6e6aee7a74583ac9da22b2a8be10d71d
arm-001 score SHA-256: f744984e695f5907739c08d22152218dd843a95c914e6711d4e008eb0f5a1db5
final-results-audit.md SHA-256: cb20ce0fdf63e35c60d91e74f989d6017c7b44eb14bb89b4e36b82cc0c783a74
post_score_development_exclusion.json SHA-256: 3d3eec289d9cfbce8eb228038d13540b0e38faad61c9972cd4374cbb34f0a714
condition_map.json raw local-only SHA-256: 9dead079b61da037b049e2c3f1d22a52fd73ddcb7bf9ad78fba4bb6bffe2a058
The redacted handoff contains all 16 dataset row IDs, styles, source text hashes,
opaque IDs, public arm input/output hashes, score hashes, and arm-to-condition join.

## Ignored logs and compact tensors

Force-add present compact provenance logs when root wants a self-contained payload:
experiments/TRR-P01/runtime/cpu-table-20260905.log
experiments/TRR-P01/runtime/cpu-table-20260905-identity.log
experiments/TRR-P01/runtime/cpu-qualification-20260905-r2.log
experiments/TRR-P01/runtime/evaluator-final-20260905.log
experiments/TRR-P01/runtime/panel-20260905.log
experiments/TRR-P01/runtime/reconstruct-final-arm-000.log
experiments/TRR-P01/runtime/reconstruct-arm-000-20260905/freeze.log
experiments/TRR-P01/review/qualifier-cpu-20260905.log
The 10.5 MB lookup diagnostics per final arm and 21 KB prediction tensors are
compact enough for review; finish receipts bind their hashes. The 11.1 MB
method-qualification tensor is also retained for the largest-cell qualification.

## Keep local

experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors
The BF16 table is 525337024 bytes, SHA-256 51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3.
Keep public model snapshots and the historical lens local; their pinned IDs and
hashes remain in the manifest and report. Keep panel-20260905/private_truth.safetensors
and evaluator-final-20260905/evaluator_private/ local. The raw condition map is
hash-bound by the redacted post-score mapping but is not published.
No source plaintext, target weights, or private truth tensor is included.

## Failed/provisional records retained

experiments/TRR-P01/runtime/reconstruct-final-arm-000-analysis.json
experiments/TRR-P01/runtime/reconstruct-final-arm-000-command.txt
experiments/TRR-P01/runtime/reconstruct-final-arm-000-progress.txt
experiments/TRR-P01/runtime/reconstruct-final-arm-000-reservation.json
experiments/TRR-P01/runtime/reconstruct-final-arm-000.log
experiments/TRR-P01/runtime/reconstruct-final-arm-000/preflight.json
These records document the excluded exit-124 timeout before predictions; the
pre-logging runner emitted no phase telemetry, so the exact stage is unknown. The
pre-commit evaluator attempt remains provisional; neither contributes to scores.

## Root publication checks

Recheck paths and receipt hashes at staging, force-add only compact review files,
leave large/private payloads local, run git diff --cached --check, verify the exact
task/TRR-P01 ref (never shared common-repository HEAD), and confirm global STATE.json
and research/dual_benchmark_registry.json remain unchanged. Publication is pending
root commit, push, and PR.
