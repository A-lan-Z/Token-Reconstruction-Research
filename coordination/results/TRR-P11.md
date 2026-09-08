# TRR-P11 — released selection and truth-free capture handoff

Status: `PREPARATION_IN_PROGRESS_EVALUATION_CAPTURE_COMPLETE_PREDICTION_PENDING`.

TRR-P11 remains the independent successor to the blocked TRR-P10 exact-state
confirmation. The P10 exact states remain unavailable and were not reused.
PR24 remains open, draft, and unmerged against `task/TRR-P10`.

Current phase status is recorded in `experiments/TRR-P11/phase-status-r1.json` (SHA-256 `0c7648afe0f8f736e6c861123cce11603b92bd6187c838e6ad01a34802a2e84b`). The selector-bound manifest remains the exact 32,973-byte snapshot `experiments/TRR-P11/manifest.json` (SHA-256 `22e3b51e4835e85ccaf84c52248bb53c02e702c586f6d36116422e0fd5670f3a`); later phase metadata is kept in the separate receipt so the frozen selector input is unchanged.

The complete accessible exclusion release is bound by:

- `experiments/TRR-P11/exclusions/recovery_identity_audit_r17.json` — SHA-256
  `43416d0d1945821812278df8dfc3ee7639f872e9c8407abdf245fb281bda9d65`;
- `experiments/TRR-P11/exclusions/identity_union_export_r14.json` — SHA-256
  `125275eab38c66117d45f5e0df4085058dae8a5c939fff1554cab7e608eb1fe1`; and
- `experiments/TRR-P11/exclusions/root_selection_release_r1.json` — SHA-256
  `0e1816711965b0601a2e00ce4c187bded9f0c0fca4493e71ef9478a772659f95`.

The released audit reports complete coverage over the explicit accessible
inventory and zero eligible or unresolved residual rows. P03 remains sealed
and outside that inventory. Historical r1–r16 audit artifacts remain
preserved; r17 is the only current selection release.

The frozen source selector executed with seed 5011 and the registered Pile
`[0,2000)` and Finance `[20000,28000)` ranges. Its private output contains 256
records per domain and remains under ignored task-local storage:

`outputs/TRR-P11/private-evaluation/selection/source_selection.json`
(SHA-256 `fe7129e9d7230100d1900facea98b15b8000f382a0bd7035b2039b4d9429bf68`).
The executed release command and code bindings are recorded in
`experiments/TRR-P11/selector/command-r3.json` (SHA-256
`050ec7276e705fa0197bdc8309a81eea25750aa90358d92b565bf7d53203f45f1`).
Selection read public source text for trusted rerender/tokenization, wrote no
source text or token IDs, loaded no model, used no GPU, and opened no truth.

The actual two-copy restore gate passed before model evaluation. The package
is `trr0012-fixed-pair-f2b133f96f77c128`; the actual receipt,
restore manifest, and execution receipt are bound in the task manifest. The
new B0 state is `current_fixed_replication_1` at selected step 8000 and the
new B1 state is `expanded_fixed_replication_1` at selected step 13000. Both
state identities, tensor identities, secondary Windows copies, and the fixed
four-record public smoke comparison are recorded there. The old exact-state
confirmation remains blocked and is still not claimed.

A truth-free public confirmation capture is complete in private custody:

- `outputs/TRR-P11/private-evaluation/capture/confirmation-r2/capture.json` —
  SHA-256 `387669ee84828041513cb12c441671da9352f654e0a4b0a43707cd76e6d456ab`;
- `outputs/TRR-P11/private-evaluation/capture/confirmation-r2/observations.json` —
  SHA-256 `a2d3ea127412bd660512591ba1703850e6667bc20ff50f09d225550f149e9320`;
- `outputs/TRR-P11/private-evaluation/capture/confirmation-r2/panel.json` —
  SHA-256 `f350dccf69247b396c99f3eecba98b93cef57fa891831489ba759f10036c2497`.

It contains four domain-target cells, 256 records per domain, stored 128-token
sequences, and 127 scored post-BOS positions. It used public base and
`public_lora_2601` inputs, wrote no truth or token IDs, and did not access P03.
The runtime worker owns the subsequent B0/B1 and A1 prediction, prediction
freeze, and scoring stages; no prediction or score is asserted here.

The private primary four-cell receipt set also passed integrity review: `outputs/TRR-P11/private-evaluation/primary/post-run-integrity-r2.json` (SHA-256 `469c2ae652cc1774fc05a37d6011d108c18bfb34cc61adf6bc5e836235d790aa`). The four cell receipts remain private and are represented here only by opaque paths and hashes; no source IDs, predictions, or scores are reproduced.

The old exact-state confirmation remains blocked because the selected historical artifacts are unavailable and were not reused. A1 comparator prediction, prediction freeze, and the separate transfer selection/capture remain pending; truth remains unopened.

The separate 64-record transfer reservation is released for identity-only
selection but remains unselected and uncaptured. The hash-bound reservation is
`experiments/TRR-P11/transfer/opaque64_reservation_r3.json` (SHA-256
`877abff7bd6f6dc2c7a56974902bed0389bd66e3ef84bfbdc4fe8ddaaf8c47fe`) and its
truth-free capture preflight is
`experiments/TRR-P11/transfer/capture_preflight_r3.json` (SHA-256
`efec6f9bb7f8a9d348ba2180b9cf8cccb892a896abd9fa1ea904fd3bb9b0f9d6`). The
transfer uses the frozen 32-per-domain ranges Finance `[28000,30000)` and
Pile `[9000,10000)`, keeps evaluator-owned raw payloads private, and has no
Agent1 handoff yet. Its seven-variant capture command is ready after the
opaque panel/token batch and live resource preflight are supplied; no transfer
GPU job was launched in this phase.

The statistical contract remains frozen: paired source-record bootstrap seed
9009 with 10,000 draws, descriptive central 95% percentile intervals, and the
registered `paired_exact_cp` implementation. No truth was opened and no new
scientific score is available yet.
