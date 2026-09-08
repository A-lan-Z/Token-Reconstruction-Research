# TRR-P11 — primary predictions and transfer capture handoff

Status: `EVALUATION_IN_PROGRESS_PRIMARY_B0_B1_PREDICTIONS_COMPLETE_TRANSFER_SELECTION_COMPLETE_A1_QUALIFICATION_ONGOING`.

TRR-P11 remains the independent successor to the blocked TRR-P10 exact-state
confirmation. The P10 exact states remain unavailable and were not reused.
PR24 remains open, draft, and unmerged against `task/TRR-P10`.

The current phase record is `experiments/TRR-P11/phase-status-r2.json`
(SHA-256 `08a55f8c07d21f38fbd46bea1b03cf96911c701350906bee84261db5955c526f`).
The selector-bound manifest remains the exact 32,973-byte snapshot
`experiments/TRR-P11/manifest.json` (SHA-256
`22e3b51e4835e85ccaf84c52248bb53c02e702c586f6d36116422e0fd5670f3a`). The
later phase metadata is kept in separate receipts so the frozen selector input
is unchanged.

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
source text or token IDs to these records, loaded no model, and opened no
truth.

The actual two-copy restore gate passed before model evaluation. The package
is `trr0012-fixed-pair-f2b133f96f77c128`; the actual receipt, restore manifest,
and execution receipt are bound in the task metadata. The new B0 state is
`current_fixed_replication_1` at selected step 8000 and the new B1 state is
`expanded_fixed_replication_1` at selected step 13000. The old exact-state
confirmation remains blocked and is still not claimed.

The current truth-free public capture used by the prediction pipeline is
`outputs/TRR-P11/private-evaluation/capture/public-r1/capture.json` (SHA-256
`c6f6ba0c63ba758af653c261934147909dc93f03694ef1e51b7e58fc27a8a5e3`) with
observation manifest
`outputs/TRR-P11/private-evaluation/capture/public-r1/observations.json`
(SHA-256 `ee14dd3d39ea061ce0bc64d1fa09ee66563e4233fbb7026909de78ca6ae2c541`)
and panel metadata SHA-256
`50114382a2f6b980b7caa3aa84174642f077a4b2ad332d155d7b5873368afb9d`. It has
four cells and 256 records per domain; truth and P03 remain unopened.

The earlier confirmation-r2 capture is preserved as an unused prior attempt:
its capture, observation, and panel hashes are respectively
`387669ee84828041513cb12c441671da9352f654e0a4b0a43707cd76e6d456ab`,
`a2d3ea127412bd660512591ba1703850e6667bc20ff50f09d225550f149e9320`, and
`f350dccf69247b396c99f3eecba98b93cef57fa891831489ba759f10036c2497`.
Its recorded costs remain retained. Metadata-only comparison found all four
capture tensor-digest entries and all four observation-manifest tensor-digest
entries equal to public-r1; no raw tensors were read for that comparison.

B0 and B1 primary predictions are complete for all four cells. The private
integrity receipt is
`outputs/TRR-P11/private-evaluation/primary/post-run-integrity-r2.json`
(SHA-256 `469c2ae652cc1774fc05a37d6011d108c18bfb34cc61adf6bc5e836235d790aa`)
and reports PASS. The cell receipts remain private; this record reproduces no
source IDs, predictions, or scores. A1 comparator qualification remains
ongoing, so its prediction and score are not asserted here.

Transfer selection is complete and truth-free. The selected transfer receipt
is `outputs/TRR-P11/private-evaluation/transfer-selection-r2/transfer_selection.json`
(SHA-256 `8674ea5c824c5f3bc4aa10c44439422654572af546684567046f4a5ffb717c0f`)
for 32 records per domain. Its opaque panel file is
`outputs/TRR-P11/private-evaluation/transfer-selection-r2/opaque_panel.json`
(file SHA-256 `bee22ad90f4411118ce3388b4c3e8b96956160550c7f1da7ae75a615aecaebd4`,
panel binding SHA-256
`9968368eef8c8c91b220ef824276aceb5044c0b055545e2b230e66e09b64e47f`). Root
has shared the opaque panel with Agent1. The reservation is
`experiments/TRR-P11/transfer/opaque64_reservation_r5.json` (SHA-256
`afedbd2e05370bdb1cb5ac8cc81bc68b284befa607a449644a1cf78d2de3d0e6`), and
the capture preflight is
`experiments/TRR-P11/transfer/capture_preflight_r4.json` (SHA-256
`9dd71ef824314a62565bfd066c2295a73abaaf920f497f67a6b2530fd72400a3`). GPU
capture is still pending; no transfer truth has been opened.

The remaining gates are A1 qualification and prediction freeze, GPU capture of
the selected transfer panel, final prediction-hash freeze, private truth
opening, and coordinated scoring/publication. The statistical contract
remains frozen: paired source-record bootstrap seed 9009 with 10,000 draws,
descriptive central 95% percentile intervals, and the registered
`paired_exact_cp` implementation.
