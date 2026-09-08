# TRR-P11 — new-weight replication planning handoff

Status: PREPARATION_IN_PROGRESS_EXCLUSIONS_PARTIAL_RESTORE_ASSETS_PENDING.

TRR-P11 is an explicitly named successor to the blocked TRR-P10 exact-state
confirmation. The exact PR20/TRR-0011 fixed-readout states are unavailable
for reuse; this task will test one newly rebuilt B0/B1 fixed-readout pair.
The predecessor records and draft PR22 remain preserved and unmerged.

The amendment is saved byte-for-byte at coordination/requests/TRR-P11.md
with SHA-256 70a2cf13b2795916749031bcfed3a55205cd52a44926c1d160bd83508706d9b6. The persistent successor branch starts at P10
commit cab7d455c305aa01c3d3bfac225fcffa3e66640b.

Agent 1 owns exactly one paired current-bank B0 and expanded-bank B1 fit,
including the common starting decoder, preprocessing, schedule, validation
selection, durable model states, and deployment package. Agent 2 owns the
complete identity/sequence exclusion audit, prospective evaluation plan,
restore gate, and independent scoring handoff. Agent 2 has performed no
fitting, source selection, GPU work, activation capture, prediction, or truth
access. The single Agent 1 paired fit was released after the shared preflight and
frozen-input checks; root controls its current execution, and Agent 2 does not
duplicate that fit.

Preparation evidence is now recorded without opening the scientific boundary. The pure-Python selector validation receipt `experiments/TRR-P11/selector/validation-r2.json` (SHA-256 `192b26efd83592b5ba23c53c9e3cd14087752553b867b9292b16e25936e51639`) passes all 8 tests, including complete identity-union round-trip/tamper rejection, historical H40 rejection, authoritative B0/B1 input binding, and the injected trusted-selector path. It performed no source scan, model load, GPU work, or scientific prediction/evaluation truth access. The synthetic restore-gate receipt `experiments/TRR-P11/restore/execution-tests-r11.json` (SHA-256 `5ee820b3c66164d1f6d45db41f75e5b7cd4e9e98169143370ccbd67d39f274d5`) passes 12/12 tests, with the exact restore and smoke identity checks exercised synthetically; the actual Agent 1 package and independent secondary copy remain pending. The latest exclusion producer receipt is `experiments/TRR-P11/exclusions/recovery_identity_audit_r7.json` (SHA-256 `b0357082d50082231b0b53e61a77a8e9159e72351689993fbe63502d7a7491e3`); root review rejects its completeness claim because nonzero rows were proved. Its effective status is `REJECTED_BY_ROOT_REVIEW_FALSE_COMPLETENESS`, selection release is false, and a replacement audit binding is pending.

The exact B0/B1 bank and development-selection metadata are hash-bound in the manifest and parallel contract. The binding sidecar is `experiments/TRR-P11/exclusions/replication_inputs_binding_r1.json` (SHA-256 `6b1945a065d7a737b40ca0623bc5bedc909e8721836c95eed4742b64ae171c98`). This binds preparation inputs jointly with the partial audit while preserving the remaining canonical, legacy, and targetfit review concerns. No Agent 2 fit, fresh source selection, activation capture, prediction, truth opening, or P03 access has occurred; Agent 1 fit progress is recorded separately below.

The frozen evaluation geometry is 256 records per domain, 128 stored tokens
including BOS, 127 scored post-BOS positions, and four separate
domain-target cells: Pile/Finance by public_base/public_lora_2601. The
prospective A1+A2 K256 comparator, if independently restorable, uses the
first 128 records per domain with 16,256 scored positions per cell. If its
package is unavailable, the matrix is reported as qualified partial and the
comparator is not dropped or replaced based on new outcomes. Source
reservations inherit seed 5011 and Pile [0,2,000), Finance [20,000,28,000);
selection remains blocked pending complete exclusions and the frozen plan,
then may proceed independently of package restoration.

The primary new-expanded-B1 versus new-current-B0 comparison will preserve,
for every cell, the paired exact-record gains, losses, and ties before any
summary is interpreted.

The preregistered statistical contract resamples paired source records with
seed 9009 and 10,000 draws, reports descriptive central 95% percentile token
intervals, and uses the declared exact-record `paired_exact_cp` interval. The
byte-identified implementation is `scripts/trr0010_analysis.py` from source
commit `70c57db7643913eea97cc606775b3f1f3807967a`, SHA-256
`90078ac78bfcdcfb5f782a598c943b78cb0417f6895906a1e6d61056a3793cef`, with
`bootstrap_token_delta` and `paired_exact_cp` bound before truth. Agent 2
owns this scorer dependency; it is separate from Agent 1's decoder package.
There is no automatic CI or promotion gate and no extra inherited harm route.

The mandatory restore gate requires two verified durable copies, including the
proposed Windows backup boundary
/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012, with at least 10 GiB
free after the copy. This boundary protects against WSL reset and temporary
runtime loss; it is not a whole-host physical-failure backup. The gate also
requires clean-workspace retrieval, exact file/tensor identities, and a fixed
already-opened public smoke test. The smoke fixture is the first two
already-opened TRR-0010
public_base records per domain in Finance-then-Pile order, evaluated for both
new fixed models after normal checkpoint selection. A pointer, old replay, or
training-directory load does not pass the gate. P03 remains sealed and the
truth boundary is unopened.

The machine-readable contract is coordination/parallel/TRR-P11.json and the
full prospective plan is experiments/TRR-P11/planning/replication_plan.md.

The successor branch is published as draft PR24 against task/TRR-P10 and remains unmerged. The task-local Agent2 scorer source is scripts/trr_p11/scorer/trr0010_analysis.py; it is byte-identical to the registered TRR-0010 source (SHA-256 90078ac78bfcdcfb5f782a598c943b78cb0417f6895906a1e6d61056a3793cef).


Agent 1 progress is recorded separately from any scientific result. B0 has completed the native 13,000-update fit; its registered step is 8,000 under the earliest-strict-maximum rule, with all seven registered checkpoints persistent and the native audit passing. The paired B1 arm started with step 0 persisted and has latest confirmed normal-checkpoint progress at step 2,000 (`expanded_fixed_replication_1`); interval 2 is recorded as 174.70 seconds. This metadata contains no development scores. The eight-update qualification receipt is a separate qualifier, not the B0 or B1 fit: `../TRR-0012/outputs/TRR-0012/qualification_fixed_b1_v2/qualification_receipt.json` (SHA-256 `2eba2b75b64015911f9801637814e5de9698a2f00824f67980d720a14872c706`), with eight discarded updates recorded from its metadata. The source commit is `c943663356ee5016a5ed6bd83c6d714e1aae22f0`.

The latest exclusion producer receipt is now referenced as `experiments/TRR-P11/exclusions/recovery_identity_audit_r7.json` (SHA-256 `b0357082d50082231b0b53e61a77a8e9159e72351689993fbe63502d7a7491e3`), but root review rejects its completeness claim because nonzero rows were proved. The effective status is `REJECTED_BY_ROOT_REVIEW_FALSE_COMPLETENESS`; a replacement binding is pending, selection release remains false, and no source selection has started.

The nonexecuting historical runtime estimate is `experiments/TRR-P11/capture/historical_a1_runtime_estimate_r1.json` (SHA-256 `366c92b2aea776d592e1b669caf2b44811bc7cb3534c32ee3f338cef9d6a9cd1`). For frozen A1+A2 K256, row scaling gives 1,160.926 seconds of timed interval for the preregistered first-128 subset, but historical Pile receipts used 40 tokens while P11 stores 128. A geometry-aware planning proxy applies the worst observed Finance 128-token per-record total (3.5878 seconds including warmup and three measured calls) to all 512 comparator records, then adds 6.781 seconds of historical model/embedding/state startup: 1,843.747 seconds (30.729 minutes) before P11 I/O, allocator variance, watchdog overhead, and safety margin. This is a release envelope, not a fixed timeout or a P11 measurement. The largest representative cell must qualify with a live resource watchdog, and an arbitrary timeout must not truncate the matrix.

The capture adapter checkpoint `177532c` aligns the post-capture receipt with the Agent1 package: the overall capture receipt exposes four `cells`, shared observations use local `record/NNN` order and bool/int64-normalized tensor digests, and raw producer order/tensor bindings remain alongside them. Its nine pure-Python capture tests pass. This is static adapter evidence; it records no source scan, model run, prediction, truth access, or P03 access.


The selector preflight is persistent but intentionally unopened. The hash-only source descriptor is `experiments/TRR-P11/selector/public_source_inputs_r1.json` (SHA-256 `1fa6aea4485ad602d396e6a57dc53257977a0f4581375892ef22ab176d52b409`), and the exact gated command is recorded in `experiments/TRR-P11/selector/command-r1.json` (SHA-256 `7daae7d003b60a3c62f096b914d27df9e36f40e72a67d054cef182552654db5a`). Its wrapper resolves paths from the P11 repository root, uses the existing sibling `../TRR-0010`, and records an executable offline outer plan with 7,200 CPU seconds, 10,800 wall seconds, and a 60-second kill grace. The command points at r7 as an explicit rejected-by-root-review placeholder and must be rebound to the replacement audit before execution; the manifest and released-audit gate runs before source-input normalization or trusted public loading. Its output is create-only at `experiments/TRR-P11/selector/source_selection.json`; that file has not been created.


The native A1+A2 adapter now has an explicit largest-cell qualification entry for `finance__public_base`; qualification and resumed matrix entry reject any other qualification cell. A qualification-only run writes `qualification.json`; a resumed matrix must validate and reuse that receipt, so the qualified cell is retained without a redundant rerun. The default wall limit is unset. An external stdlib watchdog polls parent RSS, GPU free memory, temperature, and compute-app exclusivity between cell checks and fails closed on violations. The implementation, selector wrapper, and pure Python tests are bound in the task metadata; the combined selector, wrapper, capture, and runtime suite passes 26 tests. No native model, CUDA, prediction, or truth run has started.

The original TRR4 public payload identity recovery is complete after Agent 1 released the CPU lease. The opaque export `experiments/TRR-P11/exclusions/original_fit_1200_identity_r1.json` (SHA-256 `45700bb02bbb89e7c8006e0bc6cff60b8f3afaeea533e1d9c4c41cc19531a9da`) covers all 1,200 fitting rows, with 350 applicable H128 identities and 343 H129 identities; its recovery receipt is `original_fit_1200_recovery_r1.json` (SHA-256 `3f06a5287beaccb39efc3bf6b0ad2da9dbf09e376eb4d58fb2406152b0bd95a1`). The original 48-record validation export `validation_48_identity_r1.json` (SHA-256 `d68d98aecdfd0cf041be1e508074ffbf4ded24e0bc0a401362f1eb262e5e17ab`) has 6 applicable H128 identities and 5 H129 identities; its receipt is `validation_48_recovery_r1.json` (SHA-256 `9473f1b378add851456dc1acf120cb530738ddbb055695c46613fcb0311efc0e`). Both receipts report zero mismatches and no source text, token values, activations, model, GPU, truth, or P03 access.

Each recovery ran as a separate one-thread child under the existing CPU process-group watchdog with a 2 GiB RSS cap, 12 GiB minimum host availability, 120-second wall limit, and 15-second kill grace; the prior 2 GiB virtual-address-space cap was removed. The original-fit guard peaked at 325,758,976 bytes group RSS and the validation guard at 537,128,960 bytes; both watchdog receipts are preserved under `experiments/TRR-P11/exclusions/watchdog_original_fit_r1/` and `watchdog_validation_48_r1/`. The combined handoff index is `experiments/TRR-P11/exclusions/original_payload_recovery_handoff_r1.json` (SHA-256 `af52b27c503a2e935a7beb0978fd5fe3b07d27bebe0893aa6ecf9c34d91a5986`) for the exclusion worker's opaque identity-union integration. This does not release source selection; the canonical exclusion audit remains partial and the restore gate remains pending.
