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
access. The single Agent 1 fit is authorized only after the shared preflight
and frozen-input checks; root controls that release.

Preparation evidence is now recorded without opening the scientific boundary. The pure-Python selector validation receipt `experiments/TRR-P11/selector/validation-r2.json` (SHA-256 `192b26efd83592b5ba23c53c9e3cd14087752553b867b9292b16e25936e51639`) passes all 8 tests, including complete identity-union round-trip/tamper rejection, historical H40 rejection, authoritative B0/B1 input binding, and the injected trusted-selector path. It performed no source scan, model load, GPU work, or scientific prediction/evaluation truth access. The synthetic restore-gate receipt `experiments/TRR-P11/restore/execution-tests-r11.json` (SHA-256 `5ee820b3c66164d1f6d45db41f75e5b7cd4e9e98169143370ccbd67d39f274d5`) passes 12/12 tests, with the exact restore and smoke identity checks exercised synthetically; the actual Agent 1 package and independent secondary copy remain pending. The latest exclusion audit remains partial at `experiments/TRR-P11/exclusions/recovery_identity_audit_r4.json` (SHA-256 `2629c919e8de0b829b951826689256c5ad5bf2e824269407961fee875f49b1dd`), with selection release false.

The exact B0/B1 bank and development-selection metadata are hash-bound in the manifest and parallel contract. The binding sidecar is `experiments/TRR-P11/exclusions/replication_inputs_binding_r1.json` (SHA-256 `6b1945a065d7a737b40ca0623bc5bedc909e8721836c95eed4742b64ae171c98`). This binds preparation inputs jointly with the partial audit while preserving the remaining canonical, legacy, and targetfit exclusion-coverage gaps. No fit, fresh source selection, activation capture, prediction, truth opening, or P03 access has occurred.

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
