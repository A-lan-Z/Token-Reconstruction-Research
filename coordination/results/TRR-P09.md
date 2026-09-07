# TRR-P09 implementation checkpoint

This checkpoint provides the shared fixed/directional runner primitives at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, based on the proposed P09 adapter and streamed-bank contract; the common scientific contract is still pending. No public activation bank, model checkpoint, target observation, truth, GPU, or scientific fit was opened or run. A separate permitted public-source/tokenizer count audit is recorded below; it did not load a model or select a P09 row.

The runner now has a required random-access loader adapter for arbitrary scheduled rows, preserving order and duplicates; validates nonnegative scheduled rows, complete batch geometry, BOS/position IDs, and every sampled position's attention mask; and records lazy-load and optimizer-update timing separately. One shared optimizer contract covers all unique trainable decoder and readout-hook parameters, rejects omitted or unowned trainable parameters, clips the same complete set, and checks post-update finiteness. Checkpoint state digests include both decoder and hook state. Per-step state hashing remains opt-in and is disabled by the shared loop; checkpoint hashing is the normal path.

The synthetic directional hook test demonstrates that a trainable readout parameter receives an update and is included in clipping and state hashing. The runner retains explicit step-zero eligibility and earliest-strict-maximum checkpoint selection, and validation accepts an independently supplied sequence width for the H128 view. Domain-balanced validation aggregation remains an evaluator responsibility; the generic loop currently reports raw token totals and the caller-supplied metric.

## Agent 2 public-bank planning checkpoint

The bounded planning recommendation is a 12,000-row proportional expansion, pending the joint identity and capacity contract: 6,000 natural Alpaca, 3,000 natural Pile, 1,800 natural Finance, and 600 controlled contexts each for Pile and Finance. This preserves the published B0 construction ratio while providing 10x its 1,080 natural-parent support. The existing 1,200-row B0 must remain the byte-equivalent prefix. Controlled replacements remain in the inherited 128--191 post-BOS length stratum and four inherited position bins. No new validation pool is proposed; the natural `public_base` validation binding already opened by Agent 1 remains separate and root-owned.

A permitted public Alpaca count/tokenization audit rendered all 52,002 rows with the pinned public tokenizer. It found 50,778 rows after 1,224 keyed fit/validation identity exclusions, with 39,765/24,401/14,317/6,824 rows meeting post-BOS thresholds 64/96/128/160. These are count-only upper bounds: the exclusion metadata has a 4,193-value union containing 4,073 opaque sequence/reservation digests, and zero rendered-text SHA-256 hits against that union do not establish disjointness because the namespaces/canonicalizations differ. The audit receipt and exact command/environment provenance are `experiments/TRR-P09/planning/alpaca-capacity-audit.json` (SHA-256 `d1f11364cabe6db876bb83f09ce1d8d7db1cdc3f76ac2e8bafaed149ba33a5c9`) and `experiments/TRR-P09/planning/alpaca-capacity-audit-provenance.json` (SHA-256 `c9c6828e16d87a7316f36b10d7a832a7cd04c20a73c035b4178062e3926fc2d2`). The executed CLI is `scripts/trr_p09/audit_alpaca_capacity.py` (SHA-256 `db8b54c290a1ee5c8915a6c315c6969ee4cea3284da246f57d044b0cbbd8b80e`). The human proposal is `experiments/TRR-P09/planning/bank-proposal.md`.

The initial r1 attempt did execute source selection, but it is preserved and excluded before capture. The corrected r2 source selection has now completed under the accepted CPU-only supplement; it still produced no public-model forward, activation-bank capture, fitting, final-panel evaluation, or truth access. Capture remains blocked until the setup-owned r2 metadata is revised to remove its stale legacy ten-cycle descriptor and the capture qualification gate passes.

Validation command (from the isolated P09 worktree):

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_fixed_control_runner.py tests/test_trr_p09_fixed_control_adapter.py tests/test_trr_p09_bank_artifact.py
```

Result: 25 passed in 1.90s. Syntax validation also passed with `python3 -m py_compile scripts/trr_p09/fixed_control_runner.py src/token_reconstruction/trr_p09_fixed_control_adapter.py`. A later setup-owned integrity checkpoint receipt records the same focused module suite at source commit `eaa1f1e5aa0be1e6d0dcb896834f4d6ac4e43861`: 28 passed, 0 failed in 3.005 seconds. The receipt is `experiments/TRR-P09/setup/integrated-checkpoint-tests.json` (SHA-256 `44658a8098ad2406b98ea17111d4ce163b00e95fd03548021da24a2c4bf4ece0`); it is fixture/integrity evidence only and did not open a public bank, model, forward, fit, GPU, or truth.

Before any real qualification or fit, the study caller still must bind the finalized serialized schedule reader/digest/exposure contract, domain-balanced validation metric, checkpoint state serializer/callback, and outer fail-closed resource watchdog/CLI. The generic loop has no production asset loader, state-file writer, resource guard, or scientific constants and is therefore an implementation checkpoint, not a fit-ready scientific run. Setup-owned bank payloads and unrelated implementation files remain excluded; the planning files listed above are metadata-only Agent 2 evidence and contain no bank payload, model output, selection, or truth.

## Publication metadata

This preparation checkpoint is published as draft [PR #19](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/19) with base `task/TRR-0009` and evidence head `140ed70df03aca32814cbd4b48aa395745d57ad6`. The PR is open, draft, and unmerged; publication metadata does not change the preparation-pending scientific status above.

## Stage-1 authorization and cost-lineage update (2026-09-07)

The public stage-1 input recipe is now bound by
`experiments/TRR-P09/planning/stage1-public-bank-plan.json` (SHA-256
`bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c`) and
countersigned in
`experiments/TRR-P09/setup/stage1-plan-countersignature-r1.json` (SHA-256
`5a08ea2699ce038d7895239932312326852df5ff8514b878e150b7da622853a1`). This
authorizes guarded CPU input preparation under the signed recipe. It does not
open a model, create activations, start a fit, acquire an evaluation lease, or
open truth. The text above records the pre-receipt state; the completed r1 CPU
preparation and its pre-capture exclusion are recorded below as a create-only
receipt.

The published starting-checkpoint cost lineage is recorded separately in
`experiments/TRR-P09/planning/starting-state-cost-lineage.md` (SHA-256
`8f81b372d51d58d61f88f6c778825240ad67e95ece8d6804bcf8542307e36c9e`). It
reports full scheduled runs, selected steps, preparation/capture components,
and UNKNOWN fields without inventing a grand total or charging shared receipt
fields repeatedly.

A read-only compiler receipt audit is recorded in
`experiments/TRR-P09/review/stage1-compiler-review.md`. The r1 geometry,
quotas, source hashes, namespace counts, N, and separate diagnostics pass the
metadata review. The r1 attempt is excluded before capture because the observed
B0 replacement order differs from the signed first cycle, and because 63 of 624
non-B0 prior-development Alpaca rows were not excluded by the compiled ledger.
These findings are implementation/provenance blockers for r2, not a new
scientific gate; no model or truth boundary was crossed.


## Stage-1 CPU preparation receipt and pre-capture exclusion (2026-09-07)

The guarded CPU preparation completed with return code 0 in 32.387874 seconds
under source commit `c3cf2956bd2fe0c1254561d8dfb17ae8405bb8fd`. Its receipt is
`experiments/TRR-P09/setup/stage1-input-preparation-r1-receipt.json` (SHA-256
`4314ee55f64f15e88e25319d0a80860a152a83b7d41b02f9539858aef23d0153`). It
produced public tokenized inputs only: 12,000 rows at width 192, 1,200 B0
rows plus 10,800 new rows, 1,243,710 post-BOS positions, and `N=13,000`.
The output and records hashes are `981406fd65109bdaacd935d4eab979a4520d0a5a589f4f6da1f18a119d7de168`
and `30fe4216fd3fb6b76e4aec79324fafd73a398ed901e07b4681b815cce2dc98a4`,
respectively. The compiler loaded no model and produced no activation payload.

The quotas are 6,000 Alpaca natural, 3,000 Pile natural, 1,800 Finance
natural, and 600 controlled rows each for Pile and Finance. Receipt review
found the required separate 64-row diagnostics and namespace-specific
exclusion counts. A public metadata-only overlap check found no B0/B1
public-record-hash overlap and no overlap with the bound TRR-0009 source IDs or
dataset-scoped source indices. The separate prior Alpaca capacity audit
bound 1,224 Alpaca row keys in the public parent-exclusion manifest
(SHA-256 `bd1359f1184091570023e22a7682d1f97c08f8f05e47f69f6b3e6be089cd0181`):
600 are B0 rows and 624 are earlier development rows. The r1 scan excluded
only 600 by metadata and zero by hash; 63 of the 624 non-B0 rows occur in the
r1 B1 additions. This ledger gap is preserved as a second pre-capture blocker
for r2.

The r1 output is excluded before capture because the observed published B0
replacement sequence is a permutation of, but not in the same order as, the
signed 3,600-ID first cycle. Both sequences contain the same 3,600 unique IDs
with identical multiplicities; the first order mismatch is `624` versus
`17065`. Since IDs are assigned to concrete row/position occurrences, this
order difference is not silently treated as equivalent. The narrow addendum
under review will bind the observed B0 occurrence/offset digests and repeat
each of the 120 B0 controlled-row templates exactly nine times on new parents
matched by domain and active length, preserving the inherited position bins.
No model capture, fitting, or truth access may start until that addendum and a
fresh CPU receipt pass.

The full metadata-only audit is
`experiments/TRR-P09/review/stage1-compiler-review.md`; the r1 result is kept as
an excluded attempt for provenance.


## Corrected r2 CPU preparation audit (2026-09-07)

The corrected CPU preparation completed under the jointly countersigned narrow
addendum and template-compatible supplement. The receipt is
`experiments/TRR-P09/setup/stage1-input-preparation-r2-receipt.json` (SHA-256
`9b5b575fcd0454ca43c46f68e99cf9444cdde65c234a46aa571ef48051955798`); its
output manifest is
`/tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/preparation_manifest.json`
(SHA-256
`262d13d7cf86c72e97ef8d1ad9656a2f34603fd3c0d147f1648769c2e8c21123`) and
records are SHA-256
`367cfba0ffe78f59454861a76f23830f480a8745cc6b35e6b4a0d05eca53638b`. The
watchdog reported PASS, return code 0, 32.880936 seconds, peak group RSS
1,609,445,376 bytes, and no resource errors. The run source commit was
`15be77368dae5129a216c9354975b6347cf28720`; the focused metadata/synthetic
test command reported 73 passed.

An independent metadata-only audit found 12,000 ordered records with the
expected quotas: 6,000 Alpaca natural, 3,000 Pile natural, 1,800 Finance
natural, and 600 controlled records for each Pile and Finance. It found
1,243,710 post-BOS positions, `N=13,000`, 45,631 distinct post-BOS token IDs,
and 64-row diagnostics per bank with seed 4010 and quotas 32/16/10/3/3. The
first 1,200 record identities and source metadata match the published B0
records. Among the 1,080 controlled additions, there are 120 distinct joint
(stratum, target length, offset, replacement-token) templates and every one
is used exactly nine times. The full public parent exclusion manifest is
bound (SHA-256
`bd1359f1184091570023e22a7682d1f97c08f8f05e47f69f6b3e6be089cd0181`): the
600 B0 Alpaca rows intersect its 1,224 Alpaca keys as expected, while the B1
additions have zero record-ID, source-row-key, current-fit, or rendered-hash
overlap, including zero overlap with all 624 non-B0 prior-development Alpaca
keys.

The setup manifest retains a stale legacy field named
`full_ten_cycle_composition` and its associated identity digest, describing
the superseded signed-order construction. That field is not used as the r2
controlled-row result: the independently checked r2 template assignment and
exact-nine audit above are authoritative for this review. Setup must revise
the r2 receipt/manifest metadata before capture so the machine-readable
record agrees with the corrected construction. The r1 output remains
excluded; its B0 order permutation and 63 prior-development Alpaca overlaps
are preserved as failed provenance, and the two earlier structural-token
failures plus the pre-final slot-compatibility failure remain excluded
attempts.

## Immutable B0 loader binding

The existing monolithic B0 payload is now bound without copying or
materializing H through
`experiments/TRR-P09/setup/b0-immutable-loader-binding-r1.json` (SHA-256
`7e6d7d3ac0ceec77f8b1e961a1a74b0675fd43e6ef3e4e77dcdd368d0c27445d`) and
`scripts/trr_p09/b0_immutable_loader.py` (commit `86e66aa`). The adapter
validated the published headers only: `[1200,192,2048]` BF16 activations, U8
mask, I64 positions, and I32 tokens, with active-prefix positions and zero
padding. Its two synthetic tests passed; no real B0 H slice, model, GPU, fit,
or evaluation truth was opened.

The scientific status remains preparation-only: no P09 activation capture,
model forward, fit, or truth access is authorized by this receipt.

## Synthetic shard I/O qualification (2026-09-07)

A bounded CPU-only synthetic benchmark used deterministic data with the planned
`[64,192,2048]` BF16 geometry and the four P09 payload keys. The executable is
`scripts/trr_p09/synthetic_io_qualification.py` (SHA-256
`167702aeb5adf3262c2580027ce2f950d37b908f9cdc1b43747b47bcb9d3ee5e`) and the
successful receipt is
`experiments/TRR-P09/review/synthetic-io-qualification-r2.json` (SHA-256
`27e90d3b0d91d6307c504ff3fa95be0ca6cd8a1902c86803111204ea2930e3cc`). The
50,491,792-byte synthetic shard serialized in 0.0605 s, hashed in 0.0202 s,
read its header in 0.000168 s, and read back 50,491,392 bytes in 0.000270 s
with exact tensor equality. No public payload, model, forward, fit, or truth
was accessed.\n
For 169 shards, the measured payload size extrapolates to 8.533 GB (7.947 GiB).
Applying a conservative 2x allowance for sidecars, temporary/retry space, and
filesystem overhead gives 15.894 GiB against the 20 GiB retained cap, leaving
4.106 GiB by this synthetic storage bound. Serial extrapolated write/hash/read
costs are 10.22/3.42/0.046 seconds; these are local synthetic I/O timings and
exclude model-forward and capture scheduling. Two setup-only synthetic failures
are preserved in `synthetic-io-qualification-failure-r1.json` (missing shard
directory) and `synthetic-io-qualification-failure-r2.json` (existing scratch
directory); neither touched real P09 artifacts.
