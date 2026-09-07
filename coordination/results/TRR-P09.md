# TRR-P09 implementation checkpoint

This checkpoint provides the shared fixed/directional runner primitives at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, based on the proposed P09 adapter and streamed-bank contract; the common scientific contract is still pending. The GPU was used only for five bounded public-base qualification attempts, all excluded before activation-bank capture. The public model was loaded for the latter two attempts; no activation bank, target observation, evaluation truth, or scientific fit was opened or persisted. A separate permitted public-source/tokenizer count audit is recorded below; it did not load a model or select a P09 row.

The runner now has a required random-access loader adapter for arbitrary scheduled rows, preserving order and duplicates; validates nonnegative scheduled rows, complete batch geometry, BOS/position IDs, and every sampled position's attention mask; and records lazy-load and optimizer-update timing separately. One shared optimizer contract covers all unique trainable decoder and readout-hook parameters, rejects omitted or unowned trainable parameters, clips the same complete set, and checks post-update finiteness. Checkpoint state digests include both decoder and hook state. Per-step state hashing remains opt-in and is disabled by the shared loop; checkpoint hashing is the normal path.

The synthetic directional hook test demonstrates that a trainable readout parameter receives an update and is included in clipping and state hashing. The runner retains explicit step-zero eligibility and earliest-strict-maximum checkpoint selection, and validation accepts an independently supplied sequence width for the H128 view. Domain-balanced validation aggregation remains an evaluator responsibility; the generic loop currently reports raw token totals and the caller-supplied metric.

## Agent 2 public-bank planning checkpoint

The bounded planning recommendation is a 12,000-row proportional expansion, pending the joint identity and capacity contract: 6,000 natural Alpaca, 3,000 natural Pile, 1,800 natural Finance, and 600 controlled contexts each for Pile and Finance. This preserves the published B0 construction ratio while providing 10x its 1,080 natural-parent support. The existing 1,200-row B0 must remain the byte-equivalent prefix. Controlled replacements remain in the inherited 128--191 post-BOS length stratum and four inherited position bins. No new validation pool is proposed; the natural `public_base` validation binding already opened by Agent 1 remains separate and root-owned.

A permitted public Alpaca count/tokenization audit rendered all 52,002 rows with the pinned public tokenizer. It found 50,778 rows after 1,224 keyed fit/validation identity exclusions, with 39,765/24,401/14,317/6,824 rows meeting post-BOS thresholds 64/96/128/160. These are count-only upper bounds: the exclusion metadata has a 4,193-value union containing 4,073 opaque sequence/reservation digests, and zero rendered-text SHA-256 hits against that union do not establish disjointness because the namespaces/canonicalizations differ. The audit receipt and exact command/environment provenance are `experiments/TRR-P09/planning/alpaca-capacity-audit.json` (SHA-256 `d1f11364cabe6db876bb83f09ce1d8d7db1cdc3f76ac2e8bafaed149ba33a5c9`) and `experiments/TRR-P09/planning/alpaca-capacity-audit-provenance.json` (SHA-256 `c9c6828e16d87a7316f36b10d7a832a7cd04c20a73c035b4178062e3926fc2d2`). The executed CLI is `scripts/trr_p09/audit_alpaca_capacity.py` (SHA-256 `db8b54c290a1ee5c8915a6c315c6969ee4cea3284da246f57d044b0cbbd8b80e`). The human proposal is `experiments/TRR-P09/planning/bank-proposal.md`.

The initial r1 attempt did execute source selection, but it is preserved and excluded before capture. The corrected r2 source selection has now completed under the accepted CPU-only supplement; the CPU preparation itself produced no public-model forward or activation-bank capture. The corrected metadata revision is bound below. Five subsequent public-base qualification attempts are recorded separately; all are excluded, so activation capture and fitting remain blocked pending the implementation fix and a passing qualification.

Validation command (from the isolated P09 worktree):

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_fixed_control_runner.py tests/test_trr_p09_fixed_control_adapter.py tests/test_trr_p09_bank_artifact.py
```

Result: 25 passed in 1.90s. Syntax validation also passed with `python3 -m py_compile scripts/trr_p09/fixed_control_runner.py src/token_reconstruction/trr_p09_fixed_control_adapter.py`. A later setup-owned integrity checkpoint receipt records the same focused module suite at source commit `eaa1f1e5aa0be1e6d0dcb896834f4d6ac4e43861`: 28 passed, 0 failed in 3.005 seconds. The receipt is `experiments/TRR-P09/setup/integrated-checkpoint-tests.json` (SHA-256 `44658a8098ad2406b98ea17111d4ce163b00e95fd03548021da24a2c4bf4ece0`); it is fixture/integrity evidence only and did not open a public bank, model, forward, fit, GPU, or truth.

Before any production capture or fit, the study caller still must bind the finalized serialized schedule reader/digest/exposure contract, domain-balanced validation metric, checkpoint state serializer/callback, and outer fail-closed resource watchdog/CLI. The generic loop has no production asset loader, state-file writer, resource guard, or scientific constants and is therefore an implementation checkpoint, not a fit-ready scientific run. Setup-owned bank payloads and unrelated implementation files remain excluded; the planning files listed above are metadata-only Agent 2 evidence and contain no bank payload, model output, selection, or truth.

## Publication metadata

This checkpoint is published as draft [PR #19](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/19) with base `task/TRR-0009` and current evidence head `ebe05e88ddfebae0155fb016885c6c8938c32840`. The PR is open, draft, and unmerged; the scientific status remains pre-fit.

## Stage-1 authorization and cost-lineage update (2026-09-07)

The public stage-1 input recipe is now bound by
`experiments/TRR-P09/planning/stage1-public-bank-plan.json` (SHA-256
`bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c`) and
countersigned in
`experiments/TRR-P09/setup/stage1-plan-countersignature-r1.json` (SHA-256
`5a08ea2699ce038d7895239932312326852df5ff8514b878e150b7da622853a1`). This
authorizes guarded CPU input preparation under the signed recipe. The
subsequent root-authorized bounded qualification attempts and their exclusions
are recorded below; neither produced a persisted activation bank, fit, or truth
result. The text above records the pre-receipt state; the completed r1 CPU
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
`e8e0e2bad1bba8ba71482dd28babfff4bcdf391cf224a95551705d2643171c5d`); the
metadata-only revision is
`/tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/preparation_manifest-r2-corrected.json`
(SHA-256
`79bbd55c3e66555501e0abc886e69b5a4ff85bca5301a129f6e5d398524e844f`) and
the unchanged payload records are SHA-256
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

The corrected metadata-only revision now supersedes the original preparation
manifest. Its `r2_actual_occurrence_audit` records the actual B0 3,600
occurrences plus 32,400 r2 template-assignment occurrences, with full digest
`0e1b7726cc1e394dba8ccaf528d22216205e1d595f3b8bb853c879c3a77df53b`; the
legacy signed-order comparison remains explicitly historical. The r1 output
remains excluded; its B0 order permutation and 63 prior-development Alpaca
overlaps are preserved as failed provenance, and the two earlier
structural-token failures plus the pre-final slot-compatibility failure remain
excluded attempts.


## Public-base qualification attempts (2026-09-07)

Five root-authorized qualification attempts consumed the bounded public-base
lease and are excluded before activation capture. The compact evidence record is
`experiments/TRR-P09/setup/stage1-qualification-attempts-r2.json` (SHA-256
`21187abe870792af65b8aacc9f691ab81558b7483b0ef5e720af6a8dce621592`). The first attempt used code commit
`09ab9958ad167f17e6e5973a2ca15306a9c4b1fe` and failed during plan verification
with `STAGE1 plan capture condition/count changed`, before model load. It ran
from `2026-09-07T11:15:35.877964Z` to
`2026-09-07T11:15:36.893870Z` (1.023437 s), with peak group RSS
512,315,392 bytes and minimum sampled host availability 19,681,001,472
bytes. Its watchdog child exited nonzero without guard termination.

The retry used code commit `b6fc36290ac738a6279bec2fe3b08320eae1ef26`,
loaded the public model, completed the original and repeat checks, and then
failed with `qualification batch has no future padding pattern`. It ran from
`2026-09-07T11:20:31.289107Z` to
`2026-09-07T11:20:38.877870Z` (7.596591 s), with peak group RSS
1,694,908,416 bytes and minimum sampled host availability 18,701,070,336
bytes. Its watchdog also exited on the child failure without resource
termination. A third retry then completed two representative batches and five
forwards with exact repeat/padding checks in 0.532312922 seconds, but its
external watchdog failed closed during teardown when `/proc/92747/status` became
unreadable; the child returned 0 and the wrapper returned 125 after SIGTERM.
The entire third attempt is excluded despite the child qualification receipt
being `QUALIFICATION_PASS`. The fourth and fifth retries likewise completed
two representative batches and five forwards with exact repeat/padding checks
(0.525853747 and 0.532732332 seconds), but each external watchdog failed closed
during leader teardown with unreadable RSS; both child processes returned 0 and
wrappers returned 125. All five attempts are excluded; there is no accepted
qualification, activation-bank output, fit, or evaluation-truth access. Runtime receipts remain immutable under
`/tmp/trr-p09-runtime/stage1-qualification-r2*`.

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

The scientific status remains pre-capture: no P09 activation bank was
persisted, no fit was started, and no evaluation truth was opened. The two
public-base qualification attempts above are excluded failures, not scientific
results.

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


## Common CPU schedule preparation (2026-09-07)

The shared CPU schedule factory produced immutable B0 and expanded B1 artifacts from the corrected public input mask, with no model, H, activation-bank, fit, or evaluation-truth access. The compact receipt is `experiments/TRR-P09/setup/common-schedules-r1.json` (SHA-256 `3275af36352e4dd15ba639bdd86af439841ed0b19379aba292ad00f07115b86d`). It binds `scripts/trr_p09/fixed_control_caller.materialize_schedule_plan` and `inherited_schedule_steps`, seed `4010`, `N=13,000`, batch size `8`, position budget `512`, and the exact `SchedulePlan.semantic_sha256` contract.

The local runtime payloads are `/tmp/trr-p09-runtime/common-schedules-r1/schedule-b0-seed4010.safetensors` (27,053,864 bytes, SHA-256 `ba53cd4bad54bc88670f346a7f959eb0ce5278ec28eac8bdf6463cd44936ec6f`, semantic digest `9aaad9c030f2f9b801f91f956c97f858966580edd21229cebb350dcde358f2c3`) and `schedule-b1-seed4010.safetensors` (27,053,864 bytes, SHA-256 `be0745ebb56e0ed39c2cf7e9c05a657209126f4c1c552242d3893c52d6718801`, semantic digest `8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5`). Each file contains exactly `batch_record_indices[N,8]` (`int32`), `draw_record_slots[N,512]` (`int16`), `draw_position_slots[N,512]` (`int16`), and `used_replacement[N]` (`uint8`). B0 covers rows `0:1200` with `124,371` post-BOS valid positions; B1 covers rows `0:12000` with `1,243,710`. The runtime receipt is local-only; the compact receipt preserves the hashes and schema.


## Current stage-1 qualification and public capture update (2026-09-07)

The five qualification attempts described above remain preserved as excluded historical attempts. A sixth, clean root-authorized retry then passed: `QUALIFICATION_PASS`, five public-model forwards, exact repeat and future-padding checks, 0.532744502 seconds child elapsed, and a clean external watchdog (`6ef585c784b6898884c9fb44243c3afe1833ea211618cc957f116bbd99479b38`). Its immutable child receipt is `/tmp/trr-p09-runtime/stage1-qualification-r2-retry5/qualification-receipt.json` (SHA-256 `3b5c99cfc5a125b25339df558d3ffb6ea37249e3d49122178cf574e9f079df8c`); the compact current binding is `experiments/TRR-P09/setup/stage1-capture-binding-r1.json` (SHA-256 `caabf6d8e4983bf453cfe64b3548cc3cedaa245f7dad08bf438d81d72a65837a`).

The authorized public-base capture subsequently completed with watchdog PASS and child/wrapper return code 0. It created 169 immutable shards containing 10,800 new rows for expanded global rows 1,200--11,999, using complete `[8,192,2048]` BF16 public-model forwards. The capture child elapsed time was 55.403302 seconds and the guarded wrapper elapsed time was 62.141337 seconds. Resource high-water values were GPU reserved 2,474,639,360 bytes, process RSS 1,803,186,176 bytes, and retained output 8,545,446,344 bytes; the watchdog observed peak group RSS 1,842,196,480 bytes and minimum host availability 18,462,162,944 bytes. The bank manifest is `/tmp/trr-p09-runtime/stage1-capture-r2/bank_manifest.json` (SHA-256 `f1ad02c2836c53992299e3459d886bf87c2f0d2d437c86224ea4db689c1b2216`), and the capture receipt is `/tmp/trr-p09-runtime/stage1-capture-r2/capture-receipt.json` (SHA-256 `c314cf36fd207ae95203c3c6adda0dc7773ac2765f85b5c7fd4ed5b4f0de1618`). Raw shard payloads remain local-only and are referenced by the manifest; they were not rewritten or added to Git.

The bank manifest deliberately retains `geometry.forward_batch_qualification = PENDING_SHARED_GEOMETRY_QUALIFICATION`. The completed capture is evidence of the public forward/capture path, not a substitute for the shared loader geometry review. No optimizer step, fit, target observation, or evaluation-truth access occurred.

A CPU-only public support binding was also prepared from the compiled public token IDs and masks. B0 has 17,126 positive-support token IDs over 124,371 post-BOS positions; expanded B1 has 45,631 over 1,243,710 positions. The Agent1/TRR-0010 support digests are `f554d25681904530bf0f52752b50f92ae5da10b99998d06eb0cc30ed25fccd4d` (B0) and `f91eaf39433496ba6fd36c5c5233e1d8abbaa2527ff7d75e7b304ec4df4fbd0e` (B1). The compact binding is `experiments/TRR-P09/setup/public-support-binding-r1.json` (SHA-256 `1c202ab7518404c12726b75147e1246d72fc40bbaf7c4c043edf0792a5e65351`); support vectors are local runtime payloads, with the digest namespace distinguished from the historical TRR-0009 support digest.

This current section supersedes the earlier pre-capture status statements for the qualification/capture phase; those statements remain retained as time-scoped historical evidence.


## Qualified bank metadata revision (2026-09-07)

Root reviewed and committed the final descriptive bank-manifest revision in `experiments/TRR-P09/setup/qualified-bank-manifest-r1.json` (SHA-256 `657d73a2d9787bdb321260d1757c4a6e8c27c23dc1e07a30273bf7e2ff41c7d2`). It points to the qualified manifest `/tmp/trr-p09-runtime/stage1-capture-r2/bank_manifest-qualified-r1.json` (145,390 bytes, SHA-256 `0e14dbf74c947cfc6c586e7a5dbf76a27d8f0522fb8e54d2f98c6067bb36d196`). The revision changes only the stale descriptive qualification/status fields; the original bank manifest, 169 shard payloads, sidecars, row order, geometry, and input bindings are unchanged.

The shared forward-batch geometry gate is now recorded as `QUALIFICATION_PASS`. The capture watchdog remained PASS with 169 shards and 62.141337 seconds wrapper elapsed time. All five earlier qualification failures remain preserved and excluded, the accepted clean retry remains bound, and P09 has zero fits or optimizer steps, zero target observations, and unopened evaluation truth. GPU work is released; common schedules and the public support binding remain ready for the next reviewed fit-contract step.

## Validation and fixed-diagnostic evidence (2026-09-07)

The validation and diagnostic checkpoint is bound to evidence commit `31963b67394082ac8d67b3878920aa2789eb1921`. Public-only validation completed for 384 records (Finance 256, Pile 128; H=128) and passed the preparation and audit watchdogs. The compact audit is `experiments/TRR-P09/setup/public-validation-r1-audit.json` (SHA-256 `25d30f3cfb7e3c00c0d8ecde96db550c772f663ffa81a3a5a211fe1be166a022`); the successful preparation watchdog is `/tmp/trr-p09-runtime/public-validation-watchdog-r1` and the separate audit watchdog is `/tmp/trr-p09-runtime/public-validation-audit-watchdog-r1`. The fixed 64-row-per-bank diagnostic binding is `experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json` (SHA-256 `8b899bc168b55570e4cfd526ff4e8656b70050e840386cba9390ad6adb1f54c3`), generated by `scripts/trr_p09/bind_fixed_diagnostic.py` (SHA-256 `d1459f01e1942dd8796e124ae31abdef218ee1a7c8273f830a6246ece20fc8fd`). It preserves the 32/16/10/3/3 stratum quotas and the 1,350 B8x192 capture exposure; it reads public fitting token IDs, masks, and positions only and does not modify the bank manifests.

A root rerun was refused by the create-only preparation guard because `/tmp/trr-p09-runtime/public-validation-r1` already existed. That harmless duplicate-launch attempt is excluded and recorded at `/tmp/trr-p09-runtime/watchdog-public-validation-r1` (child and wrapper return code 125); it is distinct from both successful watchdog roots above.

P09 still has zero fits and zero optimizer steps, with target observation and evaluation truth unopened. The remaining gates are the reviewed fixed-control fit qualification, final shared fit-contract/loader integration, fixed 64-row diagnostics at step zero and each checkpoint, and full-bank start/end metrics.

## Fixed-control source freeze (2026-09-07)

The fixed-control source and production configuration are now frozen in `experiments/TRR-P09/review/fixed-control-source-freeze-r1.json` (SHA-256 `8412076b428ab1dbd14fa2cb83c6ecf5111c13bae04572f2d4139d1760d0a681`), at source-freeze commit `ebe05e88ddfebae0155fb016885c6c8938c32840`. The receipt records 34 focused tests passing, actual B0/B1 configuration checks passing, and the corrected r5 checkpoint grid `[0, 1000, 2000, 4000, 8000, 12000, 13000]`; diagnostic hooks are source-frozen for 64-row checkpoints and full-bank step-zero/final metrics.

This remains a pre-fit handoff: exactly 0 fit runs and exactly 0 optimizer steps have executed; no fit model, fit embedding, public fit forward, target observation, or evaluation truth was opened. The remaining gates are Agent1's largest-cell qualification, the final joint fit-contract/loader handoff, exactly two fixed-control fits (B0 and B1), and frozen output/metric handoff.
