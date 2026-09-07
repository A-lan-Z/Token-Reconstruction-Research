# TRR-P09 implementation checkpoint

This checkpoint provides the shared fixed/directional runner primitives at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, based on the proposed P09 adapter and streamed-bank contract; the common scientific contract is still pending. No public activation bank, model checkpoint, target observation, truth, GPU, or scientific fit was opened or run. A separate permitted public-source/tokenizer count audit is recorded below; it did not load a model or select a P09 row.

The runner now has a required random-access loader adapter for arbitrary scheduled rows, preserving order and duplicates; validates nonnegative scheduled rows, complete batch geometry, BOS/position IDs, and every sampled position's attention mask; and records lazy-load and optimizer-update timing separately. One shared optimizer contract covers all unique trainable decoder and readout-hook parameters, rejects omitted or unowned trainable parameters, clips the same complete set, and checks post-update finiteness. Checkpoint state digests include both decoder and hook state. Per-step state hashing remains opt-in and is disabled by the shared loop; checkpoint hashing is the normal path.

The synthetic directional hook test demonstrates that a trainable readout parameter receives an update and is included in clipping and state hashing. The runner retains explicit step-zero eligibility and earliest-strict-maximum checkpoint selection, and validation accepts an independently supplied sequence width for the H128 view. Domain-balanced validation aggregation remains an evaluator responsibility; the generic loop currently reports raw token totals and the caller-supplied metric.

## Agent 2 public-bank planning checkpoint

The bounded planning recommendation is a 12,000-row proportional expansion, pending the joint identity and capacity contract: 6,000 natural Alpaca, 3,000 natural Pile, 1,800 natural Finance, and 600 controlled contexts each for Pile and Finance. This preserves the published B0 construction ratio while providing 10x its 1,080 natural-parent support. The existing 1,200-row B0 must remain the byte-equivalent prefix. Controlled replacements remain in the inherited 128--191 post-BOS length stratum and four inherited position bins. No new validation pool is proposed; the natural `public_base` validation binding already opened by Agent 1 remains separate and root-owned.

A permitted public Alpaca count/tokenization audit rendered all 52,002 rows with the pinned public tokenizer. It found 50,778 rows after 1,224 keyed fit/validation identity exclusions, with 39,765/24,401/14,317/6,824 rows meeting post-BOS thresholds 64/96/128/160. These are count-only upper bounds: the exclusion metadata has a 4,193-value union containing 4,073 opaque sequence/reservation digests, and zero rendered-text SHA-256 hits against that union do not establish disjointness because the namespaces/canonicalizations differ. The audit receipt and exact command/environment provenance are `experiments/TRR-P09/planning/alpaca-capacity-audit.json` (SHA-256 `d1f11364cabe6db876bb83f09ce1d8d7db1cdc3f76ac2e8bafaed149ba33a5c9`) and `experiments/TRR-P09/planning/alpaca-capacity-audit-provenance.json` (SHA-256 `c9c6828e16d87a7316f36b10d7a832a7cd04c20a73c035b4178062e3926fc2d2`). The executed CLI is `scripts/trr_p09/audit_alpaca_capacity.py` (SHA-256 `db8b54c290a1ee5c8915a6c315c6969ee4cea3284da246f57d044b0cbbd8b80e`). The human proposal is `experiments/TRR-P09/planning/bank-proposal.md`.

No P09 source selection, public model forward, activation-bank capture, fitting, final-panel selection, or truth access has occurred. The deterministic selection recipe is documented only; it must apply exact keyed identities, namespace-matched digests, inherited Pile `[7000,10000)` and Finance `[12000,20000)` reservations, and length quotas before any later root-authorized selection.

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
