# TRR-P09 B0 replacement provenance review

This is a metadata-only audit. I read the signed stage-1 receipt from Git
commit `a6c07e7`, its declared preparation manifest, the declared public
`records.json`, the declared TRR-0007 legacy corpus plan, and the declared
stage-1 recipe. I did not read activations, hidden evaluation truth, or any
other study payload.

The receipt and manifest are bound to these public metadata files:

- receipt: `/tmp/trr0010_stage1_receipt.json`, SHA-256
  `4314ee55f64f15e88e25319d0a80860a152a83b7d41b02f9539858aef23d0153`;
- preparation manifest: `/tmp/trr-p09-runtime/stage1-input-preparation-r1/preparation_manifest.json`,
  SHA-256 `c6d6915bb0e539336928725e9c32288157f15dbb96fe08544925e84cc40474e1`;
- public records metadata: `/tmp/trr-p09-runtime/stage1-input-preparation-r1/records.json`,
  SHA-256 `30fe4216fd3fb6b76e4aec79324fafd73a398ed901e07b4681b815cce2dc98a4`.

## Finding

The discrepancy is a real identity-set change, with an ordering/provenance
mismatch as well. It is not only a canonical digest or ordering difference.
The signed receipt explicitly records
`b0_identity_cycle_matches_signed_recipe=false`. Its actual B0 replacement
digest is `8499eb0e0706b6a8c74a258d09857336e135eb345bbc3603c6940a23a991d56e`,
whereas the signed 3,600-ID recipe reference is
`0e84f625e24fb4a2038972cf417923dc20a8cc2fd85e55ac6e9b7a846f5ca576`.

From the public records, B0 contains 120 controlled rows: 60
`finance_controlled` and 60 `pile_controlled`. Every row has 30 paired
replacement positions and token IDs, giving 3,600 B0 occurrences and 3,600
unique B0 replacement IDs, each occurring once. The legacy plan declared in
the receipt contains 2,000 unique selected IDs. The actual B0 set contains all
2,000 legacy IDs plus 1,600 additional IDs; it has no legacy-only IDs. Thus the
signed 3,600-ID support list cannot be treated as the actual B0 ordered
replacement pattern.

The existing prepared records have 1,080 controlled B1 rows, 540 per stratum,
with 32,400 replacement occurrences. Each of the 3,600 actual B0 IDs occurs
nine times in B1 metadata. The B1 stream's first 3,600 occurrences are not the
B0 occurrence sequence, so this multiplicity fact does not repair the missing
B0 template binding.

## Feasibility of the narrow nested correction

The proposed correction is feasible from the declared metadata. B0 has 60
controlled templates per stratum, and every template has a distinct paired
position pattern within its stratum. Grouped by `(stratum,
target_post_bos_token_count)` (the active length), the B0 and B1 counts are:

| stratum | B0 templates | B1 controlled rows | B1/B0 by every active length |
|---|---:|---:|---|
| finance_controlled | 60 | 540 | exactly 9x |
| pile_controlled | 60 | 540 | exactly 9x |

There are 35 active-length groups in each stratum. For every group, the B1
row count equals nine times the B0 template count. A deterministic stable
assignment can therefore copy each actual B0 template's paired
`replacement_positions_after_bos` and `replacement_token_ids` to exactly nine
new controlled parents with the same stratum and active length. This preserves
the joint token/offset/length/domain multiplicities rather than only the
marginal 3,600-ID multiset.

The safe amendment is consequently to retain the observed B0 prefix and
construct only the additions by ninefold reuse of those 120 actual B0
templates. The compiler should record per-template assignment counts and a
new occurrence/position digest. It should not substitute the signed 3,600-ID
list, reselect IDs, or alter the B0 prefix. This review performed no
recompilation or mutation.
