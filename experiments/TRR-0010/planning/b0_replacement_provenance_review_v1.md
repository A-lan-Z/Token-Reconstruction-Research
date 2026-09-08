# TRR-P09 B0 replacement provenance review

This is a metadata-only audit. I read the signed stage-1 receipt from Git
commit `a6c07e7`, its declared preparation manifest, the declared public
`records.json`, the declared TRR-0007 legacy corpus plan, and the declared
stage-1 recipe. I did not read activations, hidden evaluation truth, or any
other study payload.

**Correction history.** The first version of this note incorrectly used the
legacy 2,000-ID plan comparison to characterize the B0-to-additions identity
set. That was the wrong comparison. The corrected conclusion below is based on
an executable aggregate over B0 and B1 controlled records. The earlier claim
of an identity-set or global multiplicity change is superseded.

The public metadata inputs are bound to these files:

- receipt: `/tmp/trr0010_stage1_receipt.json`, SHA-256
  `4314ee55f64f15e88e25319d0a80860a152a83b7d41b02f9539858aef23d0153`;
- preparation manifest: `/tmp/trr-p09-runtime/stage1-input-preparation-r1/preparation_manifest.json`,
  SHA-256 `c6d6915bb0e539336928725e9c32288157f15dbb96fe08544925e84cc40474e1`;
- public records metadata: `/tmp/trr-p09-runtime/stage1-input-preparation-r1/records.json`,
  SHA-256 `30fe4216fd3fb6b76e4aec79324fafd73a398ed901e07b4681b815cce2dc98a4`;
- aggregate receipt with no raw IDs:
  `experiments/TRR-0010/planning/b0_replacement_joint_aggregate_v1.json`,
  SHA-256 `6287fe214d6964b224e3473f8729f2056cbd464dc89e2d84cd09bb768707a9f0`.

## Corrected finding

The actual B0-to-B1 global token identity set and multiplicity are preserved:
B0 has 120 controlled rows (60 Finance and 60 Pile), 3,600 replacement
occurrences, and 3,600 unique token IDs, each once. B1 has 1,080 controlled
rows (540 per stratum), 32,400 occurrences, and the same 3,600 unique IDs,
each exactly nine times. The aggregate check
`B1_token_counter == 9 * B0_token_counter` is true.

The real discrepancy is the assignment structure. For each token whose B0
origin is Finance or Pile, B1 contains five occurrences in its B0 stratum and
four in the other stratum. The global support and counts therefore remain the
same, while the domain assignment changes.

The joint key
`(stratum, target_post_bos_token_count, replacement_position_after_bos,
replacement_token_id)` is not ninefold B0 repetition: B0 has 3,600 unique
joint keys, B1 has 32,290, and 35,499 union keys have counts different from
`9 * B0`. There are 3,209 B0-only and 31,899 B1-only joint keys. The ordered
B1 sequence is also not nine exact B0 cycles; only 20 of 32,400 tuple
positions match the expected repeated sequence. These are assignment/order
changes, not a new global token set.

The receipt's `b0_identity_cycle_matches_signed_recipe=false` and the
differing B0/signed-list digest fields establish that the signed recipe does
not certify the observed B0 ordered cycle, but those fields alone do not prove
an identity-set change because their canonicalization is not specified here.
The aggregate evidence above is the basis for the corrected conclusion. The
legacy 2,000-ID plan remains a historical-plan comparison and should not be
used as the B0-versus-B1 identity result.

## Feasibility of the narrow nested correction

The proposed correction remains feasible and is now more precisely motivated.
B0 has 60 controlled templates per stratum, and every template has a paired
position/token pattern. Grouped by `(stratum, target_post_bos_token_count)`
(the active length), B1 counts are exactly nine times B0 counts in all 70
groups:

| stratum | B0 templates | B1 controlled rows | B1/B0 by every active length |
|---|---:|---:|---|
| finance_controlled | 60 | 540 | exactly 9x |
| pile_controlled | 60 | 540 | exactly 9x |

A deterministic stable assignment can therefore copy each actual B0
template's paired `replacement_positions_after_bos` and
`replacement_token_ids` to exactly nine new controlled parents with the same
stratum and active length. This preserves the joint token/offset/length/domain
multiplicities that the current B1 assignment does not preserve, while keeping
the observed B0 prefix unchanged.

The narrow amendment is to reuse the actual B0 templates for additions, record
per-template assignment counts and a new occurrence/position digest, and avoid
substituting the signed 3,600-ID list or reselecting tokens. This review
performed no recompilation or mutation.
