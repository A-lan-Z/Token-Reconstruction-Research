# TRR-P09 stage-1 compiler receipt audit

**Disposition:** `R1_EXCLUDED_PRE_CAPTURE_B0_ORDER_CONTRADICTION`.

This is a metadata-only audit of the completed CPU input preparation. It is
not a second scientific gate. The compiler read public source rows and
tokenized them, but loaded no model, created no activations, opened no target
observations, and opened no evaluation truth.

## Receipt checks

Receipt: `experiments/TRR-P09/setup/stage1-input-preparation-r1-receipt.json`
(SHA-256 `4314ee55f64f15e88e25319d0a80860a152a83b7d41b02f9539858aef23d0153`).
The guarded CPU run completed with return code 0 in 32.387874 seconds; peak
group RSS was 1,647,108,096 bytes and the watchdog status was `PASS`.

The receipt has the expected 12,000-row, width-192 geometry: a 1,200-row B0
prefix and 10,800 new rows, with 64-row logical shards. The five row quotas
are exactly 6,000 Alpaca natural, 3,000 Pile natural, 1,800 Finance natural,
600 Pile controlled, and 600 Finance controlled. The compiler reports
`N=13,000`, 45,621 supported token IDs, and 1,243,710 post-BOS positions.

The controlled allocation is feasible under the proposed addendum: B0 has 120
controlled rows (60 Pile and 60 Finance), while the new rows have 1,080
controlled rows (540 per domain). Public metadata shows that every
`(stratum, target_post_bos_token_count)` count in the new controlled rows is
exactly nine times the B0 count. Thus a deterministic nine-way template
repetition can preserve the domain, active-length, and inherited position-bin
allocation without changing the natural-parent selection.

Other receipt checks pass for this purpose. The compiler verified all bound
Arrow-file hashes, preserved the mask-aware B0 tensor convention, emitted
separate 64-row diagnostic hashes for the current and expanded banks, and
reported the namespace-specific exclusion counts (1,928 H128, 1,928 public
record, 1,704 record-ID, 1,200 rendered, 384 source-index, and 10 source-file
entries). A metadata-only overlap audit found zero B0/B1 public-record-hash
overlap and zero overlap with the bound TRR-0009 source IDs or dataset-scoped
source indices. These checks do not replace the signed exclusion ledgers.

One additional exclusion discrepancy is concrete. The earlier public Alpaca
capacity audit bound `experiments/TRR-0007/support/broader_bank_v5/public_parent_exclusion_manifest.json`
(SHA-256 `bd1359f1184091570023e22a7682d1f97c08f8f05e47f69f6b3e6be089cd0181`)
and found 1,224 dataset-scoped Alpaca row keys: 600 B0 rows plus 624 prior
development rows (the latter are the non-B0 remainder of the TRR-0004 fit and
validation metadata). The r1 compiler scan reports only 600 Alpaca metadata
exclusions and zero hash exclusions. Comparing public source indices shows
that 63 of those 624 prior-development rows are present in the r1 Alpaca B1
additions. Thus the old development ledger was not fully applied; this is a
second reason r1 cannot be captured, and it must be bound and replayed in r2
with the namespace-correct exclusion receipt.

## Blocking B0 discrepancy

The signed plan says to “reuse frozen ordered published 3600-ID list in ten
deterministic cycles, preserving B0 first cycle.” The receipt instead records
`PUBLISHED_B0_ACTUAL_IDS_RETAINED_SIGNED_B1_SEPARATE`: the published B0 tensor
provides the first 3,600 occurrences and the signed list is used for each of
the nine B1 addition cycles. The receipt therefore correctly reports
`b0_identity_cycle_matches_signed_recipe=false`.

The discrepancy is order-only, but it is scientifically relevant. A public
metadata/tensor-ID comparison found:

* both sequences have length 3,600 and contain 3,600 unique IDs;
* their multisets and per-ID multiplicities are identical (maximum
  multiplicity difference 0);
* their order differs, with the first mismatch at occurrence 0 (`624` in the
  observed B0 sequence versus `17065` in the signed sequence);
* the observed B0 occurrence digest is
  `8499eb0e0706b6a8c74a258d09857336e135eb345bbc3603c6940a23a991d56e`.

Because replacement IDs are attached to concrete row/position occurrences,
an order permutation changes which token is injected at which offset even
when aggregate token counts are unchanged. The r1 input file must therefore
remain excluded from capture under the signed plan; it must not be silently
accepted as equivalent.

## Required narrow addendum

Before any model capture, root and Agent 1 should sign a create-only addendum
that binds the observed B0 occurrence digest and offset digest
(`3503f7d8c66660a9e3746b28c49e7400e6dc72bfb25b33cb29896bf1a5b0fbd6`) as the
immutable published prefix. It should define the B1 controlled additions as
the 120 observed B0 controlled-row templates, each with its 30 offset/token
pairs, repeated exactly nine times on fresh parents. Assignment should be a
deterministic stable order within exact dataset stratum and active length,
preserving the inherited position bins; the compiler must record the row to
template assignment and its digest. The public metadata audit above shows
that the required nine-to-one row allocation exists in every stratum/length
cell.

The amended receipt should report the actual B0 sequence as cycle 1, nine
template-derived B1 cycles, the resulting 36,000 occurrences, and all source,
plan, exclusion, and input hashes. It must also bind the public Alpaca
exclusion manifest above and show zero overlap for all 624 non-B0 prior-development
rows (including the 63 rows that slipped into r1). The r1 public source scan
can be retained as an excluded attempt; no model, activation capture, fit, or
truth access is permitted until the addendum and a fresh CPU receipt validate
this mapping.

The B0 records do not carry the same `public_record_sha256` and H128 sidecar
fields as B1 rows. The separate overlap audit found no duplicate, but this is
a provenance limitation that the addendum/r2 receipt should state explicitly;
it is not evidence that the signed exclusion namespaces may be merged.
