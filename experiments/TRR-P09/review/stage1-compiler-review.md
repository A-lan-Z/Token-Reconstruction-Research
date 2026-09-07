# TRR-P09 stage-1 compiler review

Status: **BLOCKED before public source selection**. This is a read-only review
of `scripts/trr_p09/prepare_stage1_inputs.py` and its capture adapter. No public
source rows were selected, no model or activation payload was loaded, and no
truth was opened.

The existing synthetic compiler tests pass (`7 passed` in
`tests/test_trr_p09_stage1_inputs.py`) and both task-local compiler/capture
modules pass syntax compilation. The following issues must be fixed and
covered by synthetic tests before running the compiler:

1. **Controlled-cycle cursor check is impossible.** B0 contributes one 3,600-ID
   cycle. The 1,080 additions contribute 32,400 occurrences, so the final
   cursor is 36,000. The current check compares it with `3,600 + 36,000`
   (39,600), and therefore rejects every valid 12,000-row compile. The receipt
   must also record the ordered occurrence/cycle digest and distinguish the
   immutable B0 cycle from the nine addition cycles.
2. **B0 controlled-prefix binding is incomplete.** The compiler verifies the
   B0 tensor geometry but does not compare the published synthetic rows'
   replacement positions and IDs with the frozen 3,600-ID list. When it builds
   `InputRow` objects, B0 synthetic rows leave replacement metadata empty, so
   the output sidecar cannot certify the inherited assignment. Preserve the
   published controlled traversal (Pile then Finance, descending target length,
   then slot), verify its 3,600 IDs/positions, and carry that metadata into the
   receipt without changing B0 tensors.
3. **Exact-length source allocation does not implement the signed slot order.**
   `select_candidates` iterates candidates in stable-key order and assigns each
   candidate the largest currently available length. The inherited constructor
   iterates target slots in descending target length and slot, then takes the
   first remaining source with at least `target + 1` full tokens. These orders
   can assign the same source set to different lengths. Bind the actual target
   slot order and emit the selected source's dataset ID, split, revision, row
   index, and stable key for reproducibility.
4. **Source and exclusion identities are not fail-closed.** Actual Arrow bytes
   are only `stat`-recorded; they are not hashed against the plan. Finance uses
   the signed `arrow_shard_sha256` field while the compiler reads
   `arrow_sha256`, producing `None` placeholders. The required TRR-0009 public
   development selection is silently skipped when absent. The approved ledger
   fallback resolves relative paths from the wrong repository root. Fix these
   before opening any source rows.
5. **Namespace extraction conflates identities.** The exclusion walker places
   `public_record_sha256` values into the rendered-source set and ignores
   numeric `source_indices`. The signed contract keeps public-record,
   rendered-source, sequence, and reservation namespaces distinct; source
   indices need a bound `(dataset, split, revision, row)` interpretation or a
   fail-closed missing mapping. Do not use one digest namespace to exclude
   another.
6. **B0 input copy and capture provenance need a byte-level check.** The output
   writer reconstructs the full 12,000-row tensors from active IDs/masks rather
   than copying and comparing the first 1,200 rows from the published tensor
   payload. Keep the inherited mask-aware padding/position convention, copy the
   B0 prefix, and compare a tensor-byte digest before publication. The capture
   adapter can then expose rows 1,200:12,000 as its 10,800 64-row shards; this
   adapter path is present, but it must consume the strengthened prefix and
   provenance receipt.

The review intentionally does not treat the existing synthetic tests as proof
of source capacity, namespace disjointness, B0 equivalence, or capture
readiness. Those require metadata-only fixtures for the checks above before
the authorized CPU selection step.
