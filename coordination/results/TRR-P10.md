# TRR-P10 — recovery handoff

TRR-P10 is currently preparation-blocked after execution-host recovery. No
new GPU job, fresh source selection, observation capture, prediction, truth
access, or final scientific claim occurred during recovery.

The durable task/TRR-P10 branch survives at
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P10
with the confirmation plan commit 30afcad and execution-adapter commit
f1eb31f. The incoming amendment is preserved byte-for-byte. Agent 1's
immutable inference package and published eight-cell CUDA replay metadata are
available as read-only inputs: package SHA-256
a74f687f8287cacfeafa0acc81a97f208672dedf797792c60f88ba93b0318b25 and
replay receipt SHA-256
79fda7559f0e4e426227d5af0046dd35f6b6b1db721d2d25b87895947c9cb234. The
replay receipt reports exact results for all eight GPU FP32 cells; it was not
rerun during recovery.

The runtime worker also preserved a bounded CPU test/inventory checkpoint at
commit d3a89b6111b6f74badd589716d977fb133e0ee89. Its execution-test receipts
cover the UTC orchestration interval 04:07:19–04:07:44, with tool wall time
3.312702983 seconds and pytest time 2.12 seconds. The interval encloses
orchestration observations; no exact process wall time is inferred from it.

The proposed independent primary remains the frozen expanded-fixed
step-13,000 state versus frozen current-fixed step-8,000, on Pile and
Finance under the two existing target conditions public_base and
public_lora_2601. It uses 256 records per domain and 127 post-BOS positions
per record: 32,512 token positions and 256 complete records per cell. The
frozen K256 A1+A2 comparator uses the first 128 records per domain in that
same pre-truth order, with 16,256 token positions and 128 complete records
per cell.

The prior PR20 panel is not independent. Metadata shows 128/128 record-ID,
rendered-record, and H128 final-sequence intersections with TRR-0009 in both
domains; the Finance panel is the ordered first 128 records of TRR-0009's
256-record selection, and the Pile panels are identical ordered sets. The
historical predictions and scores are preserved and reinterpreted as
selection-overlapping development evidence.

The registered source-extension candidates are Finance [20,000,28,000) and
Pile [0,2,000), with seed 5011. Exclusion-audit and selector preparation must
be rebuilt from preserved metadata before any source selection. The missing
B0/B1 state files must also be reverified by their published hashes. The
published public embedding dependency remains available at its existing
hash-verified path. No old run receipt is being recreated.

The full curator boundary, paired error inventory, candidate-failure taxonomy,
and historical runtime/memory basis are in
experiments/TRR-P10/planning/confirmation_plan.md and
coordination/parallel/TRR-P10.json. Release remains blocked by the missing
exclusion/selector preparation, B0/B1 state re-verification, the opaque
reservation agreement, and root's explicit compute release.

## Recovery checkpoint — 2026-09-08

The durable task/TRR-P10 branch and its two preparation commits were
recovered into
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P10.
The former /tmp/trr-p10 checkout and /tmp/trr-p10-runtime are missing.
The retained old Git worktree registration and index were inspected
read-only; the index contains the committed tree and no staged additions for
the known uncommitted exclusion, selector, or capacity files. No historical
receipt was recreated from a summary.

This recovery is therefore preparation-blocked. The known exclusion and
selection files, capacity receipts, and runtime files are recorded as
unlocated in the recovery receipt
experiments/TRR-P10/recovery/git_worktree_recovery_r1.json. The B0 and B1
state paths and published hashes are retained as identities, but their
current files are unlocated and not proven destroyed. The published public
embedding dependency remains referenced by its existing path; no replacement
was generated.

No new source selection, observation, prediction, truth access, model run,
or final scientific claim occurred during recovery. P03 remains sealed and
Agent 1’s worktree was not modified.

## Publication checkpoint — 2026-09-08

The recovery handoff is published on branch task/TRR-P10 at commit 7eb5de3.
Draft PR22 (https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/22)
targets task/TRR-P09, remains open and unmerged, and records the recovery
receipt, task-local manifest, result, and state. This publication checkpoint
does not change the preparation-blocked status or authorize source selection,
model execution, prediction, or truth access.

## Final preparation validation — 2026-09-08

The latest exclusion audit is
experiments/TRR-P10/exclusions/recovery_identity_audit_r4.json, SHA-256 8efd27d30fc2b8898fd6df7cc64174b8a666c0c3cdc430d799c9773ebfa8a0e4. Its status remains
PARTIAL_METADATA_ONLY_EXCLUSION_AUDIT after 48 source bindings. The matching
validation receipt is experiments/TRR-P10/exclusions/recovery_validation_r4.json, SHA-256 18cfe79841a8fec6beaf6003723064007af5140c1c175c3c62f9ced79607b9b8; its final
suite reports 13 passing checks in 0.75 seconds. The r4 audit supersedes the
preserved r1–r3 audit artifacts.

The remaining audit gaps are explicit: aggregate opened-panel and
pointer-link coverage is unresolved, and P04 fingerprint producer mappings
remain unverified. The selector has not been rebuilt or released, so this
audit does not authorize source selection or a zero-overlap claim for the
future panel.

The recoverable execution guard is
experiments/TRR-P10/recovery/execution-tests-r2.json, SHA-256 191d08f38c3619978eebc87afa05637d26d05ff68c26230a071b1557da1d25f3. It reports 10 passing checks and no
failures, including the bound post-freeze scoring artifacts. The exact B0 and
B1 state files remain unlocated and are the primary readiness blocker. No new
source selection, observation, prediction, model run, or evaluation truth was
opened.
