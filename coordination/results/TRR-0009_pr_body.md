# TRR-0009 PR body draft

**Base/task:** `task/TRR-0008`

## Result

TRR-0009 freezes and records a bounded readout-adaptation mechanism-screening pilot. It binds the root-approved decision contract and selected method states, applies the approved P08 hash-only exclusion exchange, verifies final eligible capacity, and selects 256 Finance and 128 Pile natural records with paired target conditions. The first identity-only panel is preserved and excluded before capture after one Finance H128 overlap with P08 was detected; corrected `selection_v2` applies the missing P08 source/H128 classifier inputs and passes all approved-ledger intersection checks. No source text, target labels, predictions, or truth are included in the selection artifacts.

Exact human-readable result: `coordination/results/TRR-0009.md`

Exact structured manifest: `experiments/TRR-0009/manifest.json`

## Bound evidence

- Final inventory: `experiments/TRR-0009/planning/source_inventory_final.json` — 4,123 eligible Finance and 364 eligible Pile records.
- Excluded original selection: `experiments/TRR-0009/selection/source_selection.json` and `source_exclusions.json`; one Finance H128 overlap with approved P08, no capture use.
- Corrected source selection: `experiments/TRR-0009/selection_v2/source_selection.json` — 256 Finance + 128 Pile, identity-only, zero P08/P06/TRR7/TRR8 overlaps (P04 source overlap zero; P04 H129 unavailable from selected ledgers).
- Corrected exclusions and reservation: `experiments/TRR-0009/selection_v2/source_exclusions.json` and `opaque_source_sequence_reservation.json` — 384 public-record and 384 H128 sequence hashes, hash-only.
- Repair audit and exchange receipt: `experiments/TRR-0009/selection_v2/exclusion_repair_audit_v2.json` and `experiments/TRR-0009/coordination/p08_replacement_exchange_ack.json`.
- Loader qualification: `experiments/TRR-0009/evaluation/loader_qualification_v2/qualification.json` — fixture equivalence PASS.

The 128-line P08 reader compatibility adapter accepts the exact producer-authored sanitized hash schema and preserves the legacy P06 shape. The selector repair passes P08 source/H128 sets through the existing classifier, asserts zero postselection overlap, and writes corrected outputs in the isolated `selection_v2/` namespace. These changes affect no population, range, exclusion policy, sampling, model, or decision rule. The initial unsupported-schema attempt and the original one-overlap panel are retained as excluded evidence. The v2 selection executed from `64aa610` with the seven-line output guard change, published as `55ef2d4`.

## Validation

- `tests/test_trr0009_selection.py`: 15 passed after corrected artifacts were generated.
- Inventory and corrected selection completed identity-only with truth unopened.
- Replacement reservation was independently verified by A2; original and replacement reservations have a 385-hash union in each field.
- Capture, prediction, and timing are complete before truth; fresh quality scoring and the prospective decision remain pending metadata-gate compatibility review.

## Final findings (fill after capture and scoring)

- Capture and resource receipt: `COMPLETE — 16.958598 s; bind exact receipt in manifest.`
- Four-method prediction matrix: `COMPLETE_BEFORE_TRUTH — 145.870593 s; per-cell denominators and latency table are bound in the report/manifest.`
- Timing and alias-qualified cost: `PASS — 235.260320 s precision receipt; all four candidate/fixed cost cells and alias controls pass.`
- Quality score and decision: `PENDING_METADATA_GATE — truth remains unopened until the compatibility repair is reviewed.`
