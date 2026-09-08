# TRR-P10 — planning handoff

TRR-P10 is currently metadata-only. No GPU job, fresh source selection,
observation capture, prediction, or truth access has started.

The proposed independent primary is the frozen expanded-fixed step-13,000
state versus frozen current-fixed step-8,000, on Pile and Finance under the
two existing target conditions `public_base` and `public_lora_2601`. It uses
256 records per domain and 127 post-BOS positions per record: 32,512 token
positions and 256 complete records per cell. The frozen K256 A1+A2 comparator
uses the first 128 records per domain in that same pre-truth order, with
16,256 token positions and 128 complete records per cell.

The prior PR20 panel is not independent. Metadata shows 128/128 record-ID,
rendered-record, and H128 final-sequence intersections with TRR-0009 in both
domains; the Finance panel is the ordered first 128 records of TRR-0009’s
256-record selection, and the Pile panels are identical ordered sets. The
historical predictions and scores are preserved and reinterpreted as
selection-overlapping development evidence.

The registered source-extension candidates are Finance `[20,000,28,000)` and
Pile `[0,2,000)`, with seed 5011. The exclusion worker must first apply all
accessible fitting, calibration, checkpoint-selection, development, and
opened-evaluation ledgers, explicitly including the TRR-0009 selection
manifest, and must report zero matching intersections before capture. If the
extensions cannot provide the requested 256 eligible records per domain, a
compatible extension must be registered before any replacement is selected.

The full curator boundary, paired error inventory, candidate-failure taxonomy,
and historical runtime/memory basis are in
`experiments/TRR-P10/planning/confirmation_plan.md` and
`coordination/parallel/TRR-P10.json`. Release remains pending the immutable
Agent1 package, opaque reservation agreement, exclusion audit, and root’s
explicit compute release.
