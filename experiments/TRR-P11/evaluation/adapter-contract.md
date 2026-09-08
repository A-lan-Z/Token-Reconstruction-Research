# TRR-P11 evaluation adapter contract

`scripts/trr_p11/evaluation.py` is the pretruth adapter for the restored
TRR-0012 package. It keeps the restored package output files and their sibling
receipts in place and binds them by file hash plus an explicit safetensors
`tensor_key`; it does not reserialize predictions.

The input document uses schema
`token-reconstruction.trr-p11-evaluation-manifest.v1` and status
`PREDICTIONS_READY_NO_TRUTH`. It contains hash-bound JSON files for `selection`,
`capture`, and the P11 restore receipt, one observation binding for each of the
four cells, explicit source-order hashes, state/model identities for B0 and B1,
and `trace_files` plus `cost_files`. Every prediction cell has an output file,
its exact tensor key, tensor digest, row count, and a sibling
`token-reconstruction.trr0012-prediction-receipt.v1` binding. The primary keys
are `current_fixed` and `expanded_fixed` and must have 256 rows by 128 stored
tokens. The A1 comparator has 128 rows and must declare indices exactly
`[0, ..., 127]` with the P10 `indices_digest`.

The comparator is required for a complete comparison. A technical block may be
recorded only as `comparator.status=BLOCKED_TECHNICAL` with
`preregistered=true`, a nonempty `blocker_id`, and a concrete reason. Such a
freeze is written as `QUALIFIED_PARTIAL_A1_BLOCKER`; the bound truth sidecar
may still be opened once and the B0/B1 primary result may be scored. The score
retains `a1_diagnostics.status=UNAVAILABLE` and never presents a full matrix.

`freeze_predictions` reopens and rehashes every selection, capture, restore,
observation, state, prediction, sibling receipt, trace, and cost binding. It
writes a create-only freeze with `truth_opened=false`. Before the one truth
sidecar is opened, `load_truth_after_freeze` repeats that complete validation,
checks the truth descriptor's freeze hash, and checks the sidecar hash and
geometry. `score_after_truth` then builds the P10 paired inventory and calls the
registered byte-identical scorer at
`scripts/trr_p11/scorer/trr0010_analysis.py` (seed 9009, 10,000 draws,
one-sided alpha 0.025) separately for each cell. Domains and target conditions
are never pooled.

The score report exposes primary and comparator position strata separately, hash-bound cost receipts with parsed numeric fields, and the historical P07 support definition as `PENDING_FROZEN_FIT_SUPPORT_BINDING` until the common frequency reference and native B0/B1 fit position/attention-mask counts are bound. It never derives support from post-truth target rows. The current adapter and tests only use synthetic tensors. They do not access
source text, a model, public E, a P03 holdout, or evaluation truth from the
research run.

The restore receipt must identify `source_boundary=secondary` both at the
receipt root and in its `retrieval` record for the agreed Windows persistent
copy materialization. It must explicitly report
`training_worktree_import=false`; its consumer receipt must explicitly report
both `training_worktree_import=false` and `temporary_dependency=false`. Its
`assets` object must retain the primary and secondary copy records for all deployment roles, while
`tensor_identity` must bind B0, B1, and public E to those asset hashes. Its
`consumer_receipt` must expose the actual loaded state/readout bindings and
bundled code paths and must report no training-worktree import or temporary
dependency.
