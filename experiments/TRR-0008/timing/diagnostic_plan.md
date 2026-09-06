# TRR-0008 timing diagnostic plan

This diagnostic resolves the TRR-0007 warmed-runtime ambiguity before any new
source records are scored.  It uses only the already opened public observation
manifest, the five frozen decoder states, and the archived TRR-0007 prediction
IDs.  It never reads a source sequence, truth sidecar, target label, or score.

## Fixed execution contract

The timed interval is the existing TRR-0007 row path, including CPU BF16
activation staging, FP32 current-H decoder execution, full-vocabulary
`127 x 128256` logits and finite-value validation, argmax, CPU ID transfer,
and prediction normalization.  A device synchronization occurs immediately
before the clock starts and immediately after the row call.  Model loading,
observation loading, embedding loading, output serialization, and the
equivalence check are reported separately and are excluded from warmed
latency.

The five paths are fixed:

1. `trr6__enriched_trained_diagonal_attention128` (retained native reference);
2. `current_enriched__trained_diagonal` (the registered same-weight alias);
3. `current_enriched__residual_mlp512` (current-bank residual);
4. `improved_public_bank__trained_diagonal` (new-bank diagonal control); and
5. `improved_public_bank__residual_mlp512` (primary candidate).

The alias is a loader and temporal-order control.  Its frozen state tensors are
byte-for-byte equal to the retained reference state, and its registered
positionwise current method builds the same `JointAffineAttentionDecoder`
class as the reference loader.  The residual wrapper is used only by the two
`residual_mlp512` paths.  This distinction is recorded so a warmed gap cannot
be attributed to a nonexistent zero-output MLP in the diagonal alias.

## Blocks and guard

The prospective pilot uses all four already opened cells, 32 records per
cell, ten blocked repetitions, one warmup invocation per record, and one
measured invocation per record.  The first five blocks use the five cyclic
rotations of one fixed permutation; the next five use the reversals of those
rotations.  The same schedule is offset by cell index, so every method occurs
twice in every order position for every cell.  The schedule is recorded before
execution and no order is selected after observing latency.  The default guard is fail-closed at 600 seconds,
16 GiB process RSS, 10 GiB host available memory, 8 GiB free GPU memory, and
6 GiB reserved GPU memory.  Any non-current GPU compute process, guard breach,
non-finite output, or missing GPU telemetry aborts the run and leaves a failure
receipt.

Before a timing block is valid, every one of the five methods is run over all
128 records in all four cells and compared with its archived TRR-0007
prediction tensor using exact `torch.equal`.  The alias is also compared with
the archived retained reference IDs.  A mismatch aborts before timing.

The resource bound comes from TRR-0007: approximately 1.3 GiB peak CUDA
reservation and 1.74 GiB peak RSS for 128-row method cells, inclusive of the
1.05 GB FP32 public embedding table.  Keeping all five decoder states resident
adds under 150 MB in the prior receipts.  The pilot's 32-row blocks are
therefore below the guard with ample margin; its expected warmed work is under
four minutes on the prior GPU, with a 600-second fail-closed ceiling.

## Prospective decision rule

For every cell and block, candidate/reference is the ratio of the summed
measured seconds for that cell.  A fixed two-sided 95% Student-t interval over
the ten block ratios is reported for each cell.  The candidate passes the
existing 1.25x warmed-runtime qualification only when the upper interval
endpoint is at most 1.25 in every cell; it fails when the lower endpoint is
greater than 1.25 in any cell; otherwise the timing decision is inconclusive.
The same per-cell calculation is reported for the alias, current residual, and
new-bank diagonal controls.  The pooled four-cell ratio is retained as
descriptive context only.  The identical-weight alias is valid as an order
control only when its exact prediction check passes and every cell's 95%
interval is fully contained in the fixed [0.95, 1.05] runtime-ratio band.  An
interval entirely outside that band is a persistent >5% alias deviation;
otherwise the alias control is inconclusive.  Startup and
per-record variability, block telemetry, peak memory, code/input/state hashes,
and the exact order schedule are retained in the result receipt.
