# TRR-0008 fixed 40-block precision follow-up

This is the single approved precision follow-up for the initial TRR-0008
alias-control diagnostic. It must be committed before execution and launched
only after an explicit compute release. The initial
`experiments/TRR-0008/timing/result.json` remains immutable and descriptive.
The follow-up output is a separate create-only artifact:
`experiments/TRR-0008/timing/precision40_result.json`. If the follow-up runs,
its receipt alone supplies the final prospective qualification.

## Fixed execution

Use exactly four repetitions of the original ten-block schedule. Each cycle
has five cyclic rotations of one fixed seed-8008 permutation followed by the
five reversals. The cycle is repeated in cycle order; no order is selected from
the initial timings. Every cycle is balanced per cell, so the 40-block result
has each method in each of the five order positions eight times per cell.

Keep all four already opened cells, the first 32 records per cell, all five
registered methods, one warmup invocation and one measured invocation per
record, the TRR-0007 prediction boundary, explicit pre/post synchronization,
the 600-second fail-closed resource guard, and the full 128-row exact
prediction-equivalence check before timing. Do not access truth, source text,
target labels, or candidate arrays. Do not change model states, numerical
settings, thresholds, or the initial receipt.

Pinned command (do not execute until the release condition is met):

```text
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts python3 scripts/trr0008_timing.py --device cuda --trr7-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007 --records-per-cell 32 --blocks 40 --warmup-runs 1 --maximum-seconds 600 --output /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0008/experiments/TRR-0008/timing/precision40_result.json
```

The initial run took 75.48 seconds for block execution and 89.10 seconds
overall. Four cycles are projected at approximately 315--320 seconds overall,
leaving margin under the existing 600-second guard. A guard breach, telemetry
failure, non-finite output, or prediction mismatch remains fail-closed. Preserve
any failure receipt and do not silently retry.

## Prospective decision rule

Apply this rule to the 40-block receipt only. For every cell, compute the
candidate/reference ratio for each block and its two-sided 95% Student-t
interval. For 40 blocks this uses df=39 and critical value
2.022690920036761. The candidate passes only when the upper endpoint is at most 1.25 in
every cell. It fails when the lower endpoint is greater than 1.25 in any cell;
otherwise it is inconclusive. The pooled four-cell ratio is descriptive only.

The identical-weight alias must first pass exact prediction equivalence and the
runtime control in every cell. Alias control passes only when every cell's CI
is fully contained in [0.95, 1.05]. It is a persistent-control failure when
any cell's CI is entirely outside that band; in that case the overall result is
`INVALID_ALIAS_CONTROL`. Any other non-pass alias result is `INCONCLUSIVE`.
Candidate cost qualification is valid only when alias control passes; therefore
an alias failure or inconclusive control can never be reported as a demonstrated
candidate budget failure. If the 40-block result remains alias-inconclusive or
alias-invalid, stop with that classification; no further timing expansion is
planned.

The initial ten-block result and this fixed 40-block result must be reported
separately. The follow-up is authoritative for the final prospective
qualification only because that role was fixed before execution; the initial
receipt remains the descriptive record of the original diagnostic.
