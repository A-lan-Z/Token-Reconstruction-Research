# TRR-P10 confirmation plan

This is a planning-only artifact. It authorizes no GPU, source selection,
observation capture, prediction, truth opening, retraining, or checkpoint
reselection. The machine-readable contract is
`coordination/parallel/TRR-P10.json`.

## Why PR20 cannot be the independent confirmation

The published PR20/TRR-0010 selection used the same deterministic source
range and order as TRR-0009. Metadata comparison gives the following exact
overlap:

| domain | TRR-0009 records | PR20 records | record IDs | public-record hashes | H128 sequence hashes | ordered prefix |
|---|---:|---:|---:|---:|---:|---:|
| Finance | 256 | 128 | 128 | 128 | 128 | 128 |
| Pile | 128 | 128 | 128 | 128 | 128 | 128 |

The Pile ordered record-ID and final-sequence fingerprints are identical for
all 128 records. The Finance PR20 panel is the first 128 records of the
TRR-0009 256-record panel under the same order. The prior score, predictions,
and decision receipts remain preserved, but those results are selection-
overlapping development evidence rather than an independent text-
generalization estimate.

## Frozen comparison and denominators

The prospective primary keeps the selected states unchanged: current-fixed at
step 8,000 (state SHA-256
`d2477cdf11422cb2d028196d72775825246f65f7e07394952375b3929627475a`) versus
expanded-fixed at step 13,000 (state SHA-256
`5757506003f0a4b82cb1cd6de2a82f11345f2dbcbce13c8cf18c4f515de3ec3a`). It
uses the two already registered target conditions, `public_base` and
`public_lora_2601`, on both Pile and Finance.

The panel target is 256 records per domain. Each record contributes 127
post-BOS positions, giving 32,512 token positions and 256 complete-record
units per domain-target cell. The four domain-target cells remain separate;
their scores and uncertainty are not pooled.

The frozen K256 A1+A2 comparator runs on the first 128 records per domain in
the same panel order, selected before observations and truth. Its denominator
is therefore 16,256 post-BOS positions and 128 complete records per cell. It
is a common-subset comparison, with no truth-driven subset selection.

## Source extension and exclusion contract

The previously selected panels cannot supply independent records because
their selected records overlap PR20/TRR-0009. The registered candidate
extensions are a deliberate fresh reservation: Finance rows `[20,000,
28,000)` and Pile rows `[0, 2,000)`, with a new deterministic selection seed
of 5011. This plan does not claim that the earlier ranges are exhausted.
Before any selection, the exclusion worker must apply every accessible fitting,
calibration, checkpoint-selection, development, and opened-evaluation ledger.
The TRR-0009 `selection_v2/source_selection.json` manifest is explicitly in
that set because it served as checkpoint-selection validation for PR20.

The audit compares dataset revision/split/row identity, record ID,
rendered-record SHA-256, canonical H128 final-sequence SHA-256, and the
published H129/truncated-sequence convention when available. It reports every
intersection count by ledger and field. A field that is unavailable is
recorded as unavailable; it is never silently treated as zero. If either
extension cannot provide 256 eligible records after all exclusions, the run
stops and a compatible extension is registered before any replacement is
selected.

Only opaque canonical hashes may cross the confirmation/diagnostic boundary.
Agent1’s diagnostic reservation must be disjoint from this panel by the same
hash checks. P03 remains root-owned and sealed; this task consumes no P03
payload.

## Curator and scoring boundary

The trusted producer may read public rows to construct identity metadata and
target observations. Prediction workers receive sanitized activations, masks,
positions, frozen states, and approved public resources. They receive no
source payload or IDs, target-prefix weights, or labels. The truth sidecar is
materialized separately and opened once only after every registered method’s
predictions and diagnostics are frozen.

A synthetic candidate copied from the TRR-0009 selection manifest must be
rejected before capture. The exclusion worker's regression records the
rejection reason and proves that the guard fires without opening a model,
observation, or truth payload. No confidence interval is an automatic
promotion gate, and no threshold or hyperparameter may be changed after
seeing evaluation outcomes.

After scoring, report paired token and complete-record inventories for both
correct, only expanded-fixed correct, only A1+A2 correct, and both wrong. Join
true-token rank, A1 proposal availability, A1+A2 direct-ranking failure,
B0/B1 direct-ranking failure, public fitting support bucket, and position only
where the immutable package preserves the relevant trace. A1 proposal failure
(true token absent from top-256) remains separate from A1+A2 ranking failure
(true token proposed but not selected). Missing traces are reported as
unavailable rather than reconstructed by a selective rerun.

## Runtime evidence and pending release

Prior measurements bound the primary frozen readouts at about 4.2 ms/source
and A1+A2 K256 at about 0.88 s/source. The 128-record/domain A1+A2 subset
across two target conditions is therefore about 450 seconds before setup.
Historical isolated CUDA peaks were 1.260 GiB for the primary readout and
3.303 GiB for A1+A2. These are planning bounds only. A release-time largest
cell qualification with a fail-closed guard and safety margin is required;
microbatching cannot be used as a semantic workaround without output
equivalence.

The plan remains pending Agent1’s immutable package/replay receipt, root and
Agent1 agreement on the exact extension and opaque reservation handoff, the
count-only exclusion audit, and explicit release of source selection and
compute.
