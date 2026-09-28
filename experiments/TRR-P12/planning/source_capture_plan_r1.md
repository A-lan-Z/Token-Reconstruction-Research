# TRR-P12 source and capture plan

This plan covers the bounded P12 study of one actual public target-training
trajectory. The restored B1 decoder remains frozen. The study will compare a
baseline checkpoint and a small number of checkpoints from that single
trajectory using the same public sources, cut, rendering, and numerical path
at every snapshot.

## Release status

This is a design and exclusion-boundary record. No fresh source selection,
target training, target capture, GPU execution, or P03 access has started.
Selection stays blocked until the root preregistration release binds the
target trajectory, fitting-source ledger, reconstruction-evaluation ledger,
Agent1's reservations, and the completed opaque exclusion union.

The additive union is recorded at
`experiments/TRR-P12/exclusions/identity_union_extension_r1.json`. It binds
the P11 r14 identity union, the P11 r17 audit and root-release receipt paths,
and the packet-authorized opened P11 confirmation512 and transfer64 ledgers by
path and SHA-256. It contains no identity rows or source values. The two P11
receipt hashes still need to be copied from the root release record before the
union is frozen.

## Study unit and snapshots

Use one compatible public fine-tuning trajectory with a frozen restored B1
decoder. The trajectory must have a baseline and a modest sequence of
checkpoint stages, with the exact checkpoint identifiers and changed
parameters recorded before capture. A practical default is four snapshots:
baseline, early update, middle update, and final selected update. Fewer
stages are acceptable only if the public trajectory does not expose the
intermediate checkpoint; more stages require a written reason and the same
common-source binding.

Target-fitting sources are separate from reconstruction-evaluation sources.
The target prefix and target-training source truth stay evaluator-only. The
reconstructor receives the frozen decoder, declared public metadata, and
permitted activations, never target weights, true tokens, source text, or a
target-prefix query.

## Public source panel

After preregistration release, select a modest fresh panel of 128 records per
declared domain unless the released public pool or the exclusion audit makes
that size infeasible. If a smaller panel is required, record the available
pool size, the exclusion reason, and the resulting precision before selection.
Reserve paired records across all snapshots so that every snapshot uses the
same source identity, source ordering, tokenization, positions, attention
mask, cut, rendering, and numeric execution. Do not select a record using any
snapshot-specific output or evaluator truth.

The public panel must be disjoint from the P11 union and from Agent1's
TRR-0013 reservations. The target-training fitting sources must also be
disjoint from this reconstruction panel. The final source ledger will expose
only opaque IDs or hashes and the ledger SHA-256 to downstream agents.

## Capture path

Reuse the validated P11 public capture contract through a task-owned wrapper.
The planned geometry is the P11 common full-target forward path: batch 8,
sequence length 192 with padding, capture at cut 4, and retain 128 records in
BF16. The wrapper will preserve canonical public rendering and record stage,
record identity, shape, dtype, ordering, mask, position IDs, timing, and
artifact hashes. It will not write source text, token IDs, target weights,
target-prefix outputs, or evaluator labels.

Before a larger capture, run a CPU smoke cell that proves the output schema,
record pairing, fixed cut, and hash-only serialization. Estimate peak memory
from the smoke geometry and retain a safety margin. If batching or
microbatching is considered as a resource workaround, first perform an
output-equivalence check against the canonical path; exclude any
non-equivalent attempt.

The planned task-owned modules are `trr_p12/sources.py` for source-ledger
validation and `trr_p12/capture.py` for the metadata/hash-only capture
wrapper. `scripts/trr_p11/public_capture.py` and `source_selector.py` are
references for the validated public contract. The untracked P11
`prospective_pipeline.py` is excluded from the implementation.

## Predictor check

The study may test a failure predictor based on movement relative to the
actual decoder decision boundaries. Compute paired baseline-to-stage changes
only after all stage captures and frozen reconstruction outputs are complete.
Separate baseline errors, previously correct records broken by the update,
and errors improved by the update. A large unsigned displacement is not
treated as a boundary crossing without a declared boundary test.

Choose predictor thresholds or features from a development subset and check
them on held-out records that were not used for the update or predictor
choice. Maintain paired source identity across snapshots, and report absolute
quality plus paired changes for every stage. Include the limited common A1+A2
comparator only if its inputs and exclusions are already released.

## Evidence to emit

For each stage, retain create-only metadata sufficient to reproduce the
capture: exact code commit, command, environment/dependency identifier,
public checkpoint identifier, source-ledger identifier, seed, hardware,
start/end timestamps, preparation/adaptation/inference/synchronization/I/O
times, peak RSS, tensor shape and dtype, candidate/evaluation counts, output
metrics, and SHA-256 values for every opaque artifact. Write any failed or
excluded cell with its reason. Open target truth only after reconstruction
outputs and all candidate, timing, routing, stopping, and model choices are
frozen.

The conclusion is limited to this one trajectory. Do not launch a broader
target sweep or decoder adaptation automatically after the first result.

## Gates

1. Root releases the completed P12 exclusion union and Agent1 reservations.
2. Root preregisters one compatible public trajectory and its disjoint fitting
   and reconstruction ledgers.
3. CPU smoke validates the task-owned hash-only capture wrapper and common
   snapshot contract.
4. The largest representative CPU cell passes the resource guard with a
   measured margin.
5. Capture and reconstruction outputs are frozen before any truth opening.
6. Pairwise stage analysis and the held-out predictor check are reported in
   the P12 result and manifest.
