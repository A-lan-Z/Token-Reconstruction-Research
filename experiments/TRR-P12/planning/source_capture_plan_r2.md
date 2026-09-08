# TRR-P12 source and capture plan (r2)

This record fixes the bounded P12 study before fresh selection. It follows one
actual public target-training trajectory for the restored B1 decoder and uses
the same source order, cut, rendering, and numerical execution at every
snapshot. The planned public adaptation is a LoRA update on layers 0--3 with
rank 8, alpha 16, seed 6012, batch size 2, and sequence length 128. The
trajectory stages are the baseline plus 64, 128, and 256 updates.

## Current release gate

No fresh P12 selection, target training, target capture, GPU execution, or P03
access has started. The root must release the completed opaque exclusion union,
the Agent1/TRR-0013 reservations, the fitting-source ledger, and the
reconstruction-evaluation ledger before any public row is selected. The
create-only curator is `scripts/trr_p12/sources.py`; it binds the P11 r14
identity union, r17 audit, root selection release, confirmation512 ledger, and
transfer64 ledger by verified path, byte count, SHA-256, and known producer.
Its output is `experiments/TRR-P12/exclusions/identity_union_extension_r2.json`.

The curator emits only the approved P11 identity fields and dataset namespace:
record ID, source index, rendered and tokenized record hashes, H40/H128/H129
sequence hashes, and the two declared TRR-0002 digest fields. It skips source
text, token IDs, labels, activations, logits, and weights. The r2 union remains
append-only: Agent1 reservations and P12 fitting/evaluation identities are
added as separate hash-bound layers after root release.

## Training and evaluation sources

Use the cached public `tatsu-lab/alpaca` revision
`dce01c9b...` recorded in the root inventory. Reserve 256 records for target
fitting and 64 records for validation from the training inventory. These
records are excluded from every opened P11 ledger and from Agent1's 512-record
Alpaca reservation. Root must freeze the exact opaque ranges and their SHA-256
ledger before training; the fitting and validation rows never enter the
reconstruction evaluation panel.

Use a fresh reconstruction panel of 128 records per declared domain, with 64
development and 64 held-out records per domain. The four trajectory snapshots
use identical record identity, ordering, rendering, tokenizer, positions,
attention mask, cut, and numerical path. Selection uses seed 5012 and occurs
once before any stage-specific output is observed. It is disjoint from the
P11 union, Agent1's reservations, target-fitting rows, and validation rows.

The stage pairing is part of the source contract: a record is either present
at every released stage or excluded before capture. No stage output, target
loss, target label, decoded text, token-level correctness, or weight change
may influence source selection or the final reconstruction.

## Adaptation trajectory

The target training job is one reproducible public run with the frozen
architecture and tokenizer. Bind the exact base checkpoint, LoRA insertion
points, optimizer, learning rate, scheduler, update count, seed, and changed
parameter hashes before starting. Preserve the baseline and create-only
snapshots at updates 64, 128, and 256. Record whether each update changes the
weights materially, but do not use evaluation outputs to pick a stage.

Target fitting and its validation source truth remain evaluator-only. The
reconstructor receives the frozen decoder and the permitted target
observations; it never receives target weights, true tokens, source text, a
target-prefix query, target labels, or a loss derived from source truth.

## Capture contract

Reuse the validated P11 public capture contract through a P12-owned wrapper.
The common full-target forward geometry is batch 8, sequence length 192 with
padding, capture at cut 4, and BF16 hidden states. Retain 128 positions from
each record at the cut and store the matching attention mask and position IDs.
The H observation tensor necessarily exists as the reconstruction input; the
source/training side of the capture path remains metadata/hash-only to the
reconstruction process. Raw source text and token IDs are transient
evaluator inputs and are never serialized or handed to the reconstructor.

The wrapper records only stage, opaque record slot, shape, dtype, ordering,
mask/position metadata, timing, resource samples, and artifact hashes beside
the H observations. It does not write target-prefix outputs beyond the
declared H observation, target weights, evaluator labels, source text, or
token IDs. `trr_p12/sources.py` owns ledger binding; the planned
`trr_p12/capture.py` owns hash-bound capture serialization. The validated
references are `scripts/trr_p11/public_capture.py` and
`scripts/trr_p11/source_selector.py`; the untracked P11
`prospective_pipeline.py` is excluded.

Run a small CPU smoke before any larger capture. The smoke checks the stage
pairing, fixed geometry, ordering, hash-only sidecars, create-only outputs,
and absence of forbidden payload files. It does not qualify the largest GPU
cell. Any GPU execution requires a separate root-owned live preflight, a
measured memory/time margin, an isolated restart-safe job, and a fail-closed
resource guard. If microbatching is considered, prove output equivalence to
the canonical path first and exclude any non-equivalent result.

## Early direction predictor

The proposed predictor uses only the baseline-to-64-update displacement and
the decoder decision-boundary features available at that time. It is frozen
before any 128- or 256-update observations, reconstruction outputs, or truth
are opened. The development portion (64 records per domain) may set a
threshold using the u0/u64 transition. The held-out portion (64 records per
domain) evaluates the frozen predictor on the u0/u64 transition and on the
later u0/u128 and u0/u256 transitions, which are update stages held out from
predictor choice. This checks whether early movement predicts later failures
without using future update observations to choose the rule.

Report absolute reconstruction quality at every stage and paired changes in
three categories: baseline errors, previously correct records broken by the
update, and errors improved by the update. A large unsigned movement is not
called a boundary crossing without the declared boundary test. The limited
common A1+A2 comparator is included only if its inputs are already released;
it cannot replace a blocked comparator package.

All candidate generation, adaptation, routing, stopping, timing, and final
reconstruction outputs are frozen before target truth is opened. Truth is
then used only for declared metrics.

## CPU and evidence gates

1. Root verifies the r2 P11 opaque union and appends Agent1, fitting, and
   evaluation reservations without rewriting the P11 base.
2. Root preregisters the single LoRA trajectory and all disjoint opaque source
   ledgers, including the 256 fitting and 64 validation rows.
3. The CPU smoke validates source pairing, H observation geometry, sidecar
   serialization, and forbidden-file absence.
4. Root performs the separate GPU live preflight and qualifies the largest
   representative cell before releasing the matrix.
5. All four snapshot captures and reconstructions are frozen before truth.
6. The result reports stage metrics, paired changes, resource evidence, and
   held-out early-predictor performance, with failed/excluded cells and
   artifact hashes.

The conclusion is limited to this one trajectory. No broader target sweep or
decoder adaptation starts automatically.
