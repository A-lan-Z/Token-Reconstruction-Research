# TRR-P01 independent scientific review design

This note fixes the minimum trustworthy diagnostic before any pilot score is
interpreted. It is a review artifact; it does not register a new active method
or alter the global state, registry, or dual-benchmark protocol.

## Boundary construction

Use the pinned public `meta-llama/Llama-3.2-1B-Instruct` revision
`9213176726f574b556790deb65791e0c5aa438b6`, cut depth 4, and the declared BOS
token `128000`. Construct one prototype for every tokenizer ID in the full
vocabulary of 128,256 IDs:

```text
b_v = public_prefix([128000, v])[position 1]
```

The table is preparation, not a free preparation step. Persist it in ascending
token-ID order in the public model boundary dtype (BF16 here), and record its
byte count (525,336,576 bytes for a dense 128256 by 2048 table), SHA-256,
generation time, and peak memory. Do not drop to a truth-informed candidate
subset. Run the public model in evaluation mode with fixed deterministic
settings; convert query and table chunks to float32 only for the distance
calculation, with no additional quantization. Resolve equal scores by smallest
token ID. Record the frozen table batch (256), alternate qualification batch (128),
record batch (8), and the exact output-equivalence result before accepting any
batch-size workaround.

The primary static rule is nearest prototype by cosine similarity after
float32 L2 normalization. Report raw L2 against the untouched public
input-embedding table as the secondary no-fit baseline, with the same stable
tie rule. Cosine and normalized squared L2 are mathematically equivalent, so
do not spend pilot scope on a redundant metric.

## Minimum diagnostic panel

Use one predeclared development panel for every arm. The frozen
`pilot_plan.json` selects 16 records from the pinned Pile-10k revision by a
seed-314159 permutation: the first four eligible rows in each ordered stratum
(code, numeric-plus-punctuation, unicode-plus-instruction, prose fallback),
requiring at least 39 non-BOS tokens, are truncated and prepended with BOS
128000. Every selected row therefore has exactly 40 valid tokens; no padding is
introduced. Record source provenance privately in the evaluator manifest; the
attack sees only opaque IDs and stage-local shape/mask/position metadata. Do
not change the panel after opening any truth.

Run the following in order, with the same record order and geometry wherever
the condition permits:

1. **First post-BOS identity.** Independently generate matched public
   observations for a deterministic 256-token probe set using `[BOS, v]` and
   check that the nearest full-vocabulary prototype returns `v`. This catches
   table/position/model-path errors before context is introduced.
2. **Matched multi-position context.** Generate observations from the public
   checkpoint on the fixed panel and score every valid post-BOS position. Break
   out position 1 versus positions 2--39, and report errors by position.
3. **Shifted target on identical records.** In an evaluator-only process,
   generate a second observation set with the same token sequences, masks,
   positions, and order under a public shifted-weight target (the verified
   Vikhr full-SFT revision is an available candidate). The attack process must
   not receive target weights, target callable code, source text, truth, or a
   condition label. Score both sets with the same frozen static configuration.
   Treat the paired shifted-minus-matched effect as a diagnostic, not as a
   replacement claim.
4. **Optional reference correction.** Only after the static arm is working,
   test the fixed public reference token ID `220` with
   `c_i = public_prefix(reconstructed_prefix_i + [220])[-1] - b_220`.
   Use only the already committed predicted prefix. Count one reference
   evaluation per scored position (and all cache calls); candidate simulations
   remain zero. This is a bounded context probe, not per-candidate search.

The inexpensive same-panel historical A1 control may use the frozen Alpaca
affine lens (SHA-256
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`) without
retraining. The available historical-A1 plus fixed-A2 comparator may use the published
`a1a2_43ea0bb737bc075531ca` rule: frozen Alpaca A1 proposals, fixed direct
K=256 public-A2 candidate simulations, no shortcut, gate, adaptivity,
centering, or abstention, and always commit the K=256 winner. This is an
explicitly labelled benchmark-compatible geometry port for the 16x40 panel,
with no retraining. It is a comparator control, not a new no-fit arm
or a canonical-complete method. The exact native strict-BOS
reference is tied to 128x128 inputs and a three-episode completed-record
adaptation contract, so it is `RUNNABLE_BUT_INCOMPATIBLE_WITH_DECLARED_CONDITION`
for this 40-token panel. Do not regenerate private historical traces, import
the reference as new method code, or treat its constants as a search space.

## Freeze and scoring review gate

The evaluator/target-builder and attack/reconstructor must be separate
processes and interfaces. The attack input contains only the sanitized
observation tensor, attention mask, position IDs, opaque record ID, public
model/table identity, and fixed configuration. The condition name should be
kept outside the attack input so matched and shifted runs cannot route on it.

Before truth is opened, the freeze receipt must hash and bind, in ordered form:

- the complete observation tensor artifact and its sanitized index;
- every opaque record ID, mask/position digest, and observation shape;
- each prediction sequence and its corresponding opaque ID/mask digest;
- method configuration, prototype table, public model, executable source, and
  runtime identity; and
- the prediction artifact's bytes and schema.

The scorer must reverify all hashes, one-to-one ID coverage, order, shapes,
finite values, and immutable file permissions before reading the private truth.
It must then use truth only for declared metrics. A condition label, source row
index, text hash, token length derived from source, target path, or correctness
feedback in the attack process is a separation defect.

## Metrics and resource checks

Report post-BOS/padding-excluded token accuracy, complete token-record rate,
decoded-text/source-string exactness when applicable, first-error position,
position-stratified accuracy, and static top-32/top-512 proposal recall after
truth opening. Use paired record-level bootstrap intervals for shifted-minus-
matched effects; do not resample individual tokens as the primary unit.
Include table preparation/storage, candidate-generation, reconstruction,
reference-prefix, synchronization, I/O, CPU/GPU peak memory, and the exact
number of public-prefix and candidate evaluations. Distinguish exploratory
development evidence from canonical comparison-complete results.

Before a representative run, estimate memory for model plus the dense table,
check live GPU capacity and competing workloads, qualify the largest table and
correction cell behind a fail-closed free-memory/allocation guard, and retain
the preflight. Any batch-size or numerical workaround requires an exact
prediction/prototype equivalence check on a fixed fixture; otherwise preserve
and exclude it. No heavy GPU run is needed to establish the first table
identity check.

The pilot should end with one of: static lookup warrants a follow-up; only the
fixed reference correction warrants a follow-up; or neither does. The next
uncertainty must be stated in terms of matched context sensitivity, target
weight shift, or the cost/benefit of the correction.

The exact first-BOS probe IDs, all numerical constants, and the condition-free
attack contract are frozen in `experiments/TRR-P01/pilot_plan.json`; this note
summarizes them without duplicating the 256-ID list. The qualification receipt
must state whether batches 256 and 128 produce byte-identical ordered BF16
outputs and identical lookup predictions. A fake-prefix CPU equivalence test
is useful implementation evidence but cannot replace that pinned-model check. A full-40 versus cached-39 public-prefix comparison may also be recorded as a numerical diagnostic; because the calls have different sequence geometry, it is not a gate on the native cached-39 qualification path. Shape, finiteness, cache length, and candidate-cell resource checks remain fail-closed.
