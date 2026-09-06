# TRR-P06 design review

**Status: PROPOSED; execution is held.** The supplied packet is preserved at
`coordination/requests/TRR-P06.md` and ends in the middle of its last decision
bullet. This review therefore records a bounded design for root/setup review;
it is not a freeze or an authorization to fit, capture, open truth, or score.

## Scientific choice

The informative intervention is visibility within one decoder family. Every
arm has the same trainable direct affine path,

```text
z_i = H_i W^T + b + O(sum_j attention(i,j) V LN(H_j)),
```

followed by the same float32 normalization and public full-vocabulary readout.
The direct path starts at `W=I`, `b=0`, and the added output is zero
initialized, so the three arms share the same meaningful affine function at
step zero. Q/K/V initialization is deterministic and shared within each
replicate. All arms use layer-normalized Q/K cosine scores with scale 4.0;
using the prior P04 mix of cosine causal scores and dot-product diagonal
scores would confound visibility with normalization.

The masks are the only scientific intervention:

- `p06_positionwise_diagonal` allows only the current valid key `H_i`.
- `p06_past_only` allows valid keys `H_0` through `H_i`.
- `p06_full_record` allows every valid `H_j` in the declared stored record,
  including later positions.

At the H128 boundary, the full arm has `127-i` later vectors for a full
128-position clip at score position `i`; it receives no source labels or
reconstructed-token feedback. It may precompute activation-only features for
the whole record, then emit committed predictions left-to-right. The full
arm is therefore an offline full-record method and its retained activation
and attention computation must be included in cost reporting.

The shared-family parameter count is 5,247,361 for hidden size 2,048 and
attention width 128. A diagonal mask makes Q and K structurally
non-identifiable because each valid query has one allowed key; its effective
trainable count is 4,722,817. This is a real effective-capacity difference,
not evidence that the total parameter counts are unequal. It must be shown in
the fit receipt and interpreted alongside the mask comparison.

## Fitting and qualification

Use the already public `coverage_mix_v1` fit bank and its common 48-record
public validation split. This fixes the data distribution while the three
masks vary. Each arm receives full-vocabulary same-position CE only, the same
8-record/512-post-BOS-draw schedule within a replicate, the same validation
records and 100-step validation cadence, and the same 3,000-update AdamW
recipe (learning rate `1e-3`, zero weight decay, clip norm 1.0). Use two
predeclared replicates, 6106 and 6107, with the same initialization and
schedule seed for all arms within each replicate. Select each arm's earliest
maximum public-validation token accuracy independently; no fresh panel answer
can enter selection.

Before fresh panel truth, build a fixed 256-position capacity probe from the
common validation split. It contains 64 positions in each scored-position
bin `[1,15]`, `[16,39]`, `[40,79]`, and `[80,127]`, selected only from
positions where the common step-zero affine path is wrong. Thus the initial
accuracy is 0/256 and the test is not an already-perfect subset. Every one of
the six selected arm/replicate states must correct at least 52/256 positions
and remain finite. This is a competence gate, not a model-selection rule. If
an arm fails, stop before fresh truth and report an underqualified
comparison; do not change the mask, sample, or fit recipe to rescue it.

The largest resource qualification is a full-mask backward cell with the
actual 8 x 192 batch and 512 full-vocabulary draws. The fixed geometry is
small relative to the resident 1.05 GB F32 embedding table: an F32 activation
batch is 12,582,912 bytes, a dense 8 x 192 x 192 score matrix is 1,179,648
bytes, and a 512 x 128,256 selected-logit matrix is 262,668,288 bytes. The
5.25M-parameter model is 20,989,444 bytes, with 41,978,888 bytes for Adam
moments. These are component sizes, not a claim about peak allocator use;
record live host/device measurements before releasing the six sequential
fits. Use the inherited proposed guard of 8 GiB free GPU, 6 GiB maximum
reserved GPU, 16 GiB host RSS, 10 GiB host available, and a bounded timeout.
Do not use microbatching or mask-specific numerical shortcuts without an
output-equivalence check.

## Fresh evaluation

Select 256 new natural source records per domain (`pile` and `finance`) with
selection seed 6206, excluding fitting, validation, opened public evaluation,
and accessible duplicate source/sequence identities. Require at least the
128 valid positions in the declared clip, retain natural length/context
strata in the selection ledger, and use the same 512 source records under
both public-base and public-LoRA target conditions. Capture the same H128
observations and position/mask sidecars for all arms. The changed target is a
paired transfer condition, not a second independent source sample.

The primary quality comparison is `Full - Past`; `Past - Positionwise` and
`Full - Positionwise` are secondary descriptions. Report micro token accuracy
and per-record token accuracy over post-BOS positions 1..127, exact 127-token
clip recovery, and token/record gains, losses, both-correct, and
neither-correct counts. Resample source records, not tokens, with 10,000
paired bootstrap draws seeded 6306, stratified by domain, target condition,
and frozen natural-length stratum. If both fit replicates are scored, average
the two fixed-seed deltas within each source resample rather than treating
replicates as additional natural records.

The progression threshold is a predeclared +1.0 percentage-point token gain
or +5.0 percentage-point exact-record gain for `Full - Past`, with a
95-percent bootstrap lower bound above zero in at least one public-base
domain. Every other public-base domain/outcome must stay above the retained
-1.0 pp token or -5.0 pp exact harm limit. Public-LoRA cells are reported
separately: failure of the changed target is a transfer limitation, not proof
that no later-position information exists. This is a bounded exploratory
promotion criterion, not an equivalence test or a universal benchmark claim.

Report position bins `[1,15]`, `[16,39]`, `[40,79]`, `[80,127]` for every
cell, including the actual `future_valid_count` from each mask. At record ends
that count naturally falls to zero. Invalid/padded positions are removed from
all masks and metric denominators; a public padded-mask invariance fixture
must verify that changing invalid activation values cannot change any valid
prediction. The primary panel requires 128 valid positions so exact-clip
metrics share one denominator, while natural records may still be stratified
by their longer source lengths.

A bounded native A1+A2 K=256 anchor uses the predeclared first 64 panel
records per domain under public-base only: 128 clips and 16,256 scored
post-BOS tokens, with exact denominator 128. Its candidate/search cost and
quality are reported separately. It does not select student arms and is not a
fourth visibility condition; no student prediction may use A2 or candidate
simulation.

## Main risks and stop conditions

The key remaining confound is effective capacity: diagonal Q/K gradients are
zero while past/full Q/K are active. The direct path and identical total
parameterization make the quality comparison interpretable, but the report
must not call the arms capacity-identical. Standardizing score normalization,
initialization, fit data, schedule, validation, and readout removes the
larger P04 normalization/data confounds.

The full arm's future activations are allowed observations, but this is not a
causal or streaming decoder. It cannot revise an earlier token using a later
prediction, and it cannot use later source labels. Source-record pairing,
mask digests, position IDs, end/padding invariance, and full-record timing
must be frozen before truth. Any missing competence gate, incomplete arm,
changed source/target pairing, or resource anomaly stops the run without
opening truth. No TRR-0007 result, P03 holdout, P04 teacher objective, or
private target resource is needed for this design.

The single planned outcome is either a later-activation benefit that meets
this practical gate or a bounded result that does not. Position-bin patterns,
changed-target failure, and the A1+A2 gap remain descriptive and cannot be
used to reselect a mask, checkpoint, or source panel.
