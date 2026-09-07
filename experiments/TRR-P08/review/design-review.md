# TRR-P08 design review

Status: **FROZEN_DESIGN_PRE_FIT**. This review covers the fitting mechanism only;
no fresh source selection, capture, truth access, optimizer run, or target
matrix is authorized by this document.

## Decision-level design

The study uses a 2×2 crossed family:

| visibility | joint | staged |
|---|---|---|
| positionwise (`j=i`) | 3000 complete-model updates | 1000 affine-only updates, then 2000 complete-model updates |
| past-only (`j<=i`) | 3000 complete-model updates | 1000 affine-only updates, then 2000 complete-model updates |

Each cell has seeds 6106 and 6107, for eight independent fits. Every arm
starts from the same standard identity affine state (`W=I`, `b=0`, `s=3.0`),
the same deterministic Q/K/V initialization for its seed, zero output
projection, public F32 readout, H128 crop, cosine Q/K scale 4, AdamW settings,
record batches, and 512-position draw schedule. The direct affine path is
trainable in both schedules. Staged arms keep attention parameters frozen for
the first 1000 updates and carry the optimizer state for `W,b,s` into the
complete phase; attention parameters begin accumulating state at update 1001.
The staged phase is duplicated across visibility arms and charged in full, so
there is no hidden shared fitting cost.

The primary estimand is the interaction on the new `public_base` target, by
domain:

```text
I = (Past_staged - Positionwise_staged)
    - (Past_joint - Positionwise_joint)
```

The primary panel has separate Pile and Finance estimates on 256 new natural
records per domain. The same records under `public_lora_2601` are a paired
transfer diagnostic if the changed capture is available. Changed-target
answers do not select arms or enter the primary gate. General staged-minus-joint
contrasts for each visibility mask are reported separately, because equal
staging gains in both masks indicate a fitting-procedure effect rather than a
context-specific interaction.

All four arms within a seed consume the same ordered public-fit schedule and
all have 3000 updates, or 1,536,000 query draws. The eight fits therefore use
24,000 updates and 12,288,000 query draws in total. Checkpoint selection is
full-vocabulary micro token accuracy on the disjoint public validation bank at
step 0 and every 100 updates through step 3000, with earliest-maximum tie
handling. A staged arm selected by step 1000 is valid and is reported as an
affine/transition-phase selected state; it is not forced to a later checkpoint.

## Why 1000 + 2000

The one-third phase boundary is fixed before fitting and is not one point in a
schedule sweep. It gives the affine component enough counted updates to move
from the common identity state while retaining two thirds of the budget for
context learning. The published P06 public validation curves selected their
best checkpoints for the two visibility arms retained here at steps 1700 and
2000 for past-only and 1700 and 900 for positionwise-diagonal across the two
seeds. Thus step 1000 lies inside the observed public optimization window and
leaves a substantial complete-model phase. The published full-record 1500
selection is out of scope for this positionwise/past-only design and is not
used as its schedule rationale. This is a public development justification;
no fresh evaluation answer or hidden result chooses the split.

The main comparison remains equal-budget: joint arms train every parameter for
3000 updates, while staged arms train the direct path for 1000 and the complete
model for 2000. Phase time, optimizer state, draw counts, validation, and
serialization are recorded separately so any speed difference is interpretable
as a cost of the schedule.

## Transition correction diagnostic

At update 1000, each staged seed is evaluated with an affine-only transition
model. Before phase-2 fitting or any fresh-panel prediction, two public
cohorts are frozen:

- every wrong post-BOS position in the full disjoint public validation bank,
  retained in published order; and
- up to 256 public-fit errors, targeting 64 in each fixed position bin
  `[1,15]`, `[16,39]`, `[40,79]`, `[80,127]`, retaining all available errors
  when a bin is short.

The fit cohort is a diagnostic ledger, never a perfect or post-selected subset.
A short bin is recorded and makes only the fit-side diagnostic incomplete; it
does not stop the eight-arm fit or natural evaluation. The full validation
cohort provides the held-out development check, and an empty cohort is reported
as unavailable rather than repaired by changing the rule or adding updates.

For each staged visibility arm and seed, the report compares the full decoder
at its selected checkpoint and at final step 3000 with an affine-only twin
constructed from that same checkpoint's `W,b,s`. On both frozen transition-error
cohorts it reports four transitions: transition-affine wrong to same-final-
affine correct (direct-path progress), same-final-affine wrong to final-full
correct (added-path correction), same-final-affine wrong to final-full wrong
(residual failure), and same-final-affine correct to final-full wrong
(introduced error). The same-final-affine comparison isolates added-path
correction; transition-affine versus final-full is retained only as a combined
progress diagnostic. A later rise in affine accuracy alone is not called
contextual correction learning.

## Primary interpretation gates

The interaction margin is +0.5 percentage points for token accuracy or +5
points for exact 127-token clip recovery, with a positive paired-bootstrap
lower endpoint and symmetric −0.5/−5 harm boundaries. A contextual staging
benefit requires support in both public-base domains, no harm, no materially
opposite domain estimate, and the same nonnegative interaction sign for both
seeds. A general staging benefit requires both visibility masks to meet a
registered positive margin with lower confidence bounds above zero in both
primary domains, with same-sign seed contrasts.

A useful contextual benefit is ruled out only when both domains have token
upper confidence bounds below +0.5 points and exact upper bounds below +5
points, with no seed interaction at or above either positive practical margin.
The practical support gate is evaluated first, followed by this ruled-out
gate. Small mixed signs do not override a gate already met; they are reported
descriptively. If neither gate is met, intervals crossing a registered margin,
one-domain support, an opposite seed at the relevant direction, or incomplete
arm/receipt/exposure evidence are inconclusive. A fit-cohort shortfall alone
does not make the main comparison inconclusive.

No outcome triggers extra phases, a schedule search, sample expansion, global
promotion, or reinterpretation of P07. The final report will retain Pile and
Finance separately, give token and exact metrics, paired gains/losses, source
cluster uncertainty, phase and inference cost, and the changed-target transfer
as a supporting result only.
