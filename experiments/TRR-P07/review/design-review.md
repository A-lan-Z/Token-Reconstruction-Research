# TRR-P07 design review

Status: **PROPOSED_PRE_SCORE**. This review freezes the retrospective comparison
before any new P07 prediction or score is selected.

The two panels are the complete 256-record/domain P06 natural panel and an
evenly spaced, correctness-blind 256-record/domain subset of the opened
TRR-0006 panel. For the latter, the published selection rows at indices
`6*k`, `k=0..255`, are retained in published order. This is exactly one sixth
of each 1,536-record domain panel and gives the subset a deterministic spread
over the published order. Metadata checks show zero record-ID overlap with the
P06 panel in either domain. The subset is bound by record-ID, public-record,
and final-sequence hashes in `experiments/TRR-P07/plan.json`.

Every method receives the same H128 activation tensors, masks, position IDs,
row order, and post-BOS positions 1..127 within a panel. The matrix has six
frozen states on four domain/target cells in each panel: P06 past-only and
P06 diagonal at seeds 6106 and 6107, plus the single retained TRR-0006
positionwise and causal states. The stopped P06 full-record state is excluded.
Existing prediction artifacts can be reused only after exact binding checks;
cross-panel cells require frozen-state prediction and a create-only matrix
freeze before scoring. No source text, labels, hidden records, or another
task's sealed material is needed.

The primary estimand is P06 past-only versus the retained TRR-0006
positionwise reference. Secondary contrasts retain P06 past versus its
diagonal control, P06 diagonal versus the retained reference, and P06 past
versus the older causal state. Token accuracy is micro over 127 positions;
exact recovery requires all 127. P06 replicate correctness is averaged within
each source before aggregation and is kept fractional. No logits are averaged
and no seed is selected from outcomes.

Uncertainty uses 10,000 seeded source-record bootstrap draws (seed 7007),
resampling source indices within each domain/panel and applying the same draw
to both paired target conditions and all methods. Seeds and targets are thus
not extra independent records. The predeclared interpretation uses +1.0 pp
with a positive 95% lower interval only for practical support, with a separate
negative-harm rule and an exact-clip guard. Same-sign but inconclusive panels
remain uncertain; panel percentages are not treated as a cross-panel
improvement. The task cannot promote a fresh default or authorize a new run.

Before comparison, the existing P06 qualification/native fixture and
TRR-0006 fixture-equivalence checks must pass. The report must distinguish
exact native replay from any common-runner compatible port and retain the
published numerical, mask, position, batch, normalization, and tie policies.
A cell is supported when either token delta is at least +1.0 pp with a
positive 95% lower endpoint or exact-clip delta is at least +5.0 pp with a
positive lower endpoint, provided the other metric is not harm-classified.
Harm uses the corresponding -1.0 pp token or -5.0 pp exact upper-endpoint
rule. A coherent contrast has no harm cell, support in every domain-target
group across at least one panel, support in each panel, and no materially
opposite panel point estimates within a domain-target pair. The categories
are evaluated cellwise; there is no pooled panel/domain/target percentage.

The old P06 full-record negative result remains unchanged. Differences in fit
support, checkpoint selection, crop, initialization, and attention
normalization are recorded as possible explanations, not causal findings.
The published stored-H fixture replays are sufficient for numerical checks;
no raw-192 recapture is required and no new capture is authorized.
