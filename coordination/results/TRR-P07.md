# TRR-P07 retrospective comparison

Status: **PENDING_PREDICTION_FREEZE_AND_SCORE**

This report records the bounded retrospective comparison defined in the
approved P07 plan (plan SHA-256
`a0a2339f1a4b77e02d7d1772459dc14d442a4ce24b5111a01e58622ca1ae7c3e`) from
published parent commit
`02c861dfbfc63e3c0b7684a48323fd476a3b268a`. It is exploratory and task-local;
it is not a canonical replacement for either parent benchmark and cannot
promote a new default automatically.

## Frozen design

The matrix compares two P06 replicate seeds (6106, 6107) with the retained
TRR-0006 positionwise and causal states on two paired targets (`public_base`,
`public_lora_2601`) and two domains (`pile`, `finance`). It uses the complete
256-record/domain P06 panel and the correctness-blind TRR-0006 subset at source
selection rows `6*k`, `k=0..255`, in published order. Every cell has 128 stored
positions including BOS and 127 post-BOS scored positions. The P06 replicate
joint correctness counts are averaged within each source before uncertainty
resampling; logits and seed predictions are never averaged.

The registered contrasts are P06 past minus retained positionwise reference
(primary), P06 past minus P06 diagonal, P06 diagonal minus retained reference,
and P06 past minus retained causal. The scorer reports each of the eight cells,
each P06 seed, and the within-source replicate average. It uses 10,000 seeded
source-record bootstrap draws (seed 7007), with one source-index schedule
shared across both paired targets and all methods in each panel/domain.

## Evidence and result placeholders

- Prediction replay/freeze receipt: **pending**
- Existing P06 truth sidecar and TRR-0006 truth binding: **opened only after the
  prediction freeze; pending score receipt**
- Per-seed and replicate-averaged paired token/exact metrics: **pending**
- Position/gain/loss and bootstrap intervals: **pending**
- Cellwise support/harm/inconclusive gate: **pending**
- Cost and native-versus-compatible-port notes: **pending**

The decision gate is cellwise: practical support requires either token delta at
least +1.0 percentage point with a positive 95% lower endpoint or exact-clip
delta at least +5.0 points with a positive lower endpoint, with no harm on the
other metric. Harm uses the corresponding −1.0/−5.0 point upper-endpoint rule.
A coherent contrast has no harm cell, support in every domain-target group in
at least one panel, support in each panel, and no materially opposite panel
point estimates within a domain-target pair. Domains, targets, and panels are
not pooled into an overall percentage. Same-sign but inconclusive results stay
inconclusive, and no follow-on run is automatic.

The report will retain the parent P06 full-record disposition unchanged and
will label any native/compatible-port difference, attention normalization,
fit-support difference, initialization, crop, and checkpoint-selection
difference as possible explanations rather than causal findings. No P03,
TRR-0007, hidden holdout, fresh fitting, or new capture is in scope.
