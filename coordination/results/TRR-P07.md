# TRR-P07 retrospective comparison

Status: **COMPLETE_RETROSPECTIVE_SCORED_AFTER_FREEZE**

Disposition: **PANEL_DEPENDENT_OR_UNCERTAIN**. Both P06 past-only selected checkpoints (seeds 6106 and 6107) are
token-accuracy better than the retained positionwise reference in all 16
seed-by-cell comparisons. Their replicate-average contrast is positive on all
eight frozen paired cells, with a positive source-cluster 95% lower endpoint
in every cell. The gain is small on the practical scale in Pile
(+0.677 to +0.874 percentage points), and the P06 Finance/public-base exact
point gain is only +3.125 points. The predeclared practical gate therefore
fails for the whole domain-target panel even though there is no primary sign
reversal and no primary harm cell. This supports retaining the established
reference pending a separately planned confirmation; it does not promote a
new global default and does not launch another run automatically.

## Frozen design and decision rule

This was the bounded retrospective comparison in the approved P07 plan
(plan SHA-256
`a0a2339f1a4b77e02d7d1772459dc14d442a4ce24b5111a01e58622ca1ae7c3e`) from
published parent commit
`02c861dfbfc63e3c0b7684a48323fd476a3b268a`. It compares the two P06 seeds
6106 and 6107, the retained TRR-0006 positionwise reference, and the retained
TRR-0006 causal state on two paired targets (`public_base` and
`public_lora_2601`) and two domains (`pile` and `finance`). Each panel/domain
cell contains the same 256 source records and 128 positions, with 127
post-BOS positions scored. The P06 panel uses all 256 selected rows; the
TRR-0006 subset is the correctness-blind rows `6*k` for `k=0..255`, namely
rows `0, 6, ..., 1530`, with 256 rows per domain. The P06 seed predictions are paired replicates; their correctness counts are
averaged within source before bootstrap. No
logits or token predictions were averaged.

The primary contrast is P06 past-only minus the retained positionwise
reference. Secondary registered contrasts are P06 past-only minus P06
positionwise diagonal, P06 diagonal minus retained positionwise reference,
and P06 past-only minus retained causal. Every contrast is reported per seed
and after replicate averaging. Uncertainty uses 10,000 source-record cluster
bootstrap draws with seed 7007; the same source-index draw is reused across
both paired targets and all methods within each panel/domain. The scored unit
is therefore a source record, not a seed or a token treated as independent.

The practical support rule was predeclared as either a token delta of at least
+1.0 percentage point with a positive 95% lower endpoint or an exact-record
delta of at least +5.0 points with a positive lower endpoint, with no harm on
the other metric. Harm uses the corresponding −1.0/−5.0 upper-endpoint rule.
A coherent result requires no harm cell, at least one supported cell for
every domain-target group across the two panels, at least one supported cell in
each panel, and no materially opposite panel point estimates within a
domain-target group. Domains, targets, panels, and source records are not
pooled into an overall percentage. Same-sign results below the practical
threshold remain uncertain.

The truth boundary was respected: all predictions and tie counts were frozen
before any score reader. The successful score receipt used the existing P06
truth manifest and TRR-0006 binding only after that freeze; the P06 truth
manifest SHA is `21c07d64c489e16bd9a220f4175c23d225515c786e659613ee05ec1f01770e48`
and the TRR-0006 truth-binding SHA is
`c2f96e87699bccb38802500566eebe9c85250165e9416b59dcf87f8857f7d930`. Truth
payloads were not persisted in the result artifact. The result is exploratory and task-local,
not a canonical replacement for either parent benchmark. P03, TRR-0007,
hidden holdouts, new fitting, new capture, and automatic extensions were out
of scope.

## Primary result and gate

The primary token point estimate is positive in all eight cells and every
bootstrap token interval excludes zero. Finance exact gains are +3.125,
+5.273, +7.422, and +7.422 points in the order P06/base,
P06/LoRA, subset/base, and subset/LoRA. Only the latter three Finance cells
meet the +5-point exact practical threshold (the P06/base exact interval
includes zero). The four Pile token gains are +0.712, +0.874, +0.677, and
+0.714 points in the same panel/target order; all are below the +1-point
practical threshold. The supported primary cells are P06 Finance/LoRA, subset Finance/base, and
subset Finance/LoRA; no Pile cell is supported. The scorer consequently reports
primary `no_harm=true`, `no_material_reversal=true`, panel support in both
panels, but `domain_target_coverage=false` and disposition
`PANEL_DEPENDENT_OR_UNCERTAIN`.

The diagonal control exposes a separate local diagnostic: in P06
Finance/public-base, diagonal minus retained reference is −0.092 token points
(CI −0.183 to −0.011) and −5.664 exact points (CI −9.766 to −1.562). This is
not a sign reversal of the primary past-minus-reference comparison. It is a
warning against treating the primary difference as an isolated causal effect
of visibility or positionwise structure. The secondary past-minus-diagonal
and past-minus-causal token contrasts are positive in every cell in the table
below, but their provenance differences keep this report descriptive.

### Absolute scores by P06 replicate

Each entry is token accuracy / exact records out of 256. The two P06 seeds are the two replicate arms; the retained positionwise and causal states are single published states and are repeated on each row only to keep all six absolute arms visible.

| cell | P06 past 6106 | P06 past 6107 | P06 diagonal 6106 | P06 diagonal 6107 | retained positionwise | retained causal |
|---|---:|---:|---:|---:|---:|---:|
| P06/finance/base | 99.265% / 141.0/256 | 99.246% / 131.0/256 | 99.074% / 122.0/256 | 99.000% / 105.0/256 | 99.130% / 128.0/256 | 99.133% / 122.0/256 |
| P06/finance/lora_2601 | 99.289% / 145.0/256 | 99.283% / 136.0/256 | 99.090% / 125.0/256 | 99.062% / 114.0/256 | 99.145% / 127.0/256 | 99.210% / 130.0/256 |
| P06/pile/base | 94.565% / 14.0/256 | 94.273% / 13.0/256 | 93.990% / 7.0/256 | 93.427% / 13.0/256 | 93.707% / 8.0/256 | 93.418% / 4.0/256 |
| P06/pile/lora_2601 | 94.830% / 12.0/256 | 94.590% / 9.0/256 | 94.165% / 11.0/256 | 93.569% / 8.0/256 | 93.836% / 6.0/256 | 93.891% / 7.0/256 |
| subset/finance/base | 97.841% / 81.0/256 | 97.887% / 79.0/256 | 97.770% / 73.0/256 | 97.373% / 62.0/256 | 97.699% / 61.0/256 | 97.632% / 61.0/256 |
| subset/finance/lora_2601 | 98.087% / 84.0/256 | 98.078% / 90.0/256 | 97.955% / 73.0/256 | 97.622% / 67.0/256 | 97.736% / 68.0/256 | 97.844% / 68.0/256 |
| subset/pile/base | 94.534% / 9.0/256 | 94.381% / 11.0/256 | 94.033% / 10.0/256 | 93.485% / 9.0/256 | 93.781% / 7.0/256 | 93.489% / 5.0/256 |
| subset/pile/lora_2601 | 94.673% / 13.0/256 | 94.408% / 12.0/256 | 94.104% / 9.0/256 | 93.605% / 7.0/256 | 93.827% / 6.0/256 | 93.919% / 7.0/256 |

### Replicate-averaged absolute scores

The P06 replicate counts are averaged within source before resampling, so exact-record counts can be fractional. Token denominators are 32,512 per cell (256 records × 127 post-BOS positions).

| cell | P06 past | P06 diagonal | retained positionwise | retained causal |
|---|---:|---:|---:|---:|
| P06/finance/base | 99.256% / 136.0/256 | 99.037% / 113.5/256 | 99.130% / 128.0/256 | 99.133% / 122.0/256 |
| P06/finance/lora_2601 | 99.286% / 140.5/256 | 99.076% / 119.5/256 | 99.145% / 127.0/256 | 99.210% / 130.0/256 |
| P06/pile/base | 94.419% / 13.5/256 | 93.708% / 10.0/256 | 93.707% / 8.0/256 | 93.418% / 4.0/256 |
| P06/pile/lora_2601 | 94.710% / 10.5/256 | 93.867% / 9.5/256 | 93.836% / 6.0/256 | 93.891% / 7.0/256 |
| subset/finance/base | 97.864% / 80.0/256 | 97.572% / 67.5/256 | 97.699% / 61.0/256 | 97.632% / 61.0/256 |
| subset/finance/lora_2601 | 98.082% / 87.0/256 | 97.789% / 70.0/256 | 97.736% / 68.0/256 | 97.844% / 68.0/256 |
| subset/pile/base | 94.457% / 10.0/256 | 93.759% / 9.5/256 | 93.781% / 7.0/256 | 93.489% / 5.0/256 |
| subset/pile/lora_2601 | 94.540% / 12.5/256 | 93.855% / 8.0/256 | 93.827% / 6.0/256 | 93.919% / 7.0/256 |

### Paired contrasts and source-cluster bootstrap

Deltas are left minus right in percentage points. `CI_t` and `CI_e` are percentile 95% intervals for token accuracy and exact-record accuracy. `g/l` is the replicate-averaged paired token gains/losses. The final column gives the two seed contrasts as `6106 Δtoken/Δexact; 6107 Δtoken/Δexact`.

| cell | contrast | average Δtoken / Δexact | CI_t | CI_e | g/l | per-seed Δtoken/Δexact |
|---|---|---:|---:|---:|---:|---:|
| P06/finance/base | past−reference | +0.126/+3.125 | [+0.054, +0.203] | [-0.586, +6.836] | 78.5/37.5 | +0.135/+5.078; +0.117/+1.172 |
| P06/finance/base | past−diagonal | +0.218/+8.789 | [+0.140, +0.315] | [+4.883, +12.695] | 112.0/41.0 | +0.191/+7.422; +0.246/+10.156 |
| P06/finance/base | diagonal−reference | -0.092/-5.664 | [-0.183, -0.011] | [-9.766, -1.562] | 67.0/97.0 | -0.055/-2.344; -0.129/-8.984 |
| P06/finance/base | past−causal | +0.123/+5.469 | [+0.058, +0.189] | [+1.367, +9.570] | 76.5/36.5 | +0.132/+7.422; +0.114/+3.516 |
| P06/finance/lora_2601 | past−reference | +0.141/+5.273 | [+0.069, +0.220] | [+1.562, +9.375] | 79.0/33.0 | +0.145/+7.031; +0.138/+3.516 |
| P06/finance/lora_2601 | past−diagonal | +0.211/+8.203 | [+0.134, +0.312] | [+4.688, +11.719] | 103.5/35.0 | +0.200/+7.812; +0.221/+8.594 |
| P06/finance/lora_2601 | diagonal−reference | -0.069/-2.930 | [-0.154, +0.011] | [-7.422, +1.562] | 72.5/95.0 | -0.055/-0.781; -0.083/-5.078 |
| P06/finance/lora_2601 | past−causal | +0.077/+4.102 | [+0.012, +0.143] | [+0.391, +8.008] | 61.5/36.5 | +0.080/+5.859; +0.074/+2.344 |
| P06/pile/base | past−reference | +0.712/+2.148 | [+0.489, +0.940] | [+0.000, +4.492] | 489.5/258.0 | +0.858/+2.344; +0.566/+1.953 |
| P06/pile/base | past−diagonal | +0.711/+1.367 | [+0.518, +0.921] | [-0.195, +3.125] | 507.0/276.0 | +0.575/+2.734; +0.846/+0.000 |
| P06/pile/base | diagonal−reference | +0.002/+0.781 | [-0.272, +0.274] | [-0.391, +2.148] | 491.5/491.0 | +0.283/-0.391; -0.280/+1.953 |
| P06/pile/base | past−causal | +1.001/+3.711 | [+0.769, +1.241] | [+1.367, +6.250] | 545.5/220.0 | +1.147/+3.906; +0.855/+3.516 |
| P06/pile/lora_2601 | past−reference | +0.874/+1.758 | [+0.644, +1.115] | [-0.781, +4.492] | 514.5/230.5 | +0.993/+2.344; +0.754/+1.172 |
| P06/pile/lora_2601 | past−diagonal | +0.843/+0.391 | [+0.660, +1.040] | [-1.367, +2.148] | 523.0/249.0 | +0.664/+0.391; +1.021/+0.391 |
| P06/pile/lora_2601 | diagonal−reference | +0.031/+1.367 | [-0.255, +0.311] | [-0.977, +3.711] | 507.5/497.5 | +0.329/+1.953; -0.268/+0.781 |
| P06/pile/lora_2601 | past−causal | +0.818/+1.367 | [+0.595, +1.047] | [-1.172, +3.906] | 495.5/229.5 | +0.938/+1.953; +0.698/+0.781 |
| subset/finance/base | past−reference | +0.165/+7.422 | [+0.065, +0.265] | [+3.320, +11.523] | 164.5/111.0 | +0.141/+7.812; +0.188/+7.031 |
| subset/finance/base | past−diagonal | +0.292/+4.883 | [+0.181, +0.406] | [+1.172, +8.594] | 226.5/131.5 | +0.071/+3.125; +0.514/+6.641 |
| subset/finance/base | diagonal−reference | -0.128/+2.539 | [-0.271, +0.009] | [-1.562, +6.445] | 191.5/233.0 | +0.071/+4.688; -0.326/+0.391 |
| subset/finance/base | past−causal | +0.232/+7.422 | [+0.134, +0.334] | [+3.711, +11.328] | 165.5/90.0 | +0.209/+7.812; +0.255/+7.031 |
| subset/finance/lora_2601 | past−reference | +0.346/+7.422 | [+0.245, +0.446] | [+3.516, +11.719] | 191.0/78.5 | +0.351/+6.250; +0.341/+8.594 |
| subset/finance/lora_2601 | past−diagonal | +0.294/+6.641 | [+0.195, +0.395] | [+2.930, +10.547] | 215.5/120.0 | +0.132/+4.297; +0.455/+8.984 |
| subset/finance/lora_2601 | diagonal−reference | +0.052/+0.781 | [-0.063, +0.166] | [-2.930, +4.492] | 221.5/204.5 | +0.218/+1.953; -0.114/-0.391 |
| subset/finance/lora_2601 | past−causal | +0.238/+7.422 | [+0.131, +0.344] | [+3.711, +11.328] | 161.5/84.0 | +0.243/+6.250; +0.234/+8.594 |
| subset/pile/base | past−reference | +0.677/+1.172 | [+0.506, +0.854] | [-0.391, +2.930] | 455.0/235.0 | +0.754/+0.781; +0.600/+1.562 |
| subset/pile/base | past−diagonal | +0.698/+0.195 | [+0.508, +0.901] | [-0.977, +1.562] | 489.0/262.0 | +0.501/-0.391; +0.895/+0.781 |
| subset/pile/base | diagonal−reference | -0.022/+0.977 | [-0.269, +0.217] | [-0.781, +2.930] | 462.0/469.0 | +0.252/+1.172; -0.295/+0.781 |
| subset/pile/base | past−causal | +0.969/+1.953 | [+0.786, +1.163] | [+0.000, +3.906] | 513.5/198.5 | +1.046/+1.562; +0.892/+2.344 |
| subset/pile/lora_2601 | past−reference | +0.714/+2.539 | [+0.531, +0.898] | [+0.781, +4.688] | 471.0/239.0 | +0.846/+2.734; +0.581/+2.344 |
| subset/pile/lora_2601 | past−diagonal | +0.686/+1.758 | [+0.504, +0.877] | [+0.195, +3.516] | 485.5/262.5 | +0.569/+1.562; +0.803/+1.953 |
| subset/pile/lora_2601 | diagonal−reference | +0.028/+0.781 | [-0.218, +0.272] | [-0.781, +2.539] | 474.5/465.5 | +0.277/+1.172; -0.221/+0.391 |
| subset/pile/lora_2601 | past−causal | +0.621/+2.148 | [+0.438, +0.807] | [+0.586, +4.102] | 438.0/236.0 | +0.754/+2.344; +0.489/+1.953 |

## Provenance, timing, and limitations

The prediction replay receipt is
`experiments/TRR-P07/runtime/replay-r1/replay_manifest.json`, SHA-256
`229bd0aaf5080802320de58ecf28bef54af35f4140ad5916d2615c66c0061110`, with 48
predictions and status `FROZEN_P07_PREDICTIONS_NO_TRUTH`. Replay code was
commit `9e3176efe9bb04908d5a94e1e1dd8eec8f8c424c`; the source-free largest-cell
qualification was commit `016278dd211f991efd1f04b91258e2fcea674170` and passed
with 2,661,285,888-byte peak reserved GPU memory in 12.31 seconds. The
successful score artifact is
`experiments/TRR-P07/runtime/scored-r2/results.json`, 18,877,105 bytes, SHA-256
`f60f71c3ca747c6120671458bb54e559764c462a21abf9845d99967d13c76a38`; scorer
code was commit `1617b84d529ffcae600ffb82bd0e0ff86b0e6fa2`. A prior score
attempt is preserved as `runtime/scoring-attempt-r1.json`; it failed closed on
a producer record-ID digest serialization mismatch before loading truth arrays.
The metadata adapter was corrected in commit `1617b84`, after which the frozen
score completed.

The replay took 184.153 seconds for the 48 frozen cells. Its 24 published
fixture checks and resource-watchdog guards passed before the matrix. P06 used
the published batch-eight chunked full-vocabulary path, measured at
3.968–4.173 ms per record across its 32 cell-method timings. Retained TRR-0006
states used their native one-record full-logit path, measured at 4.743–4.881 ms
per record across 16 cell-method timings. These intervals include device
synchronization and exclude observation loading; the two paths are explicitly
labelled separately, are descriptive only, and are not pooled into a speed
claim.

P07 bound the exact published state and selection metadata, including the true
fit-record IDs and checkpoint selections; the ordered 1,200 fit-record IDs are
the same in the published metadata. The P06 and retained metadata use
different digest serializations (canonical JSON list versus newline-delimited),
which is why their digest strings differ. The P06 fit/prediction path cropped
and supported H128 positions 0..127, whereas retained states were fit/validated
with H192 support and replayed on H128 prefixes. The P06 selection SHA is
`d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a` and the
TRR-0006 selection SHA is
`75909aaf0f9e40176c197d86c09651097010a11519855f1db3dc50fe5e754f43`. This binding preserves published identities and does not make the fitted
systems equivalent. The provenance review records the resulting material
differences in fit crop and support geometry, initial W/b/scale, seed and
schedule, selection metric and selected checkpoint, and diagonal score
normalization. These differences are retained as provenance limitations rather
than silently treated as an identical fitted system. The P06 and retained TRR-0006
states are therefore reported as their published arms and any checkpoint
comparison remains associative rather than a causal method claim.

The primary gains/losses are paired token changes at identical source and
position coordinates. Exact-record metrics use 256-record denominators, while
the averaged exact counts in the table can be fractional because replicate
correctness was averaged before resampling. The practical gate is an
exploratory task decision, not a universal benchmark threshold.

## Decision

Retain the established reference as the working anchor. The result supports a
bounded statement that the selected P06 past-only checkpoint is a better
checkpoint on these common frozen inputs in token accuracy, while its practical
quality margin is too small in Pile to justify global promotion. A future
hypothesis/confirmation run could separately predeclare which provenance
factor or panel coverage it is testing, but that is not authorized or
automatically triggered by this report. No claim is made that all past-only
fits, visibility mechanisms, or unseen panels behave the same way.
