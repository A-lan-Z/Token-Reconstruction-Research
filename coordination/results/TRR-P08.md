# TRR-P08 staged affine-first versus joint fitting

Status: **PENDING_FRESH_CAPTURE**. This is the reviewable report for the
approved P08 study. No fresh evaluator truth, observation payload, prediction
arrays, score output, or fresh-panel quality result is included here yet;
public development-fit evidence is reported below.

## Pre-capture source-binding deviation

A metadata-only audit found that the effective r1 exclusion collector omitted
two approved bindings. The original candidate universe had 0 source-level and
1 H128-sequence overlap with the approved TRR-0007 opaque ledger (256/256
identities), and a distinct 0 source-level and 1 H128-sequence overlap with the
approved TRR-0009 reservation (384 new identities), for 2 unique H128 overlaps
total. The audit itself used hash-only metadata and did not read or persist
source/token payloads. No observation capture, model/target load, prediction,
or truth was opened before this repair.

The r1 selection metadata is preserved but excluded before capture:
`experiments/TRR-P08/runtime/source-selection-r1/selection.json` (SHA-256
`75f8d66007d6ae5558ded8b0fc428fc88db8aa5a04f4d372bd187f5934079274`). The
repaired r2 universe and selection now bind all four required opaque ledgers
and retain the same 512-record panel, seed 8108, eight fit states, declared
source ranges, and no schedule or sample expansion. A metadata-only review
found zero public-record and zero final-sequence intersections with each
ledger and their union; its receipt is
`experiments/TRR-P08/review/source-selection-r2-review.json`. The r2 selection
is ready for the separate guarded capture gate; the original r1 metadata will
not be reused.

## Frozen scope and question

The task-local study asks whether a competent direct affine path trained first
changes reconstruction quality or the incremental value of past activations
under the fixed P08 decoder family. It compares positionwise (`j=i`) and
past-only (`j<=i`) visibility under two equal-total-budget schedules:

| visibility | joint | staged |
|---|---:|---:|
| positionwise | 3000 complete-model updates | 1000 affine-only + 2000 complete-model updates |
| past-only | 3000 complete-model updates | 1000 affine-only + 2000 complete-model updates |

There are two paired fit seeds (6106, 6107), eight fits, common H128 geometry,
common ordered record and position draws, standard identity affine
initialization (`W=I`, `b=0`, `s=3.0`), and the reviewed P06 numerical settings.
The primary target is `public_base`; `public_lora_2601` is a paired transfer
diagnostic and does not select an arm or enter the primary gate. The fresh
panel is 256 records per domain (Pile and Finance), with 127 scored post-BOS
positions per record when all positions are valid.

The primary interaction is reported separately by primary domain:

```text
I = (Past_staged - Positionwise_staged)
    - (Past_joint - Positionwise_joint)
```

General staged-minus-joint contrasts for each visibility mask and the separate
Past-minus-Positionwise contrasts are reported alongside it. These are
exploratory, task-local estimates and are not a canonical benchmark
replacement or a global promotion claim.

## Frozen analysis and decision gates

Correctness counts, exact indicators, and paired gains/losses are averaged
within source record across seeds before bootstrap resampling. Seeds are
replicates, not additional source records. Uncertainty uses 10,000 paired
source-record cluster bootstrap draws with seed 8080, with the same source
indices reused across methods and paired targets within each domain. Pile and
Finance, targets, and primary versus transfer analyses remain separate.

The contextual support gate requires, in both primary domains, an interaction
benefit of at least +0.5 percentage points in token accuracy or +5 points in
exact 127-token recovery, with a positive 95% lower endpoint, no interaction
harm, and no seed interaction at or below the corresponding harm boundary.
General staging support uses the separately registered +0.5/+5 margins for both
visibility masks in both primary domains. If support is absent, useful
contextual benefit is ruled out only when both primary-domain interaction
upper endpoints are below both registered benefit margins and no seed reaches
a benefit margin. Otherwise the result is inconclusive. Gate outcomes do not
trigger a schedule sweep, extra phase, sample expansion, automatic
confirmation, or global promotion.

Exact-record denominators, token denominators, per-seed estimates, replicate
averages, paired token gains/losses, and position-bin diagnostics will be
reported for every domain-target cell. The transition diagnostic will report
full-versus-same-checkpoint affine-only correction, residual, and introduced
errors on its predeclared public fit/validation cohorts; a fit-cohort shortfall
limits that diagnostic only.

## Execution and truth boundary

The approved execution sequence has a repaired metadata-only source
selection; capture and later truth gates remain pending:

1. Capture observations from the repaired r2 selection, then freeze all
   selected/final states, observations, masks, positions, prediction files, and
   timing metadata in a create-only no-truth joint receipt.
2. Revalidate that receipt and materialize the private truth arrays only after
   the joint-freeze gate is accepted. Truth arrays remain outside the
   reconstruction repository; the truth manifest binds the freeze, selection,
   observation, prediction, record-order, and final-sequence hashes.
3. Invoke the fail-closed artifact scorer once against that truth manifest.
   The scorer revalidates the joint receipt before loading truth, checks the
   frozen row fingerprints and geometry, scores all four domain-target cells
   and all four methods at both seeds, and writes metrics/provenance without
   embedding truth arrays.

The reviewed adapter/scorer contract is covered by synthetic fail-closed
checks for explicit execution acknowledgement, geometry and BOS validation,
truth row-order/fingerprint mismatch, prediction tampering, create-only
outputs, and post-freeze hash bindings. No real truth or score command has
been run for this template.

## Provenance placeholders

- Approved plan SHA-256: `9d1eb8dca89c76f4064c0c380636cb9d585672f7fae796d5295d6507b788caa3`
- Packet SHA-256: `8edf99fcb311eadc1f17b5694dd82001ba6053aefcf8c828a3d0e19a53bb0059`
- Root approval: `experiments/TRR-P08/review/root-design-approval.json`
- Freeze/scoring implementation: `3c6e93c`, `c801e88`
- Truth materialization adapter: `c125afe`
- Excluded r1 source selection: `experiments/TRR-P08/runtime/source-selection-r1/selection.json` (SHA-256 `75f8d66007d6ae5558ded8b0fc428fc88db8aa5a04f4d372bd187f5934079274`)
- Repaired r2 source universe: `experiments/TRR-P08/runtime/source-universe-r2/frozen.json` (SHA-256 `23a8c632e232552e3dd5a33bcb1d5b07387505a971fb6c7df55ab1f8d5316a72`)
- Repaired r2 source selection: `experiments/TRR-P08/runtime/source-selection-r2/selection.json` (SHA-256 `e8023ed00ae3efbaa1b86f1290aa4de6b23da48dd70f7b10d9e147fbf6fbfb86`)
- Independent r2 metadata review: `experiments/TRR-P08/review/source-selection-r2-review.json` (SHA-256 `fe9afbd5f1145f9fe5f7e33bb68e0fb40c5ae9423e60cf4d119e3f87705d300e`)
- No-truth joint receipt: **PENDING**
- Truth manifest: **PENDING; private and outside the repository**
- Score artifact and hash: **PENDING**
- Final evidence manifest: **PENDING**

## Results to fill after authorized scoring

### Fit, selection, and resources

**Public development fit: PASS; fresh-panel selection and capture: PENDING.**
The eight-arm public fit completed at 3000 updates per arm with the registered
1000 affine-only plus 2000 complete-model staged schedule, or 3000
complete-model joint updates. Selection used full-vocabulary validation token
accuracy at step 0 and every 100 updates, with the earliest maximum rule. The
following values are development evidence only:

| seed | arm | selected step | best validation token accuracy | final validation token accuracy | final exact / 48 |
|---:|---|---:|---:|---:|---:|
| 6106 | positionwise staged | 2200 | 0.961771 | 0.960429 | 17 |
| 6106 | past-only staged | 1200 | 0.962106 | 0.960429 | 16 |
| 6106 | positionwise joint | 1500 | 0.964453 | 0.962441 | 17 |
| 6106 | past-only joint | 1500 | 0.964453 | 0.962106 | 18 |
| 6107 | positionwise staged | 1900 | 0.964118 | 0.962106 | 15 |
| 6107 | past-only staged | 2300 | 0.963112 | 0.962777 | 16 |
| 6107 | positionwise joint | 1800 | 0.965459 | 0.965124 | 16 |
| 6107 | past-only joint | 2100 | 0.964453 | 0.964453 | 18 |

The eight arms shared each seed's ordered position schedule, standard
initialization, and 512 query draws per update. This produced 1,536,000 query
draws per arm and 12,288,000 across the matrix. Summed arm wall time was
750.484 seconds; the watchdog interval was 756.123 seconds from
2026-09-07T01:39:51Z through 01:52:24Z. The watchdog recorded peak group RSS
7,569,088,512 bytes and minimum sampled host availability 17,729,118,208
bytes. The largest arm used 3,202,351,104 bytes of reserved CUDA memory with
12,388,925,440 bytes free and 5,621,690,368 bytes process RSS. The fit review
records these within the predeclared resource limits and preserves the exact
receipt hashes.

The affine transition is competent but not perfect on public development
material: seed 6106 had 139 fit errors among 112,825 valid positions and 125
validation errors among 2,982 positions; seed 6107 had 116 fit errors and 122
validation errors. Both validation cohorts were complete. The predeclared
256-error fit quota was short (139 and 116 rows), with bin shortfalls
`[51, 42, 0, 24]` and `[58, 52, 6, 24]`; this is a nonfatal diagnostic
shortfall and did not authorize cohort changes.

For the staged validation cohorts, selected/final full-decoder correction
gains/regressions (same-checkpoint affine-only versus full) were:

| seed | arm | selected gain/regression | final gain/regression |
|---:|---|---:|---:|
| 6106 | positionwise staged | 7 / 4 | 7 / 3 |
| 6106 | past-only staged | 5 / 0 | 5 / 4 |
| 6107 | positionwise staged | 11 / 1 | 8 / 0 |
| 6107 | past-only staged | 2 / 0 | 3 / 1 |

These transition and validation curves show a measurable, highly competent
public affine baseline and a nonempty correction signal, so they qualify the
fit and diagnostic machinery. They do not establish fresh-panel quality or
select a method for the natural evaluation. Joint-arm transition comparisons
are retained as collateral same-cohort diagnostics and are not interpreted as
staged-transition evidence.

### Absolute quality and paired comparisons

**PENDING.** Report per-seed and replicate-averaged token accuracy, exact
records, denominators, position-bin metrics, and paired gains/losses for
Pile/Finance × public_base/public_lora_2601. Do not pool domains or targets
into an overall quality claim.

### Bootstrap intervals and gates

**PENDING.** Report the primary interaction, general staging, and visibility
contrasts with the registered 95% source-cluster intervals and practical gate
fields. State the deterministic disposition exactly as emitted by the scorer.

### Disposition

**PENDING.** The final disposition will be one of the registered task-local
outcomes. It will distinguish contextual interaction support, general staging
benefit, useful-benefit ruled out, and finite-sample/incomplete evidence, and
will retain the transfer and architecture limitations. No result will be
presented as a universal mechanism claim.
