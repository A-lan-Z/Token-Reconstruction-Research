# TRR-P08 staged affine-first versus joint fitting

Status: **PENDING_EXECUTION**. This is the reviewable report template for the
approved P08 study. No evaluator truth, fresh natural-panel rows, prediction
arrays, score output, or experimental result is included here yet.

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

The approved execution sequence is:

1. Complete the fixed eight-arm fit and public validation selection, then freeze
   all selected/final states, observations, masks, positions, prediction files,
   and timing metadata in a create-only no-truth joint receipt.
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
- No-truth joint receipt: **PENDING**
- Truth manifest: **PENDING; private and outside the repository**
- Score artifact and hash: **PENDING**
- Final evidence manifest: **PENDING**

## Results to fill after authorized scoring

### Fit, selection, and resources

**PENDING.** Record each arm's selected step (step 0 through 3000, earliest
maximum), public-validation curve, transition cohort counts, fit/validation
correction diagnostics, update/query-draw totals, phase-specific wall time,
inference time, peak memory, and preserved failures or shortfalls.

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
