# TRR-P03 — Validate, then compact, the frozen projected-prototype readout

## Objective

Determine whether P02's projected-prototype improvement survives a representative public evaluation and, only if warranted, whether its useful scoring geometry can be retained in a substantially smaller search-free readout.

The owner values both eliminating offline fitting and eliminating A2 at inference. This task investigates the latter using existing fitted resources. **No new fitting is not the same as a fitting-free method.** Preserve the historical lens's fitted provenance throughout.

This is an independent follow-on for agent two. Do not duplicate agent one's matched-fitting, nonlinear-decoder, or contextual-decoder experiments. You have discretion over efficient implementation and a modest experimental budget; the requested outputs and scientific separation matter more than a prescribed architecture.

## Starting point and coordination

Repository: `A-lan-Z/Token-Reconstruction-Research`.

Start from P02 publication head `7956b4357d076abce3ccfc407d3fcac832fd34f6` on `task/TRR-P02`, after verifying the current local and remote state. Create a separate `task/TRR-P03` branch/worktree. Preserve both agents' existing work and environments. Do not merge PRs or update global `coordination/STATE.json` or the active-method registry. Use task-local state under `coordination/parallel/TRR-P03.json`.

Read the charter and relevant P02 report/code. The current assignment authorizes exploratory screening; it does not authorize replacement claims or an automatic expansion into the full historical matrix. Observe the existing access boundary and resource-preflight requirements. Coordinate shared compute rather than interfering with another job. No paid compute is authorized.

## What P02 does and does not establish

P02 scored the full vocabulary on twelve predetermined public rows: three endpoint token IDs across four short contexts. Raw prototypes recovered 7/12, projected frozen prototypes 11/12, and historical A1 9/12. This is a useful hypothesis, not a representative 91.7% reconstruction result or evidence of superiority over A1 generally.

Shared reference subtraction worsened non-reference recovery from 5/8 to 4/8. Public-panel mean-centering obtained 40/40 with a restricted truth-containing candidate dictionary and 0/12 with the full vocabulary. Do not revive these variants unchanged or use restricted-candidate results to select a purported full-vocabulary winner.

The projection improved relative token separation but increased absolute cross-context L2 variation. Do not assume context invariance, low-rank structure, or an accuracy guarantee from a spectral plot.

## Stage 1 — Establish whether the uncompressed readout is worth compressing

Freeze a modest, broader public panel before scoring: natural sequences with varied token identities, context lengths, and input styles. Include substantially more than the three P02 endpoint IDs. Keep controlled token/context probes separate from natural-text metrics. Exclude P02's exact recorded tuples from new confirmation claims; do not ban those token IDs from future evaluation.

Compare the existing frozen raw-prototype, projected-prototype, and historical A1 readouts on identical observations with the full declared vocabulary and the same numerical execution settings. Add a modest existing A1+A2 anchor where affordable to keep the eventual accuracy target visible, rather than rerunning the historical policy matrix.

Use matched public observations first, plus a paired independently shifted target if resources are already available; reconstruction must not receive its unavailable prefix. No teacher-prefix input may be supplied to the readout at reconstruction. Observation generation and post-freeze scoring remain separate. Previously opened P01 or other records may be reused for explicitly retrospective checks, not represented as fresh confirmation.

Report actual deterministic top-1 predictions, ties, token accuracy, exact records, and paired improvements/regressions versus A1. Preserve per-record results and relevant uncertainty; a few token differences do not establish a reliable gain. Publish the uncompressed finding before optional compression work.

If the projected rule's apparent advantage disappears materially outside P02, stop its promotion and explain the discrepancy. A clearly bounded independent footprint benefit may still justify a small compression diagnostic, but do not use compression to disguise inadequate parent-method accuracy.

## Stage 2 — Test compactness without a new training campaign

If Stage 1 supports continuing, test a small predeclared set of compact budgets (for example ranks 128 and 256) against the full scorer. Keep ordinary A1 as a same-budget compression control so an efficiency result is not incorrectly attributed to projected prototypes.

Focus on the full deployed scoring computation, including the vocabulary table. Merely shrinking a roughly 16 MiB lens while retaining the 501 MiB BF16 / approximately 1 GiB FP32 full-width dictionary would leave much of the footprint and scoring cost untouched. Count preparation, all retained inference assets, and runtime scratch space separately. Original assets needed only during construction need not remain deployed, but that must be demonstrated rather than assumed.

A useful algebraic option, not a required implementation:

Let the frozen affine lens be `g(h) = W h + a`, the boundary prototype be `p_v`, and `z_v = normalize(g(p_v))`. Then, in exact arithmetic,

`argmax_v cosine(g(h), g(p_v))`

has the same ranking as

`argmax_v [(W^T z_v)^T h + a^T z_v]`.

This permits a single compiled affine token readout or a factorized approximation. The positive query normalization is common to every candidate; removing it preserves rankings, not the numerical cosine margins or probability calibration. Preserve prototype normalization and the affine bias. Validate the actual finite-precision implementation against the parent; fusion changes are not automatically output-equivalent.

Choose an efficient decomposition; avoid unnecessary large intermediates and do not assume that a low-rank lens automatically yields a useful low-dimensional vocabulary scorer. Do not refit on evaluation answers. Any public calibration or rank selection must be labelled and separated from held-out evaluation, and would remain additional preparation.

Evaluate token decisions and score/margin preservation, not singular-value energy alone. Define a practical quality-retention and footprint target before checking the holdout. A failure at one compact rank rejects that tested setting, not every possible compression mechanism; nevertheless keep this round bounded rather than responding with an open-ended rescue sweep.

## Handoff and decision

Retain code, tests, exact resource identities, configurations, frozen predictions, and failed attempts. Use:

- `coordination/requests/TRR-P03.md`
- `coordination/results/TRR-P03.md`
- `experiments/TRR-P03/manifest.json`
- `coordination/parallel/TRR-P03.json`

Commit and push the separate branch and open a follow-on PR against its actual parent, without merging. Lead the report with one of these outcomes:

1. The projected readout did not generalize beyond the tiny diagnostic; stop this variant.
2. It generalized, but useful aggressive compression was not established at the tested budgets.
3. A compact fitted-origin, A2-free scorer retained a specified amount of quality with a measured footprint/runtime benefit.

Keep matched-target, shifted-target, controlled-probe, and natural-text outcomes separate. No result here eliminates the historical fitting phase, establishes a benchmark replacement without the required confirmations, or settles all no-fit research. Recommend one next scientific decision based on evidence, not another automatic round of the same method.
