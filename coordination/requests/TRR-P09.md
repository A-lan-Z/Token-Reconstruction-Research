# Next shared study — Can better fitting support and a genuinely learned token readout close the reconstruction gap?

## Goal

Produce a materially stronger search-free decoder, not another marginal
score adjustment or training-schedule variant.

Test two factors:
1. More unique public token/context examples.
2. Freedom to change token readout directions rather than only token
   score gain and bias.

Keep A2 out of deployed inference.

## Ownership and starting point

Use the published TRR-0009 infrastructure and exact selected-state identities
at commit 4602f98eeb03ac121d8bbc230d7e2b219551b914 as the shared starting
resource. Select the appropriate competent fixed-readout starting state
from its manifest; do not choose a starting checkpoint using fresh answers.

Agent one owns directional readout implementation and its fits.
Agent two owns nested public fitting-bank preparation and fixed-readout
data-scaling controls.

Use separate worktrees and task-local outputs. Agree on a small common
experiment contract before fitting. Share public preparation artifacts
read-only, with exact identities.

Do not merge existing PRs, alter the active registry, open P03's holdout,
or modify each other's workspaces.

## Comparison

Use a small crossed design:

- Current bank, fixed readout.
- Current bank, trainable token directions.
- Expanded bank, fixed readout.
- Expanded bank, trainable token directions.

Retain the unchanged starting checkpoint as a reference. Count shared
pretraining and all additional fitting.

The expanded bank should contain substantially more unique examples,
not merely more repetitions or a replacement mixture of the same size.
A roughly order-of-magnitude increase is a useful planning target if
resources and source availability support it.

Build a nested, documented public sampling process with varied token
identities and contexts. Preserve natural-text evaluation. Generate
activation pairs through actual public-model forwards.

Report distinct fitting examples, token support, repeated exposures,
optimizer updates, and fitting cost separately. Include a continued-
training control so more data is not silently confounded with more compute.
Give the larger bank enough exposure for an informative learning curve.

## Directional readout

Allow token-specific changes to the scoring vectors, anchored to their
public initialization. Gain/bias-only changes do not satisfy this task.

Choose one defensible parameterization and regularization approach.
It must add genuine token-specific freedom, not only a global linear
transform that the existing decoder can absorb.

Keep unrestricted full-vocabulary prediction. Define supported/updated
rows using fitting data only, and report rare/absent-token regressions.
More trainable state must be included in memory and preparation accounting.

Prefer a design that can be stored as a single deployed readout rather
than requiring an additional routing or search component.

Do not add contextual access, staging schedules, teacher-ranking losses,
or ensembles to this comparison.

## Development and evaluation

Keep fitting, validation, and final evaluation roles distinct.
Use opened public development material for implementation and selection;
do not launch fresh confirmation after every development step.

Freeze the final small set before one common natural evaluation.
Report Pile and Finance separately, with matched public observations
and a paired changed target where practical.

Use the charter's permitted information and truth separation.
No source-token input beyond BOS, target-prefix oracle, guessed-token
feedback, or A2 fallback.

Report token errors, exact 127-token post-BOS clip recovery, paired
gains/regressions, and source-record uncertainty.

Compare against the strongest applicable fixed-readout control, not
only the old retained reference. Include A1+A2 on a declared common
subset large enough to keep the real quality gap visible. Keep that
subset's denominator explicit.

Assess how much of the baseline-to-A2 error gap is closed, along with
absolute reconstruction quality. Do not promote a method solely because
a tiny local improvement clears a statistical threshold.

Predeclare the useful pilot outcome and cost limits. Do not modify
completed studies' gates or repeatedly expand the final sample.

## Cost and stopping

This round permits a more substantial one-off public fit. It does not
authorize paid compute or unbounded scaling.

Preflight and bound compute, host/device memory, and artifact storage.
Stream or chunk activation data as necessary. Coordinate shared GPU use
and measure comparative inference without competing workloads.

Report training cost, learned/fixed state, and matched warmed inference.
A larger direct decoder may be acceptable if its measured quality gain
justifies its cost and it remains substantially cheaper than A2.

Stop scalar-calibration and staging follow-ups. If the crossed comparison
remains nearly flat on fresh data despite an informative fit, report that
clearly rather than proposing another automatic minor variant.

The final handoff should identify the best usable decoder produced,
its actual quality/cost gap to A1+A2 on common inputs, and which factor
caused the improvement.