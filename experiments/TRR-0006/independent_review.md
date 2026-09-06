# TRR-0006 independent interpretation review

Review target: `scored_v1/result.json`, SHA-256
`84b0535802a587c71553de7c14af4b9e952befd00e3ef65dfdd20a23c0ae70df`,
produced at code commit
`33dc6258614188927751ade45a0f0a2efe1f8361`. The review used only the scored
JSON, report, execution receipt, cost inputs, and frozen decision plan. It did
not open the private truth sidecar.

## Independent checks

Each cell has 1,536 source records and 127 scored post-BOS tokens per record.
The four exact events partition every record into both-correct, causal-only
(gain), positionwise-only (loss), and neither-correct; each partition sums to
1,536. Record IDs match between the two methods and between public-base and
synthetic-LoRA targets within each domain.

I recomputed one-sided Clopper-Pearson endpoints with SciPy beta quantiles at
`alpha = 0.05/32 = 0.0015625`. The recorded lower bound is
`L_CP(g) - U_CP(h)` and the upper bound is `U_CP(g) - L_CP(h)`; all values agree
to numerical precision. I also regenerated the 10,000 source-record bootstrap
draws using seed 5005 for Pile and 5006 for Finance, with the same schedule
for both target conditions in each domain. Token deltas, percentile intervals,
and practical tails match the result. These bootstrap endpoints are
approximate; the CP endpoint guarantee is marginal and exact under its
binomial marginal model.

## Cell-level results

Values are percentage points. Exact bounds use the registered CP construction;
token bounds use the registered paired source-record bootstrap.

| Cell | Both | Gain | Loss | Neither | Causal exact % | Diagonal exact % | Exact Δ | Exact [L,U] | Token Δ | Token 95% interval | Token practical [L,U] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pile / public base | 30 | 4 | 12 | 1,490 | 2.214 | 2.734 | -0.521 | [-1.669, 0.643] | -0.310 | [-0.374, -0.246] | [-0.396, -0.222] |
| Pile / synthetic LoRA | 34 | 10 | 11 | 1,481 | 2.865 | 2.930 | -0.065 | [-1.404, 1.275] | 0.076 | [0.008, 0.144] | [-0.020, 0.174] |
| Finance / public base | 312 | 49 | 38 | 1,137 | 23.503 | 22.786 | 0.716 | [-1.861, 3.285] | -0.117 | [-0.155, -0.080] | [-0.171, -0.067] |
| Finance / synthetic LoRA | 355 | 51 | 44 | 1,086 | 26.432 | 25.977 | 0.456 | [-2.231, 3.138] | 0.042 | [-0.002, 0.084] | [-0.019, 0.103] |

The descriptive Finance public-base exact point estimate is positive, but it
is only 0.716 percentage points and its practical upper bound is 3.285 pp,
below the registered +5 pp exact margin. A point estimate reaching a margin
would remain descriptive under the frozen plan.

## Decision-rule interpretation

The frozen multiplicity family contains 4 cells × 2 outcomes × 2 directions =
16 directional bounds, with no duplicate causal-versus-positionwise contrast.
The result has no supporting endpoint: no token lower bound reaches +0.5 pp and
no exact lower bound reaches +5 pp.

The upper practical bounds are below the benefit margins in every cell:

- token: -0.222, 0.174, -0.067, and 0.103 pp, all below +0.5 pp;
- exact: 0.643, 1.275, 3.285, and 3.138 pp, all below +5 pp.

Thus the all-four-cell exclusion rule is satisfied and the recorded
`positionwise_default` classification follows. This means the causal context
advantage was excluded at the registered margins for this panel and target
pair. It does not claim that the diagonal method is universally superior.

No lower token bound is below -0.5 pp, and no lower exact bound is below -5 pp.
Consequently harm is excluded under the retained harm limits; the negative
Pile public-base point estimates do not constitute material-harm evidence.
There are no material-harm endpoints.

## Cost and scope qualifications

The recorded warmed runtime ratio is 1.0268666, the maximum of the four
cell-level ratios, against the 1.25 budget. The retained preparation/training
ratio is 0.9992161 against the budget of 2. Both are qualified, with no new
training (`new_training_seconds = 0`). This qualifies the cost field attached
to this result and does not alter the quality decision.

The result retains the fixture-only qualification status and records no main
matrix qualification failure. The P04 cross-study exclusion limitation remains
material: aggregate exchange data do not provide per-record target-fit IDs,
source ranges, or replay sequence hashes. The historical A2 comparison remains
a separate denominator and was not recomputed by TRR-0006.

The execution receipt records `verified_before_truth: true`,
`truth_opened: false` for the public gate, `truth_opened_once: true`, and
`truth_payload_read_before_gate: false`. The final claim remains limited to the
two frozen enriched states, four declared cells, natural source panel, and
128-token clips.
