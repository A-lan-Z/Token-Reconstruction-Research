# Development38 reconstruction-development grid

Public qualification passed all288first-token decisions, three exact repetitions
each, with four independent full-table cosine references. This motivates testing
a causal first decision inside joint optimization; it is not a benchmark claim.

Freeze six rules: control and locked-first, each at16,32,64joint updates. All use
factor2, four scalar KL iterations, zero warm optimization steps, all128256finite
logits at each remaining position and native complete40/128-position geometry.
The first lookup scores every cachedBOS response by cosine and emits its argmax
as the reconstructed first unknown token. It supplies no proposals to A2.

The locked rule replaces only the first mixture with the emitted token's actual
embedding on every forward pass, and returns that token at every final/readout
checkpoint. Its gradient row is zero because the decision is fixed; the unused
row stays finite but may be recentered by the unchanged update. Remaining rows
are unchanged in shape and retain the full vocabulary. Loss remains the mean
over all post-BOS positions; the first residual is included as a constant.
Probability diagnostics report confidence1/purity0for the fixed token. No source
truth is used, and a wrong first decision has no oracle fallback.

Keep the original metric initialization and warm diagnostic pass. Store both
the unchanged initial mixture and actual mixture when locked. Account for the
extra table, preparation/rebuild, direct lookup and full decoder time. Stop at
each declared step budget; do not use checkpoint times from a longer run as
standalone latency. The BF16surrogate gradient is an explicitly approximate
existing method, not an exact inverse or a precision port.

Run two isolated restart-safe workers, one per rule, reusing each worker's
capture across its three stopping budgets. Largest128/64case qualifies first.
Public qualification per worker:18repeated decodes, six eager/replay checks,
two independent complete probability-gradient checks, initial-mixture/warm
anchors against archived stage32, and persistence of the direct first decision.
Every control64output and all traces must match its archived stage32output;
shorter controls must match its corresponding token/trace prefix.

Then run all48cells on the unchanged eight-record retrospective development
panel, three repeats each. Preserve full outputs before any qualification gate.
Require exact repeated outputs, fixed-token checks and all24control/24initial
anchors. Freeze the entire matrix before opening retrospective labels; use
missing-cell, changed-source and changed-prediction negative gates.

Report each condition/group separately, including every first-token error,
final token accuracy, exact records, and16/32/64costs. Source truth never changes
inference decisions. This grid is exploratory; a selected fixed contender must
be registered and run in both canonical setups before a replacement claim.
