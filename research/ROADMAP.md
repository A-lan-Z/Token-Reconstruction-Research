# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0002 has now completed both the controlled crossover/calibration phase and
the owner-requested bounded exhaustive A1+A2 configuration study. The latter
screened 512,136 unique policies, causally compared 57 finalists, and froze
fixed direct-cosine K256 with no shortcut, adaptivity, centering, or abstention.
That same policy achieved 99.64% on a wholly new isolated blind Pile replicate,
99.76% on canonical clean Pile, and 99.73% on historical Finance. It is the
current accuracy-first A1+A2 default. This is a better configuration of the
established mechanism, not a new reconstruction mechanism.

All future method comparisons must populate both registered setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`. A partial matrix is comparison-incomplete
and cannot support an overall-best claim. The active registry now contains 24
methods and 48 required canonical cells. The calibrated K32-to-K64 method and
fixed K64 remain cheaper Pareto alternatives, but neither is the accuracy
winner. The highest-value next work is to approach K256 accuracy with fewer
candidate simulations, or improve proposal recall beyond rank 256, while
holding fresh-blind isolation and dual-setup coverage fixed.

## Prior R1 priority (superseded by the dual-benchmark requirement)

TRR-0001-R1 has completed its fixed-method fresh confirmatory run and is
awaiting review. Under the exact pinned scope, causal public-surrogate selection
improved token accuracy over direct inversion at cuts 4 and 8, recovery declined
with depth, and candidate proposal recall was the dominant primary bottleneck.
The original run remains
`ACCESS_INTERFACE_NONCOMPLIANT_ORIGINAL_RUN`; its post-hoc identifier audit
passed but does not repair its access interface.

If a later packet authorizes new research, the highest-value proposed direction
is a fresh preregistered study of public-only candidate proposal at multiple
budgets, holding the validated causal selector fixed and measuring the
quality/cost frontier across multiple target updates. That proposal was
superseded by the completed TRR-0002 controlled crossover and calibration study.

## Scope note

TRR-0001's model, data, cuts, target construction, and budgets are temporary
experimental choices. They do not permanently restrict methods permitted by
RESEARCH_CHARTER.md. The revision's process isolation is a temporary enforcement
choice for this fixed confirmatory run, not a permanent ban on public auxiliary
data.
