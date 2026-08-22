# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0001-R2 has completed the required 3-method by 2-setup comparison matrix
and is awaiting review. Causal public-surrogate K16 is the strongest tested
method on the clean 64x40 Pile/LoRA setup; strict-BOS adaptive A1+A2 is the
strongest tested method on the historical 128x128 finance setup. Because the
winner changes with the setup, no method is currently accepted as the overall
replacement for the others.

All future method comparisons must populate both registered setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`. A partial matrix is comparison-incomplete
and cannot support an overall-best claim. The highest-value proposed scientific
direction is to improve cross-setup robustness, especially strict-BOS routing
and coverage on the clean setup, while preserving its historical precision.
TRR-0002 has not been assigned or started.

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
quality/cost frontier across multiple target updates. This is a proposal only.
TRR-0002 has not been assigned or started.

## Scope note

TRR-0001's model, data, cuts, target construction, and budgets are temporary
experimental choices. They do not permanently restrict methods permitted by
RESEARCH_CHARTER.md. The revision's process isolation is a temporary enforcement
choice for this fixed confirmatory run, not a permanent ban on public auxiliary
data.
