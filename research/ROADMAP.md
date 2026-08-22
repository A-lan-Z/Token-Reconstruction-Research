# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0002 has completed the controlled 44-cell component crossover, public-only
confidence calibration, a fresh isolated blind confirmation, and the calibrated
method's two canonical cells. The frozen calibrated A1/causal method achieved
98.72% on the fresh blind 64x40 Pile/LoRA replicate, 99.16% on the canonical
clean setup, and 99.21% on the historical finance setup. It therefore satisfies
the preregistered simultaneous success rule and replaces historical strict A1+A2
as the current robust default. Always-K64 A1/causal remains the highest-accuracy
tested variant at greater cost.

All future method comparisons must populate both registered setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`. A partial matrix is comparison-incomplete
and cannot support an overall-best claim. The active registry now contains 23
methods and 46 required canonical cells. The highest-value next work is to
improve A1 proposal recall beyond rank 64 or reduce adaptive causal cost while
holding the frozen blind protocol and dual-setup coverage fixed.

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
