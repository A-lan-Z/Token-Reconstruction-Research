# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0002 has now completed both the controlled crossover/calibration phase and
the owner-requested bounded exhaustive A1+A2 configuration study. Owner
revision R2 additionally ran a frozen 12-policy target/surrogate transfer panel
because the public selection split had saturated. The target was the existing
generation-300 Finance-Instruct model, while A1 and A2 retained only public
surrogate resources.

The transfer panel produced 11 distinct accuracies. Centered K512 reached
99.9285% and exactly matched A1 top-512 proposal recall; direct K512 reached
99.8570%, centered K256 99.8070%, and direct K256 99.7284%. The multistage
centered finalist reached 99.7713% with only 1.68% of direct K256's logical
candidate simulations. Thus target mismatch reveals both a centering benefit
and a much stronger quality/cost point that the public ceiling concealed.

The R2 panel is retrospective because the Finance truth was already open. The
official fresh-blind/dual-canonical default therefore remains the owner-R1
fixed direct-cosine K256 policy, not because R2 found it best on the target, but
because only that policy was frozen before the completed blind confirmation.

All future method comparisons must populate both registered setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`. A partial matrix is comparison-incomplete
and cannot support an overall-best claim. The active registry now contains 24
methods and 48 required canonical cells. The calibrated K32-to-K64 method and
fixed K64 remain registered cheaper controls.

The highest-value next work is a preregistered fresh hidden Finance-target
update that confirms centered K512 and the multistage finalist without tuning,
while retaining both canonical setups. If centered K512 again reaches its
candidate-recall ceiling, further accuracy work should improve A1 recall beyond
rank 512 rather than add selector complexity. If the multistage result repeats,
it becomes the leading candidate for a lower-cost active method.

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
