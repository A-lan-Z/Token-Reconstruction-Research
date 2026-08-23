# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0002 owner revision R3 completed a 24-cell strict-surrogate and
target-shift study. It retained both canonical setups and added paired
GrandMaster observations from the untouched public checkpoint and a verified
heavy full-SFT derivative on the same 64 hidden records.

The zero-fit checkpoint-identity proposer did not replace the historical public
Alpaca lens. On the heavy target, centered K512 reached 69.37% with the strict
proposer and 77.46% with the Alpaca control. The same control reached 83.07% on
the matched target, so heavy fine-tuning caused a 5.61-point loss. Every
fresh-panel method reconstructed 0/64 complete inputs.

The heavy winner's candidate recall was 78.97%, while A2 was 98.09% accurate
when the true token was present. The next accuracy work should therefore improve
public-only candidate generation under target shift rather than merely enlarge
or complicate the selector. A scientifically useful follow-up would
preregister several non-arbitrary public proposer constructions or fitting
corpora, justify each resource, and freeze them before a new hidden
fine-tuned-target panel exists. That panel should use an external corpus with
documented provenance disjoint from the target's declared fine-tuning corpus if
it is intended to measure generalization to unseen text.

Whole-input recovery is now a primary practical metric. Future studies should
optimize and report token-complete inputs, decoded-text-complete inputs, errors
per failed input, and length-stratified recovery, not token accuracy alone.

All compared methods must still populate both registered canonical setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`, as well as every declared auxiliary
target. The active registry remains 24 methods and 48 required canonical cells.
Owner-R1 fixed direct K256 remains the official fresh-blind accuracy default;
calibrated K32-to-K64 and fixed K64 remain cheaper controls.

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
