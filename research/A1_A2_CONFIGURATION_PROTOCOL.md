# A1+A2 configuration-search protocol

This protocol governs any claim that a particular configuration is the best
member of the historical A1 proposal plus public causal candidate-simulation
family. `RESEARCH_CHARTER.md` remains the sole authoritative scientific
definition. This document is an evaluation discipline, not a restriction on
future method ideas.

## What “all configurations” means

A literal search over every real-valued threshold is impossible because there
are infinitely many. Before scores are inspected, each study must therefore
freeze:

- the candidate-budget grid and all increasing schedules to be enumerated;
- every candidate-comparison rule;
- immediate A1 acceptance choices;
- every routing-confidence definition and its finite threshold construction;
- final rejection, fallback, and stopping behavior; and
- deterministic tie-breaking and a total winner rule.

The complete Cartesian product of those frozen choices is the exhaustive
configuration family for that study. The result must report its exact size,
deduplication rule, omissions, and full-table artifact hash. A short hand-picked
list must not be called exhaustive.

## Required separation

Configuration fitting and selection must use public auxiliary trajectories.
Canonical observations, canonical truth, prior private evaluation truth, and
fresh-blind truth must not influence the configuration choice.

Use three disjoint roles:

1. a public component surface for exhaustive screening;
2. a disjoint public causal split for exact left-to-right finalist selection;
3. a held-out public condition and fresh blind evaluation used only after the
   winner is frozen.

Truth-prefix simulation is permitted only on public auxiliary data and must be
labelled a component diagnostic. The official winner must be selected by an
exact causal run that begins with only the declared BOS token and subsequently
uses only its own reconstructed prefix.

## Comparability and cost

After selection, register the single frozen winner as an active method and
follow `research/DUAL_BENCHMARK_PROTOCOL.md`. The identical serialized policy,
constants, and decision rule must run on both canonical setups. Search variants
remain exploratory configurations rather than active methods unless separately
registered.

Report accuracy, coverage, exact records, candidate recall, selected candidate
budgets, logical and executed candidate simulations, synchronized proposal and
selection time, peak memory, and persisted state. A faster scheduler may batch
independent rows, but it must not change decisions; verify the historical anchor
against the pinned native implementation.

## Winner freeze

The preregistration must define a total, deterministic winner rule before the
search. Commit the winning policy, fitted numeric gates, code and input hashes,
predictions, and selection table before any held-out or fresh-blind truth is
opened. Held-out failure may weaken or reject the claim, but must not trigger an
unregistered method swap.

## Completed TRR-0002 owner-revision study

TRR-0002 owner revision R1 froze and evaluated the complete declared family:
512,136 unique policies, 57 exact-causal finalists, and one deterministic
accuracy-first winner. The winner is
`a1_a2_exhaustive_configuration_winner` / policy
`a1a2_43ea0bb737bc075531ca`:

- historical public A1 candidates;
- a fixed candidate budget of K=256;
- direct cosine comparison against each candidate simulation;
- no immediate A1-confidence shortcut;
- no adaptive routing or fitted threshold;
- no candidate-group centering; and
- no abstention: always commit the K=256 winner.

This is a configuration of the established A1+A2 mechanism, not a newly
invented reconstruction mechanism. It won because the preregistered rule
prioritized worst-domain and mean accuracy before runtime. Direct and centered
K=256 were equally accurate on public causal selection; direct cosine was
slightly faster and therefore won the deterministic tie-break.

The frozen policy was not revised after selection. It achieved 99.92% on the
held-out public Pile condition and 100% on held-out public Finance, 99.64% on a
new isolated blind Pile replicate, 99.76% on canonical clean Pile, and 99.73%
on canonical historical Finance.

For questions about the best tested A1+A2 setup, the accuracy-first answer is
therefore fixed K=256, no adaptivity, no immediate A1 acceptance, and direct
per-candidate comparison. The former K32-to-K64 calibrated method remains a
lower-cost balanced point, and fixed K64 remains an intermediate point. They
must not be relabelled as the accuracy winner: on the canonical setups K=256
used four times as many logical candidate simulations as K64 and about
1.8--2.2 times its measured compute time, while recovering 11 additional clean
tokens and 65 additional historical tokens.

Future configuration studies must freeze a new finite Cartesian product and
repeat the same public-selection, winner-freeze, fresh confirmation, and
dual-canonical procedure. They must not tune a replacement on the held-out,
fresh-blind, or canonical results recorded here.
