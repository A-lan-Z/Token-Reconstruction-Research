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
