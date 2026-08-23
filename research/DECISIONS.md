# Research Decision Log

This file records scientific decisions when future work produces evidence for
them. Administrative relay events are kept separate from scientific decisions.

## Scientific decisions

TRR-0001-R1 supports the following decisions within its exact pinned scope:

- causal public-surrogate selection improves token accuracy over direct
  inversion at cuts 4 and 8 under the shared candidate budget of 16;
- recoverability declines from cut 4 to cut 8 for both methods;
- candidate proposal recall is the dominant primary bottleneck, while causal
  ranking is nearly perfect conditional on inclusion;
- exact non-embedding sequence recovery is not established; and
- this single target update does not establish a general unavailable-prefix
  mismatch penalty.

The original run remains preserved as
`ACCESS_INTERFACE_NONCOMPLIANT_ORIGINAL_RUN`. Its successful post-hoc
identifier noninterference audit does not turn it into accepted blind evidence.
These decisions rely only on the fresh R1 access-compliant run and remain scoped
to its model, data, target update, cuts, sequence geometry, and budget.

TRR-0001-R2 adds the following cross-setup decisions:

- every active method must be evaluated on both registered benchmark setups;
- causal public-surrogate K16 beats direct inverse K16 on both setups;
- strict-BOS adaptive A1+A2 wins on the historical finance setup but loses on
  the clean Pile/LoRA setup, so there is no setup-independent winner;
- the strict method's clean-setup weakness is primarily abstention/coverage:
  its top-512 candidate recall is 99.84% and its selective accuracy is 100%,
  but it covers only 13.26% of scored tokens; and
- the R2 backfill is retrospective compatibility evidence and does not replace
  the fresh blind-confirmatory status of the R1 clean run.

These decisions remain scoped to the two registered setups and tested method
configurations. They do not impose a scientific method ban.

TRR-0002 adds the following decisions:

- public A1 proposals plus causal public-prefix selection are robust across both
  registered setups; at K32 they reached 98.72% clean and 98.93% historical;
- the fixed-budget historical A2 core fails mainly because its raw confidence
  scale and suffix-abstention rule do not transfer, not because A1 stops finding
  the correct token: A1 K32 recall remained 98.76% clean and 99.03% historical;
- the residual-affine proposal and the A1/residual union did not improve on pure
  A1 under the causal selector in the controlled crossover;
- the public-only calibrated successor uses a scale-normalized K32 margin to
  expand uncertain positions to K64, never abstains, and was frozen before its
  fresh hidden selection existed;
- that same frozen method achieved 98.72% on the fresh blind replicate, 99.16%
  on the canonical clean setup, and 99.21% historically, exceeding the 98.22%
  historical strict A1+A2 result and the 83.97% prior new-setup causal result;
- always-K64 A1/causal remains slightly more accurate (99.32% clean and 99.26%
  historical), while the calibrated method retains most of that gain at lower
  compute cost; and
- every new active method must still populate both canonical setups. Fresh
  replicas supplement rather than silently replace the canonical cells.

These decisions are scoped to the pinned public model, lens, target-update
family, cuts, datasets, and tested geometries. They do not ban other methods.

TRR-0002 owner revision R1 adds the following decisions:

- “all configurations” means the complete preregistered finite Cartesian
  family, not the impossible set of all real-valued thresholds; this study
  evaluated all 512,136 declared unique policies;
- the accuracy-first winner among the tested A1+A2 configurations is fixed
  direct-cosine K256 with no immediate A1 shortcut, no adaptive routing, no
  candidate-group centering, and no abstention;
- the winner is an improved configuration of the historical A1+A2 mechanism,
  not a new reconstruction mechanism;
- the frozen policy achieved 99.64% on a new isolated blind Pile replicate,
  99.76% on canonical clean Pile, and 99.73% on historical Finance;
- compared with fixed K64 on the canonical records, K256 recovered 11 more
  clean tokens and 65 more historical tokens, with no record-level regression,
  but used four times the logical candidate simulations and about 1.8--2.2
  times the measured compute time; and
- K256 is the tested accuracy default, while calibrated K32-to-K64 and fixed
  K64 remain explicitly labelled lower-cost alternatives rather than winners.

These owner-R1 decisions remain scoped to the frozen finite search family and
the recorded public, blind, and canonical conditions. They do not claim that
K256 is globally optimal over untested mechanisms or continuous policies.

## Administrative entries

- TRR-0000 bootstraps and validates the repository-backed relay. It selects no
  scientific method, experiment direction, or permanent methodological
  restriction.
- TRR-0000 was accepted and merged as commit
  `0f641e5f071dd38331d2e2b7821d40fc74941c2e`, preserving accepted head
  `b087365766f00432077476bf32a6afdf2e854841` in main ancestry.
- TRR-0001 is assigned to establish blind direct and causal-surrogate baselines
  plus a cut-depth map.
- Review TRR-0001-R1 relabels the original run
  `ACCESS_INTERFACE_NONCOMPLIANT_ORIGINAL_RUN` and requires a fixed-method clean
  confirmatory run. This is a revision of TRR-0001, not a new scientific
  direction, and TRR-0002 remains unauthorized.
- User-directed revision TRR-0001-R2 establishes the durable dual-benchmark
  protocol and backfills every existing method across both setups.
- The repository owner explicitly superseded the stale proposal-only TRR-0002
  packet with the controlled component-crossover and calibrated-selector study.
  The exact override is preserved at
  `coordination/requests/TRR-0002-OWNER-OVERRIDE.md`.
