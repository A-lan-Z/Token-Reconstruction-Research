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
  protocol, backfills every existing method across both setups, and leaves
  TRR-0002 unassigned and unstarted.
