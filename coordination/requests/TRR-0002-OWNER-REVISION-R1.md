# TRR-0002 owner-directed exhaustive configuration revision R1

Date received: 2026-08-23 (Australia/Sydney)

The repository owner directed the still-open TRR-0002 study to continue with a
systematic search over the complete bounded A1-based decision family rather
than treating the current K32-to-K64 rule as the endpoint.

Verbatim owner direction:

> Yeah, I'm not asking you to tell me which one is better right now. I'm asking you to design a test to test all possible configurations, all possibilities of the A1 + A2 (or if you call that your new method) and come up with the best one.

After receiving the proposed exhaustive design, the owner authorized execution:

> Sounds good, proceed

This revision authorizes continued work on `task/TRR-0002` and pull request #3.
It does not authorize merging that pull request or starting TRR-0003.

The bounded family keeps the frozen public A1 proposer unchanged and varies the
complete decision system after proposal. The preregistered search must cover:

- fixed candidate budgets and increasing adaptive schedules within A1's
  top-512 output;
- direct and candidate-group-relative comparison rules;
- disabled and fixed confidence-grid immediate A1
  acceptance;
- historical winner strength, raw top-two separation, scale-adjusted
  separation, with A1 confidence tested by the immediate route;
- expansion, forced-choice, A1 fallback, and historical suffix-stop outcomes;
- a preregistered empirical-quantile threshold grid plus the exact historical
  and current thresholds, with behaviourally identical configurations deduplicated;
- the exact historical native A1+A2 control and an equivalent common-runner
  implementation so decision quality is separated from implementation speed.

All search choices and the winner rule must be frozen before canonical or fresh
blind truth is used. Every search configuration must be evaluated on disjoint
public development replicas of both benchmark families. Active finalists and
the frozen winner must follow `research/DUAL_BENCHMARK_PROTOCOL.md` on both
canonical setups. Runtime finalists must be rerun independently rather than
timed through shared search computation.

The official winner is selected lexicographically from untouched public
validation data by:

1. highest token accuracy on its worse benchmark family;
2. highest mean token accuracy across the two families;
3. highest exact-record accuracy;
4. lowest independently measured runtime per 1,000 scored tokens;
5. lowest peak memory, fewer candidate simulations, then simpler decision rule.

The frozen winner must then undergo fresh blind confirmation without revision.
If it fails to generalize, the failure is reported; a runner-up may not be
substituted after truth reveal.

`RESEARCH_CHARTER.md` remains the sole authoritative scientific definition.
