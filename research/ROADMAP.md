# Research Roadmap

This file records prioritized research work as it is authorized and supported
by the repository-backed relay.

## Current scientific priority

TRR-0002 owner revision R4 completed a 45-cell exact-input target-only bridge.
It reused all 128 historical inputs and the exact old post-BOS metric while
holding the public A1 proposal source, public A2 surrogate, cut, policies, and
numerical execution fixed. Only the target-model weights changed among the
untouched public checkpoint, Finance generation 300, and a verified heavy full
SFT.

The best robust quality/cost point was adaptive K256-to-K512: 99.7999%,
99.7999%, and 99.8213%, with 118, 118, and 115 completely reconstructed inputs.
It used about 23% of fixed-K512 candidate simulations. Fixed centered K512 was
the accuracy-first point at 99.9285%, 99.9285%, and 99.7856%, with 124, 124,
and 102 complete inputs. Thus heavier target fine-tuning alone did not destroy
historical A1+A2; the historical strict policy's larger 2.60-point loss came
from confidence-gate and suffix-abstention transfer.

R4 also confirms that the plain zero-fit checkpoint-identity proposer is not a
replacement for the fitted public A1 lens: it recovered only 73.55%--75.13%
and 0/128 complete inputs. The next accuracy work, if authorized, should
preregister non-arbitrary public-only proposal constructions and test them on a
new blind external input panel using paired untouched and fine-tuned targets.
Holding inputs and all non-target components fixed is mandatory for a claimed
target-weight effect.

Whole-input recovery remains a primary practical metric. Future studies must
report complete inputs alongside token accuracy and cost. All active methods
must still populate both registered canonical setups under
`research/DUAL_BENCHMARK_PROTOCOL.md`; target-only bridge cells supplement
rather than replace that matrix. R4 is retrospective evidence and does not
start TRR-0003 or replace the owner-R1 fresh-blind default.

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
