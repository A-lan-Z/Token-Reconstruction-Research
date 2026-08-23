# Dual-benchmark comparison protocol

Status: mandatory evaluation procedure from TRR-0001-R2 onward.

`RESEARCH_CHARTER.md` remains the sole authoritative research definition. This
protocol operationalizes its like-for-like evaluation requirement and does not
ban any charter-permitted method.

## Purpose

Every active reconstruction method must be evaluated in both canonical setups.
The two setups answer different questions, so a complete matrix is more useful
than choosing one setup and losing comparability with earlier work.

Claims about the best configuration inside the A1+A2 family must additionally
follow `research/A1_A2_CONFIGURATION_PROTOCOL.md`. Exploratory search variants
are not active methods merely because they were enumerated; the single frozen
winner becomes active and must then add one cell in each canonical setup.

## Canonical setups

### clean-pile-lora-64x40

- 64 records, 40 tokens per record including the known BOS, and 39 scored tokens
  per record;
- pinned Pile-10k records and Llama-3.2-1B-Instruct public surrogate;
- the unavailable rank-4 target-prefix LoRA condition is primary;
- cuts 0, 4, and 8 are retained for the direct and causal depth map, with cut 4
  as the common cross-setup comparison boundary;
- clean runs use cryptographic selection commitment, sanitized observations,
  fail-closed reconstruction isolation, output freeze, and truth reveal only
  after all registered methods have frozen their outputs;
- no persistent state enters the first evaluated record unless an explicitly
  registered adapted arm preregisters its permitted earlier-record state.

### historical-finance-strict-bos-128x128

- 128 right-padded Finance-Instruct rows, up to 128 positions each, at the
  generation-300 target's layer-4 input;
- exactly one known seed token (BOS); every valid post-BOS token is scored;
- the pinned public Alpaca affine lens and public Llama layers 0--3 are used by
  the historical strict-BOS cascade;
- the exact native historical runner remains the reference execution for that
  method and setup;
- this opened source is retrospective evidence. A future fresh replacement must
  retain the same declared geometry and scoring contract before it can replace
  the historical setup in this protocol.

## Active method registry

| Method ID | Frozen decision rule | Native setup |
|---|---|---|
| `direct_inverse_k16` | public-data affine inverse, full-vocabulary cosine top-16 proposal, choose proposal rank 1 | clean 64x40 |
| `causal_public_surrogate_k16` | the same frozen top-16 proposals, re-rank with the public prefix and reconstructed prefix, no target-prefix calls | clean 64x40 |
| `strict_bos_adaptive_a1_a2` | public Alpaca A1 top-512, confidence fast path 0.999, progressive public-prefix A2 tiers 32/128/512, normalized-winner threshold 2.0, then suffix abstention | historical 128x128 |
| `a1_scale_calibrated_adaptive_causal_k32_to64` | public A1 top-64; causally score K32; expand to K64 when the frozen scale-normalized top-two gap is at or below 1.2544946670532227; never abstain | dual-setup successor |
| `a1_a2_exhaustive_configuration_winner` | public A1 top-256; direct causal cosine at fixed K256; no fast path, routing, centering, or abstention | dual-setup accuracy winner |

The word `adaptive` in `strict_bos_adaptive_a1_a2` refers to its per-token tier
and route selection. Learned online A1 adapters are separate methods and must be
registered as separate rows if reactivated.

## Required matrix

Every result must contain these six base-control cells:

| Method | clean 64x40 | historical 128x128 |
|---|---:|---:|
| direct inverse K16 | required | required |
| causal public-surrogate K16 | required | required |
| strict-BOS adaptive A1+A2 | required | required |

Adding an active method adds one required cell in each setup. Adding a canonical
setup adds one required cell for every active method. A task with a missing cell
is incomplete for overall comparison, even when its available cells succeeded.

### TRR-0002 controlled crossover extension

TRR-0002 adds the following five component combinations at each fixed candidate
budget 8, 16, 32, and 64, with every resulting method required in both setups:

| Proposal | Selector |
|---|---|
| public Alpaca A1 | fixed-budget historical A2 core |
| public Alpaca A1 | causal public-prefix cosine |
| residual-affine inverse | fixed-budget historical A2 core |
| residual-affine inverse | causal public-prefix cosine |
| round-robin deduplicated A1/residual union | causal public-prefix cosine |

The fixed-budget A2 core uses the historical centered-cosine score,
`K * softmax(score)[winner]` confidence, threshold 2.0, and suffix abstention.
The original A1-confidence fast path and progressive 32/128/512 tiers are not
part of the symmetric factorial selector; they remain represented by the exact
`strict_bos_adaptive_a1_a2` control.

The causal selector uses the same public prefix and greedy reconstructed prefix,
but selects the maximum uncentered cosine at every position without abstention.

The crossover's frozen v2 registry is preserved at
`experiments/TRR-0002/preregistration/dual_benchmark_registry.v2.json` and
contains its 44 required setup-method cells. The active v3 registry adds the
calibrated successor, producing 46 required cells. The crossover is
retrospective; the calibrated successor was fitted only on public development
updates, frozen, and then evaluated in a fresh isolated blind target-update run.

### TRR-0002 calibrated successor

The calibrated successor replaces the historical A2 raw confidence threshold
and suffix abstention. Its confidence is the K32 causal top-two score gap divided
by the RMS deviation of all 32 scores. This statistic is invariant to a common
score shift or positive rescaling. Positions at or below the frozen public-only
threshold expand to ranks 33--64; every position emits a token.

The method must remain byte-identical across both canonical setup cells. A fresh
clean confirmation may use a new cryptographically hidden, disjoint 64x40 split,
but that replicate supplements rather than deletes the canonical clean cell.
TRR-0002 froze the method before the new split existed and confirmed 98.72% on
the fresh blind replicate, 99.16% on the canonical clean setup, and 99.21% on
the historical setup.

### TRR-0002 owner-R1 exhaustive configuration winner

The owner-requested follow-up enumerated all 512,136 policies in its frozen
finite A1+A2 configuration family on public component surfaces, causally ran 57
deterministically selected finalists on disjoint public trajectories, and froze
one winner before held-out, fresh-blind, or canonical access. The exact winner
uses direct cosine at fixed K256, no immediate A1 shortcut, no adaptivity, no
centering, and no abstention.

The same serialized policy achieved 99.76% on canonical clean Pile and 99.73%
on canonical historical Finance. A wholly new disjoint isolated blind replicate
scored 99.64%. This makes K256 the accuracy-first default among tested A1+A2
configurations. The calibrated K32-to-K64 method remains a cheaper balanced
alternative; these are different operating points and must retain their own
method IDs and cost records.

## Native executions and benchmark-compatible ports

An exact native execution uses the method's frozen source, assets, constants,
and native geometry unchanged.

A benchmark-compatible port may change only:

- tensor packing, padding, or row batching;
- record identifiers and input/output serialization;
- loop bounds implied by the benchmark's declared record and sequence geometry;
- scoring adapters needed to express the common metrics.

A port must not change candidate budgets, thresholds, candidate ordering,
scoring functions, cache semantics, abstention behavior, inverse or lens state,
or truth access. Each port must record its differences and pass a semantic check
against the native implementation on a shared compatible fixture when one
exists. Reports must label exact and ported executions separately.

If a method genuinely cannot be ported without changing its decision rule, the
cell records a failed exact attempt and the task remains comparison-incomplete;
incompatibility is not silently treated as a completed comparison.

## Execution and truth opening

For a fresh confirmatory run:

1. preregister the complete active-method matrix;
2. commit method identities, fixed assets, constants, seeds, and port hashes;
3. prepare only charter-permitted observations for each setup;
4. run and freeze every cell before opening that setup's truth;
5. verify the freeze and matrix completeness;
6. reveal truth once, score every frozen cell with the same evaluator, and do
   not rerun selectively;
7. preserve failed runs and deviations.

Backfills on already opened records must be labelled retrospective and cannot be
promoted to fresh confirmatory evidence. They are still required to restore the
comparison matrix and to test implementations.

## Common reporting

Report within each setup:

- post-BOS end-to-end token accuracy, with abstentions counted incorrect;
- exact record/row count and rate;
- coverage and selective accuracy for methods that can abstain;
- candidate proposal recall when candidate sets exist;
- candidate simulations, public-model evaluations, runtime phase breakdown,
  peak memory, and implementation complexity;
- exact method and port identity, artifact hashes, environment, and failures.

Use paired record-level differences and uncertainty where the records permit it.
Compare methods within each setup. Across-setup differences measure robustness
to the setup change; do not pool their tokens or present a single averaged score.

## Claim rule

`better overall`, `best method`, and replacement claims require the complete
matrix. A method may still be called better on one named setup when that cell is
valid and the claim is explicitly limited to that setup. Runtime claims use the
same timing boundary and hardware within a setup.

## TRR-0001-R2 backfill status

TRR-0001-R2 backfills all six existing cells after both source truths were
already available. These results are retrospective comparability evidence, not
a new blind confirmation. The clean TRR-0001-R1 direct and causal outputs remain
the governing blind results for their native cells; R2 reruns test reproducibility
and add the previously missing ports.

## TRR-0002 status

TRR-0002 completed all 44 preregistered crossover cells, the two calibrated
method cells, and the two frozen exhaustive-winner cells. Both fresh clean
confirmations were isolated and frozen before truth reveal. The active v4
registry therefore contains 24 methods and 48 canonical setup-method cells;
future active methods must likewise add one cell per canonical setup.

### Owner-R2 target/surrogate diagnostic status

Owner revision R2 did not register 12 new active methods or add a third
canonical setup. It froze a diverse exploratory shortlist from the completed
public configuration table and ran a retrospective stress diagnostic inside
the existing historical Finance setup, whose target prefix is generation-300
Finance-Instruct while A1/A2 retain public-only resources.

Centered K512 reached 99.9285%, centered K256 reached 99.8070%, and the
multistage centered finalist reached 99.7713% with 60,200 logical simulations.
These single-setup target-shift results differentiate promising configurations
but cannot support an overall-best or replacement claim. The active registry
and its 48 required cells therefore remain unchanged.

A future replacement study must preregister the selected policy as active,
populate both canonical cells, and confirm it on a newly hidden target update.
The saturation and freeze requirements are specified in
`research/A1_A2_CONFIGURATION_PROTOCOL.md`.
