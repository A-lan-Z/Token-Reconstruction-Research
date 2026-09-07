# TRR-0010 common evaluation and timing proposal

> **Historical planning note — superseded by the canonical v2 contract.** Use `shared_contract.proposal.json` and `shared_contract.proposal.md` for the current starting state, crossed design, readout parameterization, panel, and thresholds. This retained note records earlier alternatives and availability observations; its old unchanged-state binding, timing draft, and unapproved A1/A2 assumptions are not active decisions. No fitting, source selection, truth access, or timing run is authorized by this note.


Status: planning only.  This note records the reusable evaluation boundary,
the source-availability constraints visible in the published TRR-0009
metadata, and the preflight estimate.  It performs no source selection,
truth access, fitting, inference, or timing run.  It does not bind an A1/A2
artifact that is not present in the allowed metadata.

## Frozen starting point and scope

The task-local starting reference is the TRR-0009 published infrastructure at
commit `4602f98eeb03ac121d8bbc230d7e2b219551b914`.  The competent fixed-readout
starting state selected by that task is:

| item | value |
| --- | --- |
| loader | `token_reconstruction.trr0007_positionwise.load_positionwise_model_state` |
| state | `experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors` |
| state bytes / SHA-256 | 29,390,492 / `5cada4a3d04bb5477e7af0be25ed8d8ac25a89283223e9ba14b18fa10416bee14` |
| contract | current activation `H_i` only, full-vocabulary tied `E`, no history, A2 disabled |
| geometry | hidden 2048, vocabulary 128,256, BOS 128000, 128 stored tokens, 127 scored post-BOS tokens |
| public embedding | 1,050,673,488 bytes, SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1` |
| published bank metadata | 1,200 fitting records, 124,371 valid post-BOS positions, 48 validation records |
| published support | 17,126 token IDs, support digest `8090b9231042d6c9f9c52f3a0a9d8733b8ec1b739a4f28b12ab1057a7126969d` |

The four crossed arms all start from the same selected fixed state and use
the same declared update budget.  The unchanged method is the separately
published current-bank residual anchor recorded by the shared contract:
`experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors`,
29,390,628 bytes, SHA-256
`2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`.  It is
loaded without TRR-0010 updates.  If the root freezes a different unchanged
identity, that state binding must replace this row before fitting; it cannot be
inferred from final answers.

| bank | fixed readout | trainable directions |
| --- | --- | --- |
| current 1,200-record bank | `current_fixed` | `current_directional` |
| expanded nested bank | `expanded_fixed` | `expanded_directional` |

The fixed arms continue the decoder with the public readout `E` fixed.  The
directional arms use the same base decoder and update schedule while adding
the directional readout parameters.  This makes `current_fixed` the
continued-training control for the data-size comparison and keeps bank size
and direction freedom crossed.  The unchanged state remains a separately
reported starting reference.  The retained TRR-0009 gain/bias state is not a
TRR-0010 directional arm.

The shared-contract proposal in
`experiments/TRR-0010/planning/shared_contract.proposal.json` currently uses
batch 8, 512 post-BOS positions per update, seed 4010, 20,000 updates,
checkpoint/validation interval 1,000, gradient clip 1.0, zero weight decay,
and decoder learning rate 2e-4.  Those values remain proposal-level until
the root freezes them; every arm must receive the same exact update budget and
schedule once frozen.  The shorter TRR-0009 3,000-update schedule is a
runtime reference, not a silent substitute.  Receipts must report unique
records, valid positions, exposures, updates, validation size, preparation
time, optimizer bytes, state bytes, and wall time separately for each arm.

## Directional readout proposal

Use one single deployed readout state containing the inherited decoder,
support IDs/counts, and a token-specific additive row delta.  For fitting-only
support (S), let Delta[S] have shape `[|S|, 2048]` in FP32.  The shared
contract currently proposes a direct table; no low-rank or gain/bias
alternative is substituted in this evaluation path:

```
delta_E[S] = Delta[S]
logits_v(H) = base_logits_v(H) + H @ Delta[v]          for v in S
logits_v(H) = base_logits_v(H)                         for v not in S
```

The prediction still materializes the complete 128,256-way logits and takes
one argmax.  It does not route, search, rank teachers, use context beyond the
current `H_i`, or materialize a second full dictionary.  Each supported row
has an independent coefficient vector, so this is a genuine token-specific
direction change rather than a gain/bias or a global hidden-space transform.
Rows absent from fitting labels have no delta and remain exactly the public
readout rows.  Support IDs and counts are derived only from the fitting
split, frozen in the state, and checked against the bank digest at load time.

Initialize Delta to zero, so the deployed logits are exactly the shared
starting state at step zero.  Use the shared-contract proposal's fixed
frequency-weighted relative L2 anchor against the public rows, with
`lambda=1e-4` and no post-hoc tuning:

```
1e-4 * mean_v((1 + 4/sqrt(count_v)) *
              ||Delta[v]||_2^2 /
              (2048 * rms(E[v])^2 + eps))
```

The public row itself is the anchor; the zero initialization and saved
starting-state digest make the initialization exact.  A synthetic zero-state
check must show bitwise-equivalent predictions before fitting.  After fitting,
the state must be loadable as one readout and must preserve full-vocabulary
output for unsupported IDs.

The deployed directional state adds approximately

```
4 * (2048 * |S|) bytes
```

for FP32 parameters, plus metadata.  It is about 140.3 MB at the current
17,126 support IDs and about 1.05 GB if nearly the whole vocabulary is
supported.  Adam and gradients add about 561.2 MB and 4.20 GB respectively
during fitting; they are not part of deployed inference.  These figures are
reported explicitly rather than hidden in a materialized embedding artifact.

## Public bank and source-range constraints

The only exact source ranges visible in the allowed TRR-0009 selection
metadata are half-open ranges Finance `[12000, 20000)` and Pile
`[7000, 10000)`.  TRR-0009 selected 256 Finance rows (source indices
12021--19988) and 128 Pile rows (7021--9935).  The frozen selection and
exclusion records are:

* selection v2: `c2e996514f7f45e55d7bfadc27fc8048bdb07a2c979b208de8716972d8d1def2`;
* identity exclusions: `bba988428e6e1d13c2916b5e9c001acd86aa37d838bda553744eecd0f88d806f`;
* the selection uses the producer-defined `public_record_sha256` rendered
  record hash and opaque P04/P06/P08 ledgers, with no source text or token IDs
  opened.

These facts are availability metadata, not authorization to select those rows
again.  Rows used for a final panel cannot enter a fitting bank.  If the
Finance range is used for fitting, a separate final range must be reserved;
if it is used for final evaluation, every fitting row and every prior ledger
must be excluded.  The same role decision must be made before Agent 2 builds
the expanded bank.

The current Pile source is `NeelNanda/pile-10k`; it has no source indices at or
above 10,000.  Therefore “Pile beyond 10000” is not a valid range for that
dataset.  A compatible extension requires a separately identified public
dataset/revision, rendered-record hash convention, and an opaque zero-overlap
receipt against P04/P06/P08, TRR-0009, and the final panel.  No extension is
claimed here.  Finance `[12000, 20000)` can provide a large public range only
after the role assignment and all identity exclusions are frozen.

For the expanded bank, target 12,000 unique records (at least 10,000 for an
order-of-magnitude claim), preserving natural domain proportions and nesting
the exact current bank.  If approved public availability yields less than an
informative eightfold expansion, record the shortfall and do not silently
substitute repetitions for unique examples.  Agent 2 must provide the exact
selected record-hash set, source revision, fit/validation split, support
digest, and all opaque exclusion receipts before any fit or final selection.

## Common evaluation and A1/A2 comparator boundary

Use one public-model forward capture per frozen final panel.  The capture
must emit matched `public_base` and `public_lora_2601` observations for each
source, with the current-H/full-vocabulary geometry above.  Reuse the TRR9
capture/prediction boundary and its numerical settings:

* BF16 activation input staged to FP32;
* FP32 decoder and embedding; autocast disabled;
* CUDA matmul TF32 disabled, cuDNN TF32 enabled, FP32 matmul precision
  `highest`;
* chunked activations, no source prefix beyond BOS, no history, and A2 false.

The five scientific methods are `unchanged`, `current_fixed`,
`current_directional`, `expanded_fixed`, and `expanded_directional`.  All
must consume the same frozen observation files and the same record order.
For every method, compare the first 32 records in each cell against its
frozen prediction digest before timing; recheck code, state, observation,
support, and method bindings before writing the timing receipt.  A1 and A2
are separate comparator paths.  A2 is allowed its declared causal
prediction-prefix input for that comparator only and is never loaded into a
TRR-0010 deployed method or its timing denominator.

The allowed TRR-0009 metadata contains no authoritative A1 or A2 method,
state, source-panel, model-configuration, code-commit, or runtime records.
The following must be bound by the root before freeze; no path or hash is
inferred from another study:

```
A1: method/state/loader/code commit; source-selection and observation
    records; target conditions; numerical settings; runtime and memory.
A2: same fields, plus its explicitly permitted causal-prefix input and
    inference contract; deployment exclusion must be true.
```

This is the remaining historical-availability dependency.  It is a blocker
to an A1/A2 claim, not a reason to fabricate a comparator from TRR-0009.

## Timing design

The five scientific methods require one extra identical-state alias to retain
the TRR-0009 execution-path control.  Thus the scientific method count is 5
and the timing path count is 6.  The alias is a harness path, not a sixth
scientific method.  The recommended fixed plan is:

* 32 predetermined records per cell, four cells, one warmup and one measured
  call per record;
* 48 blocks, seed 8008, six rotations followed by six reversals per complete
  cycle (four complete cycles), with no outcome-dependent order selection;
* one alias of the declared primary fixed denominator, preferably
  `expanded_fixed`, with exact prediction equivalence required before timing;
* all candidate/fixed ratios reported by cell with Student-t block CIs;
  qualify a directional candidate only when every cell's upper 95% CI is at
  most 1.25; alias failure invalidates cost qualification, while alias
  inconclusive remains cost-inconclusive.

Forty blocks cannot be an exact rotation/reversal balance for six paths.  The
48-block choice is prospective and avoids silently using a partially balanced
40-block schedule.  It yields 36,864 measured and 36,864 warmup calls across
the six paths.  If both fixed controls require independent aliases, the
seven-path/56-block design must be frozen before timing; it is not an
adaptive extension.

The timing boundary remains the TRR9 `predict_current_h` call with explicit
device synchronization, full-vocabulary logits, CPU argmax, and prediction
digest checks.  Guards run at phase/block boundaries plus lightweight checks
between records, not an expensive `nvidia-smi` call for every record.  Record
block ratios, per-record variability, startup/preparation time, CPU/GPU
telemetry, temperature/utilization/load when available, memory, code/state/
input hashes, and all failures.  Timing stops on a resource anomaly.

## Scoring and denominator

Score each domain and target condition separately.  For each source, report
token accuracy over the 127 scored positions and exact 127-token recovery;
retain paired source deltas between each crossed arm and its fixed control.
Frequency strata use fitting-label support counts: token accuracy is summed
correct divided by summed exposed tokens, with whole-source bootstrap units.
Absent and rare strata are `UNKNOWN` when their source denominator is below
the predeclared floor; they are never pooled to manufacture a pass.

The A1/A2 comparison denominator is the set intersection of public record
hashes after all approved ledgers:

```
D_domain,target = TRR0010_final ∩ A1_final ∩ A2_final
```

The same source must have both matched target conditions when the paired
contrast is reported.  The useful target is 256 sources per domain and
condition (512 total source records); 128 per domain is the minimum acceptable
common subset for a bounded result.  The final receipt must print the exact
count and digest per domain/condition.  If a domain falls below 128, report
that A1/A2 gap closure as unavailable rather than pooling domains or target
conditions.

For accuracy (q), define the per-cell gap closure only when the A2--A1
denominator is positive and estimable:

```
gap_closed = (q_candidate - q_A1) / (q_A2 - q_A1)
```

Report the absolute candidate quality, paired gain/regression, and the
denominator alongside this ratio.  A nonpositive or uncertain denominator
produces `NOT_COMPUTED`; it is not evidence of closure.  A1/A2 comparator
timing is reported separately from the five deployed current-H methods.

## Smallest code adaptations

Keep TRR-0009 source files unchanged and import their low-level behavior
through task-local adapters:

1. Parameterize a TRR-0010 contract with the five method IDs, bank/support
   bindings, state file records, panel counts, full-vocabulary geometry, and
   A1/A2 comparator records.  Require bytes and SHA-256 for every source,
   state, support, embedding, observation, and code binding.
2. Add one directional-readout model/loader implementing the factorized
   `Delta[S]` rows and zero-state equivalence.  The loader must reject missing,
   stale, or mismatched support IDs/counts and must report unsupported-token
   behavior explicitly.
3. Adapt the capture/runner registration to accept the five methods while
   reusing `predict_current_h`, numerical configuration, guards, and
   create-only receipts.  A1/A2 comparator input is kept in a separate
   adapter and cannot enter the deployed method list.
4. Generalize timing schedule generation from five paths to the frozen six
   paths (or the separately frozen seven-path variant), while retaining the
   TRR9 Student-t CI and exact prediction-equivalence checks.
5. Extend scoring to the two bank factors, support-frequency strata, paired
   target deltas, and the explicit A1/A2 denominator.  Truth access remains
   after complete public freeze, gate validation, and prediction integrity.

## Honest preflight estimate

TRR-0009 provides the following measured anchors on the same public model:

* one 4-method, 768-record prediction run took 145.87 s, with approximately
  1.35 GB CUDA reserved and 1.74 GB process RSS;
* its 5-path, 40-block timing run took 235.26 s;
* the largest 3,000-step public fit used 3.92 GB peak CUDA reserved and
  5.23 GB peak RSS, with the 1.05 GB embedding resident;
* the fixed and supported-readout arm walls were 355.8 s and 393.8 s.

Five report methods should take about 182 s for the same 768-record panel
before the directional table overhead.  The recommended six-path,
48-block timing scales the measured TRR9 timing to about 339 s; allow about
380 s for directional arithmetic and startup, bounded by a 600 s phase cap.
Capture is expected to be on the order of 20--30 s for the same geometry.
The expanded-bank fit must stream activation chunks; a tenfold copy of the
published 947 MB fit activation payload would be roughly 9.5 GB and must not
be loaded whole.  The fit and evaluation guards remain at 6 GiB reserved GPU,
16 GiB RSS, 8 GiB minimum free GPU, and 10 GiB minimum host available unless
a new measured preflight is approved.

Before any heavy run, qualify the largest expanded-bank training batch and a
full-vocabulary directional inference record on the live device.  Require
finite loss/gradients, zero-state output identity, support digest identity,
and memory/time margin.  A failure leaves a receipt and stops the run; no
batching or method order is changed after seeing an outcome.

## Freeze prerequisites

Before Agent 2 begins the expanded fit, the root should bind:

1. the exact current-bank fit/validation manifest and nested expanded-bank
   source-hash set;
2. the disjoint final source range, with opaque P04/P06/P08/TRR-0009 ledgers
   and a declared Finance/Pile count;
3. the exact A1/A2 state, loader, source, observation, code, runtime, and
   permitted-input records;
4. rank, initialization sketch, regularization, update seed/schedule, and
   the six-path timing plan; and
5. the final public panel and support bindings before one common truth
   evaluation.

Until these records exist, this proposal is a reusable design and an honest
availability assessment, not an authorization to select sources or run
fitting/evaluation.
