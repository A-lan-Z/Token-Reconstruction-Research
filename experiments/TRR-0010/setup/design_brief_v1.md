# TRR-0010 Agent1 directional-readout setup brief

Status: setup and synthetic implementation planning only. No TRR-0010 fitting,
source selection, fresh capture, truth access, or GPU training has started.

## Shared starting resources

The isolated worktree is `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010`, branch `task/TRR-0010`, at the exact shared starting commit
`4602f98eeb03ac121d8bbc230d7e2b219551b914`.

The proposed common decoder start is the TRR-0009 continued-fixed selected step
400 state:

- `../TRR-0009/experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors`
- 29,390,492 bytes, SHA-256
  `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`
- current-H only, full-vocabulary tied-E, hidden width 2,048, context width 128,
  vocabulary 128,256, A2 disabled.

The shared public normalized embedding is
`/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`, shape `[128256,2048]`, FP32, 1,050,673,488 bytes, SHA-256
`ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
The current published fitting bank is TRR-0007 `coverage_mix_v1`:
`../TRR-0007/experiments/TRR-0007/support/broader_capture_v2/enriched_manifest.json`, 1,200 records, H `[1200,192,2048]`, 124,371 valid post-BOS positions, manifest SHA-256
`c7a857e545a2f252ce8b3ab71bb2336e552fd212c7c434e39cef66f233b77a08`.
The expanded bank and its support set remain Agent2-owned and are not selected
or opened by Agent1 in this setup phase.

The inherited shared schedule is
`../TRR-0007/experiments/TRR-0007/improved_fit_v1/improved_public_bank/schedule.safetensors`, file SHA-256
`dcf439e2221bf34cd526a2b3fa0e5ebba0e6393e6ee2d4525d3b7edf6e9eea94`, semantic digest
`5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472`.
It contains 3,000 steps, batch size 8, and 512 position draws per step. The
actual stored position slots range from 1 through 191; 131,576 of 1,536,000
draws are at positions 128--191. The fit mask is width 192 with 124,371
post-BOS valid positions, so TRR-0009 training used the full width-192 mask;
its final 128-token evaluation clip does not define training geometry.

## Proposed one directional parameterization

Use a sparse supported-row additive direction table, initialized to zero:

```text
E_eff[v] = E[v] + Delta[v]       for v in the fitting-bank support S
E_eff[v] = E[v]                    for v outside S
logit[v] = logit_scale * <z_i, E_eff[v]>
```

`Delta[S]` is a trainable `[|S|,2048]` FP32 table. It changes each supported
token's scoring vector directly, preserves the public E initialization exactly,
and is not a gain/bias correction or a global absorbable linear transform.
Absent rows have no parameter and remain fixed. The full vocabulary is scored
for every position. The registered `score_rows` hook materializes `E_eff` and
performs one merged full-vocabulary matmul. A separate `sparse_score_rows` /
`sparse_forward` path exists only as a diagnostic for equivalence and memory
comparison; it is not an active training, validation, or deployment method.

The proposed anchor is a single frequency-weighted relative L2 penalty, with
`lambda=1e-4` inherited as the one fixed anchor strength:

```text
lambda * mean_v[(1 + 4/sqrt(count_v)) *
                ||Delta[v]||_2^2 / (H * rms(E[v])^2 + eps)]
```

There is no hard gain/bias bound and no row renormalization at zero; zero
initialization must preserve logits and decisions bit-exactly. The common
trainer will own the optimizer, sampler, schedule, validation, checkpointing,
and fit/validation accounting. Agent1 will provide the directional readout,
penalty, state export, materialization-equivalence hook, and synthetic tests.
Both directional arms must start from the same fixed state as their matched
Agent2 controls. Final schedule and support binding remain subject to the
shared contract before substantive fits.

The merged primary path includes one full materialized E_eff and its dense
backward gradient; the exact current and full-support bounds are recorded in
the Revised readout memory accounting section below. The sparse implementation
remains diagnostic-only. The largest actual expanded support must pass a live
guard and merged-primary qualifier before substantive fitting.

## Required initialization and causality checks

Before any fit, synthetic tests will check zero-correction logits/argmax
identity, directional-column-only changes, absent-row invariance, current-H
causality, finite gradients and penalty, sparse-hook versus materialized-E
output equality, state/export round trip, and the parameter/memory formula.
No fit or fresh evaluation starts until Agent2's bank manifests, the common
contract, compute lease, and largest-cell resource preflight are frozen.

## Shared hook and scoring-path refinement

The task-local adapter now matches the shared Agent2 `ReadoutHook` surface. The
common decoder produces normalized current-H rows `q` and inherited scalar
`exp(s)`. Agent1's `score_rows(query_rows, logit_scale, embedding,
base_logits=None)` receives only those rows, the immutable public E, and an
optional already-computed full-vocabulary base score. The registered hook
materializes `E_eff` and scores every row with the same dense matmul used by
deployment. It never routes on target labels, source IDs, position choices, or
sampler state; public E is never renormalized.

The merged path is the primary training, validation, and registered-inference
path. Full-vocabulary CE remains the only data loss, plus the fixed
frequency-weighted relative L2 anchor (`lambda=1e-4`). The direct Delta group
has one fixed learning rate of `1e-4`; the shared decoder group uses the
common continuation rate, currently proposed as `2e-4`. No rate sweep is
planned. The adapter exposes `optimizer_param_groups`, `loss_terms`, and
`metadata` for the common runner.

`score_rows` accepts `base_logits` for protocol compatibility and geometry
validation, but does not silently substitute the sparse score. The old sparse
correction is retained as `sparse_score_rows` / `sparse_forward` for a
diagnostic comparison only. Before any state is selected, the sparse diagnostic
and dense primary score must be compared on declared validation rows with
recorded max-absolute-logit and argmax tolerances. No sparse result may enter
the registered method matrix without a contract amendment.

The dense `E_eff` is created from immutable public E plus the trainable Delta
rows on every primary score call; this keeps one hook semantics while exposing
the real dense workspace to the resource qualifier. The create-only exporter
records the same effective rows as a separately bound full
`[128256,2048]` FP32 artifact.

## Revised readout memory accounting

The 1,050,673,152-byte FP32 materialized E_eff and a same-sized dense
backward gradient are included in the primary bound. The materializer uses one
out-of-place `index_copy` from detached public E; it does not retain a second
full clone. Two support-row buffers (gather and add) are counted separately.
Using the prior 3,917,479,936-byte base reserved peak and 512 sampled rows:

- Current support K=17,126: sparse diagnostic estimate 4,513,738,752 bytes
  (4.204 GiB); merged-primary estimate 6,860,603,392 bytes (6.389 GiB).
- Full-vocabulary support upper bound K=128,256: sparse diagnostic estimate
  8,382,840,832 bytes (7.807 GiB); merged-primary estimate 12,322,865,152
  bytes (11.477 GiB).

The estimates include Delta parameters, its FP32 gradient and two Adam
moments, one full E_eff tensor, its dense backward gradient, and the two
K-by-2048 support-row materialization buffers. The N-by-K sparse score buffer
is counted only in the diagnostic phase. E_eff is derived from public E and is
not a separate optimized dense parameter, so no dense E_eff Adam moments are
counted; Delta's optimizer state is included. The primary optimizer should use
`foreach=False` or an equivalent bounded-scratch setting when the common
runner freezes that option. The largest expanded support must still pass a
live resource guard and merged-primary qualifier before any substantive fit.
Large Delta/E_eff files are task-owned immutable artifacts referenced by
manifests and hashes; no duplicate copies are placed in source control.
