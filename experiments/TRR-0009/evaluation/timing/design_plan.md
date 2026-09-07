# TRR-0009 evaluator/timing/integrity reuse plan

Status: **DESIGN_ONLY_NO_EVALUATION**. Recorded UTC: `2026-09-07T00:11:47Z`. This plan was prepared after the restored assignment and the training interface design; it does not authorize training, selection, prediction, timing, truth, or scoring.

## Frozen scientific interface received from training

The four reported methods are `unchanged_anchor`, `continued_fixed_readout`, `continued_adaptable_readout`, and the retained published `published_reference` context. The three research arms share the published TRR-7 current-bank residual checkpoint selected at step 2100 (`trr0007_residual_mlp512`, SHA-256 `2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`, 29,390,628 bytes); the published reference is not fitted. The adaptable arm keeps the current-H transform and fixed public embedding `E`, then applies the supported-token bounded row correction defined by the training design: two zero-initialized scalars per supported token, with absent IDs left at the anchor (`gain=1`, `bias=0`). The training design binds 17,126 supported IDs and 34,252 trainable scalars.

Every inference path uses only the current `H`, matching mask/positions, the declared BOS, and the full vocabulary. There is no A2 path, token history, earlier hidden state, candidate simulation, teacher ranking, or source selection in the evaluator.

## Lean reuse boundary

The implementation should keep TRR-0008’s trusted low-level behavior and replace only hardcoded task bindings:

| TRR-0008 component | TRR-0009 reuse/adapter | Required TRR-0009 change |
|---|---|---|
| tensor/file helpers, normalized prediction, create-only writes | import low-level helpers where geometry matches | task/schema IDs and generic method/cell validation |
| `predict_current_h`, synchronized full-vocabulary row scoring, finite-logit checks | import or wrap the current-H primitive | verify direct post-logit zero-correction initialization and exact output equivalence for all declared states |
| runner chunking, resource guard, warmup/measured recording | thin parameterized runner | method rows come from the owner registration; complete registration binding includes `bytes`, path, and SHA before first write |
| public gate and pretruth freeze | parameterized gate adapter | validate the frozen four-method matrix, code/input/state hashes, run-manifest registration bytes, prediction/timing tensor digests, and truth-free flags |
| CP and paired record-bootstrap primitives | import arithmetic primitives into new scorer | owner method roles, per-cell contrasts, fit-frequency strata, and no pooled target conditions |
| balanced timing, synchronization, CI summary, startup/telemetry evidence | parameterized timing adapter | schedule is fixed for the frozen method count; init-equivalence receipt and full-vocabulary output equivalence are mandatory |

Do not copy the TRR-0008 task-specific 10k-line surface or silently accept its four-method hardcoding. The new files should expose task-local schemas and call reusable pure primitives only where their assumptions are explicitly checked.

## Registration and integrity sequence

1. The owner freezes a plan and registration before any fresh evaluation. Registration records method IDs/roles, exact loader and state bindings, public observation/capture/source-selection bindings, fitting-frequency binding, numerical settings, resource guard, timing-plan record, and the full code-binding list. Every file record has `path`, `bytes`, and `sha256`.
2. The runner verifies registration, code bindings, input manifests, states, and embedding before model load; it records the same bindings again after the last prediction and before writing success. Any drift fails closed. It writes create-only prediction and timing artifacts and emits a run manifest whose nested registration file record includes `path`, `bytes`, and `sha256` from the start.
3. The gate verifies the complete method×cell matrix, tensor shapes/dtypes/BOS/padding, artifact hashes and tensor digests, timing rows, method/state/input/code bindings, warmup/measured exact-ID equality, and all truth-free flags. It rejects a missing registration byte count, missing/corrupt/different prediction root, altered state/input/code, source labels, or candidate arrays before truth.
4. The gate emits a create-only public freeze receipt. Only that receipt authorizes truth preparation. Truth preparation remains outside the runner and opens no labels before this point.
5. The scorer verifies the freeze, binding hashes, sidecar metadata, and prediction root, then opens the truth sidecar exactly once. It writes only serialized metrics; it cannot update states, route records, revise timing, or alter predictions.

## Score contract

All quality results are cell-local: each domain×target condition is scored separately and target conditions are never pooled. Exact recovery is the 127-token post-BOS clip endpoint. Token results use paired source-record means. Primary comparisons are adaptable versus fixed-readout, with unchanged-anchor versus fixed/adaptable and any optional published-reference contrast labelled according to the owner-frozen decision contract.

The fitting frequency map is derived only from public fitting labels and is bound before evaluation. After the truth gate opens, each cell’s truth IDs are assigned to declared fitting-frequency bins (`unseen_0`, `seen_1_4`, `seen_5_9`, `seen_10_49`, and `seen_50_plus`). The `unseen_0` and `seen_1_4` bins are safeguards: each cell reports its own weighted correct/exposed-token ratio and 95% source-record bootstrap lower bound, with a lower bound of at least -0.01 required; fewer than 32 exposed source records is `UNKNOWN`, and a zero denominator is `NOT_COMPUTED`. These gates cannot influence prediction, routing, stopping, or method selection. Each cell/stratum keeps its own denominators and paired source-record uncertainty; no target condition is pooled or fabricated.

For a declared composite exact alpha `alpha`, the paired gain/loss bound uses two component tails of `alpha/2`: `L_gain = beta.ppf(alpha/2, gains, n-gains+1)` and `U_loss = beta.ppf(1-alpha/2, losses+1, n-losses)`, with explicit zero/all-success endpoints, then `LCB = L_gain - U_loss`. Token bounds use the owner-frozen one-sided per-record bootstrap seed/draws. Focused tests must compare both directions and endpoint cases with SciPy; no pooled target-condition bound is permitted.

## Timing contract

Timing uses identical current-H inputs for every method, full-vocabulary scoring, explicit pre/post device synchronization, one fixed warmup and one measured invocation per row, exact warmup/measured ID equality, and separate startup/readout materialization/I/O accounting. The initial adaptable readout must be checked against the unchanged anchor at zero correction with exact prediction equality before any timing or quality claim. The final method states are then timed under the same cell/order workload.

The scheduler is deterministic and balanced for five timing paths: the four reported methods plus an identical-state/loader alias of `continued_fixed_readout`. It uses 40 blocks, 32 records per cell, one warmup and one measured invocation per row, and four complete cycles of five rotations plus their reversals. The alias must have exact prediction equivalence and each cell's 95% runtime CI must be fully contained in [0.95, 1.05]. Candidate/fixed cost qualifies only when every cell's upper 95% ratio CI is at most 1.25; an alias failure is `INVALID_ALIAS_CONTROL`, and an inconclusive alias makes cost evidence inconclusive. No block count, order, or result can be selected after observing latency. The authoritative inference path applies the bounded correction directly to post-logit rows, avoiding materialization rounding. Full dictionary materialization is outside this round's inference path and is excluded from the timing claim. The timing receipt reports per-cell ratios and CIs, direct-path preparation and optimizer/training costs, per-record variability, telemetry, peak memory, and all code/input/state hashes.

## Test gates before any GPU release

- complete registration records including `bytes` and actual hash verification;
- synthetic successful runner/gate path plus rejection of missing/corrupt/different prediction roots, state/input/code drift, truth flags, and warmup/measured mismatch;
- exact zero-state adaptable-readout equivalence to the unchanged anchor;
- the four reported methods and five timing paths, with no A2/history fields;
- per-cell CP alpha/2 regression, zero/all-success endpoints, paired bootstrap, and no pooled target keys;
- absent/rare fitting-frequency bins bound to the public fit map only, empty-stratum handling, and post-gate truth ordering;
- balanced timing schedules for three and four methods, synchronized warm/measured boundary, fail-closed guards, and deployment/preparation cost accounting.

The read-only bound and prior evidence are in [`resource_preflight.json`](resource_preflight.json). Evaluator, capture, gate, score, truth-boundary, and timing implementations plus CPU synthetic integrity tests are ready. Remaining readiness dependencies are the trained selected states and the owner’s resource lease; no evaluation, capture, timing, truth, or scoring run has been authorized here.
