# TRR-0010 / TRR-P09 Agent-1 integration and capture preflight review

**Status:** `CONDITIONAL_RECOMMENDATION_NOT_EXECUTION_READY`

This is a read-only review of the authorized Agent-1 handoff and the P09
common runner. No Agent-1 files were edited. No model, data, public weights,
truth, or GPU tensor was loaded by this review, and no fit or capture was
run.

## Reviewed identities

The reviewed Agent-1 contract is
`experiments/TRR-0010/planning/shared_contract.proposal.json`, SHA-256
`3b0a68a4a05304e285c1fedcf8a2982bf4bef831e2e5bc0cebb20b8b2dba08b3`, schema
`token-reconstruction.trr0010-shared-contract-proposal.v4`, with status
`PROPOSAL_V4_READY_FOR_ROOT_REVIEW_BEFORE_A2_HANDOFF`.

The directional implementation is
`scripts/trr0010_model.py`, SHA-256
`9eb38939ba0541f51ce7a964080a058c313081f7ef7b25e2b51f91bac92310db`.
The thin P09 integration wrapper is from Agent-1 commit
`40219147cdf393d79436770a9a0cc9f0535e7c85`:

- `scripts/trr0010_p09_integration.py`, SHA-256
  `d648ca4f33b7fdf2c51ebf58f055291e6cb3bbaa0d663cef138181eb0ba7505c`;
- `tests/test_trr0010_p09_integration.py`, SHA-256
  `079183935c713dd413f68df0953228760daf532395f9e2f6d9f3aebb82bb921a`;
- `experiments/TRR-0010/setup/p09_integration_checklist_v1.md`, SHA-256
  `0d8f382f15003acc19659cb3e5c0b73c4bfd7a08b52849870f0779954147967a`.

The handoff JSON claims `head_commit=4602f98e...`, while the authorized
Agent-1 worktree currently resolves to `871ae84478b4957919774faa10bfcf126b06d6d9`.
The execution receipt must record both the inherited starting parent and the
actual worktree commit; the handoff field alone is not a sufficient source
identity.

The P09 common runner reviewed here is `scripts/trr_p09/fixed_control_runner.py`
at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, file SHA-256
`0a54945968f5f797b53b09df72150f4f44dd2e623ae2929db68751d60389cc2e`.
The current P09 branch has later metadata-only commits, including
`788f906b1de680e853d6b4c971e803cccc6bbb98`; the runner source hash remains
the binding identity for this interface review.

## Integration decision and production wiring

The proposed query-aware interface is compatible with the P09 row dispatcher.
The directional arm should be wired once per arm as follows:

```text
base = directional.base
hook = directional
directional.to(device)
E = E.to(device)
directional.bind_embedding_statistics(E)
groups = directional.optimizer_param_groups(base, base_learning_rate=2e-4)
optimizer = AdamW(groups, weight_decay=0.0, foreach=False)
run_training(..., decoder=base, hook=hook, compute_base_logits=False)
```

`compute_base_logits=False` is required for the directional path. The
Agent-1 hook's registered `score_rows` path materializes the merged effective
embedding and performs the full-vocabulary query-by-`E_eff` matmul. It accepts
`base_logits` for interface compatibility but does not reuse them, so passing
`True` would perform a redundant public-E projection. The fixed arm should use
the inherited fixed hook and its exact base-logit path; it should not be
silently implemented by the directional class.

The public embedding must be staged on the same device as the decoder and
hook parameters before the loop. `bind_embedding_statistics` must run after
device placement and before the first loss, because the anchor penalty
requires the bound RMS vector. The shared runner's optimizer coverage check is
compatible with `base` plus the nested hook, and the Agent-1 wrapper exposes
the intended decoder and directional parameter groups. `foreach=False` should
be an explicit execution receipt field and a checked optimizer construction,
because the contract currently says `PENDING_A2_AGREEMENT` and the full merged
table creates a large transient allocation.

The registered primary path is the merged dense `E_eff` path. The sparse
supported-column routine is diagnostic-only: it may be used for an output
equivalence check, but it cannot be substituted for fit, validation, or
deployment.

## Amendments required before a fit

1. **Selection metric:** `run_training` currently records micro-averaged raw
   token accuracy and accepts a single string metric. The contract requires
   the equal-domain-weighted validation metric
   `mean(Finance token accuracy, Pile token accuracy)` on 256 Finance + 128
   Pile public-base records at H128. Production integration must provide a
   domain-aware validation callback or precomputed metric object and make the
   checkpoint selector consume that declared metric. It must not select on
   the runner's raw micro-average by accident.

2. **Optimizer scratch policy:** construct and receipt-check
   `torch.optim.AdamW(..., foreach=False, weight_decay=0.0)` for all arms,
   unless root explicitly freezes an equivalent bounded-scratch policy.
   The directional group is base LR `2e-4` and Delta LR `1e-4` under the
   current proposal. All unique trainable decoder and hook parameters must be
   in the one optimizer, with gradient clipping applied to that union.

3. **State binding:** the generic P09 `method_state_digest` hashes the base
   decoder once as `decoder::` and again through the nested directional hook
   as `readout::base::`. This is internally deterministic but differs from
   Agent-1's directional state serialization digest. Use the Agent-1
   `save_directional_state` binding (or an explicitly declared adapter digest)
   as the authoritative directional checkpoint identity, and record the P09
   generic digest only as a separate diagnostic if retained.

4. **Capacity qualification:** Agent-1's 6.389-GiB current-support and
   11.477-GiB full-support values are estimates, not qualifications. Before a
   full matrix, qualify the actual expanded support with B=8, width=192,
   512 sampled rows, merged dense `E_eff`, backward, AdamW state allocation,
   `foreach=False`, and the selected activation dtype. Preserve the public
   output/argmax equivalence of zero-Delta step zero and a serialize/reload
   equivalence check. Stop on allocator, driver, thermal, or guard failures;
   do not microbatch as a memory workaround without a numerical equivalence
   receipt.

5. **Contract status:** exact B0/B1 recipe and exclusion identities, the
   validation-panel source receipt, the common optimizer policy, the A1+A2
   comparator identity/cost, and the actual resource caps remain pending in
   the Agent-1 proposal. This review therefore recommends wiring review only,
   not execution readiness.

## Inherited public capture path

The existing producer path is the one to reuse. It loads the pinned
`meta-llama/Llama-3.2-1B-Instruct` snapshot
`9213176726f574b556790deb65791e0c5aa438b6`, verifies the full model weight
file (2,471,645,608 bytes, SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`), and
calls `AutoModelForCausalLM.from_pretrained(..., local_files_only=True,
dtype=torch.bfloat16, attn_implementation="sdpa")`. It runs the first four
layers through `ContiguousPublicPrefix.forward_full`, in `eval()` and
inference mode, with no gradient.

The exact capture geometry is:

- padded input IDs: batch 8, sequence width 192, `torch.long`;
- full forward output: `[8, 192, 2048]`, BF16;
- retain compact H128 output `[records, 128, 2048]`, BF16;
- store attention-mask and position-ID sidecars for the compact output;
- invoke the existing resource check after every batch and capture one style
  cell at a time.

The producer does not pass a separate attention mask into
`forward_full`; the fixed right-padded IDs use the inherited causal-mask
path, and the sidecar mask is used to zero/validate padded output rows. The
qualification changed future padding IDs and compared active outputs with
`torch.equal`: the fixed 8x192 path passed with maximum absolute difference
zero. The unpadded/batch-1 diagnostic was non-equivalent and is explicitly
excluded from capture. Therefore no batch-1, unpadded, or microbatched capture
substitution is permitted.

The exact inherited helper bindings are:

- `scripts/trr0005_produce_confirmation.py` trusted loader, SHA-256
  `7289ee5db30bc0b2d3c1b50b1d5b6ca1ac235001371f61b92adf10e2dd979e2b`;
- `scripts/trr0004_produce_confirmation.py` capture loader, SHA-256
  `b1ae4e199de62f350df0ce9890caa90a70b0611a22051ce6bc304e0ccc4e1b2a`;
- `src/token_reconstruction/public_prefix.py` native prefix implementation;
- `src/token_reconstruction/public_activation.py` fixed-batch capture helper.

The inherited 8-row qualification measured public-base peak CUDA allocated
`2,472,678,400` bytes and reserved `2,478,833,664` bytes; the corresponding
public-LoRA qualification reserved `2,499,805,184` bytes. A broader inherited
capture receipt observed maximum reserved `3,546,284,032` bytes and maximum
process RSS `5,816,393,728` bytes. These are reference measurements, not a
P09 guarantee, but they support retaining the inherited fail-closed guard.

## Capture geometry and bounded resource proposal

One full 8x192 BF16 hidden batch is
`8*192*2048*2 = 6,291,456` bytes (6 MiB); the retained 8x128 compact batch is
`4,194,304` bytes (4 MiB). For the inherited largest 1,536-record domain
cell, full H192 storage is `1,207,959,552` bytes (1.125 GiB), compact H128
storage is `805,306,368` bytes (0.75 GiB), and simultaneous full-plus-compact
host tensors are about `2,013,265,920` bytes (1.875 GiB). Capture must free
the full cell before proceeding to the next condition/cell. For a 384-record
development cell, the corresponding full-plus-compact bound is 0.46875 GiB;
the exact frozen record count should be recorded rather than inferred.

Use these inherited hard caps for a future largest-cell qualification and
capture, pending root's stage-1 contract sign-off:

| resource | fail-closed cap | current read-only snapshot |
| --- | ---: | ---: |
| GPU free before/through each check | at least 8,589,934,592 B (8 GiB) | 12,465 MiB free; 3,513 MiB used of 16,303 MiB |
| GPU reserved by child | at most 8,589,934,592 B (8 GiB) | not allocated by this review |
| child/process RSS | at most 17,179,869,184 B (16 GiB) | no capture child; Codex host process was ~1.31 GiB RSS |
| host available | at least 10,737,418,240 B (10 GiB) | 20,109,090,816 B available |
| capture filesystem free | proposed at least 20 GiB before each cell | 287,708,635,136 B free on `/tmp/trr-p09` filesystem |

The snapshot was taken at `2026-09-07T08:34:05.229111540Z` with
`nvidia-smi`, `/proc/meminfo`, `free -b`, and `df -B1`; it is a planning
snapshot, not a reservation. A future run must refresh it immediately before
loading the model and must preserve the actual live values in its receipt.
The disk threshold is a proposed bounded cap because the inherited receipt
does not define a disk guard; the expected compact output for a 1,536-record
cell is 0.75 GiB before sidecars, and cells must be written sequentially.

## Required largest-cell qualification sequence

After stage-1 sign-off and a fresh resource reservation, execute exactly the
existing producer qualification before the matrix:

1. Refresh GPU free/used, host available/RSS, disk free, and process lists;
   fail closed if any cap is not met.
2. Bind the frozen model snapshot, tokenizer, source selection, source code
   hashes, and public-base condition. Load the prefix with the inherited
   BF16/SDPA/cut-4 backend.
3. Run the existing changed-future-padding test at B8x192 and require
   `torch.equal` on active outputs. Record allocated/reserved CUDA peaks,
   external GPU free minimum, host RSS HWM, wall time, and output shape/dtype.
4. Run one largest frozen cell through the same `capture_public_prefix` path,
   saving only the truth-free compact H128 artifact and sidecars. Check
   finite values, mask/position identities, artifact SHA, and the exact
   inherited geometry. Do not replace it with unpadded or smaller-batch work.
5. Only after the qualification receipt passes may sequential cells run. Keep
   one condition/cell resident, preserve partial/failure receipts, and never
   open truth during capture.

The inherited `scripts/trr0009_eval_capture.py capture --execute` entry point
is the intended production entry point once its P09 selection/panel bindings
are signed. This review does not launch it.

**Disposition:** Agent-1's query-aware adapter and P09 dispatcher are
integration-compatible under the wiring above. The domain-balanced selection
callback, explicit optimizer scratch policy, adapter state digest binding,
actual merged-W capacity qualification, and contract/provenance freeze remain
required before any fit or public capture.
