# TRR-P09 streamed public-bank proposal (r0)

This is a preparation contract proposal, not a capture authorization. It
binds the shared loader and the artifact invariants while leaving source
selection, model forwards, and the final fit schedule for the root/design
review.

The bank is stored as complete public sequences. The proposed fit loader
minibatch is 8 records, while the public-forward batch grouping remains
configurable until geometry qualification. Every forward consumes complete
192-token sequences and stores `activations` (H, BF16, width 2048),
`token_ids`, `attention_mask`, and `position_ids` together. The natural
validation width is a separate 128-token view bound to the already-opened
TRR-0009 `public_base` contract; it is not a new 48-row panel and it is not
used to redefine the fitting bank geometry.

The current bank is B0: exactly 1,200 records and 124,371 post-BOS fitting
positions. An expanded bank candidate has 12,000 records, so its first 1,200
rows must be copied byte-for-byte from B0. The prefix comparison covers record
order, parent and sequence IDs, token IDs, masks, positions, and H. The
producer records the five digests in the manifest and fails closed if the
expanded prefix is regenerated, reordered, or numerically changed. The
expanded target is a planning value pending the common source and capacity
contract; no source row has been selected by this proposal.

At 12,000 rows the BF16 H payload is
`12000 * 192 * 2048 * 2 = 9,437,184,000` bytes (8.79 GiB), before token,
mask, position, sidecar, and filesystem overhead. A 64-record shard is about
50 MiB of H and is divisible into eight-record forwards, giving 188 shards.
The loader keeps one CPU batch and one decoder batch resident. The entire bank
is never moved to the GPU. The preflight runner reports disk, host memory, and
GPU margins without importing a model; an authorized capture must then use the
external fail-closed watchdog and inner resource checks.

Each shard is published through a unique staging directory, a safetensors
payload, a hash-bound sidecar, and a `COMPLETE` marker. A completed shard is
verified and skipped on resume. Existing payloads and sidecars are never
replaced; an incomplete staging directory remains as failure evidence. Sidecar
rows contain only public fitting identities and hashes (`record_id`,
`parent_record_id`, `sequence_id`, source/sequence digests, active count,
position exposure, and global row). Plaintext source is not part of the shard
payload; a separate public cache, if later justified, requires its own exact
binding. Public fitting H, token labels, masks, and positions are permitted training
inputs. Final evaluation source rows, target labels, and truth payloads never enter
this fitting bank; Agent 1 owns their post-freeze scoring boundary.

The shared implementation handoff is
`StreamedBankLoader.get_records(global_indices)` from
`scripts/trr_p09/prepare_streamed_bank.py`. It returns
`activations[N,192,2048]` BF16, `token_ids[N,192]`, boolean
`attention_mask[N,192]`, integer `position_ids[N,192]`, and identity metadata
in the caller's order, including repeated rows and indices crossing shard
boundaries. `iter_batches()` is only a sequential convenience view; it is not
the fit schedule. The implementation owner can consume this adapter in the
common training loop, while the directional readout remains an Agent1-owned
module. No second sampler or hidden full-bank loader should be added.

The fixed public training diagnostic is represented in the schema as a
predeclared index list and digest, but remains unbound until the common fit
contract is frozen. It must be selected from public fitting metadata before
fitting and cannot be chosen from fresh answers or post-score behavior. The
same schedule artifact records repeated exposures, optimizer updates, and
cost; shard order alone is not treated as a training schedule.

The runner currently exposes only `--mode preflight` and `--mode fixture`.
The fixture creates tiny deterministic tensors to test natural-row/prefix
byte equivalence, exact 8-row streaming, mask/position validation, and
create-only behavior. There is deliberately no production capture mode in
this task-local patch. Root must first bind the public snapshot hashes,
geometry, schedule, source exclusions, and resource receipt before an adapter
can call the public model.

Open review items for the shared contract are the exact public model/tokenizer
snapshot records, the B0 payload/record-order digests, the 128-token natural
validation record digest, fixed-diagnostic indices, and the final expanded
source manifest. Their absence is represented as pending fields rather than
invented hashes.
