# TRR-0010 STAGE1 capture-cap addendum

**Status:** `READ_ONLY_STAGE1_CAP_PROPOSAL_NO_CAPTURE`

This addendum narrows the earlier capture review to the STAGE1 public-base
activation artifact. It does not authorize or run capture, load the model, or
open truth.

## Scope and storage geometry

STAGE1 has one target condition, `public_base`, and 10,800 new records. The
forward remains the inherited padded batch of 8 records by 192 tokens. Each
storage shard contains 64 records, which is only an I/O boundary: each shard
must be computed as eight native B8 forwards, and a final 32-record shard as
four B8 forwards. The storage shard must never become a B64 forward or a
microbatching substitution.

STAGE1 stores full H192 BF16 activations only. It does not create a duplicate
H128 activation artifact. Record and parent mappings, rendered/sequence
hashes, masks, and position IDs are separate truth-free metadata/sidecar
artifacts; source text, target labels, and token IDs are not written to the
capture output.

The tensor sizes are:

- one B8x192x2048 BF16 forward output: 6,291,456 bytes;
- one 64-record H192 shard: 50,331,648 bytes (48 MiB);
- 10,800 records: 169 shards, with the final shard containing 32 records;
- all H192 activations: 8,493,465,600 bytes (about 7.91 GiB), before sidecars
  and file metadata;
- mask and int64 position sidecars for all records: 18,662,400 bytes before
  serialization overhead.

The writer must stream one 64-record shard at a time and release its full
activation tensor after the shard is durably written and hashed. The existing
TRR9 packager is a paired-condition/H128-retention adapter and currently
accumulates a complete cell; it is therefore a reference for provenance and
validation, not an unchanged STAGE1 writer.

## Exact inherited producer bindings

The inherited public model is
`/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`.
The producer calls `AutoModelForCausalLM.from_pretrained` with
`local_files_only=True`, `dtype=torch.bfloat16`, and
`attn_implementation="sdpa"`, retains the cut-4 contiguous prefix, and calls
`ContiguousPublicPrefix.forward_full` in eval/inference mode.

Bind these exact source paths and SHA-256 values in the future capture
receipt:

| role | path | SHA-256 |
| --- | --- | --- |
| STAGE1 adapter reference | `/tmp/trr-p09/scripts/trr0009_eval_capture.py` | `3b0338062c62a049caedcc968a2a5d505135b6117ad65706f32dd624aa1394d3` |
| public capture helper | `/tmp/trr-p09/src/token_reconstruction/public_activation.py` | `3be43dd5b6a5918a48f289971ab2a833977a7f9d87094e6001c5c7cdc60ab76f` |
| public prefix backend | `/tmp/trr-p09/src/token_reconstruction/public_prefix.py` | `b9dca1d8d56c7c07413015aea4bde79e8ceac827c32ee9bc0d1a236ea1dd35f6` |
| trusted producer loader | `/tmp/trr-p09/scripts/trr0005_produce_confirmation.py` | `7289ee5db30bc0b2d3c1b50b1d5b6ca1ac235001371f61b92adf10e2dd979e2b` |
| trusted prefix loader | `/tmp/trr-p09/scripts/trr0004_produce_confirmation.py` | `b1ae4e199de62f350df0ce9890caa90a70b0611a22051ce6bc304e0ccc4e1b2a` |
| padding qualification helper | `/tmp/trr-p09/scripts/trr0004_prepare_public_activations.py` | `6f1a9b6d7ee4aad89ee8152021e6e431b8652a6f7c162167a6180229feed70e8` |
| inherited capture producer | `/tmp/trr-p09/scripts/trr0006_capture_public.py` | `ade20eda4cc6ca3d25861558e5256d1b119ad1419e2bf909c1715fed612f20c7` |

The pinned model weight file is 2,471,645,608 bytes with SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.

## Qualification and fail-closed caps

Before the 10,800-record run, qualify at most three representative B8x192
public-base batches, including a batch representative of the longest/padded
input pattern. Repeat each exact batch through the same `forward_full` path.
Require finite BF16 H192 outputs, unchanged active-row geometry, and
`torch.equal` on the repeated active outputs. The existing future-padding
equivalence check remains required. Do not use an unpadded or B1 comparison as
a batching workaround. The qualification has a hard wall cap of 600 seconds,
including model load, guards, synchronization, and receipt writes.

The STAGE1 production capture has a 7,200-second hard wall cap, including
model load, selection/hash verification, all 169 streamed shards, full H192
serialization, hashing, synchronization, and cleanup. A projected runtime
above this cap is a fail-closed condition; no partial output is promoted to a
complete capture. Preserve a failure receipt and partial shard hashes.

Use the following caps:

| quantity | cap/requirement |
| --- | ---: |
| GPU reserved by child | <= 8 GiB (`8,589,934,592` bytes) |
| GPU free immediately before launch | >= 8 GiB |
| GPU free during the child | >= 2 GiB; stop and preserve failure below this floor |
| child/process RSS | <= 12 GiB |
| host available immediately before launch | >= 10 GiB |
| host available during the child | >= 6 GiB; stop below this floor |
| retained capture-data footprint | <= 20 GiB, including H192 shards, sidecars, manifests, and failure evidence |
| filesystem free before launch and each shard | >= 20 GiB |
| qualification wall | <= 600 s |
| STAGE1 capture wall | <= 7,200 s |

The read-only snapshot at `2026-09-07T08:34:05.229111540Z` was GPU free
12,465 MiB / used 3,513 MiB, host available 20,109,090,816 bytes, and
filesystem free 287,708,635,136 bytes. It passes the proposed prelaunch
floors, but it is not a reservation and must be refreshed immediately before
execution. The 20-GiB data cap leaves roughly 12 GiB above the calculated
7.91-GiB H192 payload for sidecars, manifests, temporary serialization, and
preserved failures.

The runtime GPU-free floor is intentionally lower than the prelaunch floor,
but it remains a hard safety stop. A child that reaches the 8-GiB reserved
cap, 12-GiB RSS cap, 6-GiB host-available floor, 2-GiB GPU-free floor, or
20-GiB retained-data cap must terminate fail closed rather than shrink the
forward batch or silently drop shards.

## Receipt requirements

The future receipt must bind the public-base condition, the 10,800-record
selection and parent-mapping hashes, all 169 shard descriptors, the exact
producer/helper hashes above, the model snapshot/weight hash, B8x192/BF16/
SDPA/cut-4 geometry, qualification result, per-shard wall and I/O timing,
whole-run wall, peak reserved GPU, minimum GPU free, RSS HWM, minimum host
available, disk free, and `truth_opened=false`. No H128 duplicate is expected
or permitted for this STAGE1 artifact.
