# TRR-P09 STAGE1 capture interoperability readiness

Status: `CPU_SYNTHETIC_PASS_NO_MODEL_RUN`.

This review covers the producer/consumer boundary only. It does not load the
public model, read evaluation truth, or allocate GPU memory.

## Bound interfaces

The setup compiler output is consumed through
`token-reconstruction.trr-p09-stage1-public-inputs.v1` at
`/tmp/trr-p09-runtime/stage1-input-preparation-r1/preparation_manifest.json`.
Its immutable payload has keys `token_ids`, `attention_mask`, and
`position_ids` with 12,000 rows by 192 tokens; records remain a separate
identity sidecar. The capture adapter hashes and validates that manifest,
exposes only the local B1 slice of 10,800 rows, and translates local rows
`0:10800` to expanded rows `1200:12000`. It does not rewrite the compiler
payload or write token IDs to activation output shards.

Both the capture adapter and common streamed-bank loader now use the inherited
position convention: `position_ids[t] == t` where the attention mask is active,
and `0` in inactive padding. The common loader accepts an expanded B1 row
origin and verifies contiguous ranges from that origin. `CombinedStreamedBankLoader`
composes a B0 manifest ending at the B1 origin while preserving arbitrary
order and repeated schedule indices.

## External watchdog

The execution-ready wrapper is the existing fail-closed process-group
watchdog `/tmp/trr-p06/scripts/trr_p06/resource_watchdog.py`, SHA-256
`2384b51e3220d6bbd0c6b723d414e2f3350fceacb974ec88fa0757ffbcc7d652`.
The P09 capture validator now requires the watchdog receipt to bind this
actual executable path, byte count, SHA, command invocation, future lease
expiry, P09 plan/input hashes, wall/GPU/RSS limits, and post-child race-safe
handling. The wrapper itself enforces wall timeout, process-group RSS, host
MemAvailable, and fail-closed post-child evidence. P09's inner guard remains
responsible for GPU and retained-output caps.

The root must create the P09 watchdog and authorization receipts for the live
lease. No receipt is fabricated by this implementation checkpoint.

Qualification command, to be run only after the root lease and receipts are
present:

```text
python3 /tmp/trr-p06/scripts/trr_p06/resource_watchdog.py \
  --output-root /tmp/trr-p09-runtime/watchdog-stage1-qualification-r1 \
  --timeout-seconds 600 --poll-seconds 0.5 \
  --max-rss-bytes 12884901888 --min-available-bytes 6442450944 \
  --cwd /tmp/trr-p09 --label TRR-P09-stage1-qualification-r1 -- \
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 scripts/trr_p09/stage1_public_capture.py --mode qualify \
  --input-manifest /tmp/trr-p09-runtime/stage1-input-preparation-r1/preparation_manifest.json \
  --stage1-plan experiments/TRR-P09/planning/stage1-public-bank-plan.json \
  --output-root /tmp/trr-p09-runtime/stage1-qualification-r1 \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --device cuda \
  --plan-sha256 bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c \
  --authorization-receipt /tmp/trr-p09-runtime/authorization-stage1-qualification-r1.json \
  --watchdog-receipt /tmp/trr-p09-runtime/watchdog-stage1-qualification-r1.json
```

The command has not been executed in this receipt. The production wrapper
uses the same structure with a 7,200-second cap and a distinct create-only
output root, after a passing qualification receipt is bound.

## Synthetic gate

The focused command was:

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_*.py
```

Result: `58 passed` in `2.70 s`, CPU-only. This includes the end-to-end
synthetic compiler-output/input-parser -> fake-prefix qualification ->
create-only capture -> composed B0+B1 loader smoke. The smoke checks expanded
row translation, active-prefix/zero-padding positions, arbitrary order, and
repeated rows. It also checks that a real watchdog executable binding and
future lease are required.

## Source evidence

At the time of this review:

- `scripts/trr_p09/stage1_public_capture.py` SHA-256
  `ff81e244945ab0752dd58525e03a324fa7022cffe17b39707b671b5d1b169cbc`.
- `scripts/trr_p09/prepare_streamed_bank.py` SHA-256
  `e76a27f94328648f8fe3da6c6fae48fffd27006b213f9b70b976bcb5f4077c4b`.
- `tests/test_trr_p09_stage1_public_capture.py` SHA-256
  `977d4f11c84476a1e36174b39363906fd9f369c5031b37d9a91994970234c252`.
- `tests/test_trr_p09_bank_artifact.py` SHA-256
  `20bb9aab149c8cb35e4d1164febaea5dd98a2b048e744e5ed86a2993265e370b`.
- Signed Stage1 plan SHA-256
  `bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c`.

No public capture, qualification, truth opening, or scientific fit is part of
this evidence.
