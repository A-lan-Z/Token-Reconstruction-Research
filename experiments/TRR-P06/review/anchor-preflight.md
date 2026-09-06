# TRR-P06 published A1+A2 anchor preflight

Status: **preflight only; no anchor inference was launched by this task**.
Root must provide the explicit timing-window grant and a fresh exclusive-device
check before using the command below. The anchor source change is in
`69a77e822272a8b7d5e17aa15de11ee345c6f9ce` (the follow-up focused-test commit
is `ec3907f`); use the reviewed full descendant when launching.

The accepted source-free observation manifest is
`experiments/TRR-P06/runtime/public-capture-r2/observations.json`, SHA-256
`922c63438611ed7d6d8a55ac41d7548edfd561da3370dcd4df280973f7e7c146`. Its
public-base observation file hashes are pile
`d997bc7c3fa1040e64a48c86f6e21e823de116a5c002dcb7012d7682a33f0ae5` and
finance `e6828483fe8a35666f5b8878f2188c7054aee7d0acc5a67ecb82590e61cc3266`.
The loader binds source order at the cell level (`record_ids_sha256`) and
validates the nested tensor descriptor after that check. This matches the
accepted capture-r2 schema.

## Frozen assets and guard

- Selection: `experiments/TRR-P06/runtime/source-selection-r1/selection.json`, SHA-256 `d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a`.
- Normalized public embedding: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`, 1,050,673,488 bytes, SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
- Retained A1 lens: `experiments/TRR-0004/evidence/comparators/public_a1_lens.pt`, SHA-256 `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`.
- Published parent adapter: `experiments/TRR-0004/evidence/comparators/round001_teacher.py`, SHA-256 `10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd`.
- Public model snapshot: `/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`, revision `9213176726f574b556790deb65791e0c5aa438b6`; model weights SHA-256 `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.
- Guard: 8 GiB minimum GPU free, 6 GiB maximum GPU reserved, 16 GiB maximum process RSS, 10 GiB minimum host available, 1,800 seconds child timeout. Use four CPU threads and one inter-op thread in the environment; the parent adapter remains CUDA.

## Exact guarded launch (do not launch here)

Use fresh create-only roots `experiments/TRR-P06/runtime/anchor-r1` and
`experiments/TRR-P06/runtime/watchdog-anchor-r1`:

```text
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/resource_watchdog.py --output-root experiments/TRR-P06/runtime/watchdog-anchor-r1 --timeout-seconds 1800 --poll-seconds 0.5 --max-rss-bytes 17179869184 --min-available-bytes 10737418240 --cwd /tmp/trr-p06 --label TRR-P06-anchor-r1 -- env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/run_anchor.py --repository-root /tmp/trr-p06 --selection experiments/TRR-P06/runtime/source-selection-r1/selection.json --observation-manifest experiments/TRR-P06/runtime/public-capture-r2/observations.json --embedding-path /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors --lens-path experiments/TRR-0004/evidence/comparators/public_a1_lens.pt --reference-path experiments/TRR-0004/evidence/comparators/round001_teacher.py --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --output-root experiments/TRR-P06/runtime/anchor-r1 --device cuda --minimum-free-gib 8 --maximum-gpu-reserved-gib 6 --maximum-host-rss-gib 16 --minimum-host-available-gib 10 --max-seconds 1800
```

The anchor evaluates the retained `frozen_a1_a2_k256` rule: fixed A1 proposal
K=512, fixed A2 selector K=256, first 64 public-base records in each of the
pile and finance domains. That is 128 records total and one warmup plus one
measured adapter call per record, 256 calls total; candidate arrays and truth
remain omitted. The command records observation load, row staging, warmup and
measured intervals separately and requires warmup/measured IDs to match.

The published parent resource receipt
`experiments/TRR-0005/final_prediction_run_evidence_v1/resource_receipt.json`
reported 527.4821984938171 measured seconds across four cells, 3,546,284,032
bytes maximum CUDA reserved, 5,816,393,728 bytes maximum process RSS, and
8,326 MiB minimum external GPU free. This P06 anchor has two public-base cells
and 128 records, so the parent result is a conservative planning reference,
not a P06 measurement. Preserve the child and watchdog receipts and report
actual P06 peaks and per-domain work before any truth gate.
