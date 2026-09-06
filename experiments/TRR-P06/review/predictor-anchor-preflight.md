# TRR-P06 predictor and anchor launch preflight

Status: **preflight only; no prediction or anchor launch was made by this task**.
The commands below require root's explicit timing-window grant, a fresh live
resource check, and create-only output roots. The source binding for this
review is commit `69a77e822272a8b7d5e17aa15de11ee345c6f9ce` (or a reviewed
root descendant that preserves the same numeric source). The accepted public
capture is `experiments/TRR-P06/runtime/public-capture-r2/observations.json`
with SHA-256
`922c63438611ed7d6d8a55ac41d7548edfd561da3370dcd4df280973f7e7c146`.

Both runners must be executed sequentially under fresh watchdog roots. The
child runners enforce the GPU free/reserved, host RSS, host available-memory,
and elapsed-time guards; the outer watchdog enforces process-group RSS and host
available memory. The canonical limits are 8 GiB minimum GPU free, 6 GiB
maximum GPU reserved, 16 GiB maximum host RSS, 10 GiB minimum host available,
and 1,800 seconds. CPU settings are four intra-op threads and one inter-op
thread. The student runner's first full panel cell is the largest
representative qualification; halt on any guard, allocator, driver, thermal,
or source-binding anomaly.

## Pinned assets and accepted receipts

| Asset | Path | SHA-256 or binding |
| --- | --- | --- |
| Student fit receipt | `experiments/TRR-P06/runtime/main-r1/main_fit_receipt.json` | source commit `f1b35756fc535b5e3350e4edd4feff9e46f80321`, status `PASS` |
| Student states | `experiments/TRR-P06/runtime/main-r1/seed-{6106,6107}/.../selected.safetensors` | six state descriptors are bound by the fit receipt |
| Public observations | `experiments/TRR-P06/runtime/public-capture-r2/observations.json` | `922c63438611ed7d6d8a55ac41d7548edfd561da3370dcd4df280973f7e7c146` |
| Source selection | `experiments/TRR-P06/runtime/source-selection-r1/selection.json` | `d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a` |
| Normalized public table | `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors` | 1,050,673,488 bytes; `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1` |
| Anchor lens | `experiments/TRR-0004/evidence/comparators/public_a1_lens.pt` | `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742` |
| Anchor parent reference | `experiments/TRR-0004/evidence/comparators/round001_teacher.py` | `10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd` |
| Public model snapshot | `/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6` | revision `9213176726f574b556790deb65791e0c5aa438b6`; model weights `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f` |
| Accepted qualification | `experiments/TRR-P06/runtime/qualification-r2/qualification_receipt.json` | `PASS`, two-update full-record cell |
| Accepted main fit | `experiments/TRR-P06/runtime/main-r1/main_fit_receipt.json` | six selected states, `PASS` |

The observation manifest is source-free and truth-free. The anchor loader now
binds each source-order hash at the accepted manifest cell level, beside the
nested tensor descriptor; no observation tensor should be opened before the
selection and manifest hashes pass.

## Student full matrix command (do not launch here)

Use a fresh output root and a matching fresh watchdog root:

```text
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/resource_watchdog.py --output-root experiments/TRR-P06/runtime/watchdog-predictions-r1 --timeout-seconds 1800 --poll-seconds 0.5 --max-rss-bytes 17179869184 --min-available-bytes 10737418240 --cwd /tmp/trr-p06 --label TRR-P06-student-predictions-r1 -- env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/run_predictions.py --repository-root /tmp/trr-p06 --fit-receipt experiments/TRR-P06/runtime/main-r1/main_fit_receipt.json --observation-manifest experiments/TRR-P06/runtime/public-capture-r2/observations.json --embedding-path /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors --output-root experiments/TRR-P06/runtime/predictions-r1 --device cuda --torch-threads 4 --torch-interop-threads 1 --batch-records 8 --projection-chunk 512 --minimum-free-gib 8 --maximum-gpu-reserved-gib 6 --maximum-host-rss-gib 16 --minimum-host-available-gib 10 --max-seconds 1800
```

This is six states × four observation cells, full-vocabulary, batch 8,
projection chunk 512, with one warmup and three measured passes per cell. The
fit receipt's six training arms measured 3.00–3.24 seconds for 31 validation
checkpoints over 48 records; scaling the same full-vocabulary work to 24
prediction cells gives a rough 40-second readout estimate before table/state
startup. This is only a planning estimate; the first student cell records the
actual peak and timing, and the remaining matrix stops if it violates the
frozen guard.

## Published-parent A1+A2 anchor command (do not launch here)

The anchor is the retained `frozen_a1_a2_k256` rule, first 64 public-base
records per domain, with A1 proposal K=512 and A2 selector K=256. It is a CPU
normalized-embedding port of the published public-prefix adapter; it does not
use student states or a fallback method.

```text
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/resource_watchdog.py --output-root experiments/TRR-P06/runtime/watchdog-anchor-r1 --timeout-seconds 1800 --poll-seconds 0.5 --max-rss-bytes 17179869184 --min-available-bytes 10737418240 --cwd /tmp/trr-p06 --label TRR-P06-anchor-r1 -- env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:src:scripts /usr/bin/python3 scripts/trr_p06/run_anchor.py --repository-root /tmp/trr-p06 --selection experiments/TRR-P06/runtime/source-selection-r1/selection.json --observation-manifest experiments/TRR-P06/runtime/public-capture-r2/observations.json --embedding-path /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors --lens-path experiments/TRR-0004/evidence/comparators/public_a1_lens.pt --reference-path experiments/TRR-0004/evidence/comparators/round001_teacher.py --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --output-root experiments/TRR-P06/runtime/anchor-r1 --device cuda --minimum-free-gib 8 --maximum-gpu-reserved-gib 6 --maximum-host-rss-gib 16 --minimum-host-available-gib 10 --max-seconds 1800
```

The retained parent receipt
`experiments/TRR-0005/final_prediction_run_evidence_v1/resource_receipt.json`
measured the same A2 adapter at 527.482 seconds across four cells, with
3,546,284,032 bytes maximum CUDA reserved and 5,816,393,728 bytes maximum
process RSS. P06 runs only two public-base cells and 128 records, so this is a
conservative upper-bound reference, not a P06 measurement; the new receipt
must report actual warmup/measured work, row staging, observation loading,
GPU peaks, and guard samples.

After each child exits, preserve both child and watchdog receipts before
reviewing the next phase. These commands produce predictions only; no truth or
source text is opened. The separate probe replay with
`--retain-probe-predictions` is a future logging-only action and is not part of
this launch preflight.
