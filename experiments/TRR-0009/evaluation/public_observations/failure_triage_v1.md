# TRR-0009 public capture failure triage

The single authorized attempt used evaluator HEAD `1f8577d5d32a1268a00600bf3ffdbe9346ca7db4` and started at `2026-09-07T02:32:17Z`:

```text
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 scripts/trr0009_eval_capture.py capture --execute --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0009 --selection experiments/TRR-0009/selection/source_selection.json --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --lora-config /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/public-calibration/generation.json --lora-update /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors --output-root experiments/TRR-0009/evaluation/public_observations --device cuda
```

The preserved receipt is `failure.json`, SHA256 `a0bc96f9d2746df5194c7fcc8231b412d9ccda4afb102c528ceb69c87dc3546f`, with status `PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH`, error `CaptureError: public_base public-prefix load failed`, and end time `2026-09-07T02:32:21Z`. It confirms no source text, token IDs, observations, or truth were written.

The failure is raised by `scripts/trr0009_eval_capture.py::_capture_condition_with_producer` around the call to `trusted._capture_prefix(condition="public_base", ...)`. That wrapper currently replaces the chained producer exception with the generic message, so no deeper traceback was serialized. The preserved output directory contains only `failure.json`; no retry or loader rerun has been made. The pinned model weight file was inspected read-only and matches the registered 2,471,645,608-byte SHA256 `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.

A CPU-only diagnostic of the exact `trr0004_prepare_public_activations._load_public_prefix` call passed at `2026-09-07T02:34`-ish UTC (same snapshot, BF16, `cut_depth=4`, no source rows/truth and no CUDA allocation): hidden size 2048, vocabulary 128256, 16 layers. Therefore the preserved production failure is CUDA-specific or occurs at the CUDA model transfer/initialization path; no competing CUDA diagnostic or capture retry was started.
