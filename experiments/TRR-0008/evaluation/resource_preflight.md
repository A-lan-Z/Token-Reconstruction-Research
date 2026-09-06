# TRR-0008 resource preflight

Status: **complete, no evaluation performed**. This receipt measures the current runner guard cost and estimates the already-planned matrix. It does not open model observations, predictions, selection data, or truth.

## Inputs and evidence

- Runner: `scripts/trr0008_eval_runner.py`, source SHA-256 `fc84d09c7563b2c27a665b182902f1a76952a506b98cbbb25c37a9e6e6fe9409`, repository commit `53f661751eddb4cdae8ea1c69393c8b5859d92e8`.
- The source-free actual-runner qualifier passed 4 methods × 4 cells × 128 rows: `experiments/TRR-0008/evaluation/qualifier/runner_qualifier.json`, SHA-256 `351113f462bedd6d5be955a3cf328984c23c3c12790f0fb311d13f068e3dcf75`, 26.6639 s, 2,048 decoder calls, and 279 guard checks.
- The frozen 40-block timing receipt measured **25,600 measured calls plus 25,600 warmup calls** (51,200 decoder calls total) in 322.1985 s. Its measured-call sum was 106.1742 s and warmup-call sum was 107.0170 s: `experiments/TRR-0008/timing/precision40_result.json`, SHA-256 `a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`.

## Planned work and guard overhead

The requested full matrix has 2,816 rows per method and four methods. The current runner therefore performs 11,264 row iterations, 11,264 warmup calls, 11,264 measured calls, and 22,528 decoder calls. `_guard` runs once per row plus two setup checks, for 11,266 checks. Each CUDA guard executes one `nvidia-smi --query-compute-apps` subprocess in addition to RSS, host-memory, CUDA-availability, and allocator checks.

A timestamped guard-only live probe ran 24 checks without loading a model or opening data. The command was:

```text
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts python3 <guard-only probe>
```

The probe called `runner._guard(device=cuda, guard=registered_limits, stage=resource_preflight_i)` 24 times and only read host/CUDA telemetry. It ran from `2026-09-06T13:28:16Z` to `2026-09-06T13:28:17Z`; all 24 checks passed. The first check took 310.55 ms while CUDA initialized. Across the 23 subsequent checks, mean time was 26.78 ms, median 24.59 ms, p95 35.16 ms, and maximum 43.55 ms. An earlier 24-check probe is retained in the JSON because ordinary telemetry variance gave steady mean 26.68 ms and p95 27.80 ms; no model or data was accessed in either probe.

Using the timestamped probe, guard time is 301.94 s at the mean estimate (including one initial check), 396.42 s at the steady p95 estimate, and 490.86 s if every subsequent check repeats the observed steady maximum. The 40-block receipt gives 4.147 ms per measured call and 4.180 ms per warmup call, or 93.80 s for the planned 22,528 calls. Combining these terms gives 395.74 s at the mean estimate, 490.22 s at p95, and 584.67 s under the observed per-check maximum, before model startup and ordinary I/O. The earlier probe’s corresponding combined estimates were 394.62 s, 407.31 s, and 554.82 s; both are retained for audit.

The registration and timing receipts use the existing 600 s maximum. The mean estimate fits with substantial margin; p95 and maximum telemetry stress leave progressively less time for startup and I/O. The run must retain the fail-closed time guard and stop on any resource anomaly or expiry. This preflight does not change the limit, batching, order, or numerical contract.

## Live resource snapshot

At the timestamped probe, the host had 32 logical CPUs, 20,916,621,312 bytes available, and 677,720,064 bytes resident in the probe process. CUDA reported 15,679,356,928 free of 17,066,033,152 total bytes, with zero reserved/allocated bytes in the idle probe and no foreign compute applications. `nvidia-smi` reported an RTX 5080 at 32 °C and 0% utilization. The completed qualifier’s loaded-state final guard recorded 14,144,241,664 free CUDA bytes, 1,457,520,640 reserved bytes, 1,741,484,032 process RSS bytes, and 20,233,240,576 bytes of host availability; these remained inside the registered guards.

The structured measurements, including both guard-only probes and their exact timestamps, are in `experiments/TRR-0008/evaluation/resource_preflight.json`.
