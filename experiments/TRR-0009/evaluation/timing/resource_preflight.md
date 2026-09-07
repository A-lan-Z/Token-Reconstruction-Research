# TRR-0009 timing/evaluator resource preflight

Status: **PREFLIGHT_AND_DESIGN_NO_EVALUATION**. Recorded `2026-09-07T00:11:47Z`. No model, observation, prediction, source, selection, training, or truth payload was accessed.

Live snapshot: `32` logical CPUs, load average `[0.439453125, 0.47998046875, 0.41162109375]`, MemAvailable `20207812 kB`, and GPU query:

```text
0, NVIDIA GeForce RTX 5080, 16303, 2982, 12996, 0, 28, P8
```

The GPU compute-app query was recorded as:

```text

```

The execution guard must repeat all telemetry immediately before and during any future run; this snapshot does not authorize compute.

The training interface binds a 17,126-ID supported frequency set and a 34,252-scalar readout correction. Its largest training geometry is `[8,192,2048]` H with full-vocabulary logits; the design envelope is 1.6–3.6 GiB reserved GPU, subject to largest-cell qualification.

For the truth-free TRR-0008 runner baseline (4 methods, 2,816 records per method), the prior preflight measured 11,264 measured and 11,264 warmup decoder calls, 11,266 guard checks, and estimated decoder-plus-guard time of 395.7 seconds at the mean, 490.2 seconds at p95 guard cost, and 584.7 seconds at the observed guard maximum. The completed run took 475.091 seconds, with peak reserved CUDA 1,352,663,040 bytes and peak RSS 1,999,114,240 bytes.

Scaling the decoder and measured guard terms only, a provisional three-arm matrix at the same 2,816 records would require 8,448 measured + 8,448 warmup calls and approximately 296.5 seconds at mean guard cost, 367.5 seconds at p95 guard cost, or 438.3 seconds at the observed guard maximum. An optional fourth method returns to the four-method baseline. These are planning estimates; final counts, optional reference, readout materialization, startup, and schedule must be frozen before execution. The 600-second fail-closed ceiling is retained provisionally.

Evidence bindings and raw numbers are structured in `resource_preflight.json`. No GPU run or truth access occurred.
