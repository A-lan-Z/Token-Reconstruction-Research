# TRR-0010 largest-cell qualifier scope v1

This is preparation only. No TRR-0010 public bank payload, GPU run, fit, capture,
evaluation, or truth has been opened by this entrypoint.

`scripts/trr0010_p09_qualifier.py` is a thin qualifier around the reviewed A2
runner. It does not implement sampling or a second training loop. An A2-owned
provider must return the hash-bound decoder, public normalized E, support
vectors, `RandomAccessLoaderSource`-compatible source, `ScheduleStep` stream,
`RunnerConfig`, validation-batch factory, and imported adapter/runner/loader
modules. The qualifier checks that those module `__file__` paths match the
three immutable source bindings before constructing the arm.

The binding manifest must be finalized with:

- `FROZEN` contract, `VERIFIED` bank, `FROZEN` schedule, selected base state,
  public E, support IDs/counts, and exact bytes/SHA-256 for every artifact;
- exact A2 source paths, full commit IDs, and source SHA-256 values;
- explicit geometry and optimizer settings, including `probe_steps`, merged
  primary `compute_base_logits=false`, base LR, directional LR, weight decay,
  anchor strength, and validation geometry; and
- the schedule seed, semantic digest, exposure metadata, and the provider's
  actual imported modules.

The lease file must state `GRANTED`, `exclusive=true`, the CUDA device, wall
limit, GPU reserved/free caps, host RSS/available caps, and disk-free floor.
There are no implicit resource defaults. The guard samples host `MemAvailable`,
current RSS, disk, GPU free/allocated/reserved/max-reserved bytes before and
after the model probe, at each A2 validation checkpoint, and after optional
checkpoint/export I/O.

The probe sequence is fixed by the binding manifest:

1. build the selected residual decoder plus exact-zero supported-row Delta and
   bind E statistics;
2. on the first scheduled public fitting batch, compare the unchanged base
   decoder's full logits with A2's directional merged-E logits and require
   bit-exact logits and argmax IDs;
3. call A2 `run_training` for exactly `probe_steps` discarded updates with
   `compute_base_logits=false`, the exact schedule prefix, and the shared
   validation-batch factory. Validation and checkpoint callbacks remain active,
   so the measured largest cell includes the common B8x192 geometry where the
   provider declares it, not only the 512-row sampled loss path;
4. require Adam state allocation, record update/load/validation timing and
   peaks, and discard the internal selected point; and
5. optionally invoke a caller-bound create-only checkpoint/export probe while
   gradients and optimizer state remain allocated. Its artifacts are qualifier
   evidence only and cannot become a fitted contender.

A2 still needs to provide the concrete provider/factory for the final bank and
schedule. The existing snapshot exposes `run_training`, `train_one_step`, and
`RandomAccessLoaderSource`, but does not provide a production schedule reader
or validation factory. The qualifier therefore fails closed until that provider
is handed off; it does not reconstruct those pieces. The planned command after
that handoff is:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=.:src:scripts python3 -m scripts.trr0010_p09_qualifier \
  --bindings experiments/TRR-0010/qualification/bindings.json \
  --lease experiments/TRR-0010/qualification/lease.json \
  --provider <A2_PROVIDER_MODULE>:build_directional_qualification_inputs \
  --output-root experiments/TRR-0010/qualification/largest_cell_v1
```

The command is not authorized for execution yet. A successful receipt is
`qualification.json`; any anomaly or missing provider/resource binding writes
`failure.json` create-only and stops. The selected base-only deployment state
is written separately through the established TRR-0007 positionwise saver, and
the full effective readout remains a separate `effective_readout_w` artifact.
