# TRR-0010 production-caller requirements v1

This scaffold is intentionally fail-closed and contains no training loop,
sampler, bank reader, public payload, GPU launch, or truth access. A2 remains
the owner of `StreamedBankLoader`, immutable schedule iteration, validation,
checkpoint selection, and resource guards.

The caller must first bind three accepted artifacts: a `FROZEN` shared
contract, a `VERIFIED` streamed-bank manifest, and a `FROZEN` schedule. It
must also bind a numeric `PASS` qualifier for the largest directional cell,
including GPU reserved peak and limit, host RSS peak and limit, host available
memory and floor, and wall time and limit. Every field is checked before the
directional hook or optimizer is constructed. `verify_production_bindings`
then checks the actual artifact bytes and SHA-256 values and matches the
imported protocol module `__file__`; source commits are required to be full
40-character identities. The imported A2 source map must contain exact commit
and SHA-256 bindings for:

```text
src/token_reconstruction/trr_p09_fixed_control_adapter.py
scripts/trr_p09/fixed_control_runner.py
scripts/trr_p09/prepare_streamed_bank.py
```

For a production invocation, these bindings must point to the actual imported
A2 runner and streamed-bank loader modules used by the caller, rather than to
protocol-only copies or descriptive placeholders. The caller must pass those
exact imported runner/loader paths in `source_paths`; `verify_production_bindings`
checks their bytes and source identities, and also checks that the imported
protocol module `__file__` resolves to its bound adapter path. A successful
syntax or protocol-surface check alone is not an execution binding.

`prepare_directional_runtime` validates the A2 hook surface and calls
`verify_production_bindings` before constructing a run. It constructs the
directional hook around the exact shared decoder object, moves the newly
created Delta/buffers to that decoder device before binding public-E row
statistics, requires all parameters and E on one device, and creates
`torch.optim.AdamW` with named decoder/directional groups, frozen weight
decay, and explicit `foreach=False`. The A2 runner then receives this
optimizer and hook through its existing `run_training` API; the caller does
not implement a second loop.

The A2 checkpoint callback receives no optimizer argument. The provided
serialization-only callback therefore writes compact model checkpoints and
records `optimizer_state_external=true`; it cannot claim a resumable update.
`restore_selected_and_export` verifies checkpoint path, bytes, SHA-256, and
selected-step/contract/bank/schedule/source-binding metadata before loading,
then writes the full effective-E artifact and verifies deployment
logits/argmax through the existing model tests. A resumed fit requires a
separately hash-bound optimizer-state artifact and the same `foreach=False` /
group validation; absent that artifact, resume is rejected by contract.

The only local validation command is synthetic CPU coverage:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=.:src:scripts python3 -m pytest -q \
  tests/test_trr0010_p09_caller.py tests/test_trr0010_p09_integration.py \
  tests/test_trr0010_model.py
```

No production command is supplied until A2 publishes the final source,
contract, bank, schedule, and resource-qualifier bindings and root grants the
exclusive compute lease.
