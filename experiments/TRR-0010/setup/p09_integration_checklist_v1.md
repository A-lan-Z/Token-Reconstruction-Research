# TRR-0010 / TRR-P09 integration checklist v1

This is a handoff checklist, not a training command or an execution approval.
The common runner owns the decoder, streamed bank loader, immutable schedule,
validation, checkpoint selection, optimizer stepping, resource guards, and
run receipts. The task-owned wrapper is `scripts/trr0010_p09_integration.py`;
it builds `DirectionalTokenReadout` and forwards one batch to A2's
`shared_decoder_rows` hook.

The reviewed A2 protocol surface is
`src/token_reconstruction/trr_p09_fixed_control_adapter.py` from API commit
`0aee09c4abb8d65b6ed3a42dc8ae93789509a518`, present in the latest reviewed
snapshot `eaa1f1e5aa0be1e6d0dcb896834f4d6ac4e43861`. The wrapper requires
`BankContract`, `TrainingContract`, `ReadoutHook`, `shared_decoder_rows`, and
`contract_digest`; it does not vendor that source. The exact dispatch is:

```text
shared_decoder_rows(
    decoder, hook, activation, valid_mask, record_slots, position_slots,
    embedding, target_ids, compute_base_logits=False
)
```

The directional hook receives normalized current-H query rows and the
inherited scalar logit scale. It scores the full vocabulary through one
materialized `E_eff` matmul. Targets reach `loss_terms` only. A2's runner must
construct the optimizer from `optimizer_param_groups`; the wrapper never
creates a second loop or sampler. Use `compute_base_logits=False` for the
directional merged-primary path so a duplicate public-E projection is not
created. The fixed control can use A2's own `FixedPublicReadoutHook`.

The largest representative qualifier must use the complete fit geometry:
`B=8`, `T=192`, `H=2048`, BF16 activation storage, `V=128256`, and the frozen
position budget (currently proposed as 512). Bind the actual A2 support count
before estimating; the current-support estimate is 6.389 GiB and the
full-vocabulary support upper bound is 11.477 GiB under the recorded FP32
merged-primary accounting. Require the live fail-closed GPU/host guards and a
measured margin. Keep optimizer `foreach=False` or an equivalent bounded
scratch mode, retain one `E_eff`, its dense backward gradient, support gather /
add temporaries, Delta gradient and Adam state, and do not switch to the
diagnostic sparse `N*K` path as a memory workaround. The B1 full-output
export/reload equivalence check is separate from this B8 qualifier.

Before any authorized fit, bind and record the exact state, public embedding,
fit/validation manifests, streamed-bank shards, schedule semantic digest,
support IDs/counts digest, source commits, numerical settings, and resource
snapshot. Save selected decoder-plus-Delta state and a separate exported full
`E_eff`; verify reloaded full-vocabulary logits and argmax IDs exactly. Do not
duplicate the full embedding in every checkpoint or source control.

The bounded local smoke command is:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=.:src:scripts python3 -m pytest -q \
  tests/test_trr0010_p09_integration.py tests/test_trr0010_model.py
```

The eventual fit command must be the exact A2 runner invocation after the
shared contract, bank artifacts, and resource lease are frozen; this task does
not invent a parallel runner CLI. Record that command verbatim in the A2 run
receipt together with preparation, update, validation, peak-memory, and
failure timings.
