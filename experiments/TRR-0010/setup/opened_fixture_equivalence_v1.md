# Opened-fixture zero-Delta equivalence diagnostic (prepared; not launched)

This is a bounded, truth-free diagnostic for the already-opened TRR-0009
`public_base` activation fixture. It checks the first two records of each
`pile` and `finance` cell at the retained 128-position clip. It compares the
published fixed decoder with public `E`, the TRR-0010 zero-Delta decoder using
one materialized effective `E`, and a freshly reloaded fixed decoder using the
serialized effective-`E` export. It records only aggregate exact-equality and
maximum-absolute-difference values. It does not load labels, source text,
truth, token IDs, or accuracy metrics, and it performs no optimizer update.

The prepared entrypoint is:

`scripts/trr0010_opened_fixture_equivalence.py`

The exact guarded command, to run only after root grants the separate exclusive
GPU lease, is:

```bash
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:src:scripts timeout --signal=TERM --kill-after=15s 240s python3 scripts/trr0010_opened_fixture_equivalence.py --repository-root . --output-root experiments/TRR-0010/evaluation/opened_fixture_zero_delta_v1 --device cuda --records-per-domain 2 --max-seconds 240
```

The entrypoint itself fails closed on CUDA pre-load free memory below 8 GiB,
runtime free memory below 2 GiB, reserved memory above 4 GiB, host RSS above 6
GiB, host `MemAvailable` below 10 GiB, or output-disk free space below 20 GiB.
It binds these immutable inputs before loading tensors:

- selected fixed state: SHA-256 `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`;
- public normalized `E`: SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`;
- B0 public support IDs/counts: support digest `8090b9231042d6c9f9c52f3a0a9d8733b8ec1b739a4f28b12ab1057a7126969d`;
- already-opened TRR-0009 `pile__public_base` and `finance__public_base` observation payloads, with their frozen SHAs in the entrypoint.

The diagnostic uses the established positionwise saver to create a separate
base-only decoder state, then reloads that state and the serialized effective
readout. The first materialized effective dictionary is released before the
CPU exports and reloaded dictionary are moved to CUDA, so the diagnostic does
not retain two full effective readout copies on the GPU. Any non-exact comparison
writes `FAIL_NONEXACT` and exits nonzero; it is never reported as PASS. A successful receipt
would be created under `experiments/TRR-0010/evaluation/opened_fixture_zero_delta_v1/`;
failures are preserved in a sibling `_failure` directory. No command has been
launched yet.
