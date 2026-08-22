# Frozen strict-BOS comparator

This directory exists only to make comparisons with the previously implemented strict-BOS A1+A2 system reproducible. It is benchmark code, not a proposed architecture for new work.

The implementation intentionally retains historical fixed constants, routing rules, adapter settings, and a four-layer prefix. Do not treat any of them as a supported search space, default, or constraint. New methods should depend on the neutral primitives in `src/token_reconstruction/`, not import this directory.

Included:

- `round001_teacher.py`: the row-serial strict-BOS A1+A2 reference and its causal rank-8 adapter update.
- `round003_wavefront.py`: the production-wavefront implementation used to compare scheduling/runtime while preserving reference decisions.
- `preflight_wavefront.py`: a CPU-only semantic and fail-closed smoke test. Its import path alone was changed so the three files can remain together.

Not included: target traces, truth, model weights, a trained lens, trained adapter state, experiment schedules, result files, reports, or conclusions.

## Provenance

The two implementation files are byte-for-byte copies from the local research source. SHA-256:

```text
10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd  round001_teacher.py
e8aa73e0168fb4a2622b93af480b357e25e692f3d4e10b110171fbc75dff8989  round003_wavefront.py
```

The source preflight hash was `f19e92be42e2a79dcd21fea72fa509941bdfb5afcd0002ef68e2d8cb7ce20c71`. After the import-only relocation its hash is:

```text
027dd5a6405350da51ce299e908096df26ec3cb41ef0f64f275faeed6b855f1d  preflight_wavefront.py
```
