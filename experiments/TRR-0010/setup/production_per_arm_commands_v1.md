# TRR-0010 per-arm production commands

Status: ready for root lease review; no CUDA fit has been launched from this
receipt. The source-bound command uses the signed stage-3 contract
`experiments/TRR-0010/planning/shared_stage3_contract_v1.json` (25,643 bytes,
SHA-256 `23e4bde4475082cc004e7ef787c22fed6301275ce2ff23a0396d975b439faf9d`)
and source commit `e76ebcd2ecd59369207c067543cbb578c5adbd23`.

Each arm receives its own lease, process/output root, 7,200-second deadline,
and fail-closed guard. The current arm binds B0 and the expanded arm binds B1;
there is no combined two-arm production budget.

```text
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts \
  python3 scripts/trr0010_directional_fit_cli.py --arm current_directional \
  --provider trr0010_production_provider:build_inputs \
  --binding-current experiments/TRR-0010/setup/production_arm_binding_current_stage3_v1.json \
  --diagnostic-binding /tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json \
  --lease experiments/TRR-0010/training/directional_fit/current_directional/lease.json \
  --output-root experiments/TRR-0010/training/directional_fit/current_directional \
  --device cuda --deadline-seconds 7200
```

```text
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts \
  python3 scripts/trr0010_directional_fit_cli.py --arm expanded_directional \
  --provider trr0010_production_provider:build_inputs \
  --binding-expanded experiments/TRR-0010/setup/production_arm_binding_expanded_stage3_v1.json \
  --diagnostic-binding /tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json \
  --lease experiments/TRR-0010/training/directional_fit/expanded_directional/lease.json \
  --output-root experiments/TRR-0010/training/directional_fit/expanded_directional \
  --device cuda --deadline-seconds 7200
```

The guard limits are: 10 GiB CUDA reserved, 12 GiB host RSS, 8 GiB host
available, 2 GiB free GPU memory, 20 GiB free disk, and 5 GiB output bytes.
The deadline includes provider preparation, diagnostics, checkpoint/export, and
runner time. The runner result is persisted before restore/export, and each
receipt separates provider preparation, runner, W restore/export, and base
export timing.

The largest CPU assembly qualification measured 3,384,340 KB RSS and 9.14 s
wall for B0, and 3,384,288 KB RSS and 17.12 s wall for B1. These are assembly
bounds only; they do not qualify CUDA fit memory or fit duration. A fresh
exclusive lease and live guard receipt are required before any production fit.
