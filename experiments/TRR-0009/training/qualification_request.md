# TRR-0009 qualification request

No qualification or fit has been run from this worktree. After root commits the
reviewed model/trainer sources, run the bounded largest-cell qualifier with one
exclusive GPU lease:

```bash
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  python3 scripts/trr0009_train.py \
  --device cuda \
  --max-seconds 1800 \
  --qualification-only \
  --output-root experiments/TRR-0009/training/qualification_v1
```

The command binds the current-bank selected starting state
`2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`, the
improved public fit/validation manifests, public normalized `E`, and the
published schedule semantic digest
`5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472`. It uses
the full 3,000-step schedule file but executes exactly two representative
updates. The largest update must report activation `[8,192,2048]`, draw `[512]`,
finite loss/gradients, and the zero-correction equivalence check uses a separate
B=1 full-logit fixture. The receipt must retain numerical settings, code/source
bindings, preparation/equivalence/update times, optimizer bytes, and resource
peaks.

The fail-closed guard is GPU reserved <= 6 GiB, free GPU >= 8 GiB, host RSS <=
16 GiB, and host available >= 10 GiB. Any anomaly leaves the create-only
qualification failure receipt and stops the lease. The full three-arm run is a
separate create-only command after a PASS:

```bash
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  python3 scripts/trr0009_train.py \
  --device cuda \
  --max-seconds 1800 \
  --output-root experiments/TRR-0009/training/run_v1
```
