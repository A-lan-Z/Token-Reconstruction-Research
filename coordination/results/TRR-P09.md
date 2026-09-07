# TRR-P09 implementation checkpoint

This checkpoint provides the shared fixed/directional runner primitives at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, based on the approved P09 adapter and streamed-bank contract. No public bank, model checkpoint, target observation, truth, GPU, or scientific fit was opened or run.

The runner now has a required random-access loader adapter for arbitrary scheduled rows, preserving order and duplicates; validates nonnegative scheduled rows, complete batch geometry, BOS/position IDs, and every sampled position's attention mask; and records lazy-load and optimizer-update timing separately. One shared optimizer contract covers all unique trainable decoder and readout-hook parameters, rejects omitted or unowned trainable parameters, clips the same complete set, and checks post-update finiteness. Checkpoint state digests include both decoder and hook state. Per-step state hashing remains opt-in and is disabled by the shared loop; checkpoint hashing is the normal path.

The synthetic directional hook test demonstrates that a trainable readout parameter receives an update and is included in clipping and state hashing. The runner retains explicit step-zero eligibility and earliest-strict-maximum checkpoint selection, and validation accepts an independently supplied sequence width for the H128 view. Domain-balanced validation aggregation remains an evaluator responsibility; the generic loop currently reports raw token totals and the caller-supplied metric.

Validation command (from the isolated P09 worktree):

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_fixed_control_runner.py tests/test_trr_p09_fixed_control_adapter.py tests/test_trr_p09_bank_artifact.py
```

Result: 25 passed in 1.90s. Syntax validation also passed with `python3 -m py_compile scripts/trr_p09/fixed_control_runner.py src/token_reconstruction/trr_p09_fixed_control_adapter.py`.

Before any real qualification or fit, the study caller still must bind the finalized serialized schedule reader/digest/exposure contract, domain-balanced validation metric, checkpoint state serializer/callback, and outer fail-closed resource watchdog/CLI. The generic loop has no production asset loader, state-file writer, resource guard, or scientific constants and is therefore an implementation checkpoint, not a fit-ready scientific run. Setup-owned bank files and unrelated planning files were intentionally excluded from this commit.
