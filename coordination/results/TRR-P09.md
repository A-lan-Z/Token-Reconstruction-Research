# TRR-P09 implementation checkpoint

This checkpoint provides the shared fixed/directional runner primitives at source commit `0c7279e30a41a5016ff97c9bcaa5e282db2dabe0`, based on the proposed P09 adapter and streamed-bank contract; the common scientific contract is still pending. No public activation bank, model checkpoint, target observation, truth, GPU, or scientific fit was opened or run. A separate permitted public-source/tokenizer count audit is recorded below; it did not load a model or select a P09 row.

The runner now has a required random-access loader adapter for arbitrary scheduled rows, preserving order and duplicates; validates nonnegative scheduled rows, complete batch geometry, BOS/position IDs, and every sampled position's attention mask; and records lazy-load and optimizer-update timing separately. One shared optimizer contract covers all unique trainable decoder and readout-hook parameters, rejects omitted or unowned trainable parameters, clips the same complete set, and checks post-update finiteness. Checkpoint state digests include both decoder and hook state. Per-step state hashing remains opt-in and is disabled by the shared loop; checkpoint hashing is the normal path.

The synthetic directional hook test demonstrates that a trainable readout parameter receives an update and is included in clipping and state hashing. The runner retains explicit step-zero eligibility and earliest-strict-maximum checkpoint selection, and validation accepts an independently supplied sequence width for the H128 view. Domain-balanced validation aggregation remains an evaluator responsibility; the generic loop currently reports raw token totals and the caller-supplied metric.

## Agent 2 public-bank planning checkpoint

The bounded planning recommendation is a 12,000-row proportional expansion, pending the joint identity and capacity contract: 6,000 natural Alpaca, 3,000 natural Pile, 1,800 natural Finance, and 600 controlled contexts each for Pile and Finance. This preserves the published B0 construction ratio while providing 10x its 1,080 natural-parent support. The existing 1,200-row B0 must remain the byte-equivalent prefix. Controlled replacements remain in the inherited 128--191 post-BOS length stratum and four inherited position bins. No new validation pool is proposed; the natural `public_base` validation binding already opened by Agent 1 remains separate and root-owned.

A permitted public Alpaca count/tokenization audit rendered all 52,002 rows with the pinned public tokenizer. It found 50,778 rows after 1,224 keyed fit/validation identity exclusions, with 39,765/24,401/14,317/6,824 rows meeting post-BOS thresholds 64/96/128/160. These are count-only upper bounds: the exclusion metadata has a 4,193-value union containing 4,073 opaque sequence/reservation digests, and zero rendered-text SHA-256 hits against that union do not establish disjointness because the namespaces/canonicalizations differ. The audit receipt and exact command/environment provenance are `experiments/TRR-P09/planning/alpaca-capacity-audit.json` (SHA-256 `d1f11364cabe6db876bb83f09ce1d8d7db1cdc3f76ac2e8bafaed149ba33a5c9`) and `experiments/TRR-P09/planning/alpaca-capacity-audit-provenance.json` (SHA-256 `c9c6828e16d87a7316f36b10d7a832a7cd04c20a73c035b4178062e3926fc2d2`). The executed CLI is `scripts/trr_p09/audit_alpaca_capacity.py` (SHA-256 `db8b54c290a1ee5c8915a6c315c6969ee4cea3284da246f57d044b0cbbd8b80e`). The human proposal is `experiments/TRR-P09/planning/bank-proposal.md`.

No P09 source selection, public model forward, activation-bank capture, fitting, final-panel selection, or truth access has occurred. The deterministic selection recipe is documented only; it must apply exact keyed identities, namespace-matched digests, inherited Pile `[7000,10000)` and Finance `[12000,20000)` reservations, and length quotas before any later root-authorized selection.

Validation command (from the isolated P09 worktree):

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_fixed_control_runner.py tests/test_trr_p09_fixed_control_adapter.py tests/test_trr_p09_bank_artifact.py
```

Result: 25 passed in 1.90s. Syntax validation also passed with `python3 -m py_compile scripts/trr_p09/fixed_control_runner.py src/token_reconstruction/trr_p09_fixed_control_adapter.py`.

Before any real qualification or fit, the study caller still must bind the finalized serialized schedule reader/digest/exposure contract, domain-balanced validation metric, checkpoint state serializer/callback, and outer fail-closed resource watchdog/CLI. The generic loop has no production asset loader, state-file writer, resource guard, or scientific constants and is therefore an implementation checkpoint, not a fit-ready scientific run. Setup-owned bank payloads and unrelated implementation files remain excluded; the planning files listed above are metadata-only Agent 2 evidence and contain no bank payload, model output, selection, or truth.
