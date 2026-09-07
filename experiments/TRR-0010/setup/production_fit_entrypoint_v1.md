# TRR-0010 directional-fit entrypoint v1

Status: concrete provider and arm-scoped launcher implemented; CPU-only
configuration and public-bank assembly passed for B0 and B1. No optimizer
update, fitting, GPU launch, source selection, or evaluation truth access has
occurred.

The production path reuses the published A2 pieces in this order:

1. `trr0010_production_provider:build_inputs` verifies one explicit arm and
   bank (`current_directional`/B0 or `expanded_directional`/B1), the signed
   stage-3 contract, per-bank support count/digest, schedule bytes, source
   hashes, state bindings, public validation descriptors, and the fixed 64-row
   diagnostic binding.
2. It loads the shared step-400 decoder state, public E, the matching B0/B1
   support vectors, and the existing streamed bank loader. It constructs A2's
   `ProductionBindings`, `DirectionalRuntime`, exact 13,000-step schedule, and
   `make_fitting_metric_callback` diagnostic callback.
3. `trr0010_directional_fit.py` validates the runner configuration, creates
   the exact `CosineAnnealingLR(T_max=13000)` with 2e-4/1e-4 groups, and keeps
   diagnostics outside checkpoint selection. The A2 runner result, including
   learning curve, exposure, checkpoints, and timing/resource decomposition, is
   persisted immediately after the runner returns before restore/export.
4. The selected checkpoint is restored and exported as one effective W plus a
   base-only decoder. Provider preparation, runner, restore/export, and base
   export timings are separate receipt fields.

The signed stage-3 contract binding is:

- `experiments/TRR-0010/planning/shared_stage3_contract_v1.json`
- 25,643 bytes, SHA-256
  `23e4bde4475082cc004e7ef787c22fed6301275ce2ff23a0396d975b439faf9d`

Its directional support bindings are B0=17,126 rows with digest
`f554d25681904530bf0f52752b50f92ae5da10b99998d06eb0cc30ed25fccd4d` and
B1=45,631 rows with digest
`f91eaf39433496ba6fd36c5c5233e1d8abbaa2527ff7d75e7b304ec4df4fbd0e`.
The shared starting state is the selected step-400 checkpoint with SHA-256
`5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`.

The approved fixed diagnostic descriptor is
`/tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json`,
130,098 bytes, SHA-256
`8b899bc168b55570e4cfd526ff4e8656b70050e840386cba9390ad6adb1f54c3`.
B0/B1 index digests are respectively
`2dba4fdb243f742496cc010985820205a6d36f8c29b7ed3058369322d181f907` and
`670624a8ef439fe11b250f1ba432d87922d6ff53da38ff8514203880237b6933`.
Rows are mapped by `global_row`; descriptive rank order is never zipped.

The launcher supports `--arm current_directional` and
`--arm expanded_directional` for separate leases/output roots, plus the
existing `--arm both` sequential helper. The concrete per-arm command and
resource contract are recorded in
`production_per_arm_commands_v1.json` and
`production_per_arm_commands_v1.md`. The current CPU-only stage-3 metadata
smoke command is:

```text
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 scripts/trr0010_directional_fit_cli.py --configuration-dry-run \
  --arm current_directional \
  --provider trr0010_production_provider:build_inputs \
  --binding-current experiments/TRR-0010/setup/production_arm_binding_current_stage3_v1.json \
  --diagnostic-binding /tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json \
  --lease /tmp/trr0010_prod_smoke/lease.json \
  --output-root /tmp/trr0010_prod_smoke/cli_single_current_v1 --device cpu --deadline-seconds 30
```

This command passed metadata-only with B0 support 17,126. The full B0/B1
assembly receipt, exact source hashes, temporary input identities, preserved
failed attempts, and CPU resource logs are in
`production_provider_cpu_assembly_smoke_v1.json` and its adjacent evidence
files. The receipt records source commit `ceef4e46e8f7ca6d6b7f2e4a3a4a0cf39f263548`
for the successful provider assembly and `e76ebcd2ecd59369207c067543cbb578c5adbd23`
for the current raw-runner receipt implementation.
