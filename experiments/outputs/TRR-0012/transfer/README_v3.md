# TRR-0012 evaluator transfer capture approved-seed compatibility

Use `capture_plan_exec_v2.json`, `capture_config_v3.json`, and `capture_commands_v3.json`. The immutable approved design is `experiments/TRR-0011/transfer/variant_plan_v2.json`: seeds are 1101, 1111, 1101, 1111, and 1113 for the five artificial conditions. The current runner truthfully executes seed 1101, but the actual constructor uses recipe_seed only through `(index + seed) % 2`; all approved seeds are odd and therefore produce byte-identical updates. `seed_compatibility_test_receipt_v2.json` records the focused actual-constructor proof, accepts the corrected plan, rejects the incompatible prior plan, and includes an even-seed control.

The old v1/v2 setup files are excluded before capture in `seed_plan_exclusion_receipt_v1.json`. No runner source was changed. Full capture remains deferred until Agent2's opaque panel and fresh live guarded preflight are available.
