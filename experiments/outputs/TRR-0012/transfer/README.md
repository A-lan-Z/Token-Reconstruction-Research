# TRR-0012 evaluator transfer capture handoff

This is a new namespace for the approved TRR-0011 evaluator-side capture continuation. Original files under `experiments/TRR-0011/transfer` are unchanged; byte-identical small copies are under `inherited/`.

The committed `scripts/trr0011_capture.py` is reused. `capture_plan_exec_v1.json` is generated create-only by that runner and passes its frozen-plan validator. The inherited `capture_recipe_v1.json` remains provenance and is not edited; its seed mismatch with the current validator is recorded in `capture_config_v1.json`.

The evaluator performs the original B8 x 192 BF16 full forward at cut 4, then retains positions 0:128 including BOS. Masks and positions are sliced to 128 before decoder handoff. The panel is exactly 32 Finance and 32 Pile records in paired order, but Agent2 must supply its reserved opaque descriptor and evaluator-owned token batches before launch.

`capture_commands_v1.json` contains deferred argv templates. No GPU, model, source panel, truth, target training, decoder fitting, teacher, or download was used in this preparation phase. Actual transfer wall time remains unknown.
