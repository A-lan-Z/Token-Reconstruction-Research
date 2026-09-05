# TRR-P03 source-panel integration review

This review used CLI help, static inspection, Python compilation, and the model-free frozen-panel parser test. No model, prototype table, observation payload, truth scorer, or full evaluation was run.

The frozen evaluator panels use schema `token-reconstruction.trr-p03-setup-panel.v1`, with row stage aliases `s1` and `s2`. The current `scripts/trr_p03/generate_observations.py` accepts that schema and its metadata-only `_read_panel(..., open_truth=False)` path resolves the sibling stage-specific truth location without opening the sidecar. The reconstruction CLI must receive generated evaluator output `public/observation_index.json` (schema `token-reconstruction.trr-p03-observation-index.v1`), not the setup-time `observation_index.json` template.

The following integration issues remain for root to resolve or explicitly guard before observation generation:

1. `generate_observations.py --stage all` groups Stage-1 and Stage-2 rows with the same length into one safetensors descriptor. The descriptor is labelled with the first row's stage and its filename uses `stage1`, so a 12-record mixed-stage bundle can violate the separate Stage-1/Stage-2 interface. Use separate invocations/output roots for the two stages or reject `all` until grouping is stage-aware.
2. The grouped public index descriptor has no `mask_digest` or `position_digest`; `io.validate_observation_index` synthesizes empty strings when it expands groups. The setup interface declares these per-record digests as required, so either emit verified digests or revise the runtime contract before certification.
3. `bundle-id` is not bound by the generator to an expected target identity. `load_model` checks geometry, while the public index writes the matched base model identity for both bundles. The evaluator command or a CLI guard must verify bundle-a uses the pinned base snapshot and bundle-b uses the pinned shifted Vikhr snapshot, preserving the historical P01 label only as provenance.
4. An explicit single `--truth` path can be reused for both stages under `--stage all`; reject that combination or require distinct stage paths. The default sibling-path resolution is stage-specific.

The current source also supports a predeclared `--record-ids` JSON subset for qualification. The root must ensure any subset is fixed before target observations and that the generated public index is the only reconstruction input. The public-index source-panel hash is now kept in evaluator evidence rather than the public index.
