# TRR-0005 bounded runnable handoff

The study has two freezes. First, before fresh source selection, root freezes
method IDs, rules, six selected states, executable code commit, public
validation selection, and the decision plan. This method freeze has no fresh
panel dependency and contains no holdout IDs or source content. After that
freeze, coverage reserves and selects the two 128-record pools, captures the
four public cells, and writes the panel-bound registration. The panel-bound
registration adds panel and observation file hashes, then the complete
prediction matrix is frozen before truth.

The canonical registration schema is
token-reconstruction.trr0005-confirmation-registration.v1. Its status is
FROZEN_METHOD_REGISTRATION, method_ids is the exact eight METHOD_IDS order,
and code_commit is a full lowercase 40-character commit. Each state_bindings
entry uses the TRR4 core groups panel, method_state (one path/bytes/sha256
record), code (one or more path/bytes/sha256 records), and code_commit.

Runtime assets are part of every state binding. All methods bind exactly one
absolute external descriptor under runtime_assets.public_embedding_table
(the normalized public E table). Only frozen_a1_a2_k256 binds the additional
public_prefix_checkpoint and public_prefix_config descriptors. Standalone A1
and all six joint decoders must not be required to bind or load P0. The
pretruth scorer rehashes E for all eight methods and P0 only for A2, both
before prediction artifacts are accepted and again before truth.

The coverage producer owns the post-method-freeze source reservation, paired
128-record panel, and four cut-4 observation descriptors. The panel uses
token-reconstruction.trr0005-fresh-confirmation-panel.v1, status
FROZEN_FRESH_CONFIRMATION_PANEL, four ordered cells, and 128 identical
record IDs between each domain's two target conditions. Each observation
descriptor supplies a regular file record, shape [128,128,2048], and public
attention_mask and position_ids. No target truth or source plaintext enters
the frozen reconstruction root.

The decoder driver owns one method at a time. It calls the predictor wrapper
with exactly one warmup and one measured call for each record and requires
identical IDs. It calls write_prediction_artifact with cell_id, method_id,
predictions, the registered binding, panel_sha256, selection_plan_sha256,
observation_sha256, and repository_root. The output artifact uses
token-reconstruction.trr0005-fresh-confirmation-prediction.v1 metadata and
exactly one safetensors tensor named predictions with shape [128,128]. The
fitting geometry remains [1200,192,2048]; it is not the fresh prediction
shape.
Metadata binds panel, plan, observation, cell, method, geometry, binding,
candidate_policy, and candidate_output. BOS is 128000; scored IDs are in
[0,128256); padding is -1; A2 candidate arrays are omitted.

The predictor and timing descriptor maps together contain exactly 32
cell/method keys. Each timing row has warmup_runs_per_record=1,
measured_runs_per_record=1, and warmup_output_exact_match_measured=true,
plus measured interval, synchronization, I/O, and peak-memory evidence.

The footing freeze adapter is scripts/trr0005_freeze_confirmation.py. When
writing a receipt it must receive actual panel_path, registration_path,
plan_path, output_root, and receipt_path. The receipt plan record and
metadata hashes bind the requested files and executable commit. The scorer
gate is validate_before_truth, called through score_with_truth_loader with
all paths, observations, the frozen public validation selection, and the
complete descriptor maps. It rehashes state, code, runtime E/P0, panel,
registration, plan, observations, and all 32 prediction tensors before
invoking the truth loader. The scorer CLI is
scripts/trr0005_score_confirmation.py and accepts prediction/timing
descriptor manifests, observations, truth, receipt, and optional paired
frequency references.

Ownership is intentionally split: coverage edits only its producer and
panel/capture outputs; decoders edit only the prediction driver and timing
outputs; footing owns predictor writer, freeze, scorer, and negative tests;
root owns the final freeze, resource windows, truth opening, commits, and
publication.

The post-fit attention diagnostic is scripts/trr0005_attention_diagnostic.py.
After the helper is committed and root grants the post-fit window, run it with
exactly one selected causal state for each distribution:

```text
PYTHONPATH=src:scripts .venv-trr0005/bin/python scripts/trr0005_attention_diagnostic.py \
  --validation-manifest <common-public-validation-manifest.json> \
  --state original=experiments/TRR-0005/joint_fit_v1/original/affine_causal_h_attention128/selected.safetensors \
  --state enriched=experiments/TRR-0005/joint_fit_v1/enriched/affine_causal_h_attention128/selected.safetensors \
  --output experiments/TRR-0005/attention_diagnostic.json \
  --hash-inputs
```

The helper reads only validation observations H and validation validity masks.
It verifies both state metadata as affine_causal_h_attention128/causal, then
recomputes layer_norm(eps=1e-5), Q/K projections, scaled scores, right-padded
causal masking, and safe softmax. It reports query-weighted current-position,
earlier-position (excluding BOS), BOS mass, entropy in nats, and strict
self-mass > 0.99 fraction overall and in the fixed post-BOS position bins.
It records truth_accessed=false and embedding_table_loaded=false.
