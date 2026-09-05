# TRR-0005 joint-fit input interface

The joint runner consumes one public activation manifest for each fitting
distribution and one common public validation manifest:

```text
PYTHONPATH=src:scripts .venv-trr0005/bin/python scripts/trr0005_fit_joint_decoders.py \
  --original-manifest <original-fit-manifest.json> \
  --enriched-manifest <enriched-fit-manifest.json> \
  --validation-manifest <common-public-validation-manifest.json> \
  --embedding-table <normalized-public-embedding.safetensors> \
  --retained-affine-state <TRR-0004-large-affine.safetensors> \
  --output-root experiments/TRR-0005/joint_fit \
  --device cuda --preflight-only
```

After the source-only forecast passes and the shared GPU window is granted,
run the actual largest-cell qualifier before the full matrix:

```text
# workdir: /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005
PYTHONPATH=src:scripts .venv-trr0005/bin/python scripts/trr0005_fit_joint_decoders.py \
  --original-manifest <original-fit-manifest.json> \
  --enriched-manifest <enriched-fit-manifest.json> \
  --validation-manifest <common-public-validation-manifest.json> \
  --embedding-table <normalized-public-embedding.safetensors> \
  --output-root experiments/TRR-0005/qualification \
  --device cuda --qualification-only --qualification-steps 2
```

Qualification loads only the two public fit manifests and the common public
validation manifest, runs the causal arm for two real AdamW updates per
distribution using the first two steps of the registered 3000-step schedule,
and writes measured timing, guard, host/GPU peak, conservative forecast, and
cross-distribution schedule identity receipts. It does not require the
retained affine state, and it never opens final-evaluation resources.

The two fit manifests use the public-fit schema
`token-reconstruction.trr0005-public-fit-data.v1` (the TRR-0004 adapter schema
is also accepted).  Their `resources` object provides the following logical
resources, each as `{path, tensor_key, shape}` where `path` is resolved
relative to the manifest:

```text
fit_observations   -> [records, 192, 2048] floating activations
fit_truth          -> [records, 192] integer public token IDs
fit_valid_mask     -> [records, 192] binary right-padded mask
fit_records        -> JSON records with unique record_id and optional domain/style
embedding_table    -> [128256, 2048] normalized public embedding table
```

The common validation manifest provides the corresponding `validation_*`
resources.  A combined safetensors artifact is supported by assigning the
keys `activations`, `token_ids`, and `attention_mask` to the three tensor
resources.  Every valid row begins with BOS `128000`; only positions after
BOS are supervised.  Fit and validation record IDs must be disjoint.  The
validation records should carry `style` or `domain` values (`alpaca` and
`pile`) so selection uses the declared unweighted style-balanced accuracy.

The coverage producer's token plan is not an activation manifest.  For the
original-like arm, the existing public TRR-0004 activation artifact may be
bound directly.  For `coverage_mix_v1`, every constructed sequence must first
receive a fresh public-model forward at cut 4; rows from another sequence may
not be spliced into the artifact.  The producer should emit the same tensor
keys and record metadata above, plus a metadata-only resource record showing
the source corpus plan and coverage summary.

The saved position-schedule receipt reports `total_draws`,
`unique_draws_within_step`, `repeated_draws_within_step`, and replacement-only
unique/repeated counts.  Thus a small fixture that requires replacement can
show the repeated exposures explicitly while the registered 1200-record fit
stream reports 512 unique pairs in each sampled step.

The runner keeps each distribution's tensors only for its sequential fit,
then releases them before loading the other distribution.  It writes
`memory_preflight.json`, `position_schedule.safetensors`,
`pretraining_diagnostic.json`, one learning curve and selected state per arm,
and `run_evidence.json`.  It never loads a final holdout panel.
