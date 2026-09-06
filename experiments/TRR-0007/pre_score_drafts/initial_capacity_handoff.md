# TRR-0007 capacity study handoff

## Status

The positionwise implementation, full-geometry qualifier, and both 3,000-step
existing-enriched fits are complete. The improved public activation bank is not
yet materialized, so the crossed two-bank comparison and fresh natural
selection remain pending. No improved-bank alias was used, and the GPU lease is
released.

The registered methods are:

- `trr0007_current_positionwise`: the retained strict-diagonal trainable
  positionwise affine decoder;
- `trr0007_residual_mlp512`: the same trainable diagonal affine path with a
  per-position `LayerNorm -> GELU` residual `2048 -> 512 -> 2048` path.

Both arms use the same neutral TRR-0005-style initialization: identity `W`,
zero `b`, `s=3`, deterministic diagonal Q/K/V seed 4005, and zero output
correction. The residual up projection is zero initialized. The two arms have
exactly equal step-zero projected rows and full-vocabulary logits on the same
512 gathered positions. The published selected diagonal state is retained as a
separate frozen reference; it is not the initialization of either crossed arm.

## Implementation and interface

The frozen implementation is in
`src/token_reconstruction/trr0007_positionwise.py` and
`scripts/trr0007_train_positionwise.py`, from code commit
`0ca98c7a7d89ad945c033cca3d0787f0732f628a`. The selected state loader is
`load_positionwise_model_state(path, method_id, hidden_size=2048,
vocabulary_size=128256, context_width=128, bottleneck_size=512)`. Each model
exposes `projected_hidden`, `logits_from_rows`, and `forward`; inference uses
only current `H_i` and the fixed normalized public `E`, with full-vocabulary
cross entropy during fitting. The evaluator binding is recorded in
`experiments/TRR-0007/evaluation_interface.md`.

The deterministic recipe is 3,000 updates, seed 4005, AdamW learning rate
`1e-3`, weight decay `0`, gradient clip `1`, record batch size 8, sequence
width 192, hidden width 2048, position budget 512, and validation every 100
steps. The materialized fit geometry is `H [1200,192,2048]`, validation
`H [48,192,2048]`, gathered activation `[8,192,2048]`, labels and masks
`[8,192]`, and selected rows `[512,128256]`. Selection is the earliest maximum
validation style-balanced accuracy, including step zero. The challenge subset
contains 2,048 seeded rows initially wrong under the common neutral state.

## Inputs and source bindings

Both completed runs used the actual TRR-0005 source manifest in the sibling
worktree, rather than an absent TRR-0007-relative copy:

```text
../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_manifest.json
```

The source manifest SHA-256 is
`8a8a9490e549181959711496ce9a18c63651de3f7cd849b90716d5cf4531db78`.
Its fit activation resource is SHA-256
`191cb77dae8d002402bcf3f126a20c5d8d34111a6e6871d66507503ca6725a99`, its
validation activation resource is
`a8e7633ffb369864af33754c5ebb2d9a4ca9d6e7d4550731e8ff26e20c8200cf`, the fit
record metadata is
`7f197b077aa0aa66edfdd8d92c8daa5b4cb2ae36bdbb80938e4ab8f1de117943`, and the
validation record metadata is
`30b422b681bef5e7af4c26d339e57dfb3571ecef8077bdc4be5d960ef05c9777`. The
fixed normalized public embedding table has shape `[128256,2048]` and SHA-256
`ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
The retained selected reference has SHA-256
`696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2`.

## Resource qualification

The first qualifier attempt is preserved at
`experiments/TRR-0007/qualification_enriched_v1/failure.json`. It failed before
materializing tensors because the TRR5 manifest's relative activation path
resolved to an absent TRR7-local `outputs/TRR-0005` path. No fit or gradient was
started. The corrected two-step command was:

```text
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 scripts/trr0007_train_positionwise.py --banks enriched --qualification-only --qualification-steps 2 --device cuda --fit-manifest ../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_manifest.json --validation-manifest ../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_manifest.json --retained-reference experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors --output experiments/TRR-0007/qualification_enriched_v2
```

The corrected qualifier passed in 34.5225 seconds at code commit
`0ca98c7a7d89ad945c033cca3d0787f0732f628a`. It materialized the actual
`[8,192,2048]` batch and eight unique records, with 512 draws, finite gradients
for both methods, and exact step-zero equivalence (`max abs delta = 0` for
projected hidden and logits). CUDA peak allocated memory was 3,597,054,976
bytes against the enforced 4,413,456,384-byte conservative floor. At
completion, CUDA free memory was 15,555,624,960 of 17,066,033,152 bytes; host
available memory was 20,125,904,896 bytes and process maximum RSS was
5,618,692,096 bytes. The qualifier used a two-step cosine schedule for resource
qualification only; it is not a prefix-equivalent main-fit curve.

The neutral-state challenge was nonempty: 84,826 public fit rows were initially
wrong, with 2,048 selected by seeded uniform sampling (`mask_sha256`
`b69af41d790fcaf63af93b6a2b062d9bab873a5aca666dd0c6e4dc8aed2be87b`, seed
11012). The frozen selected reference measured 17 fit-token errors out of
124,371 (`0.9998633122`) and 1,185 exact records out of 1,200. Its public
validation token accuracy was `0.9674433450`, with 18 exact records out of 48
and style-balanced accuracy `0.9547109695`.

## Existing-enriched fits

The authorized full-fit command was:

```text
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 scripts/trr0007_train_positionwise.py --banks enriched --steps 3000 --device cuda --fit-manifest ../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_manifest.json --validation-manifest ../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_manifest.json --retained-reference experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors --output experiments/TRR-0007/enriched_fit_v1
```

It completed in 262.5008 seconds, from
`2026-09-06T08:27:11.961882+00:00` through
`2026-09-06T08:31:34.114922+00:00`, at the same code commit. CUDA peak allocated
memory was 3,599,152,128 bytes (81.55% of the conservative floor); completion
left 15,555,624,960 bytes free. Host available memory was 20,053,188,608 bytes
and process maximum RSS was 5,618,475,008 bytes. The run had no failure receipt,
and a recursive finite-value scan found no nonfinite floats.

| Method | Parameters | Selected step | Fit token accuracy | Fit errors | Exact fit records | Challenge accuracy | Selected validation style-balanced | Validation token accuracy | Final step-3000 style-balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current positionwise | 5,247,361 | 1600 | 0.9998633122 | 17 | 1,185/1,200 | 2047/2048 | 0.9547109695 | 0.9674433450 | 0.9515848632 |
| residual MLP 512 | 7,347,073 | 2100 | 1.0000000000 | 0 | 1,200/1,200 | 2048/2048 | 0.9569962828 | 0.9693584424 | 0.9562345117 |

The current arm reproduces the inherited validation curve and selected step
1600. Its initially-wrong challenge reaches 100% by step 1800. The residual arm
also reaches 100% challenge accuracy by step 1800 and has the higher selected
style-balanced validation score by 0.0022853133 (0.2285 percentage points).
This is an existing-bank result; it does not establish a capacity effect over
the improved bank until that bank is materialized and fitted under the same
recipe.

The selected state artifacts are:

- current: `experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_current_positionwise/selected.safetensors`, SHA-256
  `b7615d077cb403cb06ebbfbdeb188f8d4a1682feefdffa22cc55430b2c87a5d4`,
  serializer state SHA `927ce84484e23f1670857a285c4d8b2186c008378c5988a57ab21d4da28e8878`;
- residual: `experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors`, SHA-256
  `2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`,
  serializer state SHA `d92fa751363113401bb067b9e2ef70ad274bceed8166d52a304d95668f1a7e3f`.

The full-fit schedule is 3,000 steps, seed 4005, with SHA-256
`5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472`; both
methods used the same schedule. The schedule had 1,536,000 draws, 1,535,800
unique within-step draws, and replacement only on one step where the selected
batch had fewer than 512 eligible positions.

A CPU comparison of the selected current state against the retained TRR-0005
state found all 11 parameter keys equal byte-for-byte (`torch.equal` true for
W, b, s, Q/K/V weights and biases, and output weight/bias), with maximum
absolute difference zero. This confirms the inherited affine arm's exact
state/output identity despite different state container metadata. The residual
state round-trips through the TRR-0007 loader with 7,347,073 parameters.

## Support-bank review and pending work

The support package currently present under
`experiments/TRR-0007/support/broader_recipe_v1` is explicitly marked
`METADATA_RECIPE_PENDING_REAL_P0_FORWARD`; its `support_summary.json` remains
`PUBLIC_FIT_SUPPORT_DESCRIPTIVE; NO_PRIVATE_TRUTH`. It specifies 120 controlled
records, 60 per domain, 3,600 replacement occurrences, position bins covering
post-BOS positions 1–127, and a proposed 3,600-token identity pool while
retaining the 1,080 natural rows. It does not yet provide an improved fit
activation tensor or a TRR5-compatible activation manifest. Therefore no
improved-only fit has been launched. Once support materializes real P0 forwards,
compatibility must be checked for schema, regular-file paths, `[1200,192,2048]`
fit geometry, `[48,192,2048]` validation geometry, fixed normalized-E hash, and
the same training opportunity budget.

## Validation and evidence locations

The focused model/trainer and TRR5 tests passed before the lease; the combined
TRR-0007 capacity/evaluation/support suite passed with 28 tests in 1.84 seconds.
The qualifier and fit receipts, curves, challenge receipts, schedules, state
files, input hashes, and runtime evidence are under:

- `experiments/TRR-0007/qualification_enriched_v1/` (preserved path failure);
- `experiments/TRR-0007/qualification_enriched_v2/` (passing qualifier);
- `experiments/TRR-0007/enriched_fit_v1/` (two complete existing-bank fits);
- `experiments/TRR-0007/manifest.json` (structured task evidence).

The executable and model code remained unchanged after commit `0ca98c7`; output
and evidence retention was committed by the parent at `1e0264bbf897d681af6e6fdf15bb3328de91c8f7`.

A retrospective TRR-0006 joint style × position × frequency error table was
requested after this fit. The public TRR-0006 score receipt safely exposes
per-record totals and aggregate cell scores, but not token labels or an error
mask. The task charter also prohibits opening another task's sealed holdout;
the environment blocked the requested `/tmp/trr0006/private/truth.safetensors`
read. The aggregate diagonal errors retained in the public score receipt are
Pile public base 12,195, Pile public LoRA 12,103, Finance public base 4,653,
and Finance public LoRA 4,477, with exact clip counts 42, 45, 350, and 399
respectively. No joint retrospective claim is made from those aggregates.
