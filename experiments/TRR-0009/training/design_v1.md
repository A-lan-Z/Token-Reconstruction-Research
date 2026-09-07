# TRR-0009 anchored supported-token readout design

This is the pre-fit design for TRR-0009. No GPU fit has been run.

## Starting point and public data

All three arms start from the same published TRR-0007 current-bank
`trr0007_residual_mlp512` selected state. The fitting and validation bank
remains the published improved public bank below; only the competent starting
checkpoint is changed to the current-family continuation state so a second
continuation is not tested after a zero-error ceiling.

- State: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007/experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors`
- State SHA-256: `2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`
- State bytes: 29,390,628; selected step 2100 from a 3000-step fit; residual MLP512; current-H only; full-vocabulary CE.
- Normalized public `E`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`, shape `[128256,2048]`, 1,050,673,488 bytes, SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
- Fit manifest: `experiments/TRR-0007/support/broader_capture_v2/enriched_manifest.json`, 8,156 bytes, SHA-256 `c7a857e545a2f252ce8b3ab71bb2336e552fd212c7c434e39cef66f233b77a08`.
- Fit activation/label/mask payload: `.../broader_capture_v2/enriched_fit_cut4.safetensors`, shape `[1200,192,2048]` plus labels/mask, 947,176,760 bytes, SHA-256 `a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d`.
- Validation manifest: `experiments/TRR-0007/support/broader_capture_v2/original_manifest.json`, 8,355 bytes, SHA-256 `e0b3182c10dad06dae6c54be7bc1bba05f867020d6ffc419f6964563a0e64109`.

The fit bank has 124,371 valid post-BOS positions. Counting only valid
positions 1..191 gives 17,126 supported vocabulary IDs and 111,130 unseen IDs.
The supported frequency strata are:

| Fit count | Token IDs | Post-BOS rows |
| --- | ---: | ---: |
| 0 | 111,130 | 0 |
| 1--4 | 14,257 | 22,783 |
| 5--9 | 1,587 | 10,165 |
| 10--49 | 1,061 | 19,463 |
| 50+ | 221 | 71,960 |

The frequency vector, support IDs, and their SHA-256 digests will be written
into the training receipt. They are derived from the public fit labels only;
no evaluation truth is involved.

## Adaptable readout

The adaptable arm wraps the published residual MLP512 decoder. The three arms
use the same architecture and the continued arms continue fitting the
decoder parameters from the selected checkpoint. For each supported
vocabulary row `v`, the adaptable arm adds two trainable scalars initialized
to zero:

```
gain_v = 1 + 0.25 * tanh(raw_gain_v)
bias_v = 0.25 * tanh(raw_bias_v)
logit_v = (base_logit_v * gain_v) + bias_v
```

Rows absent from the fit bank have no trainable entry and remain exactly at
gain=1, bias=0. The bias is applied after the bounded multiplicative logit
scale. The correction is token-specific and cannot be represented by one
global hidden-space affine map: it independently changes vocabulary rows
outside the existing H-to-hidden parameterization. The bounds preserve the
public dictionary around its initial behavior. The exact frequency-weighted
zero-anchor penalty is
`1e-4 * mean((1 + 4/sqrt(count_v)) * (raw_gain_v^2 + raw_bias_v^2))` over
supported rows. The full vocabulary is still scored on every training draw;
support-only means only the readout correction parameters, not the CE
vocabulary, are restricted.

The model interface provides row-scored logits for the training path, full
`[records,positions,vocabulary]` logits for inference, and an explicit
materializer that returns the effective dictionary plus bias. The materializer
is not used to hide deployment cost; its full `[128256,2048]` FP32 output and
preparation time will be recorded. At zero correction, row and full-logit
outputs must be exactly equal to the anchor, and state save/load must preserve
the support/count binding.

## Arms and optimization

1. `unchanged_anchor`: evaluate the loaded published state without updates.
2. `continued_fixed_readout`: train the same loaded state with the published
   `E` fixed.
3. `continued_adaptable_readout`: train the same loaded state plus the
   supported-token correction.

The continued arms use one shared deterministic schedule with semantic digest
`5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472`, seed
4005, 3,000 steps, record batch 8, 512 post-BOS draws per step, learning rate
`2e-4` for the inherited decoder parameters, AdamW weight decay 0, gradient
clip 1, cosine decay, and validation every 100 steps. The adaptable arm uses a
separate `1e-3` cosine group only for its newly zero-initialized gain/bias
scalars; the base group remains exactly the same `2e-4` continuation as the
fixed arm. The lower base continuation rate is a frozen gentle restart from
the selected step-2100 checkpoint, whose historical cosine schedule is already
late in its run; this is shared by both continued arms and is not a sweep.
The unchanged arm receives the same step-zero validation and the same
selection rule. Step zero is eligible for every arm; continued arms select the
earliest maximum validation style-balanced token accuracy, while the unchanged
anchor is explicitly selected at step zero.

The initially-wrong challenge is recomputed from the actual selected
step-2100 residual MLP state against the actual public fitting labels, using
seed 11012 and cap 2,048. The trainer remeasures the actual starting-state fit
errors and records the result, including whether the selected-state challenge
is empty, without substituting another mask. No neutral-model challenge mask
is substituted. The older TRR-0007 neutral masks (`046cc...` and `b69af...`)
are retained only as historical diagnostics and are excluded from TRR-0009
selection or claims. The earlier improved-bank starting-state proposal
(`08056c...`) is preserved in `design_v1_prefit_improved_state_superseded.md`
and is not an experimental result. An empty starting-state challenge is
reported as such; held-out validation and frequency-stratified fit curves
remain the informative comparisons.

Selection is earliest maximum validation style-balanced token accuracy. Each
curve records fit accuracy/loss, held-out validation, challenge recovery, and
frequency-stratified fit metrics at every validation checkpoint. The final
receipt reports row gains/regressions, exact 127-token records, unseen/rare
strata, preparation/materialization cost, state size, optimizer memory, and
warmed inference cost.

## Resource and failure policy

The largest gathered training batch is `[8,192,2048]` float32 H with 512
full-vocabulary logits (`512 * 128256 * 4` bytes, about 263 MB). The separate
full-logit initialization equivalence check uses B=1, where two retained
full-output tensors are about 197 MB; it is not folded into the B=8 training
batch estimate. The fixed
normalized E is about 1.05 GB; the residual decoder and Adam state are about
118 MB, and the adaptable readout adds 34,252 parameters (about 0.13 MB
FP32, about 0.52 MB including gradients and Adam state). The conservative
planning envelope is 1.6--3.6 GB reserved, using TRR-0007's measured 3.60 GB
peak as the upper reference, with a 6 GiB reserved cap, 8 GiB free-GPU floor,
16 GiB RSS cap, and 10 GiB host-available floor.

A CPU metadata preflight checks all manifest shapes, hashes, support counts,
schedule digest, starting-state identity, and output containment. A measured
largest-cell qualification must pass finite loss/gradients, exact zero-state
equivalence, actual `[8,192,2048]` geometry, and live guard margins before
any 3,000-step fit. Every arm is create-only and restart-safe; anomalies leave
failure evidence and stop the lease.
