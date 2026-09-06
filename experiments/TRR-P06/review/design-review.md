# TRR-P06 design review

**Status: PROPOSED; execution remains held.** The complete control packet is
preserved at `coordination/requests/TRR-P06.md` (SHA-256
`6f2883f1ec078877358c78fe5d05566ef845a1329f4e8f7a4a96aa69c5c5f992`). This
review records the bounded design for root/setup approval; it is not a freeze,
a resource grant, a fit, a capture, or a truth-scoring authorization.

## Scientific choice

The intervention is visibility within one compact decoder family. Every arm
uses the same trainable direct affine path and one attention correction,

```text
z_i = H_i W^T + b + O(sum_j attention(i,j) V LN(H_j)),
```

followed by the same float32 normalization and public full-vocabulary readout.
The common direct path starts from the published public TRR-0004
`historical_affine_ce_no_vocab_bias` fit, selected at public-fit step 1900:
`experiments/TRR-0004/evidence/affine/selected_states/fit_large_v1.historical_affine_ce_no_vocab_bias.safetensors`, SHA-256
`09c5b852373d8555b06508a79bb00c94041202702b61b121b35fa2b6f9f64e65`.
Its `W`, `b`, and inherited `s` (3.5380859375 in the pinned F32 state)
are loaded identically into all three P06 arms; it is a public trained affine
baseline, rather than an unqualified identity map. Q/K/V are deterministic and shared within each replicate, and the
attention output weight and bias are zero initialized. The initializer is
public-only and does not load a target checkpoint or evaluator resource.

The masks are the sole intended visibility intervention:

- `p06_positionwise_diagonal` permits only the current valid key `H_i`.
- `p06_past_only` permits valid keys `H_0` through `H_i`.
- `p06_full_record` permits every valid key in the declared current record,
  including later positions.

All three arms use layer-normalized Q/K cosine scores with scale 4.0. This
avoids the P04 normalization confound. At the fixed H128 boundary, a full
valid query at scored position `i` has `127-i` later vectors available. The
full arm receives activation observations only: it has no source labels,
plaintext, target prefix, guessed-token feedback, or later reconstructed
answers. It may precompute activation-only features for the whole record and
then emit immutable predictions in order. It is an offline full-record method,
not a token-streaming decoder.

The shared parameter count is 5,247,361 for hidden size 2,048 and attention
width 128. Under the diagonal mask, Q and K are structurally gradient-inactive
because each valid query has one key, giving 4,722,817 effective trainable
parameters. Total counts match, but effective capacity does not; both counts
are reported and this difference is part of the interpretation.

## H128 fitting and capacity qualification

The existing public `coverage_mix_v1` manifest is a 192-token source artifact,
but P06 declares an H128 crop for **every** arm and every public selection
operation. Positions 0 through 127, including BOS, are cropped before mask
construction, schedule construction, fitting, validation, or metric
calculation. Source positions 128 through 191 are never keys, queries, labels,
or denominators. The resulting public fit geometry is 1,200 x 128 x 2,048 with
112,825 valid post-BOS positions; the common 48-record validation geometry has
2,982 valid post-BOS positions. The crop and its mask/token/record digests are
recorded in the P06 input receipt.

Before the main fits, run one bounded trainability probe on public fitting data.
Run the pinned affine `W,b,s` over the frozen H128 fit rows and construct a
public-only ledger of exactly 256 positions where its prediction is wrong,
with 64 positions in each scored-position bin `[1,15]`, `[16,39]`, `[40,79]`,
and `[80,127]`. If a bin cannot supply 64 errors, fail closed. Freeze the
direct affine path and train only each arm's added attention path for 300 fixed
updates, using fixed 8-record batches and 512 query draws **with replacement**
from the ledger errors belonging to each batch, seed 6106, and the same ordered
batch/draw schedule for all masks. The actual diagonal, past-only, and full
masks remain active during this probe. Every arm must stay finite and
correct at least 52/256 of these initially wrong public-fit positions. Probe
states are discarded and cannot select or initialize the six main fits. This
qualifies added-path trainability on known public-fit errors; it does not claim
natural-panel transfer or serve as a validation model-selection rule.

The six main fits then start from the same pinned affine direct path, with all
three arms training under their actual masks. Use the same public fitting
records, labels, H128 crop, 8-record/512-query schedule, and validation records
for each arm. Use AdamW at learning rate `1e-3`, zero weight decay, gradient
clip norm 1.0, cosine scheduling, and 3,000 updates. Replicates use seeds 6106
and 6107 with the same schedule seed across arms within each replicate.
Validation is checked every 100 updates; with the explicit
`--selection-metric token_accuracy` setting, each arm/replicate selects its
earliest maximum full-vocabulary micro token accuracy over all valid public
validation positions, including step zero.
No fresh answer or changed-target result enters selection.

After a source-only resource preflight, run a disposable two-update
full-record backward qualifier on the actual 8-record x 128-position cell with
all direct and attention parameters trainable. Discard its state and require
finite loss/gradients and live resource receipts. Only then run the frozen-direct
public-fit-error probe and release the six main fits. The largest qualification
is the same actual full-mask backward cell with 512 post-BOS draws and the
full-vocabulary readout. The
resident normalized public embedding table is 1,050,673,488 F32 bytes; the
H128 activation batch is 8,388,608 bytes and the dense attention score tensor
is 524,288 bytes. The selected-logit matrix remains 262,668,288 bytes, model
parameters 20,989,444 bytes, and Adam moments 41,978,888 bytes. These are
component sizes, not peak-use claims. Measure live device and host margins
with the fail-closed guard before releasing the six sequential fits. Use the
proposed minimum 8 GiB free GPU, maximum 6 GiB reserved GPU, maximum 16 GiB
host RSS, minimum 10 GiB host available, and 1,800-second timeout. Do not use
microbatching or mask shortcuts without an output-equivalence check.

## Fresh panel and metrics

Select 256 new natural records per domain (`pile` and `finance`) with seed
6206, excluding fitting, validation, opened public evaluation, and accessible
duplicate source identities. Require 128 valid positions including BOS and
score positions 1 through 127. Reuse each source record under paired
`public_base` and `public_lora_2601` target conditions; the changed target is a
transfer condition, not an independent source sample. Capture the same H128
observations, validity masks, and positions for all arms. The source universe is stratified by frozen domain only; setup binds the exact
source universe and exclusion ledger before prediction. No target label or fresh
answer enters selection.

The primary contrast is Full minus Past. Secondary contrasts are Past minus
Positionwise and Full minus Positionwise. Report micro token accuracy,
per-record token accuracy, exact 127-token clips, token gains/losses, and
record-level both/gain/loss/neither counts. Report position bins `[1,15]`,
`[16,39]`, `[40,79]`, and `[80,127]` with actual future-valid counts. Invalid
and padded rows are excluded from keys, queries, losses, and denominators; a
public padding-invariance fixture verifies that changing invalid activations
cannot affect valid predictions. Full-record activation computation and
retained state are included in timing and memory costs.

Uncertainty uses 10,000 source-record cluster bootstrap draws (seed 6306),
stratified by frozen domain only. For each domain, one resampled source-index
vector is reused for both target conditions, keeping base and changed targets
paired; the target conditions are never independently resampled. Two fixed training replicates are averaged
within each resampled source, rather than counted as new natural records.

The registered exploratory progression gate is a Full-Past gain of at least
+1.0 percentage point token accuracy or +5.0 percentage points exact recovery,
with a 95-percent paired-bootstrap lower bound above zero in at least one
public-base domain. Every public-base domain/outcome must have a 95-percent CI
lower bound greater than -1.0 pp token and -5.0 pp exact, and changed-target
cells are reported separately. The threshold is a practical decision criterion for this panel,
family, public fit, target pair, and H128 geometry, not a universal benchmark
or equivalence test.

A bounded published-parent A1+A2 K=256 quality anchor uses the first 64 panel
records per domain under public-base only: report pile and finance separately
with 64 clips and 8,128 scored tokens each (128 clips and 16,256 tokens total;
exact denominator 64 per domain). It is the reviewed CPU embedding port of parent method
`frozen_a1_a2_k256`, state SHA-256
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`. The
report must call it a benchmark-compatible port, state its adaptations, and
not call it a native rerun. Its cost and quality denominator remain separate
from all student arms.

## Decision and stop conditions

A public-base gain meeting the benefit and harm limits supports advancing this
tested later-activation observation model. A gain that fails under the paired
changed target is a transfer limitation, not a broad success claim. A finite,
well-qualified negative result means that every public-base token and exact
95-percent CI upper bound is below its registered +1 pp or +5 pp benefit
margin; it supports retaining the simpler model within the tested scope. If no
promotion gate passes and any such upper bound reaches its benefit margin, the
result is imprecise/inconclusive, not evidence that later activations contain
no useful information. No large confirmation, architecture sweep, or redesign
launches automatically after this pilot; the handoff returns one
evidence-backed next decision.

A missing public-fit error quota, failed residual capacity probe, incomplete
arm, changed mask/geometry, unpaired target/source panel, failed resource
margin, or truth access before freeze stops the run without opening fresh
truth. This is an exploratory task-local P06 natural-panel study, not a canonical
dual-benchmark replacement or an overall-best claim; no active registry update
is made. The diagonal effective-capacity limitation, offline full-record
status, and changed-target transfer interpretation must remain explicit. No P03
holdout, P04 teacher/ranking objective, TRR-0007 result, private target
resource, or A2 student fallback is part of this study.
