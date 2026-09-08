# TRR-0011 controlled-target transfer diagnostic — design v2

Status: **design and synthetic validation only**. No source panel has been
selected, no target-side perturbation has been run, no prediction matrix has
been opened, and no truth has been read. `design_v1.md` is retained as a
superseded proposal history; this file is the corrected design receipt.

The TRR-0010/PR20 panel overlaps TRR-0009 checkpoint-selection validation, so
it remains selection-overlapping development evidence. TRR-0011 uses a new
opaque reservation after Agent 2 and root approve exclusions covering fitting
banks, calibration/checkpoint panels, opened records, and duplicate
record/sequence identities. The proposed, unselected ranges are Finance
`[28000,30000)` and Pile `[9000,10000)`, at most 32 records per domain,
selection seed 5011. These ranges are rejected if the opaque exclusion receipt
does not prove disjointness; no post-selection filtering is allowed.

## Frozen methods and target variants

The two mandatory fixed decoders are TRR-0010 `expanded_fixed` and `current_fixed`. Both use the same cut-4 public observation interface and
public normalized embedding table. Agent 1 does not duplicate the A1+A2
confirmation; Agent 2 owns that comparator and its error inventory.

Each natural source is rendered once per target variant with the same tokenizer,
architecture, cut depth, numerical settings, and record order. The future
producer binds H, masks, positions, source-order digests, code, and decoder
resources without exposing source text or labels to the decoder process. The
panel is 32 Finance plus 32 Pile at most, paired across variants.

The target-side artificial families are fixed before any outcomes:

- early prefix perturbation at layer 1 with relative parameter L2 `1e-3`;
- early prefix perturbation at layer 1 with relative parameter L2 `1e-2`;
- near-cut prefix perturbation at layer 3 with relative parameter L2 `1e-3`;
- near-cut prefix perturbation at layer 3 with relative parameter L2 `1e-2`;
- one after-cut suffix perturbation at layer 4 with relative parameter L2
  `1e-2`, used as a fixed null control.

The after-cut control must preserve H and frozen predictions bit-for-bit. A
failure invalidates the null variant; it is not treated as a small effect.
Amplitudes and recipe seeds are predeclared and never tuned against outcomes. Where layer tensor geometry permits, the early/near-cut pair shares the same sign-direction seed at each amplitude so location is the changed factor; the actual quantized relative delta is recorded by the producer.

The historical `public_lora_2601` file is a bound read-only, historical trained
benchmark calibration condition. Its published generation metadata describes 30
gradient-training steps on the first 32 `target_update_train` records; its
calibration-condition role was threshold fitting. Asset availability does not
establish an independently adapted target. It remains mandatory in the seven-condition
matrix, but no independent or novel held-out target claim is made.

## Geometry and metrics

The selected P09 fixed state is loaded through
`scripts/trr0010_p09_fixed_loader.load_p09_fixed_state` as the strict
15-tensor `ResidualMLPPositionwiseDecoder`. Its state contains the residual
positionwise decoder and scalar logit scale, with no vocabulary gain or bias
tensors. For the projected feature `u`, the public readout is
`z_j = exp(s) * <u, E_j>`.

The truth-free diagnostic records raw H displacement, relative raw H
 displacement, projected-feature L2/cosine displacement, clean predicted-class
margin, pairwise clean predicted-class versus clean runner-up hyperplane
distance, and feature displacement divided by that distance. The top/runner
boundary is explicitly pairwise: the runner-up is not asserted to be nearest
among all vocabulary boundaries, and the ratio is not a certificate or a
fitted deployment threshold. Target-side parameter L2 is reported separately.

After the public gate and only in the evaluator, paired true-token counts must
include explicit `baseline_wrong`, `broken` (baseline correct, changed wrong),
`improved` (baseline wrong, changed correct), unchanged-correct, and
unchanged-wrong counts. These counts are truth-dependent and non-deployable.
For fixed explanatory comparison, `evaluator_only_fixed_auc` reports a
predeclared higher-distortion-is-broken AUC for raw H displacement, relative H
displacement, margin-normalized feature shift, or parameter relative L2. Its
risk set is baseline-correct rows: broken rows are positives and remaining
correct rows are negatives; baseline-wrong/improved rows stay in the paired
inventory but are excluded from this risk AUC. It fits no threshold and
returns UNKNOWN when either risk class is absent or the score spread is within
the fixed near-zero FP32 tolerance. True-token margin statistics are likewise
evaluator-only. Any future predictive threshold requires disjoint update
families for development and held-out validation; this task does not fit one.

## Executable boundary and resource bound

The task-owned path is `scripts/trr0011_transfer.py`:
`validate_transfer_observation_manifest` checks actual safetensor headers,
paired source-order digests, code/state/resource hashes, and false truth flags
before a decoder is loaded. `run_registered_transfer_predictions` then calls
the existing TRR-0010 registration, embedding, method, and row runner; it
writes create-only transfer predictions plus directly computed projected
features/top-runner logits and remains truth-free. A separate
`validate_transfer_prediction_matrix` gate requires the declared transfer
matrix (two fixed methods × every available declared target condition × Finance
and Pile, 32 paired records per domain), exact prediction keys, observation
hashes, and one matching decoder-geometry artifact per cell. It is deliberately
separate from the frozen TRR-0010 public-output gate, whose six-method/128-record
shape cannot represent this diagnostic. Truth remains closed until the complete
transfer matrix gate passes; an unavailable historical target is reported as
unavailable rather than silently replaced by a partial matrix.

The largest planned cell is 64 records × 192 capture tokens × 2048 hidden
values in BF16 (50,331,648 bytes per variant), processed one variant at a
time. The prior public-prefix qualification supports a conservative 6 GiB
working estimate, 8 GiB reserved-GPU ceiling, and 11 GiB minimum free GPU.
Use a 12 GiB host-RSS cap, 8 GiB host headroom, a 20 GiB disk floor plus the
bounded output allowance, and a 900-second outer watchdog. These are
preflight limits; a representative largest cell must qualify first. No GPU,
selection, capture, or truth access is authorized by this design receipt.

## Bound assets and synthetic readiness

The corrected variant plan is `variant_plan_v2.json`; the panel template is
`panel_descriptor_template_v2.json`. The public Llama snapshot and TRR-0010
runner/gate/capture bindings remain inherited and are rehashed when a future
producer manifest is assembled. The P09 loader source and exact 15-key state
schema are included in that future binding. The CPU suite covers variant
validation, two fixed amplitudes, after-cut equality, pairwise geometry,
source-paired observation headers, fail-closed manifest rejection, the actual
runner bridge call boundary, and explicit evaluator-only error accounting.
This is not a public run, source selection, truth result, or independent
text-generalization claim.
