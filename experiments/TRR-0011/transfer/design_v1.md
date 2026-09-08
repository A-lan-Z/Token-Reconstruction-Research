# TRR-0011 controlled-target transfer diagnostic — design v1

Status: **design and synthetic validation only**. No source panel has been
selected, no target-side perturbation has been run, no prediction matrix has
been opened, and no truth has been read. The task-owned executable is
`scripts/trr0011_transfer.py`; it validates the variant/panel declaration and
summarizes paired transfer geometry supplied by a future truth-free producer.

## Why this diagnostic is separate

The amendment corrects the interpretation of the TRR-0010/PR20 panel: it
shares records with TRR-0009 checkpoint-selection validation, so it is
selection-overlapping development evidence rather than an independent
text-generalization estimate. TRR-0011 keeps those predictions and scores
unchanged. This diagnostic will use a new source reservation after Agent 2
and root approve an opaque exclusion receipt covering fitting banks,
checkpoint-selection/calibration panels, prior opened sources, and duplicate
record/sequence identities. The current panel declaration is therefore
explicitly pending and cannot select records itself.

## Frozen decoder and matched baseline

The primary method is the frozen TRR-0010 `expanded_fixed` current-H decoder
(B1 selected state) and the secondary control is the frozen `current_fixed`
(B0 selected state). Both consume the same cut-4 observation interface and
same public normalized embedding table. A1+A2 is run only as a historical
comparator on the same newly reserved common subset; its denominator is 32
records per domain (64 total) if the reservation is approved. No decoder
state, readout, optimizer, threshold, or inference rule changes in this
study.

Each source record is rendered once per target variant with the same public
tokenizer, architecture, cut depth, numerical settings, and record order.
The future target-side receipt must bind the record-order digest and every
observation tensor identity without exposing source text or labels to the
decoder process. The evaluator receives H, mask, positions, and public
metadata only. Candidate generation, routing, timing, and outputs are frozen
before any truth boundary is crossed.

The smallest credible panel is 32 Finance and 32 Pile records (64 natural
sources total), paired across every target variant. A 64-record total panel
is the upper bound for this bounded diagnostic. Exact row IDs and sequence
fingerprints will be supplied only through the separately approved opaque
reservation; no range is asserted here. A proposed range that fails the
combined exclusion receipt is rejected rather than filtered after selection.

## Controlled target variants

The target-side producer creates one clean public-base target and fixed,
predeclared artificial deltas with the same relative parameter L2 (`1e-3`)
and independent seeds:

1. **Early prefix:** perturb one public target layer before the cut (layer 1).
2. **Near cut:** perturb the last public prefix layer (layer 3).
3. **After-cut null:** perturb a suffix layer beginning at layer 4. Since the
   suffix starts after the observed cut, the producer must show bitwise
   equality of H to the clean target and the frozen decoder must show exact
   prediction equality. Any difference fails the null control and invalidates
   that variant; it is not silently treated as a small effect.
4. **Independent adapted target:** use the already locally bound compatible
   `public_lora_2601` asset when root confirms its exclusion and target-side
   provenance. Its generation config is 18,824 bytes (SHA-256
   `6bed8aca6668bd749bbad03c0c85bbace306085dac3141895dad7685f03d0682`),
   and its update is 427,632 bytes (SHA-256
   `eea7bb49f801b61df2e26a8f59af7c3096f6f3a2604404e16e589443bcfba595`).
   It requires no download or paid compute. If its target provenance or source
   reservation is not independently valid, this arm is reported unavailable;
   no artificial variant substitutes for it.

Artificial deltas are explanatory stress tests and are reported separately
from the adapted target. They are generated on the target side from a fixed
recipe and never optimized against evaluation outcomes. The after-cut delta
must still be recorded with its nonzero target-side parameter distance; its
zero H/prediction shift is the causal null check.

## Metrics and error inventory

For each non-null variant and each frozen decoder, retain only the paired
truth-free inputs needed for summary:

- raw activation displacement `||H_variant - H_clean||_2` and relative raw H
  displacement;
- projected-feature displacement `||u_variant-u_clean||_2` and cosine distance;
- clean predicted-class logit margin to the clean runner-up; and
- the clean predicted-class hyperplane distance
  `(z_top-z_runner)/(exp(s)*||E_top-E_runner||_2)`.

The main explanatory metric is feature displacement divided by that clean
hyperplane distance. It uses the actual normalized tied-E geometry of the
fixed decoder. Raw H L2 and target-side parameter relative L2 are reported
separately; neither is substituted for the margin-consistent measure. The
module accepts only the clean top/runner readout rows, so it does not need a
second full-vocabulary logit matrix.

No threshold is fitted in v1. The fixed ratio reference of 1.0 is a geometric
interpretation only, not a deployment rule. If a later predictive threshold
is proposed, whole update families must be split into development and held-out
families before target outcomes are opened; the module validates disjoint
family IDs but does not fit thresholds. Any true-token margin or paired error
inventory is evaluator-only after the public gate and is labelled
non-deployable. The post-truth inventory has four required cells: both
correct, only expanded-fixed correct, only A1+A2 correct, and both wrong.
Candidate ranks, proposal failures, and direct-decoder ranking failures stay
in separate fields.

## Target and public assets

The compatible local target asset is the already bound Llama 3.2 1B public
snapshot revision `9213176726f574b556790deb65791e0c5aa438b6`, whose model file
is 2,471,645,608 bytes (SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`). The
TRR-0010 public-prefix callable is reused; it runs the embedding and layers
`[0, cut_depth)` and returns cut activations without retaining target truth.
The frozen decoder equations were checked in
`src/token_reconstruction/trr0005_joint_decoder.py` and
`src/token_reconstruction/trr0007_positionwise.py`: normalize the affine or
positionwise residual feature, then multiply by the normalized public E with
positive `exp(s)` scale.

The adapted target binding is practical locally, but availability does not
establish independence. The reservation and exclusion receipt must prove it
is not reused as fitting, calibration, checkpoint-selection, or prior opened
material before any capture. P03 sealed holdout remains untouched.

## Execution boundary and resource bound

A future run will be a truth-free producer/capture stage, followed by the
inference package and inherited public gate. Only after all variant H files,
decoder predictions, resource receipts, and code/state/input hashes pass the
gate may root authorize truth preparation and paired error scoring. This
module has no truth loader and cannot open the sidecar.

The largest planned cell is 64 records × 192 capture tokens × 2048 hidden
values in BF16: 50,331,648 bytes (about 48 MiB) per variant. One variant is
processed at a time. The known public model file is 2.47 GB; the prior
A1+A2 qualification measured approximately 3.303 GiB isolated CUDA peak, and
TRR-0010's public E is about 1.051 GB. With one decoder, one E/readout, one
variant H tensor, and temporary output buffers, use a conservative 6 GiB
working estimate, an 8 GiB reserved-GPU ceiling, and at least 11 GiB free GPU
before launch. Use a 12 GiB host-RSS cap, at least 8 GiB host headroom, a 20
GiB disk floor plus the bounded output allowance, and a 900-second outer
watchdog per variant/cell. These are preflight limits, not a reason to shrink
the panel; root must qualify the largest cell before any matrix run.

The target-prefix capture process is evaluator-side separate from the frozen
decoder process. It must preserve the inherited truth-free capture contract,
record CPU/GPU clocks and peak memory, and fail closed on allocator,
thermal, disk, or output-equivalence anomalies. No GPU or fresh capture is
authorized by this design receipt.

## Source bindings and synthetic readiness

The task-owned metric implementation binds these reviewed source interfaces
by hash at design time:

- `scripts/trr0010_eval_runner.py` — `77c294c57ef60c9ced5284422047bf4283072632aff976e01e3f12b9181b3920`;
- `scripts/trr0010_eval_gate.py` — `ac2ce64375690f6f2a743e79f07f5d2941ee4865dcd8d500a8a209540794157d`;
- `scripts/trr0010_eval_capture.py` — `d6d6fddc6a8c01dacdf205c528056cf171c33d8ffbb89eb9f41750895ac2ee78`;
- `src/token_reconstruction/public_prefix.py` — `b9dca1d8d56c7c07413015aea4bde79e8ceac827c32ee9bc0d1a236ea1dd35f6`;
- `src/token_reconstruction/trr0005_joint_decoder.py` — `9cab3cf5ad664c37d91bcd90fe1852df71da25800ddf37291a57000908d3d38f`;
- `src/token_reconstruction/trr0007_positionwise.py` — `89fcd036a57407c8c49294aa5fd15ccef46f02b6fabcc35435d097f1213de485`.

The synthetic suite covers readout-consistent margin geometry, strict
after-cut equality, family-level held-out validation, panel reservation
requirements, truth-free flags, and evaluator-only paired error accounting.
The suite currently passes 8 tests on CPU. This receipt does not claim a
public run, source selection, target capture, truth access, or independent
text-generalization result.
