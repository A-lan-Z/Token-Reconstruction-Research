# TRR-0007 scientific report skeleton (pre-score)

**Status:** pre-score handoff. This file is a factual scaffold for the final
report; it does not replace `coordination/results/TRR-0007.md` and contains no
fresh-panel decoder score, truth result, or primary decision.

## Question and scope

TRR-0007 asks whether, at the fixed TRR-0005 fitting opportunity budget,
broader public support or a modest current-position capacity extension improves
reconstruction on a fresh natural Pile+Finance panel. The result is exploratory
and is not a canonical replacement or a dual-benchmark-complete claim.

The frozen design uses 128 records per domain, public base and synthetic-LoRA
targets on identical records, all 127 post-BOS positions for token outcomes,
and a deterministic first-32-per-domain public-base subset for the bounded A2
anchor. The four direct factorial edges are reported separately by domain and
target; domains and paired target conditions are not pooled as independent
sources.

## Frozen methods and training evidence

The retained reference is the TRR-0005 selected diagonal state. Four TRR-0007
students cross two banks and two decoder capacities:

| bank | method | selected step | fit token accuracy | fit errors / 124,371 | exact fit records | held-out token accuracy | held-out style-balanced accuracy | held-out exact records | initially-wrong challenge accuracy | fit wall seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current enriched | trained diagonal | 1600 | 0.9998633122 | 17 | 1,185/1,200 | 0.9674433450 | 0.9547109695 | 18/48 | 0.9995117188 | 117.2371 |
| current enriched | residual MLP-512 | 2100 | 1.0000000000 | 0 | 1,200/1,200 | 0.9693584424 | 0.9569962828 | 19/48 | 1.0000000000 | 123.1396 |
| improved public bank | trained diagonal | 2400 | 1.0000000000 | 0 | 1,200/1,200 | 0.9696776253 | 0.9578370758 | 17/48 | 1.0000000000 | 118.0701 |
| improved public bank | residual MLP-512 | 2100 | 1.0000000000 | 0 | 1,200/1,200 | 0.9687200766 | 0.9568477216 | 20/48 | 1.0000000000 | 123.5021 |

All four arms use the fixed 3,000-step, seed-4005 recipe and 1,536,000
post-BOS training draws per arm. The fit geometry is `[1200,192,2048]`, the
validation geometry is `[48,192,2048]`, and the selected states are immutable
under `experiments/TRR-0007/method_freeze.json`. Fit wall times are full-arm
training receipts; shared model-load and embedding preparation are not counted
once per method.

The static learning-curve figure and source data are support-owned at
`experiments/TRR-0007/figures/fit_learning_curves.png` and
`experiments/TRR-0007/figures/fit_learning_curves_plot_data.json`. The figure
marks the four selected checkpoints and separates held-out development from the
initially-wrong challenge; the fit handoff JSON remains authoritative.

## Public-bank and preparation evidence

The final broader bank contains 1,200 rows and 124,371 post-BOS positions:
1,080 retained natural rows plus 120 controlled rows. Its 3,600 controlled
replacement IDs comprise 2,000 preserved current-bank IDs and 1,600
current-enriched-unseen additions, used once each. The captured improved-bank
activations are `[1200,192,2048]`; the support verification compared all
activation values on 1,080 natural rows with `torch.equal`, found zero
mismatches, and measured maximum absolute delta 0.

The descriptive public coverage comparison reports:

- current enriched: 15,602 distinct post-BOS IDs, 14,693 in positions 1--127,
  and 909 late-only IDs;
- improved public bank: 17,126 distinct post-BOS IDs, 16,237 in positions
  1--127, and 889 late-only IDs;
- overlap: 14,381 post-BOS IDs, with 1,221 current-only and 2,745
  improved-only IDs.

The accepted CPU preparation subtotal is 23.6664545 s for the source-bound
candidate scan and final v5 construction. Repeated superseded preparation is
57.9091953 s and prior public diagnostic preparation is 19.8768633 s, for a
recorded campaign total of 101.4525131 s. Candidate scanning and bank
construction were model-free; public-development projections used the frozen
decoder on CPU. The improved public activation capture took 8.2246617 s, with
2,472,678,400 bytes peak CUDA allocation, 2,480,931,816 bytes peak reserved, and
6,066,634,752 bytes maximum RSS.

The separate A2 adapter qualifier passed on the published fixture with exact
A1/A2 agreement, 7.70 s elapsed, 3,540,137,472 bytes peak allocation, and
3,690,987,520 bytes peak reservation. It is a numerical-port qualification,
not a fresh truth evaluation.

## Fresh panel state before scoring

The identity-only selection is frozen at
`experiments/TRR-0007/selection/source_selection.json` (SHA-256
`0adf45078ee017ab1877e5d8b905261583d9916cf59c9237591166d0b39c431c`) with
128 Pile and 128 Finance records. The inventory found 877 eligible Pile and
5,323 eligible Finance records after all declared fitting, opened-public,
duplicate, P04, TRR-0006, final-v5, and public-fitting-prefix exclusions.
The exclusion receipt SHA-256 is
`7547ac0b85052955d355b58ab83bdf5ba24f9d621f514da480a336b01858760e`.

The opaque hash-only P06 reservation is
`experiments/TRR-0007/coordination/p06_opaque_source_sequence_reservation.json`
(SHA-256
`09e845fec244a38873c5bf127f6d984af91af503fb42d3a8411451ce41cdedf4`), with
256 public-record and 256 final-sequence hashes. It contains no source text,
record IDs, source indices, target labels, token IDs, weights, or truth.

The four public observation artifacts are complete and truth-free. The
observation manifest SHA-256 is
`f32ea855f4693454340b5556cc4b304e20272e680720477c0d6ef13a6ca7e483`; the
capture receipt SHA-256 is
`0b0d544d2fa55ce32bb537f28a8db7ed90866e4373ed4245917fa56567a5267f`.
No fresh decoder prediction or truth-sidecar score belongs in this skeleton;
those fields are filled only after registration, the complete 22-artifact
public run, the public freeze gate, and the single permitted truth read.

## Preregistered scoring and decisions

The primary family is four direct factorial edges, each evaluated in four
(domain x target) cells, for token and exact outcomes in both directions: 64
directional bounds total. The edges are:

1. support at trained-diagonal capacity: improved minus current bank;
2. support at residual capacity: improved minus current bank;
3. capacity on current enriched: residual minus trained diagonal;
4. capacity on improved public bank: residual minus trained diagonal.

The scorer uses 50,000 shared record-bootstrap draws (seed 5007), token one-sided
tail alpha `0.05/64`, and exact paired Clopper-Pearson component alpha
`0.05/128`. Interaction details, student-vs-reference contrasts, and A1 versus
A1+A2 gaps are descriptive 95% intervals outside the primary family.

An exploratory-useful point estimate is at least 1 percentage point in token
accuracy (about 1.27 fewer errors per 127-token clip) or 5 percentage points in
exact-record rate, with no paired point harm above the corresponding negative
margin in any reported domain/target cell. A margin is exceeded only when the
one-sided lower bound reaches the positive margin. Material harm requires the
one-sided upper bound to be below the negative margin; harm is excluded only
when the lower bound reaches the negative margin. Point estimates alone are
useful exploratory evidence, not confidence that a margin is exceeded.

Cost qualification uses same-cell measured intervals, full-arm training wall
time and 1,536,000 draws for training ratios, and method-specific preparation
for preparation ratios. The reused reference preparation is reported without a
ratio. The preregistered limits are a maximum 2.0 student fit/preparation ratio
and 1.25 student warm-runtime ratio; budget misses remain visible.

## Fields to fill after the public run

- registration SHA, prediction run manifest, timing receipt, and code commit;
- all 20 decoder prediction entries plus both A1/A2 anchor entries;
- gate receipt and truth-binding metadata;
- fresh per-cell token errors/accuracy over 127 positions and exact records;
- all four primary edges with point estimates, paired intervals, corrected tails,
  and decision labels;
- descriptive reference, interaction, frequency-bin, and 32-record A2-anchor
  tables;
- actual measured prediction/preparation costs and resource peaks;
- final `experiments/TRR-0007/manifest.json` and
  `coordination/results/TRR-0007.md`.

## Limitations to retain

The panel is exploratory and has 128 records per domain. The A2 anchor is
bounded to 32 public-base records per domain and cannot establish architecture
confirmation. The public LoRA target is paired to the same selected source
records, so target conditions are not independent samples. The support bank's
coverage comparison is descriptive and does not by itself imply a quality
improvement. All final claims must use the post-score receipt and scorer output,
not these pre-score fit or coverage values alone.

## Evidence index

- Frozen design: `experiments/TRR-0007/evaluation_plan.json`.
- Frozen method ledger: `experiments/TRR-0007/method_freeze.json`.
- Four-fit handoff: `experiments/TRR-0007/improved_fit_v1/improved_fit_handoff_v2.json`.
- Final bank and preparation accounting: `experiments/TRR-0007/support/support_handoff_v2.md`.
- Coverage: `experiments/TRR-0007/support/coverage_comparison.json`.
- Curves: `experiments/TRR-0007/figures/fit_learning_curves.png` and its plot data.
- Fresh selection and capture receipts: `experiments/TRR-0007/selection/` and
  `experiments/TRR-0007/evaluation/public_observations/`.
