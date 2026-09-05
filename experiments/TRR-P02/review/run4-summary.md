# TRR-P02 completed public diagnostic

Run4 is the retained execution of the frozen TRR-P02 public teacher-prefix
geometry diagnostic. It used source commit
`470b6f1becfaa6da110048302938feddd7204c30`, plan
`experiments/TRR-P02/plan.json` (SHA256
`9f191d0376d29a9bd46241060f5466738d55b375beb1baf78cb920b72594e030`),
model `meta-llama/Llama-3.2-1B-Instruct` revision
`9213176726f574b556790deb65791e0c5aa438b6`, seed `314159`, CPU with eight
Torch intra-op threads and one inter-op thread, CUDA hidden, and prototype
chunk size `8192`.

The exact process ran from `2026-09-05T12:23:25Z` through
`2026-09-05T12:23:37Z`, exited `0`, and did not time out. Outer `/usr/bin/time
-v` reported Max RSS `6,324,564 kB` and zero swaps. Internal peak RSS was
`6,476,353,536` bytes; all five 8-GiB RSS checks passed, and the 10-GiB
available-memory preflight passed. The full command and runtime details are
in `runtime/cpu-public-geometry-20260905-run4/run_manifest.json` and the
copied `outer-*` files.

The public panel has 46 unique known activation rows: C0-C4 at eight
candidate endpoints, plus repeated-13 C5/C6 rows with C0/C1 reuse. The
candidate IDs are `13, 32, 198, 220, 2048, 4096, 16384, 29871`; reference ID
is `220`. The twelve predeclared full-vocabulary rows are C1-C4 crossed with
`220, 2048, 4096`. Public model cost was 106 prefix calls / 336 input-token
evaluations, 30 full calls / 251 input-token evaluations, and 76 cached calls /
85 input-token evaluations. Target-model calls and candidate simulations were
both zero.

For the twelve targeted full-v rows, the same dictionary and strict rank rule
were used for each arm (`rank = 1 + count(score > true_score)`). Raw boundary
scores gave top-1 on 7/12 rows (58.33%), mean rank 5.0, and mean true-other
margin `0.0195361`. One streamed frozen-lens projection of the reused P01
prototypes gave top-1 on 11/12 (91.67%), mean rank `1.08333`, and mean
true-other margin `0.119229`. The historical A1 frozen lens gave top-1 on
9/12 (75%), mean rank `1.41667`, and mean cosine-equivalent true-other margin
`0.101620`; its native `FrozenAffineLens.forward` score scale was
`exp(s)=35.0647507`, so native margins are retained separately and are not
compared directly with cosine margins. The explicit public-panel oracle mean
centering control gave 0/12 top-1 and mean rank `1261.83`; it is labelled an
oracle diagnostic and is not a deployable correction.

On the fixed local dictionary of eight nearest OTHER P01 BOS prototypes plus
the known true ID (nine candidates), raw boundary gave 35/40 top-1 (87.5%),
reference-subtraction control 36/40 (90%), and opposite-sign control 30/40
(75%). Public-panel oracle centering gave 40/40 on this local dictionary and
remains an explicitly non-deployable diagnostic.

The shared-offset check defines `delta(C,v)=z(C,v)-b_v`. Mean offset norms for
C1-C4 were `6.83305, 6.89992, 6.89984, 6.87319`; mean residual norms after
subtracting each context mean were `12.1375, 12.1530, 12.1539, 12.2845`.
Across the 28 token pairs per context, relative pair deformation means for
C1-C4 were `0.9044, 0.8260, 0.9141, 0.9990`, with pair-cosine-to-baseline
means `0.4755, 0.5165, 0.4313, 0.3277`. For equal-position C1-C4 geometry,
raw same-token cross-context cosine distance averaged `0.482042` versus
`0.707028` for different tokens within a context (ratio `0.681786`); the
corresponding L2 means were `3.360802` versus `4.170195` (ratio `0.805910`).
The projected-prototype values were cosine `0.465287` versus `0.855795`
(ratio `0.543690`) and L2 `4.332999` versus `5.958140` (ratio `0.727240`).
The separate repeated-13 context/position control (C0,C1,C5,C6) had cosine
means `0.238806` versus `0.759670` and L2 means `2.15713` versus `4.48995`.

Wiring checks passed: recomputed public C0 rows were exactly equal to reused
P01 rows after dtype matching (both bfloat16; max and mean absolute difference
zero); subtraction sign self-checks and persistent-cache checks were recorded;
C6 (the longest declared context) passed exact ordered batched-vs-single
qualification for all eight endpoints. C0 is position 1, C1-C4 are position 2
with equal visible length, and C5/C6 are explicit positions 3/4 controls.

Retained artifacts are under
`experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4`: activation
panel SHA256 `e63026f56063083fe009fe3211548875310dd3295e7c205f0e3759f1ae5a15ca`,
diagnostics SHA256
`7352573df457804b2702a419571a9feb100ae5863d32238eab6f38f19a9586c4`,
manifest SHA256
`2ad5a6049c988940acbe0e1ef4b62320ad094c1b2e1673ca6f8e5edcc7f7f710`,
offset figure SHA256
`6796c56602983dac2340e6ae8992e7adc2246ff621908bbb0c3b46d07129e721`, and
lens figure SHA256
`fc03d6f48d19c694ac7bc5bbac716b08ad85e9bc95881a61a2265cb217698cc7`.
The P01 prototype table and historical lens assets passed their declared
identity hashes in preflight and diagnostics.

