# TRR-P01 — no-fit boundary-prototype pilot

## Result

The complete predeclared eight-key matrix was reconstructed on both opaque
public arms, jointly hash-validated before truth access, and scored afterward.
The run is an exploratory 16-record diagnostic: 39 post-BOS positions per
record, 624 scored tokens per arm, and four style strata with four records each.
All score files report complete 16-record coverage. The historical fixed
A1+A2 geometry port is the strongest comparator at 613/624 tokens (98.2372%)
and 15/16 exact records on both arms. The no-fit boundary and raw-embedding
methods are materially lower; reference correction lowers them further.

## Provenance and scope

- Worktree: /tmp/trr-p01; branch: task/TRR-P01
- Exact starting commit: 6b618760f50055dc5c8a62e830ab7a9761190cfe
- Parent task/branch: TRR-0002 / task/TRR-0002
- Final reconstruction source commit: e43a595d0f4300d5db8f93c86881b455dfa30ea4
- Control packet SHA-256: 27d6a27f935a1b84e2faf8a12e8d58cd87331e5d0e8c131502025fcaf6d697d2
- Pilot plan SHA-256: 2e68b8ce7514c9f8338d47f8c3cc56f957259453b057a4c5a3cef1273268bfce
- Setup note SHA-256: 3a7220a31ef72c48624ddd3314520063ee60ac4a011f08fac59e0a730bd7e977
- Independent final-results audit: experiments/TRR-P01/review/final-results-audit.md; SHA-256 cb20ce0fdf63e35c60d91e74f989d6017c7b44eb14bb89b4e36b82cc0c783a74.
- Model: meta-llama/Llama-3.2-1B-Instruct at revision 9213176726f574b556790deb65791e0c5aa438b6; cut 4, BOS 128000, hidden 2048, vocabulary 128256.
- Target resource: Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct at revision 7fa9d06a59246629244cdd3b6b92e4fc756baa0f; it is a public full-SFT checkpoint retained under the legacy condition ID shifted_target_lora.
- Table: model-native BF16, 128256 × 2048, raw payload 525336576 bytes; artifact SHA-256 51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3.
- Environment: Ubuntu 24.04.4 WSL2, Ryzen 9 9950X3D (32 logical CPUs), 30 GiB RAM, Python 3.12.3, torch 2.10.0+cu128, transformers 5.3.0, safetensors 0.7.0.
- Hardware: both final runs used CPU with CUDA hidden. RTX 5080 device 0 remained reserved for TRR-0003; no TRR-P01 CUDA allocation or job interruption occurred.

## Predeclared methods and access gate

The eight prediction keys were boundary cosine/L2, raw-embedding cosine/L2,
reference-corrected cosine/L2 using public token 220, frozen historical Alpaca
A1, and the labelled fixed_k256_direct A1+A2 geometry port. The exact native
A1+A2 contract remains excluded because its 128x128 geometry is incompatible
with this 40-slot panel; BOS is retained in every final 40-slot run. The port
preserves the published rule and is reported separately from the native contract.
All eight keys for each arm were serialized and hash-validated before opening truth.
No condition label, source plaintext, source token IDs, target weights, or
correctness feedback reached reconstruction.

Joint validation: JOINT_HASH_VALIDATION_PASS_BEFORE_TRUTH_OPEN at
2026-09-05T10:07:44.076606Z; 28 file records,
both arms, all eight methods. Scores were then created at
2026-09-05T10:09:32Z (arm-000) and 2026-09-05T10:09:48Z (arm-001).

The formal public post-BOS construction identity diagnostic froze its prediction
artifact before label parsing: cosine and L2 each matched all 256/256 ordered
probe IDs, with zero ties and zero collisions, and opened no truth. The result is
at experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_identity.json
(SHA-256 64091cd037d06f39287c1a8ecb63a161fd461e8b2b481a57d231426a5813faa8) and
the freeze receipt is at experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_freeze.json
(SHA-256 5ed08c85e08dd606dd6070f994a0a6be0edcb9a014064a8a620b8e7b0e7ac096).

## Final paired eight-row quality matrix

| Prediction key | arm-000 correct / 624 | arm-000 exact records | arm-001 correct / 624 | arm-001 exact records | shifted minus matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| boundary.cosine | 255/624 (40.865%) | 0/16 | 243/624 (38.942%) | 0/16 | -12 (-1.923 pp) |
| boundary.l2 | 243/624 (38.942%) | 0/16 | 236/624 (37.821%) | 0/16 | -7 (-1.122 pp) |
| raw_embedding.cosine | 231/624 (37.019%) | 0/16 | 231/624 (37.019%) | 0/16 | +0 (+0.000 pp) |
| raw_embedding.l2 | 175/624 (28.045%) | 0/16 | 180/624 (28.846%) | 0/16 | +5 (+0.801 pp) |
| reference_corrected.cosine | 81/624 (12.981%) | 0/16 | 80/624 (12.821%) | 0/16 | -1 (-0.160 pp) |
| reference_corrected.l2 | 74/624 (11.859%) | 0/16 | 80/624 (12.821%) | 0/16 | +6 (+0.962 pp) |
| historical_a1.cosine | 513/624 (82.212%) | 0/16 | 510/624 (81.731%) | 0/16 | -3 (-0.481 pp) |
| historical_a1_a2_port.cosine | 613/624 (98.237%) | 15/16 | 613/624 (98.237%) | 15/16 | +0 (+0.000 pp) |

Every row has 16/16 joined record IDs and 624/624 scored positions. Exact
record rate is zero for every row except historical_a1_a2_port.cosine, which
has 15/16 on both arms. Full per-record and per-position arrays remain in the
two score JSON files and are represented structurally in manifest quality_matrix.

## First post-BOS, position, and style effects

The first position below is the first predicted token after BOS. Position
summaries are arm-000 / arm-001 token accuracy; first-error summaries are over
the 16 records and use post-BOS positions.

| Prediction key | first post-BOS | position 2 | position 39 | first-error position |
| --- | ---: | ---: | ---: | --- |
| boundary.cosine | 1.000/1.000 | 0.375/0.250 | 0.438/0.438 | median 2, range 2-6 (16/16 records) / median 2, range 2-6 (16/16 records) |
| boundary.l2 | 1.000/1.000 | 0.250/0.188 | 0.500/0.438 | median 2, range 2-6 (16/16 records) / median 2, range 2-6 (16/16 records) |
| raw_embedding.cosine | 0.312/0.375 | 0.375/0.312 | 0.562/0.562 | median 1, range 1-2 (16/16 records) / median 1, range 1-2 (16/16 records) |
| raw_embedding.l2 | 0.312/0.312 | 0.188/0.188 | 0.438/0.438 | median 1, range 1-2 (16/16 records) / median 1, range 1-2 (16/16 records) |
| reference_corrected.cosine | 1.000/1.000 | 0.188/0.250 | 0.125/0.125 | median 2, range 2-4 (16/16 records) / median 2, range 2-4 (16/16 records) |
| reference_corrected.l2 | 1.000/1.000 | 0.188/0.250 | 0.125/0.125 | median 2, range 2-4 (16/16 records) / median 2, range 2-4 (16/16 records) |
| historical_a1.cosine | 0.938/1.000 | 0.625/0.625 | 0.812/0.812 | median 4.5, range 1-24 (16/16 records) / median 4.5, range 2-33 (16/16 records) |
| historical_a1_a2_port.cosine | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | median 5, range 5-5 (1/16 records) / median 5, range 5-5 (1/16 records) |

| Prediction key | prose (arm-000 / arm-001) | code | numeric + punctuation | unicode + instruction |
| --- | ---: | ---: | ---: | ---: |
| boundary.cosine | 0.423/0.417 | 0.417/0.404 | 0.385/0.359 | 0.410/0.378 |
| boundary.l2 | 0.417/0.410 | 0.372/0.359 | 0.365/0.346 | 0.404/0.397 |
| raw_embedding.cosine | 0.429/0.455 | 0.327/0.295 | 0.346/0.359 | 0.378/0.372 |
| raw_embedding.l2 | 0.321/0.346 | 0.218/0.231 | 0.276/0.295 | 0.308/0.282 |
| reference_corrected.cosine | 0.103/0.109 | 0.115/0.115 | 0.128/0.141 | 0.173/0.147 |
| reference_corrected.l2 | 0.103/0.103 | 0.103/0.122 | 0.128/0.141 | 0.141/0.147 |
| historical_a1.cosine | 0.878/0.885 | 0.776/0.769 | 0.891/0.891 | 0.744/0.724 |
| historical_a1_a2_port.cosine | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.929/0.929 |

The static boundary rows are near 39–41% token accuracy, raw embedding rows
range from 28.0–37.0%, and reference correction ranges from 11.9–13.0%.
Historical A1 is 82.21% matched and 81.73% shifted; the fixed port is 98.24%
on both. The position table describes context/position effects: later positions also
receive the changed reconstructed context, so these are not isolated positional
estimates. The independent audit confirms that reference correction is correct
for 16/16 records at position 1 but only 3/16 matched and 4/16 shifted at
position 2, with first errors at position 2 for 13/16 and 12/16 records.
These are descriptive pilot effects, not population estimates.

## Cost and resource matrix

| Phase or method | arm-000 seconds | arm-001 seconds | accounting |
| --- | ---: | ---: | --- |
| load prototype table | 0.166056 | 0.188678 | shared BF16 table artifact |
| load model and public prefix | 3.027164 | 1.781330 | CPU model load/prefix setup |
| static full-vocabulary phase | 6.940531 | 7.116231 | four zero-candidate-simulation methods |
| reference-220 correction phase | 528.912351 | 595.388496 | 1,248 reference evaluations per arm |
| historical fixed-K256 port phase | 32.571828 | 34.604473 | A1 proposal timing is reused inside the port |
| output I/O and artifact hashes | 0.017013 | 0.017185 | finish-receipt accounting |
| grouped process peak RSS | 5239828 KiB | 5244048 KiB | individual method peaks not measured |

| Method timing | arm-000 seconds | arm-001 seconds |
| --- | ---: | ---: |
| boundary.cosine | 1.707795 | 1.770361 |
| boundary.l2 | 1.783691 | 1.839210 |
| raw_embedding.cosine | 1.725751 | 1.715113 |
| raw_embedding.l2 | 1.685949 | 1.750492 |
| reference_corrected.cosine | 243.296648 | 287.821600 |
| reference_corrected.l2 | 285.599230 | 307.550588 |
| historical_a1.cosine | 0.911704 | 1.028638 |
| historical_a1_a2_port.cosine | 31.851544 | 33.836110 |

Each run made 2686 public-prefix calls and 162912 public-prefix input-token evaluations. Historical control used 159744 candidate simulations, 159744 candidate cache commits, and 640 persistent cache commits per arm. Reference correction used 1248 evaluations, 1248 probe commits, and 1280 persistent commits per arm.

The historical A1 timing is the proposal component reused inside the fixed
A1+A2 port; it is not added as an independent run. The production table build
used batch 256: 501 forward calls and 256512 template input-token evaluations,
with 44.429441744 seconds in the table-construction phase and 47.087785290
seconds total for the builder, including its other measured work. Its separate
256-probe qualification compared one batch-256 forward with two batch-128
forwards; the ordered outputs were exactly equal (maximum absolute difference
0.0), and batch 256 was selected for production.

Separately, the largest representative eight-record K256 method cell qualified
cached-39 versus full-40-token public-prefix execution at position 39 with exact
equality (maximum absolute difference 0.0). It used 2,048 candidate simulations,
0.464625 seconds for candidate simulation, 0.017232 seconds for the reference
probe, 0.903068 seconds for full-vocabulary lookup, and 5.413900 seconds total,
with 5,475,844 KiB peak RSS. This is resource qualification, not panel accuracy.
Table-build and qualification timings were collected on a shared CPU host and
are not uncontended comparative timing; the final matched and shifted matrix ran
in the coordinated CPU window.

## Source exclusion and post-score condition handoff

The panel was selected from Pile-10k revision 127bfedcd5047750df5ccf3a12979a47bfa0bafa
with permutation seed 314159, first four eligible rows in each style, and
BOS prepended to 39 source tokens. The redacted handoff at
experiments/TRR-P01/runtime/post_score_development_exclusion.json (SHA-256 3d3eec289d9cfbce8eb228038d13540b0e38faad61c9972cd4374cbb34f0a714) binds the condition join,
dataset revision, source row identities, styles, text hashes, public artifact
hashes, prediction hashes, and score hashes. It contains no source plaintext
or private truth tensor.

| Opaque record | dataset row ID | style |
| --- | ---: | --- |
| p01-r0001 | 8429 | prose |
| p01-r0002 | 3467 | prose |
| p01-r0003 | 3819 | prose |
| p01-r0004 | 5440 | prose |
| p01-r0005 | 3921 | code |
| p01-r0006 | 2968 | code |
| p01-r0007 | 2751 | code |
| p01-r0008 | 3911 | code |
| p01-r0009 | 2530 | numeric_plus_punctuation |
| p01-r0010 | 3608 | numeric_plus_punctuation |
| p01-r0011 | 3185 | numeric_plus_punctuation |
| p01-r0012 | 1305 | numeric_plus_punctuation |
| p01-r0013 | 7502 | unicode_plus_instruction |
| p01-r0014 | 8005 | unicode_plus_instruction |
| p01-r0015 | 6137 | unicode_plus_instruction |
| p01-r0016 | 5796 | unicode_plus_instruction |

The post-score condition join is arm-000 = matched_public and arm-001 =
shifted_target_lora. The raw evaluator condition map is retained locally and
bound by SHA-256 9dead079b61da037b049e2c3f1d22a52fd73ddcb7bf9ad78fba4bb6bffe2a058; the private truth tensor, target weights, and evaluator-private raw directory
are not part of the publication payload.

## Excluded and provisional attempts

The bounded 210-second arm-000 attempt under 6e05fee timed out at 08:46:50Z
before producing predictions, exit 124; collection at 08:48:28.985094Z is not
the termination time. The pre-logging runner emitted no phase telemetry, so the
exact stage is unknown. It opened no truth. Its raw command, progress,
reservation, and loader log are retained unchanged for provenance, but the
preserved progress line is not treated as phase evidence. The earlier evaluator
attempt ran before its source was committed and remains provisional. Neither
attempt contributes to the scored matrix. No GPU run was launched and no other
job was stopped.

## Next-round decision

Do not advance either current static or reference-correction variant to another
unchanged scientific round: their contextual exact-token rates are 11.9–40.9%
and no record is fully correct. The reference offset collapses at position 2
despite position 1 being correct for every record, so a new round would first
need to test whether deterministic context-dependent deformation can be separated
from positional effects on controlled public inputs. Retain boundary, raw
embedding, and reference correction as baselines/negative diagnostics. Retain
the historical fixed-K256 A1+A2 geometry port as a labelled comparator, not as
a no-fit result. This 16-record pilot does not justify a broad GPU allocation or
generalization claim.

## Publication status

Reviewed evidence commit `025fc39a1888c7cd2cc8ea9c4ac335633b79f8be` is published on
`task/TRR-P01`. [Pull request #5](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/5)
is open against the actual parent `task/TRR-0002` at `6b618760f50055dc5c8a62e830ab7a9761190cfe`.
The task-local publication metadata is recorded in a subsequent commit on the
same branch; the PR head identifies the final handoff.

Final publication checks passed: 53 JSON files parsed, 74 compact manifest/state
path-hash bindings verified, verbatim packet and parent ancestry confirmed, and
`git diff --cached --check` passed. The independent curation review found no
disclosure or reproducibility blocker. The complete task-owned publication list
contains 136 paths, including the three already committed focused test files.
Global STATE.json, shared protocols, and the active-method registry were not
modified. Nothing was merged.
