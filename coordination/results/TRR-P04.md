# TRR-P04 result and decision record

**Status:** exploratory comparison incomplete; no student-method winner.

The completed public stages establish that the proposed student graph can be
optimized on a bounded public error set, that the largest planned cell fits the
available GPU guard, and that the privileged public teacher produces an
informative bounded signal. All eight public training fits are now serialized
under the common schedule design. They do not establish that a GRU,
hard-confusion loss, or teacher score information improves reconstruction on
fresh records. Fresh evaluator capture, native anchor outputs, and truth-gated
fresh evaluation remain pending.

## Decision summary

The r5 capacity receipt passes the corrected, two-part gate. On the full public
256-record correction pool, the frozen affine initialization has 4,185 wrong
positions out of 45,596 (token accuracy 0.908216), satisfying the teacher-
selection feasibility threshold. The fixed eight-record probe contains 1,000
wrong positions out of 1,503 (accuracy 0.334664); after eight Adam updates its
accuracy is 0.806387 (1,212 correct positions). The probe has zero completely
reconstructed records both before and after fitting.

That large probe improvement is an optimization and capacity check on records
selected for high public affine error. It is not evidence of generalization,
teacher usefulness, or a comparison with S/H/D. The earlier r3 run has the
same learning result but omitted resource, source-commit, and UTC timing
evidence; it is retained as prior evidence, while r5 is the auditable receipt.

The disposition is **teacher qualification PASS for bounded public training
evidence; public eight-fit training comparison complete; fresh paired comparison
pending**. The teacher receipt supports using its frozen relative scores in D's
training objective. It does not support a native BOS-only accuracy claim. Both
training seeds and all four arms are now serialized, but no fresh predictions or
evaluation metrics exist yet.

## Public data and candidate preparation

The public capture uses the PR7 public-fit replay (1,200 records), a disjoint
256-record correction pool, and a disjoint 192-record validation pool. The
capture is at cut depth 4 for `meta-llama/Llama-3.2-1B-Instruct`, revision
`9213176726f574b556790deb65791e0c5aa438b6`, with hidden size 2,048 and a
fixed 192-token execution geometry. Public labels are present only because
these are public training/development pools. Capture evidence records no
evaluator truth, target-update weights, or fresh evaluation records.

Candidate preparation covers 1,456 rows (replay plus correction), 192
positions, proposal K=512, and a frozen candidate prefix K=32. The numerical
candidate tensors were generated under `1cc095d88b7006134c8778bbf205f3f0e39b8664`.
The r2 metadata-completion artifact reuses those tensors and binds the P04
training proposer to `p04_public_affine` / `pr7_public_affine_state` with the
declared tie rule **descending score, then ascending token ID**. Its artifact
SHA-256 is
`ecf031a606a8bce22116b65d2dcbf5bc25596c80961767a40924081f8f03b1e5`.

This training proposer tie rule is separate from the native A1+A2 anchor,
which retains the PR7 published proposal order and first-argmax behavior.

## Qualification history

The preserved failure records are part of the evidence:

| Attempt | Outcome | Scientific meaning |
| --- | --- | --- |
| `capacity-qualifier-r1` | Fail-closed: cuDNN RNN backward was called in evaluation mode | Source/runtime mode defect; no truth access and no result. |
| `capacity-qualifier-r2` | Fail-closed: 210 probe errors did not meet the then-coded 256-error probe gate | Superseded gate defect; optimizer did not start. The ≥256 requirement belongs to the full correction pool, not eight rows. |
| `capacity-qualifier-r3` | Learning pass on the eight-row high-error probe | Useful capacity evidence, but missing the later resource/source/time receipt fields. |
| `capacity-qualifier-r5` | Pass | Corrected full-pool feasibility plus probe-improvement and resource evidence. |
| `teacher-preflight-r1` | Fail-closed before CUDA/model initialization: the preflight imported `_centered_cosine_scores` from the wrong module | Helper-name defect; no truth access. |
| `teacher-preflight-r2` | Pass on K=32 at maximum active position 191 | Largest teacher cell resource-qualified; no truth access. |
| `teacher-qualification-r1` | Fail-closed: embedding path was not a regular file in that invocation | No teacher evidence and no truth access. |
| `teacher-qualification-r2` | Fail-closed GPU OOM during default difficult-row selection; the old path materialized a 45,596 x 128,256 FP32 logit matrix (about 21.79 GiB) on a 15.89 GiB device before candidate simulation | Teacher qualification did not run to completion; no teacher evidence and no truth access. |
| `teacher-qualification-r3` | Pass on 384 public correction positions | Frozen privileged teacher evidence and informative-gate diagnostics are available; no evaluator truth. |

The auditable r5 qualifier receipt is
`experiments/TRR-P04/runtime/capacity-qualifier-r5/qualifier_receipt.json`
(SHA-256
`0c3180e8762ae4639d93616f4fb8bdd832c6816244ac27390925470d487c2690`). It
records source commit `628c843ee127c0b5014803f8226d8f795c0c4579`, CUDA/Torch
`2.10.0+cu128`, and the exact graph: batch 8, sequence 192, hidden 2,048,
GRU width 256, position budget 512, vocabulary 128,256. The guard measured
15.679 GB free before the run and 12.840 GB after it, with 2.512 GB peak
allocated, 2.749 GB peak reserved, and 3.891 GB peak host RSS. The GPU was
released after the run.

The OOM was an implementation path issue, not a negative teacher or student
result. The old `_default_selection` recomputed full-vocabulary logits for the
entire correction pool even though the frozen candidate-preparation artifact
already contained the PR7-affine proposal rows. Commit `02e2b36` changed
selection to read the cached proposal top-1 column, preserving the declared
proposer and removing that redundant allocation. Teacher preflight r2 then
qualified K=32 at maximum active position 191, and qualification r3 completed
using that path with the existing GPU/RSS guard.

## Teacher qualification result

The public teacher is explicitly a **privileged public-prefix training scorer**.
It uses the known public token prefix to build the frozen reference cache and
scores the fixed K=32 candidate rows. It is not a deployed selector and its
accuracy is not a native BOS-only reconstruction result. The r3 receipt covers
256 difficult rows selected from frozen P04-proposer errors and 128 seeded
uniform-audit rows:

| Selection | Rows | P04 proposer correct | Gold in K=32 / K=512 | Teacher correct | Teacher fixes | Introduced errors | Retained pairs / omitted near-ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Difficult P04-proposer-error | 256 | 0 (0.0%) | 250 / 256 | 250 (97.7%) | 250 | 0 | 7,626 / 60 |
| Uniform audit | 128 | 115 (89.8%) | 125 / 127 | 125 (97.7%) | 10 | 0 | 3,817 / 26 |
| **All** | **384** | **115 (30.0%)** | **375 / 383** | **375 (97.7%)** | **260** | **0** | **11,443 / 86** |

The raw qualification key `a1_accuracy` is retained for schema compatibility;
here it means the frozen P04 affine proposer (`p04_public_affine`), not the
historical/native A1 method. The historical A1 lens is loaded only as a
public-prefix reference resource and does not choose the proposer rows.
Joining the immutable `rows_detail` record IDs to the public correction-record
metadata gives the following available style/length breakdown. Length here is
the capture-valid post-BOS source length (the public correction tensor is
right-padded to 192, with valid positions through 191); these are not the
fresh panel's balanced 16/32/64/128 strata. The error-driven plus uniform
selection therefore does not preserve fresh-panel balance.

| Public style | Capture-valid post-BOS length | Records / rows | P04 proposer correct | Gold in K=32 / K=512 | Teacher correct | Fixes / introduced | Retained / omitted pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pile_plain` | 191 | 46 / 313 | 49 / 313 | 304 / 312 | 304 / 313 | 255 / 0 | 9,326 / 73 |
| `finance_chat` | 128–159 | 5 / 6 | 6 / 6 | 6 / 6 | 6 / 6 | 0 / 0 | 180 / 0 |
| `finance_chat` | 160–190 | 5 / 5 | 3 / 5 | 5 / 5 | 5 / 5 | 2 / 0 | 150 / 0 |
| `finance_chat` | 191 | 24 / 31 | 30 / 31 | 31 / 31 | 31 / 31 | 1 / 0 | 922 / 8 |
| `alpaca_instruction` | 128–159 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 | 299 / 1 |
| `alpaca_instruction` | 160–190 | 11 / 13 | 12 / 13 | 13 / 13 | 13 / 13 | 1 / 0 | 386 / 4 |
| `alpaca_instruction` | 191 | 5 / 6 | 5 / 6 | 6 / 6 | 6 / 6 | 1 / 0 | 180 / 0 |

Thus the teacher audit spans all three public styles but is heavily weighted
toward `pile_plain` difficult rows (313/384 positions); its style/length table
is a qualification diagnostic, not a claim about fresh-panel performance.

All 384 score rows are finite, and every row retains at least two non-gold
adjacent pairs above the declared tie tolerance (28--31 per row). K=32
proposal misses are 9/384 (2.34%); K=512 misses are 1/384 (0.26%). The score
range is -0.793321 to 0.999936 with mean 0.003228. The frozen ranking scale is
`sigma_q=0.01452335`, tie tolerance `0.0001452335`, and capped pair weights
range from 0.010229 to 1.0. There are 86 omitted near-tie adjacent pairs; the
maximum-score tie count is one in every row. The predeclared informative gate
passes: all rows are finite, proposal misses are below one half, and every row
has at least two retained non-gold pairs.

The code review confirms that `derive_rank_scale` and `pairwise_teacher_loss`
remove the gold candidate before ranking or scale estimation. The D ranking
term uses a difference of two candidate log probabilities, so its shared
normalizer cancels and it contributes no gradient to the gold logit. The
ordinary full-vocabulary CE and H label-derived term still use authoritative
public labels as declared. The focused objective tests cover tie omission and
that teacher score ordering changes D only.

The teacher r3 receipt records source commit
`f22d3b05295bff9e3879bb7544502f881054ed9f`, 12,288 candidate simulations, an
86.23-second qualification phase, 5.369 GB peak reserved GPU memory, 13.259 GB
free after the run, and 3.670 GB peak host RSS. This is sufficient to release
the next bounded student-fit stage, subject to the existing freeze and
truth-access protocol.

## Public training fairness and cost audit

The serialized training source receipt binds one public data path and one
teacher/candidate provenance path for the matrix. The runner creates one
position schedule per seed before entering the arm loop and passes that same
schedule to `affine_same_data`, S, H, and D. S and the affine reference receive
no candidate IDs; H receives the frozen candidate IDs but no teacher scores; D
receives those same IDs plus the qualified teacher arrays. The candidate
artifact is `ecf031a606a8bce22116b65d2dcbf5bc25596c80961767a40924081f8f03b1e5`,
with proposer `p04_public_affine` / `pr7_public_affine_state` and the declared
descending-score, ascending-token-ID tie rule. The D teacher binding is to
`teacher_evidence.safetensors` SHA-256
`0f15a16978d0daa6bbf8a7771109350a633a7fc03a108db5fd8d782255ba84f9`, with the
frozen `sigma_q` and tie tolerance from the qualification receipt.

Both serialized seed schedules have shape 3,000 updates by eight records and
carry the same concatenated pool-order hash
`b53f032dfaf2f69e2a26a55f71ce6da2ccf3e746c0079bb3d3b92ae4c88b89b3`. Every
update contains exactly six replay and two correction records: 18,000 replay
and 6,000 correction record exposures per seed, exactly 75/25 by records. The
mask selects 1,508,895 post-BOS positions per seed: 1,124,895 replay and
384,000 correction positions, or 74.5509%/25.4491% by tokens. Every replay
record appears exactly 15 times; correction records appear either 23 or 24
times (144 and 112 records, respectively) in each seed schedule. Correction
records contribute 128 positions per update; replay contributes 332--384,
so the total selected positions range from 460--512 (seed 1737) or 461--512
(seed 2711), with mean 502.965. The token imbalance is therefore a measured
consequence of unequal replay lengths, rather than a change to the declared
record mixture.

The 384 required teacher positions are all present in both schedules. Mapping
teacher record IDs to the correction-pool offset and counting selected masks
shows 9,033 teacher-position exposures for seed 1737 (201 rows repeated 24
 times and 183 repeated 23 times) and 9,006 for seed 2711 (174 repeated 24
times and 210 repeated 23 times). Every qualified teacher row is therefore
available to D on every scheduled occurrence of its record; the teacher is not
silently sampled once per fit.

The actual implementation uses `validation_every=100`, producing the common
0, 100, ..., 3,000 validation grid for every arm and seed. This is a recorded
deviation from the proposed 200-step grid, applied uniformly before any fresh
truth access. Each arm still uses the same style-balanced public-validation
selection metric and earliest-step tie rule; selected steps can differ by arm.
The eight selected public-only checkpoints are:

| Seed | Arm | Selected step | Style-balanced public validation accuracy | Delta vs same-data affine |
| ---: | --- | ---: | ---: | ---: |
| 1737 | `affine_same_data` | 1,600 | 94.6573% | +0.0000 pp |
| 1737 | S | 600 | 94.7740% | +0.1167 pp |
| 1737 | H | 1,700 | 94.5564% | -0.1008 pp |
| 1737 | D | 1,300 | 93.7912% | -0.8661 pp |
| 2711 | `affine_same_data` | 1,900 | 94.6137% | +0.0000 pp |
| 2711 | S | 1,200 | 94.5803% | -0.0335 pp |
| 2711 | H | 1,400 | 94.5669% | -0.0468 pp |
| 2711 | D | 300 | 93.4317% | -1.1820 pp |

These are public validation selection diagnostics, not fresh-panel results.
On seed 1737, S is 0.1167 percentage points above the trainable affine
reference, H is 0.1008 points below it, and D is 0.8661 points below it. On
seed 2711, S and H are 0.0335 and 0.0468 points below affine, while D is 1.1820
points below. This exploratory pattern does not determine fresh performance or
justify selecting a winner before the paired evaluator comparison.

The original training process serialized all eight arm fits, then failed only
while constructing its aggregate with `NameError: TRAINING_SCHEMA is not
defined` (followed by an exception-handler name error). No fit was rerun. A
separate finalizer under source commit `d9805ef35f7b6c35552d081d7a8893ee65e18d9f`
verified the 26 preserved artifacts; the fit source remains
`1aefc307ebdd4cd5002ac6ac0cdc5a1fc696aa68`.

The D curve confirms that the ranking objective is numerically active rather
than silently zero. Among its 30 serialized post-update checkpoints, 20 have
nonzero `rank_rows`; those checkpoint batches contain 1--33 teacher rows and
29--984 retained pairs. At active checkpoints, the raw rank reduction ranges
from 0.2343 to 1.3170 (median 0.4509), so the fixed rank weight 0.25 contributes
0.0586--0.3293 (median 0.1127) to the total. The corresponding CE values are
0.00310--0.04902 (median 0.00626), and H's hard term is 0.000244--0.002782
(median 0.000441) over the ordinary selected rows. This apparent scale is
intentional but must be read with its reduction semantics: CE and H average
selected positions, while D's rank term is a weighted mean over only the
teacher-masked rows. It is not evidence that D has a comparable per-token
loss contribution. The fixed objective has not been swept or refit.

The per-arm fit receipts report 146.8009, 189.5909, 207.0576, and 179.9868
seconds for seed 1737 (affine, S, H, D), and 118.9694, 153.9741, 174.0851,
and 180.5299 seconds for seed 2711. Their eight-arm sum is
1,350.994774457 seconds. The seed-level `wall_seconds` fields are cumulative
from the process start, so the seed-2711 value is not a standalone second-seed
duration. The aggregate `training_result.json` field `wall_seconds=0.202251`
and the finalization receipt's 0.208573 seconds are finalizer-only timings
recorded after fitting, not fitting durations. Selected state artifacts are
approximately 16.79 MB for the affine reference and 25.98 MB for each student.
The fit receipts report a common host high-water mark of 5,882,933,248 bytes,
but no per-arm PyTorch allocator peak.

An external watchdog was attached late, at 2026-09-06T02:38:19Z, and ended at
02:53:47Z after the target process exited. It is therefore supplementary rather
than a start-to-finish guard. It sampled whole-device GPU usage, not PyTorch
`max_memory_reserved`: maximum used was 7,808,745,472 bytes, minimum free was
8,945,401,856 bytes, maximum host high-water mark was 5,882,933,248 bytes, and
maximum temperature was 76 degrees C. It observed no guard violation, but the
late start and whole-device measurement prevent it from substituting for an
in-process allocator peak or a complete start-to-finish guard.

## Deployment audit

The current activation-only prediction path is consistent with the required
zero-selector deployment contract. `scripts/trr0004_p04_predict.py` accepts
only activation/mask tensors, record metadata, a frozen student state, and the
public embedding table. It rejects source/truth fields in both record metadata
and the observation artifact. `src/token_reconstruction/p04_student.py`
computes unrestricted full-vocabulary logits from the causal activation pass;
the prediction helper has no candidate IDs, teacher scores, public-prefix
calls, A2/search fallback, source tokens, or guessed-token feedback. It resets
recurrent state per record batch and emits the declared lowest-ID top-1 rule
for exact finite-precision ties. The prediction receipt explicitly records
`uses_source_tokens=false`, `uses_teacher_or_candidates=false`,
`uses_prefix_calls=false`, and `full_vocabulary=true`.

The focused student tests cover right-padding, projection chunks crossing a
boundary, lowest-ID tie counts, and state round-trip prediction equivalence.
No deployment blocker was found in this static audit. The training-only
candidate artifact is not loaded by this prediction path.

## Native anchor pre-execution provenance review

The initial static review found a source-level provenance bug in
`scripts/trr_p04/native_anchor_runner.py` before any native anchor execution:
the shifted branch loaded the evaluator-private LoRA despite claiming that the
reconstructor did not receive target weights. Commit
`6753dff3331481e587044c94bbab8f8d6db0cdf` removes that loader and the
`--target-update` CLI input. The fixed `_load_reference_resources` path loads
the same pinned public model snapshot and public/historical A1 resources for
both `public_base` and `p04_evaluator_target_update_v1`; its receipt records
that the evaluator update was not loaded. The focused truth-free evaluator
contract tests pass (4 passed), including identical public-reference identity
across the two conditions. The evaluator observation-capture process remains
the sole private-update loader.

The anchor is therefore clear on this access-contract review, subject to the
usual receipt binding at execution. No native anchor has run before or after
the fix, so there is no anchor prediction, metric, or private-update exposure
to invalidate. The numerical result must retain its exact PR7 A1+A2 K=256
algorithm on the P04 input-panel port: 12 anchor records, 32 scored positions
per record, 384 scored positions per target, published proposal order and
first-argmax ties, and no canonical dual-benchmark claim. The student paths
retain their separate lowest-token-ID tie rule.

A stale unused `PR7_TARGET_UPDATE` path constant remains in the source file;
it is not passed, loaded, hashed, or emitted by the fixed anchor flow. Remove
it before publication if the final source certification requires that no private
path string occur anywhere in the anchor source.

## Prediction freeze provenance review

The current `freeze_predictions.py` gate correctly requires exact coverage for
all eight student/reference groups in both target conditions and both native
anchor groups, validates prediction lengths and panel coverage, and reads no
truth. The student prediction receipt already records output, state,
observation, record, and embedding hashes plus method/seed/condition and the
activation-only flags. The remaining gap is cross-binding: the freezer and
scorer currently hash prediction files and the labeled state files, but do not
require or validate a receipt for each prediction output. They therefore do not
prove that the output's state, observation condition/order, geometry, and
training provenance are the assets named by the freeze manifest.

The minimum setup addition before opening truth is one strict receipt check at
the freeze boundary. For every student output, require its prediction receipt
and verify its output hash, method/seed/condition, state hash, observation hash,
record-manifest hash, embedding-table hash, source commit, and activation-only
flags. Cross-reference the observation hash to the post-capture truth-free
receipt for that condition, including the frozen panel selection hash, ordered
record hash, mask/position geometry, 72-record/192-token/2,048-hidden shape,
and cut depth. For each state, bind the manifest row to the training
finalization receipt, state-binding receipt, and training source receipt, and
carry the verified state metadata: state schema, method, seed, architecture
JSON/digest, schedule digest, selected step, affine initialization and public
input/validation/embedding/candidate/teacher asset hashes, and fit source
commit. This binds the model/pretraining and configuration lineage without
loading tensors or truth. For each native anchor output, require the native
anchor receipt and verify its condition, panel/anchor order, observation hash,
public model/reference identity, PR7 source descriptors, tie rule, and
`target_update_loaded=false` fields. These are small receipt cross-checks; a
new generic gate is unnecessary.

The scorer's aggregate anchor table will be sufficient for totals only. The
final truth-gated report must also derive 12-record native-anchor fixes and
regressions against the selected same-data affine control and D, so those
comparisons are not inferred from the public teacher qualification table.

## Pending fresh comparison

The following sections remain intentionally open and must be filled only from
new receipts:

- **Teacher qualification:** complete for the fixed 384-row public training
  signal. The receipt and per-kind audit above are training-only evidence and
  must not be called native BOS-only reconstruction. No teacher expansion is
  authorized before the fresh paired comparison.
- **Eight-fit comparison:** complete in the public training stage for paired
  seeds 1737 and 2711, using the common 75/25 record schedule and 3,000-update
  budget. S is full-vocabulary CE; H adds the fixed label-only hard-confusion
  loss; D adds the single weighted non-gold adjacent-pair ranking loss from
  qualified teacher scores. All deployed predictions remain unrestricted
  full-vocabulary outputs.
- **Fresh metrics:** frozen predictions for the 72-record panel across
  `public_base` and the predeclared evaluator target-update condition, plus the
  separate 12-record native A1+A2 anchor. Truth can be opened only after the
  joint prediction/state freeze. Report token accuracy, exact records,
  per-style/per-length/per-target results, paired source-record uncertainty,
  and gains/regressions against the same-data controls.
- **Costs:** both seed fits and per-arm times are recorded above. Capture,
  candidate preparation, complete matrix accounting, startup, warmed inference,
  prefix calls, and final peak-memory receipts must still be reported
  separately. No amortization claim is available yet.

Evaluator setup preflights currently pass without model, target, or truth
access: the two target conditions are planned for 72 records each with 12
separate anchor records and 384 anchor positions per target. The native anchor
is an exact PR7 algorithm on this P04 input-panel port, not a canonical
dual-benchmark result. Fresh observation capture and truth-gated scoring remain
pending; the eight public fits are reported above.

## Scope and claim discipline

This is a bounded exploratory P04 study. The static/projected P03 variant
remains stopped and unmerged; P04 does not revive it or make a canonical A2
replacement claim. If teacher qualification cannot be completed under a
documented safe resource plan, that is a qualification/resource outcome, not
evidence that the student method family has failed. If the eventual D-versus-H
comparison is negative, H or S can still be retained only if their fresh,
same-data results support that simpler conclusion.

## Evidence pointers

- Panel selection: `experiments/TRR-P04/setup/public_selection-r2.json`, SHA-256
  `05f941e0dbcf29ea3efc47c7bc8abb3a7146a266eeea770f05052bb7728cde6a`.
- Public capture: `experiments/TRR-P04/runtime/public-pool-capture-r1/capture_evidence.json`,
  SHA-256 `ea430d0d98506450e4c28d1de8a1ddbf3aebc86a32514fe4f836ba5f6eeb06e7`;
  capture commit `d8c657ba834f30b6b68bf0720fbf52ca123b7509`.
- Candidate preparation: r1 receipt
  `experiments/TRR-P04/runtime/candidate-preparation-r1/candidate_preparation_receipt.json`;
  r2 metadata completion
  `experiments/TRR-P04/runtime/candidate-preparation-r2/candidate_preparation_metadata_completion_receipt.json`.
- Capacity failures and passes: `experiments/TRR-P04/runtime/capacity-qualifier-r1/failure.json`,
  `capacity-qualifier-r2/failure.json`, `capacity-qualifier-r3/qualifier_receipt.json`,
  and `capacity-qualifier-r5/qualifier_receipt.json`.
- Teacher preflight and qualification: `experiments/TRR-P04/runtime/teacher-preflight-r1/failure.json`
  (SHA-256 `0699c796300344b3c035388189351c4d8894e67bd3f92aa1d4d727348d0990ec`),
  `teacher-preflight-r2/teacher_preflight.json` (SHA-256
  `4ad906c9d2e2b1e84bae5849fa4a786d8c3ba2e0fd1b3d561a236634fda14798`),
  `teacher-qualification-r1/failure.json`, `teacher-qualification-r2/failure.json`,
  `teacher-qualification-r3/teacher_receipt.json` (SHA-256
  `8267fe11bf25db4a667bf0b6e21556d9eb6e0ca74549d3808c5e080d408d9505`),
  `teacher-qualification-r3/teacher_qualification.json` (SHA-256
  `d5519e12ba6e0e12b6cb06beec1285468fa8dc07187d095fa1b78f4fd8a21996`), and
  `teacher-qualification-r3/teacher_evidence.safetensors` (SHA-256
  `0f15a16978d0daa6bbf8a7771109350a633a7fc03a108db5fd8d782255ba84f9`).
- Public training: `experiments/TRR-P04/runtime/training-r1/source_receipt.json`
  (SHA-256 `698d91b39e211bef7ec996a42cee5d41c709995cfa44b73622c79c7949c35449`),
  `training_result.json` (SHA-256
  `2e0f9566c33fa19c1351885d4027cd00b1f22f95c4464aa1d8be4f3a55f23d19`),
  `training_finalization_receipt.json` (SHA-256
  `4ac6e3bfd482a658e62e458b5d27ec74a50350c2d6818623efc3fb645f0fba29`),
  and `late_finalization_failure.json` (SHA-256
  `5b078f399bdff9d1f335759b78fde609536793010bc187c006a347fbb443efb8`), and
  `training_state_binding_check.json` (SHA-256
  `6d1a74504c7b773693013fc56bec678db3768fe0f1453bbb59d88ae9fd9d0d12`).
  The fit source is `1aefc307ebdd4cd5002ac6ac0cdc5a1fc696aa68`; the finalizer
  source is `d9805ef35f7b6c35552d081d7a8893ee65e18d9f`. Seed-result hashes are
  `e3234f66086af6ad285e8dc9ff758ca7bbcc99ae27e6b4f5c5fe93ef8ac4b270` (1737)
  and `b20807c9f1898acd549f03a3defe327fa17b400fc9943ffc06334878cc570c44`
  (2711). Schedules have seed-1737 digest
  `887edbcef5d44ad39ad9694ef0f4e049b46ba6665a239ecc090e3febce8eedec` and
  seed-2711 digest `8fc93b18bad186cf13a20118eb4158d5e49fc5c86732ecbb9376303d7c5f7a9d`.
  Learning-curve hashes are recorded in the seed result directories. The
  external watchdog receipt is
  `experiments/TRR-P04/runtime/training-r1/external_resource_watchdog_receipt.json`
  (SHA-256 `e344773291c8076e9f75c54860a9a2e6fe8ce81a887a90135d848a99a60b4269`);
  it is a late-attached supplementary monitor, not a start-to-finish allocator receipt.
- Evaluator/anchor preflights: `experiments/TRR-P04/setup/evaluator-observation-preflight-r3/evaluator_capture_preflight.json`
  and `experiments/TRR-P04/setup/native-anchor-preflight-r3/native_anchor_preflight.json`.
- Reviewed source snapshots: `f22d3b05295bff9e3879bb7544502f881054ed9f`
  for teacher qualification, `6753dff3331481e587044c94bbab8f8d6db0cdf`
  for the native-anchor public-reference fix, and `233f310be43c0018bdd28d4d98d38a703e7d355f`
  for the setup/source lineage.


No evaluator truth, private target-update weights, or fresh evaluation source
rows were opened while preparing this record. Public correction labels used by
the teacher are training data under the declared privileged-prefix contract.
