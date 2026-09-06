# TRR-P04 result and decision record

**Status:** exploratory comparison incomplete; no student-method winner.

The completed public stages establish that the proposed student graph can be
optimized on a bounded public error set, that the largest planned cell fits the
available GPU guard, and that the privileged public teacher produces an
informative bounded signal. They do not establish that a GRU, hard-confusion
loss, or teacher score information improves reconstruction on fresh records.
The eight-fit comparison and fresh evaluation remain pending.

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
evidence; first fair student comparison pending**. The teacher receipt supports
using its frozen relative scores in D's training objective. It does not support
a native BOS-only accuracy claim. No student states, fresh predictions, or
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

## Pending first fair comparison

The following sections remain intentionally open and must be filled only from
new receipts:

- **Teacher qualification:** complete for the fixed 384-row public training
  signal. The receipt and per-kind audit above are training-only evidence and
  must not be called native BOS-only reconstruction. No teacher expansion is
  authorized before the first fair student comparison.
- **Eight-fit comparison:** paired seeds 1737 and 2711 for the same-data affine
  reference and S/H/D students, using the common 75/25 record schedule and
  3,000-update budget. S is full-vocabulary CE; H adds the fixed label-only
  hard-confusion loss; D adds the single weighted non-gold adjacent-pair
  ranking loss from qualified teacher scores. All deployed predictions remain
  unrestricted full-vocabulary outputs.
- **Fresh metrics:** frozen predictions for the 72-record panel across
  `public_base` and the predeclared evaluator target-update condition, plus the
  separate 12-record native A1+A2 anchor. Truth can be opened only after the
  joint prediction/state freeze. Report token accuracy, exact records,
  per-style/per-length/per-target results, paired source-record uncertainty,
  and gains/regressions against the same-data controls.
- **Costs:** capture, candidate preparation, teacher simulations, fitting,
  retained state/table size, startup, warmed inference, prefix calls, and
  peak memory must be reported separately. No amortization claim is available
  yet.

Evaluator setup preflights currently pass without model, target, or truth
access: the two target conditions are planned for 72 records each with 12
separate anchor records and 384 anchor positions per target. The native anchor
is an exact PR7 algorithm on this P04 input-panel port, not a canonical
dual-benchmark result. Fresh observation capture and all eight fits remain
unreported.

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
- Evaluator/anchor preflights: `experiments/TRR-P04/setup/evaluator-observation-preflight-r3/evaluator_capture_preflight.json`
  and `experiments/TRR-P04/setup/native-anchor-preflight-r3/native_anchor_preflight.json`.
- Reviewed source snapshots: `f22d3b05295bff9e3879bb7544502f881054ed9f`
  for teacher qualification and `233f310be43c0018bdd28d4d98d38a703e7d355f`
  for the current setup/source lineage.

No evaluator truth, private target-update weights, or fresh evaluation source
rows were opened while preparing this record. Public correction labels used by
the teacher are training data under the declared privileged-prefix contract.
