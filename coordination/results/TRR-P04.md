# TRR-P04 result and decision record

**Status:** exploratory comparison incomplete; no student-method winner.

The completed public stages establish that the proposed student graph can be
optimized on a bounded public error set and that the largest planned cell fits
the available GPU guard. They do not establish that a GRU, hard-confusion loss,
or teacher score information improves reconstruction on fresh records. Teacher
qualification stopped fail-closed before teacher evidence was written, so the
eight-fit comparison and fresh evaluation remain pending.

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

The disposition is **teacher qualification pending after fail-closed resource
reassessment**. No teacher scores, student states, fresh predictions, or
evaluation metrics should be interpreted from the current artifacts.

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
| `teacher-qualification-r1` | Fail-closed: embedding path was not a regular file in that invocation | No teacher evidence and no truth access. |
| `teacher-qualification-r2` | Fail-closed GPU OOM during frozen-reference candidate simulation; attempted allocation 21.79 GiB on a 15.89 GiB device | Teacher qualification did not run to completion; no teacher evidence and no truth access. |

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

- **Teacher qualification:** one privileged public-prefix teacher on 384
  correction positions (256 difficult plus 128 random audit), with candidate
  recall, proposal misses, teacher fixes/errors, score gaps, ties, and the
  frozen ranking scale. This is training-only evidence and must not be called
  native BOS-only reconstruction.
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
- Teacher failures: `experiments/TRR-P04/runtime/teacher-qualification-r1/failure.json`
  and `teacher-qualification-r2/failure.json`.
- Evaluator/anchor preflights: `experiments/TRR-P04/setup/evaluator-observation-preflight-r3/evaluator_capture_preflight.json`
  and `experiments/TRR-P04/setup/native-anchor-preflight-r3/native_anchor_preflight.json`.
- Reviewed source snapshot: `f1a7e29438f6eb1ef38406321c74bc245fc0e8a2`.

No evaluator truth, private target-update weights, or fresh evaluation source
rows were opened while preparing this record.
