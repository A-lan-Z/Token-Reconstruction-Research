# TRR-P05 — Diagnose the P04 teacher-loss regression without a new training campaign

## Objective

Explain what the failed P04 teacher-ranking objective actually did.

P04 established that D underperformed H and S on the frozen fresh panel.
Preserve that result. This task is not an attempt to overturn it, promote
distillation, or search for a better coefficient.

The central question is:

Did D learn the additional teacher-ranking task while harming token
reconstruction, or did the implementation/training setup fail to transfer
that signal meaningfully?

## Starting point and ownership

Repository: A-lan-Z/Token-Reconstruction-Research

Parent branch: task/TRR-P04
Reviewed head: c3aa40abdf33b5e794b139671fcddf6fd5a2c65e
New task/branch: TRR-P05 / task/TRR-P05

Use a separate worktree and task-local outputs/state. Preserve all prior
predictions and evidence. Do not merge any PR or modify global coordination
state, the active benchmark registry, or agent one's workspace.

Agent one continues TRR-0006 unchanged. Do not request or use its hidden
records, intermediate scores, or unpublished scientific results. Preserve
the approved hash-only coordination boundary.

P03 remains STOP_VARIANT; its sealed holdout is out of scope.

## Scope

Use existing public fitting/correction/validation resources, cached teacher
evidence, stored checkpoints, and logs.

No optimizer steps, new student fits, coefficient or temperature sweep,
new target preparation, or new hidden evaluation is authorized.

Forward/backward diagnostic calculations are allowed without applying
parameter updates. Coordinate resources and use a modest sample where
appropriate. No paid compute.

Do not turn this into a general infrastructure audit. Choose the smallest
analysis that can distinguish the explanations below.

## 1. Measure what D learned

For the initial state and the available selected/final S, H, and D states,
compare on the same public examples:

- full-vocabulary token accuracy and gold-versus-best-other margins;
- agreement with the cached teacher's non-gold pair order;
- the original teacher-ranking loss, with ties handled as in P04.

Separate the 384 teacher-supervised positions from public positions that
did not receive teacher scores. Separate difficult and uniform teacher
qualification cases.

Where stored checkpoints are unavailable, report that limitation. Do not
reconstruct an unrecorded training trajectory or refit a substitute.

The qualification teacher's high gold-token accuracy is not itself a
measure of how useful its non-gold ranking targets are to the student.

## 2. Check objective interaction and effective weighting

On a small reproducible sample of existing public batches, calculate the
separate supervised, hard-confusion, and teacher-ranking gradients.

Measure their magnitudes and alignment on shared trainable parameters,
and account for the actual reduction rules, active teacher-row counts,
pair weights, and existing gradient clipping.

Do not infer gradient balance from scalar losses or the coefficient 0.25
alone. Distinguish representative diagnostic measurements at stored
checkpoints from claims about the entire original training trajectory.

Check whether improving the ranking objective locally is aligned with
improving gold-token margins. No parameter updates are required.

## 3. Give a bounded scientific disposition

Return one evidence-backed conclusion:

- Ranking imitation improved but was not useful for reconstruction:
  retire this objective.

- A specific objective-scaling or optimization conflict plausibly explains
  the regression: describe one discriminating future experiment, but do
  not execute a new fit in this task.

- The student did not meaningfully learn the intended signal, or the
  available evidence is insufficient: keep the broader transfer question
  unresolved without weakening P04's negative result.

Do not recommend more teacher data, a larger model, or a new loss merely
because those options exist. Tie any recommendation to the measured
failure mode and the goal of accurate search-free inference.

## Deliverables

Save this assignment at coordination/requests/TRR-P05.md.
Write the result to coordination/results/TRR-P05.md and structured evidence
to experiments/TRR-P05/manifest.json, with task-local status.

Retain diagnostic code, exact inputs/state identities, commands, limitations,
and resource evidence. Open a follow-on PR against the actual parent without
merging it.

Lead the handoff with what the analysis establishes and whether another
teaching experiment is justified. A clear stop decision is a useful outcome.