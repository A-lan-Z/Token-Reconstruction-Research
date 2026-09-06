# TRR-P04 — Can offline A2 supervision improve a search-free causal decoder?

## Mission

Agent A2 owns this independent exploratory task. Here **A2 selector** means the historical public-prefix candidate-simulation algorithm, not the implementation agent.

Test whether a compact student can learn useful distinctions from the expensive selector during **offline public-data training**, then reconstruct using only activations and its own retained state. The primary question is whether teacher ranking information helps beyond equally good ground-truth supervision on the same examples. This targets search-free deployment, not fitting-free reconstruction.

The other agent owns TRR-0005's coverage-by-context comparison. Do not duplicate that matrix, wait for its results, or consume its unreported evaluation material. Static/shared-offset/projected BOS-prototype variants remain stopped; do not revive or compress them.

## Starting point and ownership

Repository: `A-lan-Z/Token-Reconstruction-Research`.

Use a new worktree and `task/TRR-P04` branch from the published learned-decoder snapshot:

```
Parent: task/TRR-0004 (PR #7)
Commit: 6e8b683e404c0acb70cd59b7dd6d6868b2061f61
```

Your previous P03 worktree and branch remain intact. The new parent is deliberate: this task needs the competent standalone fitting and evaluation machinery, not the failed prototype implementation. Reuse specific finalized helpers or read-only public assets where helpful, recording identities; do not import another agent's mutable working directory. Do not merge existing PRs, edit global coordination state, or alter the active benchmark registry. Keep status at `coordination/parallel/TRR-P04.json`.

Read the charter and relevant TRR-0004 code/results. This assignment authorizes public exploratory screening, not a canonical replacement claim. Preserve the charter's access and truth-opening separation. Choose reasonable details autonomously; this is not a request for an infrastructure audit or architecture sweep.

## 1. Use one credible compact student

A unidirectional GRU over boundary activations is the preferred starting architecture, motivated by the SIP paper's forward-inversion comparison (Appendix E.1). Preserve a competent affine/identity path rather than forcing all token information through a narrow recurrent bottleneck. A tied public-embedding output table is reasonable. You may choose another small causal architecture with a documented reason, but use the same student architecture across the supervision comparison.

The student predicts **x_i from H_0 through H_i**, with same-position labels. It must not take source tokens, teacher predictions, candidate identities, teacher scores, future activations, or guessed-token feedback as inference inputs. Reset recurrent state between records. No public-prefix call, per-token optimization, A2 fallback, or shortlist-dependent decision is permitted in deployed student evaluation. Full-vocabulary projection is permitted and its cost counts.

Start from a common reproducible fit or initialization. Demonstrate that optimization can improve on a small public set containing actual initial decoder errors; a subset already at 100% before training is not a capacity test. If the student is clearly underfit, make one justified development adjustment rather than declaring distillation or recurrence ineffective. Avoid training a residual only on rows already memorized by a frozen base.

## 2. Prepare a bounded public correction pool and qualify the teacher signal

Use a fixed public fitting/replay pool plus a separate public correction-training pool with useful mistakes or uncertain predictions from the competent decoder. Use public validation for choices, and reserve separate fresh evaluation records. Choose a sensible public mixture once; this is not another coverage sweep. Prefer a correction pool disjoint from the known fitting records of the reproducible PR #7 initialization.

All student arms must see the same records, labels, weighting/sampling schedule, and comparable optimizer and selection opportunities. Keep ordinary-example replay so difficult-example supervision does not simply create a new frequency bias. Any difficulty selection is based only on public training/development information, never on fresh or canonical answers.

Generate and cache teacher evidence only for a bounded subset of correction positions initially. Use a frozen public A2-style scorer and a modest declared candidate budget. Include both difficult positions and a representative random audit subset, so teacher quality is not reported only on preselected successes. Do not launch a huge teacher-labeling job before establishing that its scores carry useful, reliable information.

Measure proposal misses, teacher correctness, errors the teacher fixes or introduces, score spans, and ties on this public pool. Distinguish a native causal teacher using its reconstructed prefix from a training-only privileged scorer using known public prefixes. Either can supply public training supervision, but label it accurately and do not substitute its accuracy for native BOS-only reconstruction.

Keep known public token labels authoritative. Do not discard teacher mistakes or proposal misses from ordinary supervised training. Truth-included training candidate sets are allowed if explicitly labelled training-only; they must never be reported as unassisted teacher recall or used in evaluation. No private target-prefix weights or current evaluation answers may enter teacher construction.

## 3. Run a small, interpretable supervision comparison

The desired comparison uses three copies of the same student:

| Arm | Supervision |
|---|---|
| **S: supervised** | Full-vocabulary token cross-entropy on the common public data |
| **H: hard-confusion control** | S plus a label-derived loss emphasizing a fixed set of plausible incorrect tokens, without teacher numeric scores |
| **D: teacher-informed** | H plus informative relative-score/ranking supervision from the frozen public selector |

Choose one reasonable ranking/distribution objective, not a sweep of distillation losses. Freeze candidate identities for H and D from the same public decoder/resource; otherwise candidate choice itself confounds the teacher-score comparison. Pair initialization/seeds where feasible and disclose differences in training compute. A short common warm-up is fine, but continuation budgets must be comparable.

Three scientific safeguards are essential:

- Copying a correct teacher hard label merely repeats the public ground-truth label. The tested added information must be its relative scores, margins, or another clearly specified nonredundant signal.
- Raw cosine scores are not calibrated class probabilities. Inspect their scale and any temperature/normalization on public development data; near-uniform or nearly one-hot targets may make the comparison uninformative. Account for finite-precision ties and teacher errors. Do not force the student to copy an incorrect teacher top choice.
- Candidate-restricted agreement is not full-vocabulary accuracy. Retain a full-vocabulary supervised term and evaluate unrestricted predictions, including tokens absent from the teacher shortlist. Never deploy candidate generation merely because a training loss used candidates.

H versus S asks whether emphasizing confusing alternatives helps. D versus H asks whether the teacher's extra score structure adds anything beyond labels and the same hard alternatives. This distinguishes useful teaching from extra data, extra training, or relabelled supervision.

## 4. Evaluate on new records and report what was actually learned

Freeze all contenders before opening evaluation truth. Use the same fresh natural records, with multiple input styles and lengths, for the student arms and a competent standalone affine baseline. If claiming a benefit from the recurrent architecture itself, give the affine reference the same public data and comparable fitting opportunity; beating a frozen baseline after extra training does not isolate architecture. Pair matched-public and one genuinely unseen target-update condition when resources permit; target preparation stays evaluator-side, and a previously public teacher-training update is not an unseen target. Report an unavailable resource honestly rather than silently weakening the comparison.

Include a bounded native A1+A2 accuracy/cost anchor. If that anchor uses a subset, compare on that exact subset and keep its denominator separate. Keep per-domain/per-target results separate, and cluster uncertainty by independent source record rather than counting the paired targets as independent samples. Use enough independent records to avoid another tiny-endpoint result; repeat the key student comparison across seeds when the conclusion depends on a small difference.

Primary outcomes are unrestricted post-BOS token recovery, completely reconstructed records, gains/regressions against the strongest same-data student control, and measured inference cost. Teacher agreement and shortlist recall are secondary diagnostics. Inspect the subset of errors that the selector fixes, but do not select the deployed model using evaluated answers.

Count public activation generation, teacher simulations, student training, retained tables/state, startup, and warmed inference separately. Student runtime must demonstrably contain no selector. Report whether the observed preparation cost can plausibly amortize over an online stream; do not claim an amortization speedup if quality is materially worse. No actual warm-up adaptation on an evaluation stream is part of this task.

## Resource and stopping discipline

Coordinate CPU/GPU use with the running agent. Independent code and CPU preparation can proceed in parallel; heavy jobs on a shared GPU need an agreed reservation, and comparative timings need an uncontended window. Do not interrupt jobs, modify a shared environment, or incur paid compute. Qualify peak memory before larger work. Do not turn an inadequate CPU-only fit into a supposed negative result about the method family.

Publish the signal qualification and first fair student comparison before optional extensions. If the teacher carries no helpful information, or D cannot beat H under an informative setup, report that negative outcome. If H wins, retain the simpler label-only training improvement. If the recurrent S baseline itself improves recovery, report that independently of distillation. A modest gain is not an A2 replacement claim; the large remaining accuracy gap stays visible.

## Handoff

Save this assignment as `coordination/requests/TRR-P04.md`. Commit task-owned source, focused tests, configurations, model identities, compact states/predictions, and failures. Write:

```
coordination/results/TRR-P04.md
experiments/TRR-P04/manifest.json
coordination/parallel/TRR-P04.json
```

Open a follow-on PR against the actual parent `task/TRR-0004`, without merging. Lead with the decision: did the student improve because of architecture, hard-example supervision, or teacher score information; how much online cost remains; and what specific uncertainty deserves the next experiment?

## Research references (motivation, not claims about this repository)

- SIP forward inversion and GRU comparison, Sections 3.1 and E.1: https://arxiv.org/html/2409.00960v2
- Distillation with predictive distributions: https://arxiv.org/abs/1503.02531
- Target versus non-target information in distillation: https://arxiv.org/abs/2203.08679
- Distillation optimization and fidelity are not the same as generalization: https://arxiv.org/abs/2106.05945

Use only the forward, activation-based ideas relevant to the charter; gradient-based recovery and inaccessible live-model access from other papers are out of scope.
