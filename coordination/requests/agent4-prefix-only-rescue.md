# Agent 4 — Bounded rescue of prefix-only inversion

## Objective

Determine whether the corrected prefix-only method can become a useful alternative to a fitted proposer plus A2. Preserve the reported 29/30 tokens in 9.82 seconds versus A1+A2's 30/30 in 1.69 seconds as the historical pilot. Do not treat that tiny subset as either a general success or a final rejection.

The leading hypothesis is that a differentiable current-token executor plus better discrete search can avoid unnecessary full-prefix work and excessive continuous optimization. A second, bounded hypothesis is that a local curvature-aware inverse can reduce the difficult search tail.

No fitted A1/B0/B1 initializer or fallback is allowed. The method must obtain proposal information from the public or legitimately recovered prefix itself.

## Starting point and ownership

Work from your actual corrected local implementation, reported at `76c3097`; record its full commit and runtime bindings. That commit was not accessible on the remote repository during this review, so upstream findings below are checks to make—not assertions that your local code contains them.

Use a new task-local rescue branch and immutable output names. Preserve the original bugged CPU results and corrected GPU pilot separately. Do not modify the other agents' tasks, merge PRs, open P03's holdout, or infer new public-disclosure approval from this assignment. Prepare a sanitized release list under the existing approval process.

You own the task end to end. Reuse verified loaders and evaluation components. Avoid another chain of small handoffs or a broad infrastructure audit. No paid compute; qualify a bounded resource budget and coordinate shared GPU use.

## 1. One diagnostic before changing the algorithm

Reproduce the corrected pilot on its already-opened inputs. Identify the failed token's category: never proposed, proposed but rejected, incorrect final-return/timeout behavior, prior-token error, or model/numerical mismatch.

Measure per-token gradient evaluations, actual candidate forwards, vocabulary-ranking work, cache/prefix computation and runtime. Separate startup, synchronization/logging, and solver cost. Determine whether one exhausted position dominates the total. Do not optimize only that failed example and then call it fresh confirmation.

Check the upstream costs against your local code: full embedding-table clones/difference tensors, full-prefix recomputation, unused suffix/head execution, repeated token decoding or scalar synchronization, and garbage collection/CUDA cache clearing in the hot loop. Change only costs actually present and material.

## 2. Qualify one efficient current-token executor

For fixed reconstructed tokens and fixed prefix weights, cache the earlier positions' per-layer K/V as constants. Differentiate only through the current trial embedding and its current Q/K/V path. Do not mutate the committed cache during trial evaluations; commit one token once. Invalidate caches when model-prefix weights change.

Validate forward values AND input gradients against the corrected full-prefix path, across positions. Preserve actual input embeddings, rotary construction, masks, dtypes, and cut location. The supplied synthetic cache test is only a mathematical demonstration, not Llama qualification.

Use an immutable vocabulary table, a visited mask, cached row norms and matrix–vector nearest-token scoring where appropriate. Handle near ties in the qualified metric. This still reads the table; do not claim its cost vanished.

Keep this execution-only baseline separate from solver changes. Preserve its decisions or label numerical differences explicitly.

## 3. Test one main search change and one bounded alternative

### Main: discrete, forward-verified local proposals

Do not insist on solving a highly accurate continuous embedding before proposing real tokens. Use the activation-mismatch gradient to rank discrete substitutions, verify a small batch, and retain the best ACTUALLY evaluated token.

A useful surrogate is:

    q(v) = gᵀ(e_v - z) + lambda/2 * ||e_v - z||²

It can be ranked by a vocabulary matrix–vector product. With scalar lambda this equals nearest-token projection after a scaled gradient step; the formula itself is not a novel mechanism. The intervention is bounded real-token search, small-batch verification, and step/trust control based on actual improvement.

Choose one sensible schedule from public development evidence, not a parameter sweep. Keep initialization fitting-free and count its screening cost.

### Alternative: damped local inverse

If derivative qualification and cost preflight pass, test a limited matrix-free Gauss–Newton step:

    (JᵀJ + lambda I) p = -Jᵀ(F(z) - h)

Use a small capped number of JVP/VJP products, with actual-forward acceptance. Do not build a dense Jacobian at every token or hide its setup cost. If this is more expensive than the work it saves, stop it.

Check the loss normalization and effective step size. Coordinate-mean squared error scales gradients by output width; an unsuitable rate can make even a trivial inverse slow. Do not blindly multiply the current learning rate by 2,048 or adopt the toy example's identity-map rate.

## 4. Verification under numerical and model mismatch

Retain best-so-far discrete candidates even when they fail an exactness tolerance. A visited mask may avoid duplicate work; it must not erase the best answer from the output policy.

Use actual discrete forward scores at commitment, not an unverified nearest embedding at timeout. Predetermine budget-exhaustion behavior and keep failures in the denominator.

First distinguish numerical mismatch from changed target weights. Do not make observation precision or capture easier just to obtain equality. An angular-loss comparison is optional only when the diagnostics identify scale mismatch; keep it separate from optimizer changes.

A residual or score gap is not proof of correctness under an imperfect prefix. Do not loosen thresholds until the known failed token passes.

## 5. Evaluate and decide

Compare the corrected original schedule, the efficient execution baseline, and at most the two qualified solver challengers on common inputs. Use matched-public controls first, then a small unused natural panel after choices are frozen. Report a separate unusual-token stress panel.

Show token and exact-clip recovery, runtime distribution, total derivative/forward work, memory and stopping failures. Include the same-input A1+A2 comparator. If Agent 3 has a completed small-budget hybrid, it may be an additional explicitly identified practical comparator, never an initializer.

Only advance to mismatched/recovered-prefix experiments if matched-model quality and total cost warrant it. No validated recovery path currently available means that stage remains untested; actual target weights cannot substitute for recovered weights. Clearly separate supplied-prefix component performance from cold-start closed-loop tracking.

The model prefix is frozen within a record; permitted updates from that record may affect only later records. No source truth beyond BOS may affect search.

Back up actual states and required dependencies before final evaluation and verify a clean restore. Deliver one report with the measured bottleneck, changes tested, quality/cost frontier and a clear advance/stop decision. No automatic larger solver sweep or final-panel expansion.

## References

- Local research report: `RESEARCH_REPORT.md` and `analytical_checks.py` in this handoff.
- Official SIPIT source, pinned: https://github.com/giorgosnikolaou/SIPIT/tree/820683156b7257313046a4fb3c492e52519525b7
- SipIt paper: https://arxiv.org/html/2510.15511v4
- HotFlip: https://aclanthology.org/P18-2006/
- Matrix-vector nonlinear least squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
- JVP/VJP construction: https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html
