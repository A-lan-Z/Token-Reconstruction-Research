# Rescuing prefix-only token inversion

## Executive assessment

A bounded rescue attempt is justified. The strongest proposed design is not a longer continuous-optimization run. It is a small, differentiable current-token forward model with cached context, a discrete token-proposal rule tied to the observed activation mismatch, batched verification, and an optional bounded curvature-aware step for difficult positions. The public or legitimately recovered prefix supplies all proposal information. No separately fitted token inverse is introduced.

The most important distinction is between three questions: whether the observations contain the answer; whether the forward model available to reconstruction is compatible with the target; and whether the particular search routine finds the answer cheaply. A slow or unsuccessful optimizer does not settle the first two questions. Conversely, numerical inversion of the exact public operator does not establish transfer to an unavailable changing target. The supplied research review makes these distinctions explicitly. [1]

The reported corrected Agent 4 pilot recovered 29 of 30 tokens in 9.82 seconds, against 30 of 30 in 1.69 seconds for A1+A2. These measurements come from the local completion summary, not an independent reproduction in this study. The ratio calculated from the rounded times is 5.81, consistent with the reported approximately 5.82. Tying that comparator requires removing approximately 82.8% of the reported time, while also addressing the token error. Small cosmetic changes are insufficient unless a very large measured fraction of runtime is avoidable overhead.

No new Llama reconstruction accuracy or speed result is established here. The concrete new outputs are a source inspection of the upstream algorithm, numerical analyses, small synthetic verification tests, and a prioritized experimental design.

## Evidence and access boundary

Agent 4's reported local commit is `76c3097`. It was not returned by the connected GitHub commit lookup, and its branch was not in the retrieved remote branch listing. Searches of the available saved files did not locate its report, manifest, implementation or per-token traces. A local WSL path is not a remotely accessible source. Consequently, this report does not attribute a particular upstream implementation choice to Agent 4 without qualification.

The inspected upstream reference is the official SIPIT repository, pinned to `820683156b7257313046a4fb3c492e52519525b7`, especially `src/algorithm/SIPIT.py`, `src/algorithm/base.py`, and `src/utils/model.py`. [2–4] These are starting points for checking the local port, not substitutes for its missing source. The supplied research review was read as historical research context, not as evidence of Agent 4's newest execution. [1]

The model-free checks in `analytical_checks.py` use generated arrays and a small randomly initialized causal-attention model on CPU. They verify algebra and a cache/gradient construction. They do not use Llama weights, target records or a hidden evaluation set. Their output is `analytical_results.json`.

## 1. What needs rescuing

For a fixed reconstructed token prefix pi and a fixed permitted model-prefix estimate, write the next-position forward map as:

    F_pi(z) = boundary activation obtained from current input embedding z

Let h be the observed activation. Reconstruction seeks a vocabulary token v whose actual public embedding e_v makes F_pi(e_v) match h sufficiently well.

A continuous solver instead introduces an intermediate vector z, minimizes a loss such as 0.5 ||F_pi(z) - h||², and periodically proposes real tokens near z. The intermediate vector is a search device. It is not itself the recovered answer.

This distinction identifies three different causes of a failed token:

1. The search never proposes the correct discrete token.
2. It proposes the token but the verification or final-return rule fails to choose it.
3. The available forward model or committed earlier prefix makes a wrong token look better even when the correct token is tested.

The first calls for better search. The second calls for checking geometry, numeric tolerances, candidate bookkeeping or scoring. The third calls for better model-prefix recovery or a narrower transfer claim. Running more gradient steps does not automatically solve any of them.

For the one failed pilot token, record its first proposal time, best proposal rank where measurable, discrete residual, accepted/rejected history and first preceding reconstruction error. Public evaluator-only diagnostics may use the true token to identify a failure category; the reconstruction algorithm must not receive that information.

## 2. Verified upstream costs to check in the local port

The reference code is designed to demonstrate an algorithm, not necessarily to minimize runtime for a four-layer inversion service. Its relevant choices are visible in the pinned source. [2–4]

| Upstream operation | Consequence to investigate | Proposed replacement if it is actually present locally |
|---|---|---|
| Clone the entire embedding matrix for each token | Large allocation and memory traffic | One immutable table, one small visited-token mask |
| Form E minus the current vector and take all row norms repeatedly | A vocabulary-by-width temporary for every ranking operation | Cached row norms and matrix–vector scoring |
| Run continuous and discrete full-sequence forwards with caching disabled | Repeat computation for a committed prefix that has not changed | Immutable per-layer prefix K/V plus a differentiable current-token path |
| Call a causal-LM forward and then select one hidden-state layer | Potential computation of irrelevant later layers and output head | A prefix module ending at the actual observed cut |
| Decode token strings, read several scalar values and log in the inner loop | Python and accelerator-synchronization overhead | Buffered traces and synchronization at defined timing boundaries |
| Garbage collection and CUDA cache emptying after tokens | Allocation/cache-management cost inside reconstruction | Bounded retained buffers, cleanup outside the hot loop |

Some or all of these may already have been removed in Agent 4's implementation. The local source and profiler must determine which changes remain relevant. Do not claim a particular predicted speedup from this table.

At vocabulary size 128,256 and width 2,048, one FP32 matrix contains 1,050,673,152 payload bytes: approximately 1.05 GB or 0.979 GiB. A same-shaped difference temporary is equally large. A score vector is only 513,024 bytes, while a Boolean visited-token mask is 128,256 bytes. These are array-size calculations, not memory measurements of Agent 4.

For nearest Euclidean token scoring:

    ||e_v - z||² = ||e_v||² - 2 e_vᵀ z + ||z||²

The final term is identical for every token. Cache ||e_v||² and rank using the matrix–vector product. This avoids the explicit difference matrix and square root. It does not eliminate the read of the vocabulary table or its arithmetic. Near cancellation, the numeric order can differ from a direct distance calculation; recompute a small near-tie set in the qualified metric and test agreement. The synthetic float64 test found full rank agreement and a maximum distance discrepancy of 2.84e-14.

Approximate nearest-neighbour indexing is not the first change to make. It introduces another candidate-omission mechanism before exact matrix scoring has been measured. Nor should a stable full sort be retained if only a tiny shortlist is used, provided deterministic tie handling is preserved.

## 3. Caching can preserve input gradients exactly in the mathematical model

For a fixed causal prefix, earlier positions do not depend on the current provisional token. Their per-layer keys and values can therefore be computed once and treated as constants while differentiating with respect to the current embedding. The current token's query, key and value remain differentiable. This is a consequence of causal attention and is consistent with the standard KV-cache structure. [5]

The proposed path is:

    committed token prefix -> immutable per-layer K/V
    current provisional embedding -> current Q/K/V -> cached attention -> cut activation

Candidate trials must not append to the committed cache. Once a token is chosen, commit the appropriate K/V once. Reuse or recompute that chosen candidate's state under a documented equivalent path. After any model-prefix weight change, caches and local derivative state from the previous model version are stale and must be invalidated.

This is not permission to enable a generation cache blindly during ordinary model training. The prefix weights are fixed for the record, only an input is differentiated, and the evaluation function should be functional and side-effect-free. Use detached prefix tensors produced under ordinary no-gradient execution and validate the actual autodiff path. Standard cache warnings during training remain relevant to careless implementations. [5]

The included synthetic test uses two causal attention blocks, prefix lengths 1, 4 and 16, and float64. Full-prefix and cached current-token forward outputs agree within 4.44e-16; input gradients agree within 8.88e-16; the cache is unchanged. This proves only that the proposed construction can preserve the derivatives. The toy has no rotary embeddings, grouped-query attention or BF16, so it does not validate Llama's integration.

Agent 4 already reported a rotary-precision loader correction. The next cache qualification must therefore compare absolute position, rotary-frequency construction, dtype, attention masks, actual input embeddings and the intended cut, not only output shape. Do not create a numerically easier observation to make the new path pass.

## 4. A hidden scaling issue: even the identity inverse can move very slowly

The upstream continuous objective uses coordinate-mean squared error, and its reference optimizer is SGD. [3,4] This is not inherently wrong. But the numerical meaning of the learning rate depends on the output width.

For the deliberately simple model F(z)=z in d dimensions:

    L(z) = ||z - h||² / d
    grad L(z) = 2(z - h)/d

With learning rate 1, the remaining error vector is multiplied by 1 - 2/d each step. At d=2,048, 50 steps leave 95.23% of the original error norm. It takes 4,714 steps to reduce the error norm by a factor of 100. For this identity example only, the curvature-scaled step with learning rate d/2 reaches the solution in one exact-arithmetic step.

This is a mathematical counterexample to the intuition that many iterations necessarily mean a difficult inverse. It is not evidence that Agent 4 used the wrong step size. The real Jacobian, clipping, scheduler and embedding scales can change the dynamics dramatically. In particular, blindly multiplying a nonlinear solver's learning rate by 1,024 would be unjustified.

The practical diagnostic is to inspect the actual update norm relative to the current distance between vocabulary embeddings and the observed residual decrease. If many steps are tiny relative to useful token changes, test curvature-scaled or line-searched steps. Do not infer progress from a decreasing mean loss alone.

## 5. First algorithmic rescue: discrete proposals with a trust-region model

The main target is a token identity, not a highly accurate continuous vector. Let z be the current search point, r=F_pi(z)-h, and let g be the gradient of the chosen matching loss with respect to z. A simple candidate surrogate is:

    q(v) = gᵀ(e_v-z) + (lambda/2) ||e_v-z||²

The first term estimates which token substitution decreases the activation mismatch. The second discourages a jump so far away that the local approximation is unreliable. Relative to v-independent constants:

    q(v) = e_vᵀ(g-lambda z) + (lambda/2)||e_v||²

Thus a vocabulary-wide proposal ranking needs matrix–vector scoring, not a transformer forward for every token. Forward-check a small batch of the best candidates through the same current-token prefix map, retain the best actually evaluated token, and update the local search according to predicted versus achieved improvement.

This is inspired by gradient-based discrete substitution methods such as HotFlip, applied here to minimizing activation mismatch rather than maximizing a text-classification loss. [6] There is an important non-novelty qualification: with scalar lambda, the surrogate ranking is algebraically equivalent to nearest-token projection after a scaled gradient step. The expression alone is not a new algorithm. The meaningful change relative to a long proxy-search loop is the bounded, real-token-centered search schedule, local step validation, batched checking and explicit best-discrete-candidate retention.

A possible bounded configuration is a few local rounds, each verifying 4–16 candidates, with a fixed maximum work allowance. The numbers are pilot choices, not empirically established optimal settings. Count all rejected trials and all vocabulary scoring. A useful result is more correct tokens per unit wall time, not fewer nominal optimization iterations.

The synthetic surrogate test confirms that the direct quadratic expression and matrix–vector implementation produce the same ordering on generated float64 inputs. It does not establish that the nonlinear forward model will obey that ordering; actual forward verification remains necessary.

Initialization must remain fitting-free. Random-token initialization is a baseline, while simple public-only starts such as a raw-embedding match may be screened under a common budget. Do not introduce B1 as an initializer and then label the final method prefix-only. A no-fit initial guess need not be accurate enough to serve as a final decoder; its role is only to put search in a more productive region.

## 6. Second algorithmic rescue: a small local inverse, not thousands of tiny steps

A local Jacobian describes how changes in the current embedding move the boundary activation. For a provisional step p:

    F_pi(z+p) approximately F_pi(z) + J p

A damped Gauss–Newton step solves:

    (JᵀJ + lambda I) p = -Jᵀ r

It tries to explain the entire residual jointly instead of taking many repeated steps along its steepest direction. Damping limits amplification along weakly determined directions. The step should be accepted or shortened using actual forward residual improvement, not assumed correct from the linear approximation.

Large Jacobians need not be materialized. Iterative least-squares or conjugate-gradient methods can use only products Jv and Jᵀu, and modern autodiff exposes these products. Trust-region nonlinear least-squares implementations use this principle. [7,8] The inspected software documentation supports the computational construction, not a claim that it accelerates this particular inverse.

For this pilot, a dense 2,048-by-2,048 Jacobian at every token is the wrong starting point. Its storage is not the main concern; repeatedly constructing it can require far too much differentiation. Limit the number of Jacobian products and compare their measured cost with ordinary forward candidate checks.

The best use may be occasional hard-position correction after the cheaper discrete proposal rule stalls, but the first experiment should evaluate it as a distinct arm so its contribution is interpretable. If its setup cost dominates, it is not a useful rescue even if it reduces iteration count.

### Why the geometry matters when selecting a token

Nearest-token selection in the raw input space is not always consistent with boundary matching. In the local model, the relevant quadratic metric is JᵀJ:

    ||F_pi(e_v)-h||² approximately ||r + J(e_v-z)||²

A simple exact linear counterexample in the included script uses J=diag(100,1), a search point z=(0.08,0.01), a true token embedding (0,1), and a wrong token (0.08,0.30). The wrong token is closer to z in input Euclidean distance, 0.29 versus approximately 0.993. But its boundary residual is approximately 8.03, versus zero for the true token. A Gauss–Newton solve gives the true continuous embedding exactly in this constructed linear case.

This is not a performance experiment and was deliberately constructed to expose the mismatch. It shows why a solver can improve its proxy yet make poor discrete decisions if the final metric ignores how the forward model stretches directions. It motivates curvature-aware proposals or local candidate rescoring, not a claim that the actual failed token has this cause.

## 7. The verifier must change when the forward model is imperfect

The upstream reference uses a coordinatewise closeness test and excludes tried candidates that fail it. Its theoretical finite enumeration is motivated by finding an exact or sufficiently separated match under the relevant fixed forward map. [2,4,9]

Under target drift or numerical-path mismatch, the true token may have nonzero residual. Consider two public candidate states 0 and 1, while the target emits 0.01 for token 0. An exactness-style tolerance of 1e-5 rejects both. Yet token 0 remains by far the closest candidate, and the displacement is much smaller than half the candidate separation. This analytical check illustrates why failure to pass a strict equality test is not evidence that a candidate should be discarded as impossible.

Retain the best actually evaluated discrete candidate even if it fails an acceptance threshold. A visited mask can prevent redundant computation; it should not erase that candidate from best-so-far output. Do not return a newly projected but unverified token at budget exhaustion.

A mismatch-aware score or stopping criterion does not solve operator uncertainty. The true token may not be the closest under a poor recovered prefix. Likewise, a large score margin among tested tokens does not certify that no omitted token is better. Stop rules must be developed on permitted public material and evaluated with their errors included.

Because the established A2 implementation uses cosine comparison, an angular or normalized-residual objective is worth examining when amplitude changes dominate. This should be a controlled metric comparison, not an unreported change layered onto a new optimizer. Normalization can discard useful norm information; it is not universally safer than Euclidean matching. The supplied review's separation-margin argument is only a sufficient condition under a specified metric and correct prefix. [1]

Use separate numeric and model-mismatch diagnostics. A high-precision search function compared with BF16 observations may never hit a coordinatewise equality threshold even with the correct weights. A dtype change also needs consistent rotary and normalization behavior. The previously reported loader correction should remain preserved as a historical fix, not repeated as a new scientific gain.

## 8. What the later inversion literature does and does not justify

The July 2026 GPT-2 study deliberately retained continuous optimization for long inner loops and used a larger final candidate window. It reported much slower operation than the released SIPIT reference on its shared comparison. Its high-accuracy configuration also differs in several budget choices and is not evidence that adding iterations will rescue the present pilot. [10]

Qu and colleagues emphasize structured embedding priors and constrained optimization, but their broader procedure and grey-box assumptions differ from immutable left-to-right reconstruction with unavailable live weights. [11] These papers support testing a better search formulation, not importing their reported accuracy into this project.

The key theoretical boundary remains discrete injectivity versus practical, transferable inversion. SipIt's existence/termination guarantee does not bound the number of candidate tests tightly enough to make it fast here, and it does not supply the unavailable target map. [9] A successful matched-public rescue must still be tested through an imperfect prefix, then a legitimate recovered prefix, before any moving-target claim.

A modest optimization fix is not automatically a new scientific mechanism. Conversely, an efficient implementation that removes the fitted inverse and retains useful accuracy can still be a meaningful practical result without claiming a novel theorem.

## 9. Recommended bounded experiment

### Phase A: one reconstruction and cost diagnosis

Agent 4 should work from its actual local corrected source, identifying its full commit and exact input/result bindings. Reproduce the reported 30-token comparison on the same opened inputs before changing the algorithm. Preserve the old output as historical evidence.

Break out time for the continuous gradient path, discrete verification, vocabulary ranking, prefix/cache work, logging/synchronization and one-off setup. Count per-token proposals and gradients, including all unsuccessful work. Identify whether one budget-exhausting token dominates the total. Classify the failed token without feeding evaluator truth into reconstruction.

This is a compact phase, not a new audit programme. If the current implementation already uses efficient prefix slicing, safe caches and matrix scoring, record that and proceed rather than rewriting it.

### Phase B: a decision-preserving executor improvement

Apply only demonstrated removable costs. Verify forward outputs, input gradients and discrete decisions against the corrected reference on opened fixtures. Test longer prefixes as well as the first position, since recomputation costs grow with prefix length. Keep precision changes separate from performance-preserving changes.

An execution-only result answers whether the 5.81x disadvantage was mainly implementation cost. It does not establish better search.

### Phase C: two small algorithmic challengers

Use the qualified current-token executor for all arms:

- the corrected original proposal/search schedule;
- bounded discrete trust-region proposals with small-batch verification;
- a limited damped Gauss–Newton/local-inverse variant, only if derivative checks and cost preflight pass.

Keep inputs, initialization information, numerical representation and total work accounting controlled. Fix a small public development panel; freeze the chosen method before a fresh modest natural panel. Include a separate unusual-token stress panel. Do not optimize solely around the one old failed token or count it as independent confirmation after tuning.

Success requires improved reconstruction per total cost at the declared scope. Record timeout/exhaustion as failures or explicit emitted guesses according to a predeclared rule; never remove them from the denominator. Compare against A1+A2 on the same inputs and, when available, the smaller-budget B1 hybrid as a separate practical competitor. The prefix-only method may not use those fitted models for initialization or fallback.

A matched-public method that remains dramatically slower and less accurate after these bounded changes should not automatically receive another solver sweep. If it becomes competitive, continue to an imperfect-prefix component test. The absence of a validated maintained recovery path limits that stage; it does not permit substituting actual target weights.

## 10. Claim and release boundaries

A useful rescue would remove the separately fitted token decoder without introducing a hidden learned proposal model. It would still use online numerical work and, in an adaptive system, model-prefix recovery. All those costs count.

Matched-public feasibility, static-surrogate transfer, supplied recovered-prefix results and cold-start closed-loop tracking are distinct claims. Report them separately. A supplied recovered prefix trained using another method's reconstructions does not demonstrate independent bootstrap.

No target-provided labels or correctness signals may influence candidate generation, routing or early stopping. The model-prefix approximation remains fixed while reconstructing a record. Information learned from a completed record may affect only later records under the existing charter.

The current request does not authorize publishing previously blocked local coordination files. Preserve local work and prepare a sanitized release list for the existing public-disclosure process. This report and its synthetic checks do not change repository state.

## Research judgment

The prefix-only direction deserves one more carefully designed round, not because near-correct recovery must be fast in principle, but because there are identifiable distinctions the first headline result cannot resolve: redundant execution, poor step scaling, disagreement between continuous and discrete geometry, and an exact-verification rule inappropriate for mismatch.

The highest-priority rescue is the cached current-token executor plus bounded discrete proposal-and-verification. Damped Gauss–Newton is the more substantial theoretical alternative, to be kept only if its reduction in search work exceeds its derivative cost. More uninterrupted optimization, a larger arbitrary candidate window, or a new trainable decoder would not be the desired rescue.

## Sources

1. **Token Reconstruction Research Review**, supplied strategy review, 8 September 2026. Relevant sections: 3.1–3.3 (information retention, numerical stability and operator uncertainty); 6 (a structurally different fitting-free inverse); 8 (numerical versus operator shift). User-supplied file; no public URL assumed. Its PR20-era claims are historical context, not Agent 4 execution evidence.
2. **SIPIT, `src/algorithm/SIPIT.py`**, official repository, pinned `820683156b7257313046a4fb3c492e52519525b7`. https://github.com/giorgosnikolaou/SIPIT/blob/820683156b7257313046a4fb3c492e52519525b7/src/algorithm/SIPIT.py
3. **SIPIT, `src/utils/model.py`**, same pinned source. https://github.com/giorgosnikolaou/SIPIT/blob/820683156b7257313046a4fb3c492e52519525b7/src/utils/model.py
4. **SIPIT, `src/algorithm/base.py`**, same pinned source. https://github.com/giorgosnikolaou/SIPIT/blob/820683156b7257313046a4fb3c492e52519525b7/src/algorithm/base.py
5. **Hugging Face Transformers: Caching**, versioned official documentation. https://huggingface.co/docs/transformers/v4.57.0/en/cache_explanation
6. Ebrahimi, J., Rao, A., Lowd, D., and Dou, D. **HotFlip: White-Box Adversarial Examples for Text Classification**. ACL 2018. https://aclanthology.org/P18-2006/
7. **SciPy `least_squares`**, official documentation: trust-region solvers, regularization and matrix-vector-only iterative least-squares solves. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
8. **PyTorch: Jacobians, Hessians, hvp, vhp, and more: composing function transforms**, official tutorial. https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html
9. Nikolaou, G., et al. **Language Models are Injective and Hence Invertible**, arXiv:2510.15511v4, revised 13 March 2026. https://arxiv.org/html/2510.15511v4
10. Słowikowski, M., and Majewski, M. W. **Recovering Input Text from Hidden States: Study of Gradient-Based Inversion of Decoder-Only Language Models**, arXiv:2607.00852v1, 1 July 2026. https://arxiv.org/html/2607.00852v1
11. Qu, W., et al. **Prompt Inversion Attack against Collaborative Inference of Large Language Models**, arXiv:2503.09022v3 / IEEE S&P 2025. https://arxiv.org/html/2503.09022v3

### Local original analyses

`analytical_checks.py`, `analytical_results.json`, and `analytical_run.txt` reproduce the algebraic identities, memory arithmetic and synthetic cache/gradient experiment described here. These generated artifacts are not external literature and do not contain a hidden Llama benchmark. The reported 29/30 and 30/30 GPU observations remain attributed to the supplied local Agent 4 completion summary.
