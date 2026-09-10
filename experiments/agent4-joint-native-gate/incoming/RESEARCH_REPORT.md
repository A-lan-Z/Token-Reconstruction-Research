# Joint-activation proposal generation

## Executive result

A fitting-free, executable rolling-window proposal algorithm was implemented and tested. It uses multiple already-observed activations to propose new vocabulary tokens, rather than merely recombining candidates from the old per-token search. Future hypotheses remain discrete vocabulary tokens; the algorithm never receives their true labels.

The mechanism produced a positive but limited result. In a controlled proposal-only diagnostic, joint-observation gradients proposed the correct token at 3 of 29 first-error positions where the local gradient's two proposals omitted it. However, a wider ordinary sequential search was both more accurate and cheaper than the joint prototype on the common synthetic comparison. This design therefore does not yet supply the quality-and-cost case required for a large native Llama experiment.

The deliverables are an explicit algorithm, a mathematical limitation on unconstrained future-embedding optimization, runnable CPU code, retained experimental stages, and a narrowly scoped native qualification brief. No new Llama reconstruction result, actual Agent 4 rescue, or moving-target performance claim is established.

## Evidence and scope

The existing programme reports 85/92 natural tokens in 22.84 seconds for its rescued prefix-only solver, versus 92/92 in 5.20 seconds for A1+A2. Its latest reassessment attributes seven errors to two missing root proposals and five downstream errors, with 6.8% of runtime at wrong-output positions. Those figures are supplied completion summaries, not independently reproduced experiments here. The local corrected/reassessment implementations were not available in the active working directory. This prototype does not purport to replace or exactly replay them.

The supplied strategy review distinguishes information retention, numerical stability, uncertainty about the forward operator, and computational accessibility [S1]. Its prior-art and experimental cautions remain applicable. The present question is narrower: can later observed activations help generate a missing root token cheaply enough to matter?

No learned inverse, B0/B1 checkpoint, external language model, natural-language prior, offline training, or private target weights were used in these tests. The test transformer is a randomly initialized synthetic forward function. It is not a pretrained small language model and is not a validated stand-in for Llama.

The repository charter permits the full current-record activation tensor while keeping source truth unavailable and final decisions immutable [S2]. The prototype changes the previous current-position-only experimental scope. It does not revise committed tokens, update prefix weights during a record, or learn from current-record truth.

## 1. The proposed mechanism

### Intuition

An uncertain token can influence several observed activations. Rather than decide it using its own activation alone, retain a small block of provisional real tokens and ask which token substitutions make the whole observed block more consistent with the available forward model.

This is different from the rejected fixed-list block reranker. The candidate generator computes a new gradient using later observations and searches the entire vocabulary for substitutions. A token absent from an earlier shortlist can therefore enter a new shortlist.

The method is called discrete rolling-window joint search here as a descriptive label, not a novelty claim. Its components combine ordinary gradient-guided discrete substitution, bounded beam/coordinate search, forward verification, and cached causal execution. Gradient-based token substitutions have clear precedent [S3]; model-only parallel generation has related scheduling ideas but different correctness requirements [S4].

### State and inputs

At position i, the algorithm has a committed reconstructed prefix pi, the fixed public or legitimately recovered forward model F, its actual token embedding table E, and the already-observed activation window H[i:i+w]. Earlier prefix keys and values are cached as constants. The candidate window length is at most three.

The search state is a beam of at most two discrete token blocks. Every entry is a real vocabulary ID. Each initial block comes from a cheap raw activation/embedding nearest-neighbour rule, or the uncommitted suffix carried from the preceding window. This raw initializer is counted and is not asserted to be accurate. No fitted lens or correct future-token identity is supplied.

The causal forward model is evaluated with the actual input embeddings, not substituted normalized readout vectors. Both model parameters and the committed token cache stay fixed while alternatives are considered.

### Objective

For block b=(b_0,...,b_{w-1}), define

    L(b) = 1/2 sum_j ||F_pi(E[b])_j - H_j||^2 / max(||H_j||^2, epsilon).

The scale for each observed row is computed from the observation and then held fixed. It is not a correctness signal. This is the prototype's declared matching objective, not an assertion that it is optimal under target mismatch.

Differentiate this loss with respect to the provisional block embeddings. The current position's gradient includes its contribution to its own and later observed rows. This gradient is computed through the supplied forward model; it is not a target-provided backward gradient.

### Proposal rule

For a current embedding z and gradient g, rank token vectors using

    q(v) = g^T (e_v-z) + (lambda/2) ||e_v-z||^2.

The implementation obtains the same ordering by nearest-token projection after a bounded gradient step. Its radius is 0.75 times the median vocabulary embedding norm, fixed before the synthetic confirmation cohort. It is a simple development choice, not a universal theoretically optimal value.

With scalar lambda this is a proximal gradient/nearest-neighbour reparameterization, not a new inverse theorem. The substantive intervention is the use of gradients from multiple observed rows.

For each block in the beam, propose up to two replacements at each position and one simultaneous best-replacement block. Include the incumbent blocks, deduplicate candidates, evaluate every candidate through the actual forward model, and retain the best two verified blocks. There are at most four search rounds per window.

At beam M, width w and p per-position proposals, at most M(2+wp) blocks enter a verification batch before deduplication. With M=2,w=3,p=2 the upper limit is 16 blocks, or 48 newly transformed token positions per verification round. This bound exposes why small candidate counts do not automatically imply small total computation.

### Commitment and cache behaviour

After the finite search budget, commit only the first token and reuse the best provisional suffix at the next window. In the second prototype stage, a block whose actual matched-model normalized loss is below 1e-20 may be committed in full, in left-to-right order. All methods receive the corresponding no-extra-backward early-match check.

That extremely tight threshold is an FP64 matched-synthetic setting, not a deployment certificate. Under approximate weights or noisy observations a correct block may not meet it. Any native implementation needs an independently qualified numerical rule and must not claim exact correctness from a tiny surrogate residual.

Trial caches never modify the committed cache. Commit work is counted explicitly. The prototype re-evaluates the chosen committed tokens when updating the cache, rather than hiding that cost or reusing inconsistent speculative state.

## 2. Why unconstrained future vectors are dangerous

Later activations do not automatically constitute additional identifying equations if every future embedding is an unconstrained variable.

Linearize a two-part causal window:

    r_current + J_current delta_z,
    r_future + A delta_z + B delta_u.

Here delta_z changes the root embedding, while delta_u changes the future embeddings. If B is invertible, then for every delta_z there is a delta_u satisfying

    delta_u = -B^{-1}(r_future + A delta_z).

That choice cancels the entire future residual. After minimizing over unconstrained future vectors, the remaining local objective only constrains the current residual. In a square multi-position causal map, B is block lower triangular; nonsingular diagonal blocks are sufficient for this cancellation construction.

The corresponding projected cross-token Jacobian is (I-BB^+)A, which is zero in the nonsingular square case. A random eight-dimensional numerical check gave residual norm 4.44e-15 and projected cross-token norm 7.00e-15.

This is an elementary local linear-algebra result, not a claim that every transformer Jacobian has these properties or that later discrete token observations never carry useful information. It does explain why simply adding freely adjustable future embeddings can remove the very extra constraints the method was supposed to exploit.

### A discrete constructive example

Use vocabulary embeddings {-1,0,1} and

    F(z,u)=(z^3, 2z+u).

The true pair (1,0) yields observation (1,2). Starting from root z=0, the current-only squared-error gradient is zero. For every possible real-token future hypothesis u in {-1,0,1}, the joint root gradient is negative: -6,-4,-2. The fixed proximal rule therefore proposes root +1, which the local rule omits. All candidate tails are considered as hypotheses; no true future token is supplied.

An unconstrained future variable could instead take u=2 at root zero, cancelling the second residual and leaving the root gradient zero. Discreteness matters.

This favourable example does not generalize automatically. With weaker coupling F(z,u)=(z^3,z+u), a legal u=1 can explain the later observation while the wrong root remains zero. Thus even discrete future variables can sometimes hide the error. The existence of a helpful example is not a sufficiency theorem.

## 3. Executed prototype experiments

### Synthetic forward model

The prototype uses four random pre-normalized causal attention/MLP blocks, width 32, four attention heads, and 128 random vocabulary vectors. It includes residual paths and a small absolute positional term. It does not include Llama's RoPE, RMSNorm, grouped-query attention, pretrained weights, BF16 numerical path, vocabulary size, or target-update trajectory.

Each sequence contains a known BOS and eight unknown tokens sampled without using an external language model. All inversion runs are given only observations, model resources and their own earlier commitments. Ground-truth token IDs are used by the evaluator after candidate generation/prediction.

The model cache and current-window derivative implementation were verified against uncached execution: maximum output error 1.33e-15, maximum gradient error 2.22e-15, with the committed cache unchanged. These validate this toy's mathematics only.

### First fixed cohort

The initial plan was written before the first benchmark. One development example preceded random model seeds 100,101,102 with 16 sequences per seed: 48 sequences and 384 unknown tokens.

Three arms were evaluated:

- Narrow sequential: one position, one beam, two proposals per step.
- Block-verification control: width three, but each token's proposals use only that token's own observation gradient; block verification is still joint.
- Joint: the same block engine, using later-observation contributions to earlier-token proposals.

For the mechanism comparison, the initial block implementation computed individual observation gradients, so both block arms had the same type of gradient-decomposition machinery. This is more expensive than the production joint gradient, which needs one reverse pass through the summed loss.

| First cohort | Correct tokens | Exact sequences | CPU seconds |
|---|---:|---:|---:|
| Narrow sequential | 369/384 | 36/48 | 1.246 |
| Block verification, local proposals | 372/384 | 38/48 | 2.424 |
| Joint proposals | 379/384 | 44/48 | 2.306 |

Relative to the narrow sequential solver, joint gained ten tokens and eight exact sequences without a regression on this cohort. Relative to the block-verification control it gained seven tokens and six exact sequences. This supports an information effect in this construction, not an efficient Llama method.

The additional work was substantial. This prompted a documented implementation amendment, not silent tuning of the original result.

### Second fixed cohort after the implementation amendment

The amendment replaced the joint-gradient decomposition with one summed-loss reverse pass, added an early matched-block check, and permitted complete matched-block commitment. Window, beam, proposal width and round cap remained fixed. New model seeds 200,201,202 supplied another 48 sequences and 384 tokens. These are independent from the first model/record cohort, but are still only one synthetic architecture recipe.

| Second cohort | Correct tokens | Exact sequences | CPU seconds |
|---|---:|---:|---:|
| Optimized narrow sequential | 340/384 | 19/48 | 0.982 |
| Block verification, local proposals | 348/384 | 23/48 | 2.183 |
| Optimized joint proposals | 360/384 | 27/48 | 1.351 |

The joint design still improved recovery, but had not established a cost advantage. The block-verification ablation requires extra reverse decompositions only to isolate cross-position information; its runtime is not an appropriate production acceleration target. The sequential control is the relevant simpler cost competitor.

### Stronger cost control

After seeing the second-cohort work counts, a single wider sequential control was added: sixteen proposals per step, keeping the same initializer, objective, round cap, cache and stopping scheme. This comparison is retrospective; the choice of sixteen was recorded before execution, was not selected from a correctness sweep, and is not labelled a new independent confirmation.

| Same second-cohort sources | Correct tokens | Exact sequences | CPU seconds |
|---|---:|---:|---:|
| Optimized joint, two proposals per block position | 360/384 | 27/48 | 1.351 |
| Wider ordinary sequential, sixteen proposals | 382/384 | 46/48 | 0.999 |

This control defeats the current practical case for the joint design. Additional candidate width in the simpler solver did more useful work than introducing the joint window.

CPU timings are small, local implementation observations. They are not GPU forecasts or a comparison with the actual A1+A2 implementation. The work ledger independently points in the same direction:

| Count over the second cohort | Joint | Wider sequential |
|---|---:|---:|
| New positions processed in forward calls | 7,585 | 2,927 |
| Reverse-pass position count | 1,631 | 131 |
| Vocabulary row scores | 257,920 | 65,920 |
| Illustrative F+2B position proxy | 10,847 | 3,189 |

The proxy treats one reverse position as two forward positions. That is a stated sensitivity assumption, not measured FLOPs or an architecture-independent cost law. Window attention, kernel utilization, cache length, launch overhead and vocabulary size also matter. The direct counts are retained so other cost assumptions can be used.

### Proposal-only mechanism probe

A retrospective diagnostic recomputed local and joint root proposals from identical provisional blocks and the same baseline-produced prefix at each position. Labels were not supplied to either proposal generator. Correctness categories were attached after the candidates were generated.

| Subset | Positions | Local top-2 contains truth | Joint top-2 contains truth | Joint-only / local-only |
|---|---:|---:|---:|---:|
| All positions | 384 | 328 | 330 | 5 / 3 |
| Correct earlier baseline prefix | 253 | 215 | 216 | 4 / 3 |
| Baseline first-error roots | 29 | 0 | 3 | 3 / 0 |

This directly verifies that joint gradients can generate a root token absent from the local proposals. It also records harm: later observations can remove a useful proposal at other positions. The three rescued roots are not Agent 4's two actual missed roots and cannot be represented as a rescue of those cases.

## 4. Cost interpretation

At width three and beam two, the algorithm differentiates through several provisional tokens and verifies multiple complete blocks. Parallel batching can reduce launches and improve utilization, but it does not eliminate those transformed positions or vocabulary products.

The number of proposals per root is an inadequate cost measure. A forward candidate block of three tokens is not equivalent to one current-token check. Carrying a suffix may amortize some work, especially when a complete block is verified, but under imperfect model weights the exact-match shortcut may rarely fire.

The original natural-text solver needed about a 77.2% total time reduction to match its reported A1+A2 reference. Nothing in the current synthetic results supplies such a reduction. Even the optimized joint variant is more expensive than the narrower solver, and the stronger sequential control is better on both quality and measured work.

The prototype therefore meets one of the prerequisites discussed in the prior review—genuinely different proposal information—but not the other: a credible reduction in the bulk of computation.

## 5. Relationship to existing literature

SipIt supplies a fitting-free sequential inverse and distinguishes gradient-guided proposals from discrete verification [S5]. Its exact-model construction does not make a bounded, imperfect-prefix joint search correct or fast.

HotFlip supplies precedent for vocabulary-wide gradient-ranked token substitutions [S3]. Replacing its task loss with an observed-activation consistency objective and batching window proposals does not by itself establish a novel theoretical contribution.

Parallel autoregressive decoding supplies a scheduling analogy for provisional states and verification [S4]. It is a generation method; its guarantees and speedups are not imported here.

Prompt-inversion work already investigates whole-prompt continuous optimization and discretization [S6]. The meaningful distinctions here are discrete provisional tokens, a finite small-window search, no fitted token inverse, causal committed state, and explicit cost controls. This study does not assert those distinctions are unprecedented across the entire literature.

The main original analytical observation is the continuous-nuisance cancellation in a causal square local model; the main original empirical result is the executable synthetic comparison and its negative cost control. Neither is a successful real-model inversion claim.

## 6. Recommended disposition

Do not commission a full native reconstruction campaign on the strength of these toy gains. The instantiated algorithm fails its own stronger cost control. No existing agent's frozen experiment, model, output or publication permissions should be changed by this report.

A short native qualification is the only defensible possible follow-on: determine whether joint observations promote actual previously missed root tokens more effectively than spending the same work on wider local search. It should operate on already-opened diagnostic positions, have no new target training or recovered-prefix framework, and stop before full evaluation unless the actual proposal-versus-cost comparison is favourable.

That qualification is not required simply because a brief has been written. The supplied `AGENT4_NATIVE_GATE.md` states its conditions and limits. The default scientific conclusion of this report is not "rescue achieved"; it is "a mechanism exists, but this implementation has not earned the cost of a large run."

A poor synthetic cost comparison does not prove that a native Llama implementation must fail: learned geometry, vocabulary size and hardware may change the trade-off. Conversely, those differences do not justify assuming it will improve. Native evidence must establish the required trade-off rather than appeal to the possibility of better batching.

## 7. Reproducibility and limitations

Eight local unit tests passed. They cover cache and gradient equality, continuous-future cancellation, new-root proposals in the constructed example, a zero-cross-channel negative control, discrete output validity, equality of summed/decomposed gradients, unchanged model parameters, and monotone best verified block scores.

All code, traces, output tokens, work counters, plans, amendments and failed-to-justify comparisons are retained. `results_v1.json`, `results_v2.json`, `results_cost_control.json` and `proposal_probe.json` have different evidence roles. They must not be pooled into a single unseen-test claim.

The synthetic target family is untrained and small. Only matched model weights and FP64 observations were tested. There are no natural-text results, uncertainty estimates over real-language populations, target-update experiments, BF16 qualifications, recovery updates, or cold-start tracking results. CPU timing did not emulate GPU execution. All final outcomes remain a mechanistic/engineering prototype.

## Sources

[S1] Token_Reconstruction_Research_Review.docx. Supplied strategy review, 8 September 2026. Relevant sections: four-way inverse distinction; candidate separation; hypothesis-specific interpretations; whole-clip cost. Historical PR20-era evidence, not a source for the new prototype measurements.

[S2] Token-Reconstruction-Research, RESEARCH_CHARTER.md, published commit 5bbc3bf42a81c814404cf84cb46d55f0d3418667. https://github.com/A-lan-Z/Token-Reconstruction-Research/blob/5bbc3bf42a81c814404cf84cb46d55f0d3418667/RESEARCH_CHARTER.md

[S3] Ebrahimi, J., Rao, A., Lowd, D., and Dou, D. HotFlip: White-Box Adversarial Examples for Text Classification. ACL 2018. https://aclanthology.org/P18-2006/

[S4] Santilli, A., et al. Accelerating Transformer Inference for Translation via Parallel Decoding. ACL 2023. https://aclanthology.org/2023.acl-long.689/

[S5] Nikolaou, G., et al. Language Models are Injective and Hence Invertible. arXiv:2510.15511v4, 13 March 2026; ICLR 2026. https://arxiv.org/html/2510.15511v4

[S6] Qu, W., et al. Prompt Inversion Attack against Collaborative Inference of Large Language Models. arXiv:2503.09022v3 / IEEE S&P 2025. https://arxiv.org/html/2503.09022v3

Primary literature was checked through its publisher/arXiv text. The numerical outcomes in this report are from the included local scripts, not derived from those publications.
