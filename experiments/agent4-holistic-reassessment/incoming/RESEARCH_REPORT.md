# Prefix-only inversion after the rescue study

## Executive judgment

The latest result justifies not deploying the tested configuration. It does not establish that prefix-only inversion is an unpromising research family. The defensible allocation decision is to stop unchanged runs, inspect the existing failure-and-cost evidence, and permit one different, tightly bounded experiment only when that evidence identifies a plausible route to substantially better reconstruction per unit of computation.

The supplied completion summary reports 85/92 natural tokens in 22.84 seconds for discrete search, versus 77/92 in 117.67 seconds for the original solver and 92/92 in 5.20 seconds for A1+A2. It also reports 46/46 stress tokens. These are reported results, not independently reproduced measurements. The corrected rescue commit, `ce04a8d`, remained unavailable through the connected GitHub lookup during this assessment; the library search located the earlier rescue proposal, not the completed rescue report or per-token traces. It is therefore not possible to attribute the seven errors or the runtime to a specific local implementation detail from the available evidence. [1, 2]

Three conclusions should be kept separate:

1. **Current configuration:** not competitive on the supplied natural-text accuracy/runtime comparison.
2. **Further routine tuning:** not justified automatically; the remaining runtime gap is large.
3. **Prefix-only reconstruction:** still scientifically open, with narrower opportunities that differ from simply increasing search budgets.

The strongest new opportunity is a small, jointly considered block of still-uncommitted tokens, using the boundary observations of the whole block. It could reduce serial work and prevent an early mistake from corrupting subsequent search. It is not yet an established inversion algorithm in this setting, and it needs a prospective scope amendment because the completed sequential pilot was restricted to the current position. A second possibility is a bounded curvature-aware correction on measured hard positions, but the previous brief already proposed that: its actual execution status must be checked before it is presented as an unexplored rescue.

The appropriate immediate action is an evidence-to-decision pass, not another open-ended solver campaign. Reuse the existing local report and saved traces wherever possible. A lack of access by an external reviewer should not cause the implementation agent to repeat completed diagnostics.

## 1. Evidence and scope

### 1.1 Available evidence

The reported rescue outcomes and completion status come from the supplied conversation summary. The previous `RESEARCH_REPORT.md` and `AGENT4_RESCUE_BRIEF.md` are available and describe what was recommended, including cached differentiable execution, real-token proposals, and optional damped Gauss–Newton steps. They do not prove that all proposed components were implemented or qualified in the completed rescue. [1, 2]

The supplied strategy review separates information retention, numerical stability, uncertainty about the producing model, and computational accessibility. That framework is useful here, but the review predates the rescue and is not evidence of the rescue's internal failure pattern. [3]

The public charter permits the current-record activation tensor, public/recovered model resources, and prior committed reconstructions, while excluding target labels and inaccessible target-prefix computation. It also restricts cross-record state updates. The published shortlist study says the inspected A2 path used static public-prefix weights; no validated maintained recovered-model-prefix path was available there. A token-prefix cache is not evidence of model-weight recovery. [4, 5]

### 1.2 Missing evidence that changes the decision

The completion summary does not supply the number and lengths of the natural source sequences, exact-sequence recovery, per-position runtimes, first-error locations, candidate inclusion for failed positions, or the disposition of the proposed curvature solver. It does not establish a new target-transfer or cold-start tracking result.

These are material gaps. For example, seven erroneous tokens may be seven independent search failures, or mostly follow one early wrong commitment. The same total runtime may arise from expensive work at every position or a few budget-exhausting positions. Those cases warrant different actions.

No unpublished task files were modified or published. A local WSL path is not an accessible remote document, and a failed GitHub commit lookup does not prove that the local work has been lost.

## 2. The measured progress and the remaining hurdle

The following calculations use the rounded values in the supplied completion summary. [1]

| Method | Natural correct tokens | Token accuracy | Reported total time |
|---|---:|---:|---:|
| Original solver | 77/92 | 83.70% | 117.67 s |
| Discrete-search rescue | 85/92 | 92.39% | 22.84 s |
| A1+A2 | 92/92 | 100% on this panel | 5.20 s |

Discrete search reduces the token-error count from 15 to seven and is 5.15 times faster than the original. Relative to A1+A2, it remains 4.39 times slower and less accurate on this panel. Tying the measured comparator requires another 77.23% reduction in total time, in addition to addressing the errors.

This is genuine algorithmic progress, but it does not imply that another equally large improvement is available. Nor does the present accuracy estimate justify token-independent confidence intervals: tokens in a reconstructed sequence are dependent, and the number of source records was not supplied.

### 2.1 Why aggregate runtime is insufficient

Let a fraction `f` of the current runtime be changed by an intervention, and let that part become `s` times faster. The conditional total is

`T_new = 22.84 × [(1 − f) + f/s]`.

This is an accounting identity under its assumptions, not a performance forecast.

| Fraction of runtime affected | Time if that part vanished | Speedup of that part required to reach 5.20 s |
|---|---:|---:|
| 40% | 13.70 s | Impossible through this part alone |
| 60% | 9.14 s | Impossible through this part alone |
| 80% | 4.57 s | About 28.9× |
| 90% | 2.28 s | About 7.05× |

This rules out a common unfounded recommendation: a faster nearest-neighbour operation cannot rescue the whole pipeline if that operation occupies too little of the runtime. Conversely, a dominant repeated search tail could leave substantial room for an algorithmic change.

Two deliberately constructed profiles illustrate the ambiguity. If 85 positions cost 0.02 seconds each and seven cost 3.02 seconds each, the total is 22.84 seconds and those seven consume 92.6% of runtime. Reducing them to 0.10 seconds each would bring the total to 2.40 seconds, before any other changes. If runtime is distributed uniformly over 92 positions instead, making seven positions free still leaves 21.10 seconds. Neither profile is asserted to describe the actual run.

### 2.2 The offline/online trade-off

The solver is dominated on the two reported axes, accuracy and online time. It is not automatically dominated on every resource axis: removing the separately fitted inverse is a real reduction in required preparation and learned state.

A full lifecycle comparison would use `C_prefix + N × t_prefix` against `C_A1A2 + N × t_A1A2`, with common recovery costs counted consistently. The relevant `C_A1A2` is the baseline's actual required inverse preparation, not the more expensive preparation of some other B1 model. Previously paid fitting costs are sunk for an existing deployment. No numerical lifecycle break-even is established here because compatible setup costs and a representative repeated-record latency are missing.

For an ongoing fine-tuning workflow, online cost accumulates, so the measured slowdown is a serious objection. But exactly matching 5.20 seconds is a planning anchor, not a law defining elegance. An acceptable bounded slowdown in return for eliminating a fitting phase would be a prospective application decision; it should not be invented after a result fails.

## 3. What the literature supports

### 3.1 Constructive invertibility is positive evidence, not a low-latency promise

Nikolaou and colleagues' SipIt supplies a fitting-free, forward-verification construction under its stated assumptions. Its bound is at most sequence length times vocabulary size verifier trials, relying on the relevant forward map and a uniquely identifying verifier. This supports continued interest in the family, but it does not guarantee a few cheap gradient steps, success under an imperfect recovered model, or a usable finite-precision tolerance. [6]

The finite-precision distinction is substantive: a model may retain enough discrete token information while a numerical objective or tolerance fails to identify it reliably. Work on exploding inverses likewise demonstrates that formal invertibility and stable numerical inversion are different properties. Its remedies for trainable invertible architectures are not directly transferable to an arbitrary pretrained target. [7]

### 3.2 The natural/stress discrepancy does not establish an implementation bug

Słowikowski and Majewski report that common space-prefixed tokens dominate errors in their GPT-2 study, and that per-token costs have easy and budget-exhausting modes. This makes dense common-token confusions and a difficult cost tail credible diagnostic hypotheses. Their short known-weight GPT-2 setting and substantial search budgets do not predict performance for cut-four Llama. Their faster/accuracy-oriented settings also alter several factors, so the headline percentages do not isolate a single repair. [8]

The 46/46 stress result from Agent 4 therefore need not contradict 85/92 on natural text. Unusual tokens can be easier to distinguish in a particular representation or easier under shorter contexts. Without matching length, position, and context, neither explanation is established for this run.

### 3.3 A real token is a constraint, not just a rounded continuous solution

Qu and colleagues combine constrained embedding optimization with discretization and activation calibration. This supports examining whether a continuous relaxation moves into representations that fit the activation but correspond poorly to discrete tokens. Their complete method includes additional semantic proposals and grey-box machinery, so its results are not a claim that the permitted prefix-only subset will achieve the same performance. [9]

### 3.4 Parallel search is a distinct algorithmic dimension

Jacobi-style and Lookahead decoding show that existing autoregressive models can perform useful work on provisional future tokens without training an extra draft model. These are generation algorithms, not validated hidden-state inversion algorithms. They motivate a scheduling and search-organization hypothesis rather than supplying a drop-in guarantee. Naive Jacobi iteration can also fail to improve wall time, so parallel proposals are not sufficient on their own. [10, 11]

A newer structured-Newton layer-parallel framework explicitly reports that off-the-shelf pretrained models are less amenable than models trained for the procedure. It should not be imported as evidence that a generic pretrained prefix can be inverted cheaply with a few identity-Jacobian updates. It is an example of a tempting apparent remedy whose training assumptions do not meet this task. [12]

### 3.5 Curvature tools exist; their benefit must be measured

Matrix-free trust-region least squares can use local Jacobian products without storing a dense matrix. PyTorch exposes the relevant automatic-differentiation transforms. These are implementation tools, not evidence of cheap token recovery. A derivative-based improvement must save more search time than its extra derivatives consume. [13, 14]

The previous rescue brief already named damped Gauss–Newton. The completed report must identify whether it was tested, disqualified, or left unattempted. Silence in the completion summary is not a negative result for it.

## 4. The key diagnosis: where does the real answer become inaccessible to this solver?

Read the existing traces first. The useful classification is:

**Forward/numerical incompatibility.** Under a known correct prefix, the real token does not obtain the expected best verification score or cannot meet the precision-consistent acceptance rule. A different optimizer cannot fix a verifier that systematically rewards the wrong answer.

**Proposal failure.** The correct token is never among the evaluated candidates under a correct reconstructed history. This points toward search geometry, initialization, or work allocation.

**Selection/stopping failure.** The correct token is evaluated and scores best, but a threshold, final return, tie rule, or exhaustion decision emits something else. Fix the decision path rather than designing a new inverse.

**Cascade failure.** The current token is missed mainly because an earlier commitment changed the context. Independent correct-prefix diagnostics may reveal that most later positions are easy. They are privileged diagnostics, not final reconstruction results.

**Uniform excessive work.** Most positions are already correctly recovered but require too much derivative or verification work. Speed requires changing the work done per position or parallelizing it, not only repairing the seven errors.

**Concentrated excessive work.** A few exhausted positions dominate latency. A specialized bounded correction or better commitment procedure may change both accuracy and total cost substantially.

After-scoring checks using the known public tokens can identify these cases. They must remain separated from deployed candidate selection. A diagnostic that forcibly inserts the correct token or supplies true preceding tokens cannot be advertised as an improved reconstruction system.

The source-level count of first errors is more useful than treating all seven token errors as independent evidence. The report should also provide exact source reconstruction; 85 correct positions spread across entirely imperfect records has a different practical meaning from a small number of localized failures.

## 5. The strongest distinct opportunity: a small pre-commit block

### 5.1 Hypothesis

Instead of making an irreversible token commitment as soon as its own activation has been considered, temporarily maintain a few candidate continuations for a short block. Score each candidate sequence against the observed boundary vectors for that block, then commit only after the declared verification rule is met.

The prefix model remains fixed; candidate tokens remain temporary. No true future token, B1 proposal, or external language-model completion is supplied. All information comes from the allowed activation tensor, the public/recovered prefix, and computation performed on provisional candidates.

There are two possible benefits, which should be measured separately:

- Later observed activations can help discriminate an earlier ambiguous token.
- Batched candidate sequences or active records can reduce serial overhead and improve hardware utilization.

It is not necessary for both effects to hold. A method that only increases total branching cost without improving either accuracy or useful runtime should be stopped.

### 5.2 Why future observations can help even though future tokens are unknown

A deliberately constructed causal model illustrates the possibility. Let tokens take values in `{0,1}`, and define

`F(x1,x2) = (x1, 2*x1 + x2)`.

The true sequence is `(0,1)`, but the observed first coordinate is perturbed: `h=(0.55,1)`.

A greedy decision using only the first observation picks `x1=1`. Given that commitment, it next chooses `x2=0`; the sequence is wrong. Evaluating the four possible two-token sequences against both observations instead selects the true `(0,1)`.

This does not use the true second token as an input to inference: both later possibilities are enumerated. It demonstrates that an additional observation can be useful after marginalizing over an unknown token.

However, unconstrained continuous inversion exactly fits the same observation at `(0.55,-0.10)`. Rounding that solution gives the wrong `(1,0)`. Thus the toy example also shows why simply optimizing a larger continuous block is not enough. Discrete token consistency must remain meaningful.

This is an original toy check, not a Llama result or a claim of novelty for sequence search.

### 5.3 Scope and limitations

The public charter allows the full activation tensor, but the completed Agent 4 sequential pilot deliberately used current-position observations. A block experiment is an explicit prospective change to that experimental scope. Final outputs must still be committed left-to-right without revising committed tokens, and recovered model weights must remain fixed within the record. Provisional alternatives must not be represented as already committed answers. [4]

The earlier full-record learned-decoder study is not a test of this forward-consistency search. Conversely, its result does not provide evidence that the new search will work.

Candidate combinations grow rapidly, and a wrong context can contaminate multiple parallel proposals. Use a small fixed block and capped candidate/beam budget, not an expanding whole-record search. Where an exactness guarantee depends on matched weights and numerical separation, keep it specific to that setting. Under target mismatch, self-consistency is not proof of truth.

### 5.4 What would make this worth running?

This is most attractive when the trace shows plausible correct candidates before an early wrong commitment, followed by many later failures or long searches. It is less attractive when the true token is never proposed even with a correct prefix, or when the cost is uniformly dominated by a necessary expensive local solve.

A bounded retrospective check of whether a short block distinguishes existing competing hypotheses can precede implementation. If no useful alternatives exist in the saved search and no new proposal mechanism is justified, there is no case for a large block-search experiment.

## 6. Other rescue ideas and their priority

### 6.1 Curvature-aware hard-position solver: conditional, not new by default

For `r(z)=F(z)-h`, a damped step solves

`(J^T J + lambda I) p = -J^T r`.

It is attractive when gradients repeatedly make tiny progress along a badly scaled direction or when nearby real tokens produce very different forward distances. First estimate actual derivative cost and numerical reliability. Use a tightly capped solve with real-token verification. Do not build a dense Jacobian per vocabulary entry or add many restarts.

If this was already qualified and found poor in the completed rescue, retain that result. It should not be repackaged as an unexplored theory.

### 6.2 Embedding-space constraints: useful only for the relevant failure mode

Keeping continuous candidates near legitimate public token vectors can reduce meaningless continuous solutions. It can also trap search near the wrong token or introduce expensive vocabulary checks. It is relevant when continuous residuals are small but discrete tokens are wrong, not a generic replacement for failure analysis. [9]

The existing discrete solver already stays on real tokens for parts of its procedure. A new constraint must therefore identify what unresolved relaxation it changes; merely renaming its existing projection is not a new experiment.

### 6.3 Homotopy and forward-model redesign: lower priority

Gradually changing an easy identity-like map into the full prefix is mathematically conceivable. It does not guarantee that the followed branch ends at the right token, and it adds solves. Training a specially invertible surrogate could make inversion easy, but also introduces a new approximation/training problem and may violate the intended absence of an auxiliary fitting phase. Neither has enough task-specific evidence to be the automatic next rescue.

### 6.4 More sources of gradients or a fitted initializer: outside this test

Actual target gradients, an additional known token prefix, or B1 initialization would change the access or simplicity question. They should not be introduced invisibly to make the prefix-only result look successful. The current goal is to delete the independently trained guesser.

## 7. Transfer remains a separate gate

The ideal matched prefix, an untouched public surrogate, a legitimately recovered prefix, and a cold-start recovering system are different evaluations.

For the correct prefix and candidate states `s_v`, a sufficient nearest-candidate condition is target displacement smaller than half the relevant separation, with the true token included. This condition is conservative and evaluator-side when it depends on unknown truth. Its failure does not imply a wrong decision. A large displacement orthogonal to all candidate differences can leave their ranking unchanged. [3, 6]

One can see the distinction algebraically:

`||h-s_v||^2 - ||h-s_u||^2 = ||s_v||^2 - ||s_u||^2 - 2*h^T*(s_v-s_u)`.

Only the relevant directional component affects this pairwise ordering. Projecting onto the exact span of candidate differences preserves Euclidean ordering; it is not, by itself, a better selector. A change of objective needs an independent reason, such as a qualified numerical scale effect, rather than an assumption that large residual magnitude implies failure.

Even a successful block or Newton solver must then be evaluated through a genuinely imperfect prefix. Matching hidden states under copied target weights does not establish the unavailable-prefix setting. Agent 3's published report records that a validated maintained recovery path was unavailable in its inspected resources. The phrase 'recovered-prefix-only' must not conceal this missing integration. [5]

In a true closed loop, each method must depend on its own reconstructions when updating the prefix. A supplied good recovered prefix is useful component evidence, but cannot establish that the method bootstraps without another fitted inverse.

## 8. Recommended decision process

### Stage A: inspect existing evidence, without reopening the study

Have the local owner read the completed report and traces. Reuse every existing measurement. Supply only missing items needed to identify the failure class, runtime concentration, and whether the previously suggested curvature option was actually tested.

The output is a short table: first errors, downstream errors, true-token candidate availability, best verified token at exhaustion, per-stage runtime, and difficult-position share. No new fitting, no private release, and no broad audit are needed.

### Stage B: a feasibility gate before a rescue implementation

Select at most one intervention from the diagnosed failure:

- A wrong acceptance/return path or numerical mismatch: correct that issue, preserve old outputs.
- A dominant isolated optimization tail: test a capped local-inverse correction only if not already rejected.
- Early ambiguity and cascades: test one small pre-commit block.
- Hardware underutilization across independent requests: test fair batching for both methods, distinguish throughput from latency.
- Uniformly expensive correctly functioning search with no demonstrated cheaper alternative: park this solver family for now.

Use the measured cost fractions to show a plausible route to the chosen budget. Do not assume an unlimited improvement of a small component can close a whole-system deficit.

### Stage C: one bounded matched comparison, then transfer only if warranted

Select algorithm choices on opened diagnostic material. After freezing, compare on a small new natural panel with explicit source counts, matched length/context stress pairs, and the same observation precision. Include timeouts and failed records. Report a quality/work/time frontier rather than one arbitrary iteration cap.

Preserve original and rescued solvers and the actual A1+A2 comparator. If Agent 3's smaller-budget hybrid has subsequently been validated, include it as a distinct stronger comparator, never as an initializer. Use comparable batch and timing boundaries, and document required fitting and maintained-prefix resources.

Only a useful matched result earns an imperfect-prefix or closed-loop continuation. No automatic grid, sample doubling, or repeated parameter rescue is authorized by a failed test.

## 9. Overall disposition

There is insufficient evidence to promote Agent 4's current method. There is also insufficient evidence to declare the prefix-only family practically hopeless.

The defensible decision is **stop the tested configuration, hold the broader direction at a diagnostic gate, and permit one qualitatively targeted follow-up only if the existing traces justify it**. The new block-consistency hypothesis is worth that gate; it is not ready for an unconditional full campaign.

This preserves the original objective—fewer independently learned components—without treating conceptual simplicity as a reason to ignore accuracy and online cost. It also avoids the opposite mistake: treating a small failed sequential search as a proof that all forward-model-only inversion is inefficient.

## Sources

1. Agent 4 completion summary supplied in the conversation: local rescue commit `ce04a8d`; natural results 77/92 in 117.67 s, 85/92 in 22.84 s, A1+A2 92/92 in 5.20 s; stress result 46/46. Full local report not retrieved; no external URL assigned.
2. Earlier handoff artifacts, `agent4_rescue/RESEARCH_REPORT.md` and `agent4_rescue/AGENT4_RESCUE_BRIEF.md`. They concern the earlier 29/30 pilot and the proposed rescue, not proof of every component executed in `ce04a8d`.
3. *Token Reconstruction Research Review*, supplied 8 September 2026 strategy review. Sections 3.1–3.3, 6 and 8. Historical research synthesis, not the latest rescue result.
4. Research charter, pinned project source: https://github.com/A-lan-Z/Token-Reconstruction-Research/blob/5bbc3bf42a81c814404cf84cb46d55f0d3418667/RESEARCH_CHARTER.md
5. Agent 3 shortlist report, pinned source: https://github.com/A-lan-Z/Token-Reconstruction-Research/blob/3dc1c8d148c002950463fbd96e965cf936e24adc/coordination/results/agent3-b1-small-budget-a2.md
6. Nikolaou et al. *Language Models are Injective and Hence Invertible*. ICLR 2026; arXiv v4, 13 March 2026. https://arxiv.org/html/2510.15511v4 ; venue record: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5f124a1979570872ff46a6de8fbac35c-Abstract-Conference.html
7. Behrmann et al. *Understanding and Mitigating Exploding Inverses in Invertible Neural Networks*. AISTATS 2021. https://proceedings.mlr.press/v130/behrmann21a.html
8. Słowikowski and Majewski. *Recovering Input Text from Hidden States: Study of Gradient-Based Inversion of Decoder-Only Language Models*. 1 July 2026. https://arxiv.org/html/2607.00852v1
9. Qu et al. *Prompt Inversion Attack against Collaborative Inference of Large Language Models*. IEEE S&P 2025; arXiv v3. https://arxiv.org/html/2503.09022v3
10. Santilli et al. *Accelerating Transformer Inference for Translation via Parallel Decoding*. ACL 2023. https://aclanthology.org/2023.acl-long.689/
11. Fu et al. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. 2024. https://arxiv.org/html/2402.02057v1
12. Han et al. *SNLP: Layer-Parallel Inference via Structured Newton Corrections*. 2026. https://arxiv.org/html/2605.17842v3
13. SciPy. *least_squares*, versioned official documentation. https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.optimize.least_squares.html
14. PyTorch. *Jacobians, Hessians, hvp, vhp, and more: composing function transforms*. https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html

## Reproducible analytical supplement

`analytical_checks.py` generates `analytical_results.json`. The checks reproduce the aggregate arithmetic, conditional runtime limits, two invented runtime profiles, the two-token causal/discrete counterexample, an orthogonal-drift example, and an exact projection-ranking identity. They are deliberately small mathematical constructions. They contain no Llama measurements, unavailable target weights, or empirical rescue claim.
