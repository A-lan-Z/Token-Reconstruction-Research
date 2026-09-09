# Agent 4: bounded prefix-only rescue

Discrete search recovered **85/92 natural-text tokens**, **2/4 exact clips**, in **22.840s**. A1+A2 recovered **92/92**, **4/4 exact clips**, in **5.203s**. Discrete search is 5.15 times faster than the original schedule on these natural clips, but costs 4.39 times A1+A2.

The predeclared advance gate is **NOT MET**: >=95% natural-token recovery, >=50% exact clips and <=2 times A1+A2 wall. **Stop this bounded configuration; do not advance to mismatched/recovered-prefix experiments.** This does not reject prefix-only inversion generally. No fitted proposer enters either challenger.

## Historical anchor and bottleneck

The historical corrected pilot remains29/30 in9.824s versus30/30 in1.688s. Original bugged CPU results remain separate and unchanged in PR29. Rescue starts from eac972712b655cbb3712b72641614ff41c907483, preserving corrected result commit76c30972aec91aa4aa9c22b2b68135e033fc3642 and scientific execution commit ae5275561a6dd74c7e07ecbe93f26648c887c693.

The new original-schedule replay reproduced every candidate order and final token:29/30 in10.899s versus30/30 in1.782s. Its only error, stress position15, was never proposed; earlier tokens were correct and no timeout/final-return bug occurred. The genuine full-prefix residual at this error is MSE 0. That position contributes17.2% of its clip, not a dominant single-position total.

A separately synchronized diagnostic measured5.117s continuous forward/backward,2.276s discrete verification,1.363s vocabulary ranking,0.040s table setup and0.030s committed-prefix tensor setup. The remaining1.839s combines optimizer/scalar reads/guards/loop overhead. These clocks are intrusive; only uninstrumented runs support speed ratios. The remainder is not an isolated synchronization measurement.

The local code already avoids per-token table clones, per-step full vocabulary difference tensors, unused suffix/head execution, token decoding and hot-loop garbage collection. Full-prefix recomputation and repeated scalar reads were actually present.

## Qualified executor and numerical limitation

Earlier layer K/V are detached constants. Each trial creates temporary cache state; only the chosen token commits once. Parameter-version changes fail closed until caches are invalidated and rebuilt. Raw embedding magnitudes, cut4, native FP32 rotary construction and BF16 observations are preserved.

Real Llama checks cover positions1,4,15,39,127. FP32 maximum relative forward/gradient errors are 1.73e-06/4.99e-06; BF16 maxima are 0.0166/0.0447. The cached arm is a **numerical execution variant**, not a byte-identical optimization. Trial caches remained unchanged. Batch8 checks differed from singleton execution, so batching was excluded for challengers. Native comparator batching remains separately qualified.

Cached Adam returns the same29/30 historical tokens but needs3840 checks/gradients and38.758s. All30 positions exhaust the unchanged strict verifier. This is a negative execution-control result; candidate trajectories and stopping changed. No threshold was loosened to make known answers pass.

## Methods and development

- Original: corrected Adam128, lr0.01, seed4401, raw Euclidean projection and full-prefix recomputation.
- Cached: identical Adam schedule/tolerance, current-token numerical executor.
- Discrete: one random public token start;8 rounds of8 proposals, singleton actual-forward verification and best-so-far retention. Half mean-squared activation-error gradients define a quadratic substitution surrogate. Initial trust radius is twice RMS raw embedding norm; damping adapts to actual versus predicted improvement. No fitted initialization or fallback.
- Inverse:4 local steps, each4 CG iterations plus curvature estimate, with matrix-free JVP/VJP, damping and actual-forward step acceptance. Math attention is used for derivatives; numerical differences are recorded. No dense Jacobian.
- A1+A2: supplied historical public Alpaca lens, stable native top512/first256 candidates, direct cosine and own committed tokens. Comparator only.

New search arms use cached row norms, vocabulary matrix-vector scoring, direct distances for32 shortlisted rows and token-ID tie breaking before selecting8. This still reads the entire vocabulary. Mean-loss normalization is included in curvature/radius scaling; Adam's rate was not blindly multiplied.

Two generated public prose clips and one identifier clip formed development: original/cached41/45, discrete40/45 (natural30/30, stress10/15), inverse0/45, A1+A2 45/45. All five outputs froze before scoring. Inverse took23.131s versus discrete12.088s and was stopped before final selection. Its adjoint check passed, but that did not ensure useful token proposals. The post-freeze failure diagnostic reproduces its first-position candidate order and records actual continuous-step improvements. No parameter sweep or second rescue followed.

## Final natural panel

Four unused task-local public-domain paragraphs,24 positions including BOS:

| Method | Tokens | Exact clips | Reconstruction s | Median / p95 s | Candidate forwards | Gradients | Vocab scans |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 77/92 | 0/4 | 117.673 | 29.962/30.243 | 11548 | 11546 | 11546 |
| cached | 78/92 | 0/4 | 121.662 | 30.421/30.717 | 11776 | 11776 | 11776 |
| discrete | 85/92 | 2/4 | 22.840 | 5.663/5.859 | 5980 | 736 | 736 |
| a1a2 | 92/92 | 4/4 | 5.203 | 1.235/1.496 | 23552 | 0 | 92 |

## Separate unusual-token panel

Two generated identifier clips,24 positions including BOS:

| Method | Tokens | Exact clips | Reconstruction s | Median / p95 s | Candidate forwards | Gradients | Vocab scans |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 41/46 | 0/2 | 62.224 | 31.112/31.970 | 5888 | 5888 | 5888 |
| cached | 41/46 | 0/2 | 60.274 | 30.137/30.296 | 5888 | 5888 | 5888 |
| discrete | 46/46 | 2/2 | 11.177 | 5.589/5.609 | 2990 | 368 | 368 |
| a1a2 | 46/46 | 2/2 | 2.534 | 1.267/1.273 | 11776 | 0 | 46 |

Every record/token remains in its denominator, including exhausted guesses or abstentions. Natural and stress are not pooled for the advance decision. The empirical frontier under token count, exact clips and runtime is {'ordinary': ['a1a2'], 'stress': ['a1a2']}; it describes this small panel only.

## Costs and verification

| Method | Load/setup s | Post-import process s | Peak allocated/reserved GiB | Peak host GiB |
|---|---:|---:|---:|---:|
| original | 0.516 | 180.955 | 2.917/2.932 | 2.084 |
| cached | 0.515 | 182.976 | 2.917/2.926 | 2.054 |
| discrete | 0.480 | 34.794 | 2.917/2.924 | 2.013 |
| a1a2 | 0.635 | 8.866 | 2.010/2.932 | 2.062 |

Reconstruction includes table preparation, all trial forwards, derivatives, cache commits, trust control, synchronization and Python work. Cache arms additionally execute one commit per emitted token including BOS. Full-prefix arms redo earlier positions on every trial. Inverse traces separately count JVP/VJP products; loss-gradient VJPs are included in VJP totals and must not be counted twice.

Setup/process clocks start after imports; the outer watchdog includes interpreter/import time. Process clocks include prediction serialization. One public forward warmup is shared; optimizer/backward startup is charged to reconstruction. This is not a steady-state benchmark. Historical lens fitting is supplied and not remeasured: this is a supplied-prefix component comparison, not cold-start cost.

Hardware: RTX5080,16GiB, two CPU threads. Largest-workspace qualification included127 prior positions, derivatives and full FP32 vocabulary scoring. Limits:6GiB reserved GPU, >=2GiB free GPU, >=8GiB host available, <=10GiB child host RSS, <80C, <=1800s jobs. Every matrix guard passed. Actual independent prefix/tokenizer/lens copies and fresh required-dependency extraction were verified before final capture; Python-S with no global site-packages reproduced a prefix output exactly. Interpreter/stdlib, system libraries and driver remain external.

Four new cache/return invariant tests and six supplied analytical checks passed. Supplied synthetic checks demonstrate algebra only; real Llama qualification and performance receipts are separate.

## Scope and release

Only BOS, reconstructed earlier tokens, current activation and fixed public assets inform search. No model update occurs within a record. Capture, prediction and score are separate processes on one account, not cryptographic isolation. Final methods froze before source selection; all four outputs froze before truth opening. Source byte hashes, paragraph indices and seed947201 are retained. No repository-wide unseen-text or pretraining-disjointness claim is made.

Public model: Llama-3.2-1B-Instruct revision9213176726f574b556790deb65791e0c5aa438b6, blocks0-3. No actual target prefix substitutes for recovered weights. No qualified Agent3 hybrid was integrated. GPU coordination/release is documented; no other task files changed.

Canonical dual-benchmark matrix: **NOT RUN / COMPARISON INCOMPLETE**. No P03, paid compute, target training, PR merge or global registry change. No recovered-prefix or cold-start result.

Result: coordination/results/agent4-prefix-only-rescue.md.
Manifest: experiments/agent4-prefix-only-rescue/manifest.json.
Reproduction: experiments/agent4-prefix-only-rescue/REPRODUCE.md.
Brief: coordination/requests/agent4-prefix-only-rescue.md; supplied background is preserved under incoming. Its inaccessible-commit statement is historical; PR29 is now public. Recommendations were tested, not treated as measured local findings.

New public release is not inferred from the pilot approval. A sanitized package/list is prepared locally. Supplied documents, coordination data, absolute runtime paths, actual model/dependency assets and evaluator truth are excluded. No rescue branch was pushed or rescue PR created.
