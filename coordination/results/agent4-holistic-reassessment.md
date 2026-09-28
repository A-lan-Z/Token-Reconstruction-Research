# Agent 4: holistic reassessment

**Decision: keep the current configuration stopped; no further solver intervention is presently justified.** The evidence identifies two root proposal failures and five downstream context-dependent errors, alongside broadly uniform computation. Neither suggested intervention passes its evidence/cost gate. This is a bounded allocation decision, not a rejection of prefix-only inversion as a method family.

## Preserved result and error structure

Verified base: ce04a8dfb11452407df94468964723c6294b8933. The existing result remains **85/92 natural tokens, 2/4 exact clips, 22.8397s**, versus A1+A2 **92/92, 4/4, 5.2026s**. The separate stress result remains46/46 and2/2 exact. Each clip has24 positions including BOS in both groups; different text/context distributions remain, so the contrast does not establish that unusual tokens are inherently easier. No original source or output was changed.

| Record | Correct /23 | First error | Later errors |
|---|---:|---:|---|
| natural_11_0 | 23 | none | none |
| natural_11_1 | 22 | 9 | none |
| natural_84_0 | 23 | none | none |
| natural_84_1 | 17 | 18 | 19, 20, 21, 22, 23 |

Both first errors are **proposal omissions under already-correct reconstructed histories**. Neither genuine token appears among its65 tested candidates. Five later errors follow the position18 commitment. Two of those later positions did propose the true token, but its score was worse under the incorrect context; this is not a best-return bug. Every saved return equals the best actually evaluated candidate.

## Only the missing diagnostic was replayed

Two already-opened records (natural_11_0 and natural_84_1) were replayed for stage timings; every candidate ID, candidate score and final token matched saved evidence exactly. Seven error positions received privileged true-prefix and forced-genuine checks. These are **after-score diagnostics, not reconstruction results**. No true-prefix information entered a deployment method or a new confirmation panel.

| Error position | True token proposed with correct prefix? | Selected with correct prefix? | Forced genuine MSE |
|---|---|---|---:|
| natural_11_1:9 | False | False | 6.84e-07 |
| natural_84_1:18 | False | False | 9.6e-07 |
| natural_84_1:19 | True | True | 1.05e-06 |
| natural_84_1:20 | True | True | 9.7e-07 |
| natural_84_1:21 | True | True | 6.6e-07 |
| natural_84_1:22 | True | True | 1.08e-06 |
| natural_84_1:23 | True | True | 9.07e-07 |

All five downstream positions recover under the privileged correct context. Both root positions still omit the true token. Their actual forced-genuine MSE is roughly1e-6, far below the chosen wrong scores0.00156 and0.00731. No lower bound on recoverability follows from those omissions.

All seven forced genuine candidates fail the unchanged strict allclose tolerance, despite small residuals. This confirms the already-known BF16 cache/full-capture numerical mismatch. Best-so-far retention remains correct, so acceptance failure causes continued work rather than erasing a correct candidate. Loosening a tolerance cannot create the two missing root candidates; a new stopping rule requires independent precision/false-acceptance qualification. No threshold was tuned until an opened answer passed, and no unqualified repair is presented as a new inverse.

## Quantitative cost gate

All92 natural positions executed65 candidate checks,8 gradients and8 vocabulary scans. The seven wrong positions consumed **1.5616s,6.84%** of total wall. Even making them free leaves **21.2781s**. Making the seven slowest positions free leaves **21.0113s**. Median per-position time is0.2183s, maximum0.4270s; there is no dominant expensive error tail.

Matching A1+A2's historical engineering reference requires **77.22%** further total-time reduction.

| Measured component | Replay seconds | Replay share | Conditional natural-panel time if free |
|---|---:|---:|---:|
| table setup | 0.094 | 0.8% | 22.66s |
| initial cache commit | 0.413 | 3.5% | 22.04s |
| candidate forwards verification bookkeeping | 6.954 | 59.2% | 9.31s |
| input forward and gradient | 1.991 | 17.0% | 18.97s |
| vocabulary ranking and shortlist | 0.816 | 7.0% | 21.25s |
| token cache commit | 0.092 | 0.8% | 22.66s |
| unattributed optimizer guard scalar and loop | 1.383 | 11.8% | 20.15s |

These fractions were measured on two instrumented replays, not on a newly rerun92-token benchmark. The conditional floors apply T*((1-f)+f/s) with s tending to infinity and assume the sample fraction represents the original panel. They are accounting bounds under that assumption, not new performance forecasts. Phase wrappers synchronize GPU work and include associated Python/scalar bookkeeping; candidate forwards are not isolated pure kernel time.

The replay records took6.324s and5.419s, versus their historical5.892s and5.654s. Initial cache setup includes a0.411s first-record startup cost but only0.002s on the other record. The retained decision JSON gives both per-record fraction ranges. Even the larger measured candidate-forward fraction,63.1%, is below the77.2% total reduction needed; vocabulary scoring and gradients alone are much smaller targets. Faster ranking alone cannot close the gap.

The original comparison uses the same hardware, inputs and post-load reconstruction timing. A1+A2's existing native candidate batching is retained; challenger batching had failed numerical equivalence. This is sequential-record latency evidence, not a multi-record throughput claim. A fair new throughput study would need both methods batched and requalified; no such speedup is assumed here. Historical A1 fitting is supplied/sunk and was not remeasured. Model-prefix maintenance is absent from both component runs; no numerical lifecycle break-even or cold-start claim is available.

## Intervention decisions

**Short-block verification: do not implement.** The cascade makes context relevant, but both roots fail the required candidate-availability check. A beam built from the saved candidates has no correct first token to retain; joint scoring cannot select a missing candidate. A new proposal mechanism conditioned on later observations remains charter-permitted, but its useful candidates and added-work savings are not demonstrated. Correcting seven failures alone also does not address the uniform runtime deficit. No privileged true candidate was inserted into a proposed block experiment.

**Bounded curvature correction: do not repeat.** Damped Gauss-Newton was already implemented, not merely proposed:4 outer steps,4 CG products per step plus curvature estimate, matrix-free JVP/VJP, actual-forward acceptance. It passed real derivative/resource checks, then scored0/45 development tokens in23.131s versus discrete40/45 in12.088s. Its prior diagnostic showed continuous loss falling from0.01164 to0.00211 while useful tokens remained absent. No evidence here invalidates that bounded negative result.

Earlier recommendation status: caching was executed and numerically qualified, but changed BF16 trajectories/stopping and did not yield a standalone win; discrete proposal scoring was executed and final-evaluated; damped Gauss-Newton was executed and stopped; batch8 candidate execution was rejected by equivalence qualification. None is misrepresented as an unexplored option.

## Preservation, resources and reproducibility

The exact incoming brief is coordination/requests/agent4-holistic-reassessment.md. Supporting supplied files remain unchanged under experiments/agent4-holistic-reassessment/incoming. Their external-access limitation is historical context, not evidence that local results were missing.

Actual existing model/tokenizer/lens backup and independent restore bytes, plus dependency archives, were hash-verified. A new actual source archive was saved. The bounded diagnostic ran using freshly restored dependency packages from the rescue, Python-S without global site-packages, and the independent restored prefix. Exact candidate/score reproduction verifies that restore in the used execution path.

Preflight reused the measured127-position plus vocabulary/Jacobian qualification; this job used at most24 positions, singleton trials and first gradients. GPU reserved cap6GiB; at least2GiB GPU and8GiB host free; temperature below80C; job capped at600s. Measured peak reserved memory was 3.883GiB and post-import diagnostic wall 14.398s. The watchdog passed. GPU use was coordinated and explicitly released; no further GPU work is planned.

The supplied analytical examples were rerun on copied files and reproduced their supplied JSON exactly. They remain toy calculations, not Llama evidence. No new solver, parameter sweep, independent final panel, target training, recovered-prefix experiment, P03 access, paid compute, global registry edit or PR merge occurred. Canonical comparison remains incomplete.

Run scripts/agent4_reassessment/analyze.py against the pinned evidence; then the sole diagnostic command in diagnostic_guard.json. Its environment and backup_and_preflight.json bind exact source, assets and runtime. report.py rebuilds this decision from the frozen analysis/diagnostic receipts. Immutable output names require a fresh evidence directory for replay.

Result: coordination/results/agent4-holistic-reassessment.md.
Manifest: experiments/agent4-holistic-reassessment/manifest.json.
No new public disclosure was authorized. This reassessment is local and unpushed.
