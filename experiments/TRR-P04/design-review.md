# TRR-P04 scientific design review

Status: proposed design recommendation pending setup/implementation agreement and source freeze. This is an exploratory public-data diagnostic, not a canonical replacement claim. The packet is `coordination/requests/TRR-P04.md` (SHA256 `1cd77f141bf780896848438f6a2bf8ab85b2b99e51a9a7dcf2cd73872f2ed268`), and the read-only parent is PR7 commit `6e8b683e404c0acb70cd59b7dd6d6868b2061f61`.

## What the parent result does and does not establish

PR7 is a useful starting point: the controlled affine fit uses 1,200 public Alpaca records and 124,371 post-BOS positions, and its selected no-vocabulary-bias fit reaches 94.3505% on the public validation panel. The contextual pilot adds about 1.05 million parameters to that fit and gives causal attention a small fresh-panel edge, 4,943/5,312 versus 4,910/5,312 for the affine comparator. That panel contains 32 distinct records in two styles, paired across two conditions. The result is a reason to test a causal student, not evidence of a general architecture advantage.

The PR7 contextual fit cannot answer the P04 question by itself. Its `CausalResidualDecoder` freezes `W`, `b`, and `s` and trains only the added path. Its eight-record subset starts at 100% and is therefore not a capacity check. P04 must initialize from the competent affine state if useful, but continue training the complete affine map in every S/H/D arm and in the standalone affine reference. The capacity probe must contain actual initial affine errors selected from the public correction pool.

## Frozen data and evaluation design

Use three public style strata and four post-BOS length strata, with exact identities and source hashes frozen before any fresh truth is opened:

- replay: the immutable PR7 public fit, 1,200 records / 124,371 post-BOS positions;
- correction: 256 disjoint public records with useful initial affine errors or low-confidence rows;
- public validation: 192 records disjoint from replay and correction, used only for checkpoint selection;
- fresh evaluation: 72 new records, six per style-length cell (`3 styles x 4 lengths x 6`), with lengths declared as 16, 32, 64, and 128 post-BOS tokens.

Use the setup IDs exactly: `pile_plain`, `finance_chat`, and `alpaca_instruction`. These are a broader stratified public diagnostic mixture, not a claim of representativeness for all public text. Exclude every known PR7 fit/validation and prior public panel identity or truncated-sequence hash. Do not duplicate TRR-0005's coverage-by-context matrix.

The fresh records are paired under the matched public condition and one predeclared unseen target-update condition. Target preparation remains evaluator-side and its truth is not used for fitting, routing, checkpoint selection, or model choice. Keep the two conditions separate in all tables. Run the bounded native A1+A2 K256 anchor on a predeclared 12-record subset (four records per style in the 32-token stratum, 384 scored post-BOS positions per target), retaining its denominator separately from the 72-record student comparison. Preserve the native A1+A2 implementation's tie behavior and report it separately from the students' deterministic lowest-ID ties. The affine reference is evaluated on all fresh records and is fitted with the same public data and update budget as the student affine path.

Use two paired seeds, 1737 and 2711, for every arm. A seed is a repeated fit, not an additional independent evaluation record. Freeze one deterministic active-position schedule per seed and reuse that exact schedule, record order, labels, masks, and weighting across S, H, D, and the affine reference. Define the common schedule at the active-position level as a fixed 75% replay / 25% correction mixture (384 replay and 128 correction rows per 512-row update when the cap is filled), with a record batch of 8. Sample with replacement only when needed, and log the schedule digest, actual replay/correction row totals, and per-record repeat counts for every seed; do not infer token exposure from nominal record quotas. The schedule has at most 512 selected post-BOS projection rows per update. Fit for exactly 3,000 updates and evaluate the same declared checkpoint grid for every arm. Select each arm's checkpoint by maximum style-balanced public validation token accuracy, breaking exact ties by earliest step; validation CE is reported as a secondary diagnostic. There is no post-truth checkpoint or seed selection.

## Student and fair affine capacity control

For each record, the student consumes only `H_0,...,H_i` and its mask, resets its recurrent state, and predicts the current token `x_i` from `H_i`. No source token, guessed-token feedback, teacher output, candidate ID, score, future activation, public-prefix call, or A2 fallback reaches deployed inference. The full vocabulary projection is used at evaluation.

Use one unidirectional GRU with hidden width 256 and a full trainable affine bypass:

```
r_i = W H_i + b                         (W,b initialized from PR7 affine)
g_i = GRU(LN(H_i), g_{i-1}),  g_{-1}=0
a_i = r_i + U g_i                       (U initialized to zero)
z_i = exp(s) * E @ normalize(a_i)        (E is the fixed public normalized table)
```

`W`, `b`, `s`, all GRU parameters, and `U` are trainable in all three arms. The zero `U` makes the step-0 output equal to the affine initialization without freezing the affine path; gradients must be allowed into the affine parameters from the first update. The tied public table is `[128256, 2048]` and remains a runtime resource, not part of the retained student state. The affine reference uses the same trainable `W,b,s`, initialization, schedule, optimizer, and update budget with `U` and the GRU removed. This is the architecture control; a frozen PR7 affine state is only an initialization/provenance reference.

The largest-cell qualifier must exercise this exact graph at batch 8, sequence length 192, 2048 hidden units, GRU width 256, 512 projected positions, and the full vocabulary. It must allocate forward/backward and Adam state, then run a short fixed probe on the public correction errors. Require finite gradients, no allocator/driver anomaly, and a safety margin under the existing 8 GiB minimum-free / 6 GiB reserved-GPU / 16 GiB host-RSS limits before the matrix is released. A numerical batching change is a separate excluded attempt unless its predictions are bit-equivalent to the canonical graph.

The capacity receipt must report the number of step-0 affine errors and post-fit errors on a fixed correction probe, plus token loss and exact-record count. Require at least 256 step-0 wrong positions in the correction pool and an initial accuracy below 99% on the probe; a perfect or near-perfect subset cannot support a capacity conclusion. This is a public-data selection-feasibility requirement, not evidence that the student family fails. If the probe shows clear underfit after feasibility is established, allow one bounded public-only development adjustment, selected and justified from public validation, applied identically to the affine reference and all S/H/D arms, then rerun the capacity receipt before fresh-panel selection. Log that adjustment and its new schedule/configuration; do not expand the teacher or panel automatically. If fewer than 256 usable wrong positions remain after the one permitted adjustment, mark teacher qualification infeasible and report no method-family conclusion.

## Teacher qualification

Before D training, freeze all selector outputs on 384 correction positions: 256 positions selected because the frozen PR7 affine is actually wrong under public labels, and 128 positions sampled uniformly without replacement from the remaining correction positions. Preserve the style-length allocation and record IDs. The teacher run is public training evidence only; correctness is measured after its outputs are immutable. The immutable PR7 proposer ranks the public A1 logits at K=512; retain its first K=32 identities as the fixed candidate set and simulate exactly those 32 candidates with the privileged public-prefix scorer. Report both K=512 and K=32 proposal-miss/recall diagnostics, and use the K=32 set for H/D. This is a modest teacher diagnostic and is distinct from the K256 native anchor. Force this scorer to emit a finite K=32 score vector for every qualified row, including rows that the historical cascade would send through an A1 fast path; a winner-only or missing score vector is invalid D evidence.

The single training teacher is `privileged_public_prefix`: it may use the known public prefix to score the frozen first-32 identities from the K=512 A1 proposal list, and its q arrays are training-only evidence. Label its correctness, proposal misses, and any abstentions as privileged diagnostics; never report them as unassisted native reconstruction. Do not run a second native teacher for D. Native causal behavior is retained only in the separately reported K256 A1+A2 evaluation anchor.

Record, by style and length, proposal miss rate, privileged teacher top-1 correctness, fixes of frozen-A1 errors, introductions on A1-correct rows, score span and robust scale, exact and near ties, and valid/abstained rows. Public labels are authoritative for these diagnostics; teacher mistakes and proposal misses remain in the ordinary CE schedule. The native K256 anchor retains its own causal route diagnostics separately.

Pass the informative-signal gate only if all 384 rows have finite, immutable K=32 score vectors, at least 50% have at least two non-gold candidate pairs with a score gap above the fixed tie tolerance, and the K32 candidate-set miss rate is at most 50% (while retaining the K512 proposer miss rate as a separate diagnostic). Otherwise run S and H as the label-only comparison, mark D as predeclared `NOT_RUN_TEACHER_NONINFORMATIVE`, and do not expand the teacher pool this round. A signal can pass this gate while still failing the later D-versus-H quality gate.

## One objective per arm

Let `C_i` be the frozen top-32 A1 candidate identities for every scheduled replay/correction position, generated once from the canonical PR7/public-affine resource and cached before fitting. H and D receive exactly these identities; H receives no teacher scores. All arms retain the full-vocabulary CE term on every scheduled replay/correction row. The teacher score arrays exist only for the 384 qualified correction positions; the D ranking term is zero on other rows. If `y_i` is absent from `C_i`, it is included only as the labelled positive in the loss; this does not turn evaluation into a candidate-restricted decision.

```
L_S   = CE_full(z_i, y_i)
L_H   = L_S + 0.25 * mean_{j in C_i, j != y_i} softplus(1.0 + z_{ij} - z_{iy_i})
L_D   = L_H + 0.25 * L_rank
```

The H hinge is a fixed label-derived hard-confusion control. It uses only the frozen candidate identities and the public gold label; it has no teacher score or teacher top-choice input. Empty negative sets contribute zero to the hinge.

For D, use adjacent pairs among non-gold members of `C_i`, ordered by the frozen K32 teacher scores. Exact and near-tied pairs are omitted rather than arbitrarily broken. For every retained pair `(a,b)`, let `delta_q=q_a-q_b`, use target `sign(delta_q)`, and weight it by `min(abs(delta_q)/sigma_q, 1)`, where `sigma_q` is the single robust median nonzero pair-gap scale computed on the 384 public qualification rows and then frozen. The pair loss is

```
L_rank = weighted_mean(softplus(-sign(delta_q) * (logp_a - logp_b)))
```

with `logp = log_softmax(z_C / 1.0)` (student temperature fixed at 1.0) and no fitted probability interpretation for raw cosine `q`. Use one fixed tie tolerance `max(1e-6, 0.01*sigma_q)`, report the number of retained pairs, and use the original candidate order plus token ID only to serialize equal-score order; a tie never contributes a pair. Excluding `y_i` from the ranking pairs prevents D from being reduced to copying a correct hard teacher label. D therefore tests non-gold relative score structure beyond H, while the full CE term remains the authority when the teacher is wrong or misses the gold token.

The ranking scale, tie handling, candidate identity provenance, and finite-precision values must be frozen before any fresh evaluation truth. Do not sweep lambda, temperature, candidate budget, or loss family. Inspect the score scale and report it; if the informative gate passes, the above robust normalization is the only declared calibration.

## Frozen decision rule

The primary comparison is unrestricted post-BOS token accuracy on the matched public fresh condition, with paired record-cluster uncertainty stratified by the 12 style-length cells. The 1 percentage-point rule below is a predeclared practical promotion threshold; a smaller positive estimate with a lower bound above zero remains a reported exploratory gain and does not count as no effect. Use 10,000 deterministic bootstrap draws (seed 20260908), resampling complete records within each cell; average over the two paired seeds inside each selected record and never count a seed as an independent record. Report exact-record recovery, token gains/losses, per-style/length results, and worst-record regressions descriptively.

Predeclare these interpretation gates before fresh truth:

1. **Architecture:** S-GRU versus the trainable affine reference must have a point gain of at least 1 percentage point and a 95% paired record-cluster lower bound above zero. Otherwise make no recurrent-architecture claim.
2. **Hard-example supervision:** H versus S must meet the same 1-point / lower-bound rule to call label-only confusion weighting useful. If it passes while D does not, retain H as the simpler result.
3. **Teacher score information:** D versus H must meet the same rule on the matched public condition. If it is a clear regression, stop promotion and run no extension this round. If it is positive but below 1 percentage point or its interval includes zero, report the estimate and uncertainty without promotion; that gate failure is not proof of no effect. A successful D gate is still an exploratory result; it does not replace native A1+A2.
4. The unseen target-update condition is a separate transfer diagnostic. Do not pool it into the primary gate or use it to rescue a failed matched-public comparison. Report it even when uncertain, and call out any material regression by style or length.

Exact-record counts are secondary because 72 records and two seeds remain sparse for that endpoint; an exact-record loss is a descriptive warning, not a replacement for the token gate. The native A1+A2 K256 anchor is a quality/cost reference. Report its prefix calls, candidate simulations, startup, and steady inference separately from the student’s no-selector runtime and retained state.

A failed teacher gate, capacity gate, resource guard, or truth-freeze integrity check stops the corresponding comparison and is recorded as an excluded/incomplete result. There is no automatic full-pool teacher extension, architecture sweep, target sweep, or post-truth refit. The first decision is one of: D adds score information; H is the simpler improvement; S-GRU adds architecture value independent of distillation; or none of these improves the trainable affine control.

## Literature boundary

The GRU choice is motivated by the SIP/BiSR paper’s Appendix E.1 comparison, where a unidirectional GRU is the strongest model in that paper’s own LLaMA2/SensiReplaced setup; that is motivation rather than a result about this dataset. Hinton et al. describe distillation as transferring predictive-distribution information into a deployable student. Decoupled Knowledge Distillation motivates treating non-target alternatives as informative separately from the target class. Stanton et al. show why teacher fidelity and optimization details should not be treated as a guarantee of student generalization. These references support the design choices but do not establish P04 outcomes:

- [SIP/BiSR forward inversion, arXiv:2409.00960](https://arxiv.org/html/2409.00960v2)
- [Hinton, Vinyals & Dean, Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
- [Stanton et al., Does Knowledge Distillation Really Work?](https://arxiv.org/abs/2106.05945)
