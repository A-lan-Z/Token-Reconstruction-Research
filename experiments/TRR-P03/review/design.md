# TRR-P03 Stage 1 design and Stage 2 review

This is a review draft for root freeze. It defines the bounded natural-panel
diagnostic, its paired uncertainty calculation, and the continuation gate. It
contains no opened truth or reconstruction result and does not authorize a
heavy run.

## Decision question

TRR-P02 found a small apparent full-vocabulary projected-readout advantage on
12 controlled public rows (11/12 versus 9/12 for its historical A1
comparator). Stage 1 tests whether that signal persists on a broader
stratified public natural diagnostic before any compact scorer is considered.
The projected arm remains fitted-origin: the frozen public Alpaca affine lens
is applied to both query activations and boundary prototypes. Raw boundary
prototypes are a descriptive no-fit baseline, and historical native A1 is the
primary paired comparator.

## Frozen Stage 1 panel

The selector uses cached `HuggingFaceH4/no_robots` train prompts at revision
`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, with prompts used as supplied and
the pinned tokenizer's first requested post-BOS tokens. It selects 24 records,
six each at 16, 39, 64, and 128 post-BOS tokens. Each target therefore has

```text
6 * (16 + 39 + 64 + 128) = 1,482
```

scored positions. Each length has two `Coding`, two `Open QA`/`Closed QA`/
`Classify`/`Extract`, and two `Generation`/`Brainstorm`/`Chat`/`Rewrite`/
`Summarize` slots, represented as `coding`, `question_answer`, and
`creative_generation`. The selector traverses length-major and style-major
with two records per style, using Stage 1 seed `20260906` and the declared
SHA-256 key including stage, length, style, dataset index, and seed.

Candidates with exact prior overlap or any exact BOS-anchored P02
sequence/context/endpoint prefix collision are rejected as whole records.
Token IDs remain allowed. Previously opened IDs are retrospective overlap
checks only; the plan does not claim universal public-source
representativeness.

The separate `p03-s2` holdout has the same 24-record design and independent
seed `20260907`, excludes every Stage 1 source row, and remains truth-unopened
and unscored until Stage 1 disposition and compact-method freeze. Its source
truth may be prepared and hashed for isolation. Stage 1 truth is similarly
opened only after every required target/method bundle has been frozen and
validated.

The A1+A2 anchor is the four length-39 Stage 1 records at zero-based stratum
indices `[0, 2, 4, 5]`: `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and
`p03-s1-r0012`. This gives one coding, one question-answer, and two creative
records while preserving the published 40-slot geometry. It runs under its
native 256-candidate policy and is reported as an accuracy/cost reference; it
cannot promote a projected readout or rescue a failed gate.

## Conditions and access boundary

Stage 1 requires both the pinned matched public Llama target and the available
verified Vikhr full-SFT shifted target. Their selected token sequences, order,
valid lengths, masks, and positions are identical; only target weights vary.
An unavailable resource window delays the shifted arm and does not authorize
dropping it or weakening the gate.

The evaluator may use source tokens to create target activations and private
truth. The readout process receives only activations, masks, positions, cut
depth, opaque IDs, public asset identities, and fixed method configuration. It
does not receive source text or token IDs, dataset index/text hashes, style,
condition label, target model identity, correctness, or prior opened results.

Before any natural-panel truth is opened, freeze receipts must cover both
target arms and every Stage 1 method, including the A1+A2 anchor. The scorer
validates all receipt hashes, ordered opaque IDs, shapes, masks, positions,
prediction hashes, source/config hashes, and asset identities before opening
truth once. No route, candidate set, timing, or state may change afterward.

## Readout and tie rules

For activation `h`, raw boundary prototype `b_v`, affine map `g(h)=Wh+a`, and
input embedding `E_v`, the static methods are:

```text
raw:       cos(h, b_v)
projected: cos(g(h), g(b_v))
native A1: native FP32 exp(s)-scaled cosine logits from g(h) against E_v
```

Normalization and score accumulation for static arms are float32. Native A1
top-1 decisions use the FP32 logits including positive `exp(s)`; dividing by
`exp(s)` is permitted only for cosine-equivalent reporting after decisions.
The standalone raw, projected, and native-A1 ranking rule is descending score
then ascending candidate ID on exact finite-precision ties. Preserve top-1
tie counts, runner-up margins, and strict true rank
`1 + count(score > true_score)`.

The anchor retains the published `torch.topk` candidate order and first-argmax
tie behavior. Report that policy separately rather than silently unifying it
with the standalone ascending-ID rule. Do not infer finite-precision tie
equivalence from positive rescaling.

## Metrics, uncertainty, and gate

For every method, condition, length, and style, report post-BOS correct-token
counts/accuracy, exact-record counts/rate, first errors, strict rank, margins,
and tie counts. For projected versus native A1, report per-record accuracy
deltas, gain/tie/loss counts, and worst/median record deltas. Position-level
token gains (`projected correct, A1 wrong`) and losses require paired
correctness vectors; correct-count changes are reported separately when only
per-record counts are available.

The primary contrast is projected minus A1 micro token accuracy. Use 10,000
paired record-cluster draws with seed `20260905`, sampling with replacement
within each length stratum while preserving its six records and reusing each
sampled record for both methods. Report its percentile 95% CI, plus the same
length-stratified macro per-record accuracy and exact-record-rate deltas. The
exact-record CI is descriptive and is not a second gate.

The matched Stage 1 gate requires all of the following:

1. projected minus A1 micro token accuracy is at least `+1.0` percentage
   point;
2. the lower endpoint of the 95% length-stratified paired record-cluster CI
   is strictly above zero; and
3. projected exact-record count is at least A1's count.

If the shifted arm is available, require its projected-minus-A1 point estimate
to be nonnegative and its projected exact-record count to be no lower. A
positive gain below 1 point or an interval spanning zero is
`stop_no_representative_advantage`; it does not promote compression. A clear
regression is recorded separately. Any gate failure stops promotion and
leaves the holdout unopened. Failure distinguishes this tested projected
variant from representative continuation; it does not prove that every
no-fit method or compression mechanism fails.

## Optional Stage 2

Only a passing Stage 1 disposition unlocks `p03-s2`. Test exactly ranks 128 and
256 for the uncompressed projected parent, a compact projected factorization,
and a same-rank compact A1 control. Use float32 randomized range finding with
oversampling 16, two power iterations, fixed seed 8675309, QR, and small SVD;
freeze all ranks and constants before holdout truth. For
`C_proj[v,:]=normalize_fp32(g(b_v))`, store `A=U_r Sigma_r`, `B=V_r`, and row
norms, and score

```text
q = normalize_fp32(g(h))
score_r(v) = dot(q B_r, A_r[v,:] / ||A_r[v,:]||)
```

Apply the same factor precision and score kernel to `C_a1[v,:]=normalize_fp32(E_v)`.
Quality retention allows at most 1.0 point token loss, one exact-record loss,
and a paired lower CI bound of at least -1.0 point versus the parent. A
practical label additionally requires candidate-factor storage at most 25%,
resident readout storage at most 35%, and at least 20% lower full scoring time
under the same timing boundary and hardware. A failed rank gets no rescue
rank or decomposition.

## Resource and evidence boundary

Use the CPU-only default with eight Torch intra-op threads, one inter-op
thread, deterministic algorithms, float32 lookup, prototype chunks of 8192,
and query chunks of 256. Qualify the length-128 cell under the resource guard
before the matrix; record preflight, timings, peak memory, candidate counts,
hashes, commands, environment, and failed attempts. Any batching workaround
requires exact output equivalence and an excluded record if it differs. The
Stage 1 publication must precede any optional Stage 2 report.
