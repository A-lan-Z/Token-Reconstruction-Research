# TRR-P03 gate recommendation

Freeze this bounded Stage 1 decision before opening any natural-panel truth.

Stage 1 is a broader stratified public natural diagnostic using 24 cached
`HuggingFaceH4/no_robots` train prompts at revision
`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`. It has six records each at 16,
39, 64, and 128 post-BOS tokens, or 1,482 scored tokens per target. Each
length has two coding, two question-answer, and two creative-generation
slots, selected with seed `20260906` and the plan's exact category mapping.
The separate `p03-s2` holdout uses seed `20260907` and remains unopened.

Reject a candidate with exact prior overlap or an exact BOS-anchored P02
prefix tuple collision after each declared truncation. Token IDs are not
global exclusions. The four-record A1+A2 anchor is fixed at
`p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and `p03-s1-r0012`, the
length-39 stratum indices `[0,2,4,5]`; preserve its native 40-slot geometry
and report its `torch.topk`/first-argmax tie behavior separately.

Require matched and shifted target observations and all methods to be frozen
before scoring. Validate every receipt and hash, including the anchor, before
opening truth. The static raw, projected, and native-A1 arms use descending
score then ascending-ID ties. Native A1 decisions use native FP32
`exp(s)`-scaled logits; cosine-equivalent division is reporting-only after the
decision.

For projected versus A1, the primary statistic is the micro token-accuracy
delta. Use 10,000 paired record-cluster bootstrap draws, seed `20260905`,
sampling within each length stratum and reusing each sampled record for both
methods. Report the length-stratified macro and exact-record deltas and CIs as
secondary/descriptive metrics. Report per-record gain/tie/loss counts and
worst/median deltas. Position-level token gains/losses require paired
correctness vectors; per-record count changes remain labelled separately.

The matched gate passes only if:

1. projected minus A1 micro token accuracy is at least `+1.0` percentage
   point;
2. the lower endpoint of its 95% length-stratified paired record-cluster CI
   is strictly above zero; and
3. projected exact-record count is no lower than A1's.

When shifted observations are available, require a nonnegative shifted
projected-minus-A1 point estimate and no lower projected exact-record count.
Any failed criterion stops promotion and keeps `p03-s2` unopened. A small
positive gain or interval spanning zero is `stop_no_representative_advantage`;
a clear regression is reported distinctly. These are bounded exploratory
decisions about this fitted-origin projected variant and do not establish a
universal failure of no-fit methods or compression.

If the gate passes, test only ranks 128 and 256 on the untouched `p03-s2`
holdout, comparing the uncompressed projected parent, compact projected
factors, and same-rank compact A1 control. Freeze the float32 decomposition
and all constants first. A practical compact label requires no more than 1.0
point token loss, one exact-record loss, a paired lower CI bound of at least
-1.0 point, candidate-factor storage at most 25%, resident storage at most
35%, and at least 20% lower full scoring time. A failed rank receives no rescue
rank or decomposition.

The machine-readable contract is `experiments/TRR-P03/plan.json`.
