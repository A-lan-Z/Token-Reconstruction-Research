# TRR-0020: final benchmark and research reflection

The requested benchmark is complete. The current shortlist-free method is slower
and less accurate than both A1+A2 controls on both canonical benchmarks.
It satisfies the structural requirement of having no candidate proposer or
separate candidate verifier, but it does not establish the requested practical
replacement. Research is stopped at the owner's request; no new experiments
are queued. This is not a proof that shortlist-free reconstruction is impossible.

The complete record covers 50 research stages, 36 accepted development reconstruction
matrices and 2,816 method/input cells. Those 2,816 cells repeatedly use the same eight
opened development inputs; they are not 2,816 independent test examples.
The final comparison contains 1,728 replicate runs on 192 canonical inputs.
The active 35-method registry now has all 70 required benchmark cells.

## Final comparison

| Benchmark | Method | Correct / scored | Accuracy | Exact inputs | Seconds / input |
|---|---|---:|---:|---:|---:|
| Pile | A1+A2, ordinary execution | 2490/2496 | 99.760% | 60/64 | 0.2615 |
| Pile | A1+A2, optimized execution | 2490/2496 | 99.760% | 60/64 | 0.1503 |
| Pile | No shortlist, Bennett64 | 2466/2496 | 98.798% | 41/64 | 0.4011 |
| Finance | A1+A2, ordinary execution | 13952/13990 | 99.728% | 107/128 | 0.7406 |
| Finance | A1+A2, optimized execution | 13952/13990 | 99.728% | 107/128 | 0.4444 |
| Finance | No shortlist, Bennett64 | 13761/13990 | 98.363% | 14/128 | 0.7935 |

Pile has 64 inputs of 40 tokens including BOS. Finance has 128 inputs with up to 128
valid positions. Accuracy excludes the known BOS; all other valid tokens count.
Each time is the mean of per-input medians from three complete passes, with
method order rotated between passes. These are current paired timings, not
historical timing values.

Against ordinary A1+A2, Bennett64 takes 1.534 times as long on Pile and 1.071 times
on Finance. Against optimized A1+A2 it takes 2.669 times and 1.786 times as long.
It makes 24 additional token errors on Pile and 191 on Finance. Exact reconstruction
falls by 19 Pile inputs and 93 Finance inputs.

The paired 95% record-bootstrap accuracy-difference intervals against ordinary
A1+A2 are approximately -1.442 to -0.481 percentage points on Pile and
-1.581 to -1.145 points on Finance. Runtime-ratio intervals are 1.517–1.557 and
1.008–1.140. These describe the current retrospective samples, not fresh
confirmatory evidence.

The optimized A1+A2 control preserves the ordinary control's candidate arrays,
scores, MSE values and emitted tokens. It is about 42.5% faster on Pile and 40.0%
faster on Finance in this run. This remains a useful execution improvement,
but it retains the trained A1 and its 256-candidate shortlist.

Ordinary/optimized refer to execution of the established single-record
benchmark port. Right padding is removed; the archived TRR-0018 arrays are the
current anchors. This does not assert identical candidate arrays to the older
batch-of-eight historical runner. The registry and benchmark plan retain that
port disclosure.

## What the current method actually does

The pinned supplied prefix is from public Llama-3.2-1B-Instruct at cut 4
(layers 0–3). Only the declared BOS token, 128000, is known in advance.

Each unknown position holds a probability distribution over all 128,256 tokens.
The method combines token embeddings according to those probabilities, passes
one complete soft sequence through the frozen prefix, and adjusts the input
probabilities to reduce the discrepancy with the observed activations.
It updates input variables, not model parameters.

It performs 64 updates. To choose how large each update should be, it summarizes
the proposed direction using its weighted mean, variance and range and solves
a small scalar bound. This replaces repeated vocabulary-wide trial calculations
for the probability-change constraint. All vocabulary entries remain represented
and processed; there is no top-K filter and no separately proposed token list.

The final readout remembers the most probable token at each position when that
position had its smallest observed **soft-state** activation error. That token was
not independently verified as a discrete token by A2. Different positions can
come from different whole-sequence soft states; the assembled sequence receives
no final reranking or repair. The assembled discrete sequence can therefore
have an activation error that was never measured by the optimizer.

Per input, the recorded optimization work is 66 whole-sequence prefix forwards
and 64 backwards, plus initialization and graph preparation. It is not zero A2
work, and these whole-sequence counts are not directly comparable with the
baseline's individual candidate-simulation counts.

The method still constructs an initialization table from supplied prefix
weights. It fits no separate prediction model, but table construction costs
time and must be reconsidered when prefix weights change. This study did not
exercise an actually recovering prefix over time.

The earlier TRR-0019 “No-A1” variant is different: it chooses 64 embedding matches
and 64 MLP-only table matches, then fills a 256-slot shortlist with tokenizer
suffix fragments from a broader pool. It removes separately trained A1, not
candidate proposal. It never satisfied the stricter request addressed here.

## What the work established

| Research direction | Recorded evidence | Assessment |
|---|---|---|
| Full-vocabulary soft optimization, hardening and readouts | Development 1–20; three earlier registered canonical rules | High small-panel scores did not translate into a fast, accurate replacement. |
| Direct inverse and local correction methods | Development 21–28 and 39–40 | Some linear or residual improvements; no adequate complete reconstruction method. |
| Full-vocabulary probability updates and causal variants | Development 29–33 and 38 | Better-controlled updates and correct first decisions did not solve complete-sequence recovery. |
| Loss geometry and context diagnostics | Development 34–37 and the objective audit | Correct endpoints can exist behind loss barriers; a shared context correction is inadequate on tested fixtures. These are restricted findings, not impossibility proofs. |
| Profiling, fusion and reuse | Development 41–46 | Profiling found material repeated-reduction cost. Some exact execution changes saved time; an approximate fused variant changed results and worsened prose quality. |
| Bounded history mixing | Development 47–48 | The qualified stable revision added cost without an adequate accuracy gain; the original nonfinite-diagnostic run was excluded. |
| Moment-bound updates | Development 49–50 and the final canonical run | A materially cheaper update and faster full development decoder, followed by a failed larger-benchmark replacement test. |

The latest paired development comparison reduced 64-update runtime from 1.134 to
0.697 seconds for 128 positions and 0.489 to 0.406 seconds for 40 positions
(38.5% and 17.0%). Those gains are against the previous shortlist-free optimizer,
not A1+A2. The final canonical table above is the relevant baseline comparison.

The native-objective audit found that, in the checked first-error cases with
a common correct history, the baseline's correct token had lower native
activation error than the emitted wrong token. Public interpolation diagnostics
also found paths that initially increased error before reaching a correct
endpoint. Together these support optimization/readout difficulties in the tested
methods. They do not show that every feasible method will fail.

## Reflection

There was real engineering progress, but much less progress toward a practical
replacement than the number of experiments might suggest. The strongest
retained results are exact reductions in redundant computation and a qualified,
cheaper full-vocabulary update. Neither closes the discrete-token accuracy gap.

I pursued too many closely related optimizer variants on a small reused
development panel. Small gains there were weak reasons to expect a larger
benchmark gain. Profiling also came late; when performed, it identified an
avoidable cost that several earlier algorithm changes had not addressed.
More frequent comparisons against the full baseline would have made the
diminishing returns clearer sooner.

I also blurred “no separately trained A1” with “no candidate generator” in the
earlier explanation. The distinction changes whether a result meets the task.
Likewise, an isolated kernel speedup or a speedup over a weak shortlist-free
control must not be presented as an A1+A2 improvement. This report keeps all
three comparisons separate.

The numerical checks, frozen outputs, matched controls and retained failures
were useful: they prevented faster but different arithmetic from being called
an exact optimization, and prevented incomplete runs from becoming accuracy
claims. The experiment harness itself became too coupled through shared imports,
causing avoidable setup failures. Those failures were repaired and preserved,
not treated as scientific negatives.

The evidence supports stopping this line of incremental experiments now. It
supports neither claiming success nor declaring the entire idea impossible.
The current result is a documented failed contender with reusable performance
work and clearer failure mechanisms. No further search is being launched.

## Cost, validation and limitations

The final run used an RTX 5080, Python 3.12.3, PyTorch 2.10.0+cu128 and
Transformers 5.3.0. Peak reserved GPU memory was 2.934 GiB for ordinary A1+A2,
3.361 GiB for optimized A1+A2 and 4.912 GiB for Bennett64.
The successful guarded matrix took 1,359.842 seconds, retained at least 5,692 MiB
GPU free and reached at most 64 °C.

Prefix loading took 0.220–0.321 seconds across methods. Bennett engine/table
preparation took 0.390–0.775 seconds. Existing A1 asset preparation took
0.057–0.082 seconds; its original offline fit was not remeasured.
Optimized A2 warmup took 1.106–1.468 seconds and graph capture 0.771–0.817 seconds.
Inference timing includes input initialization, required geometry capture or
eviction, updates, diagnostics, synchronization and output transfer.
Preparation and disk/hash I/O are separately recorded in summary.json.

R1 passed 52 qualification checks: 8 development and 6 public equivalence checks,
plus all 38 native benchmark lengths. Both numerical environments passed isolated
import checks. All three full passes reproduced their selected outputs; novel
recorded loss traces repeated exactly, and all baseline arrays matched their
archives. Three negative truth-gate tests rejected missing cells, modified
bindings and modified predictions before the current scorer opened labels.

The first canonical matrix attempt failed during an import before any
reconstruction. Its 52 successful qualification checks and full failure record
remain. R1 separates baseline and novel imports, reruns qualification, and
completes the unchanged matrix. The earlier development 49 import failure is
also preserved; it performed no numerical work. Other excluded and repaired
attempts remain in the chronological report and research ledger.

Both canonical sources were opened historically, and the small development
panel was reused extensively. All current outputs were frozen before this
scoring pass, but that does not make the study fresh or blind. Tests used a
supplied public prefix, not a measured prefix-recovery trajectory. No claim is
made about unseen target updates or table rebuilding during recovery.

## Evidence and stop state

The archive verifies 2,565 logical artifacts stored as 2,284 unique objects.
Its 85,838,799 bytes are split into two ordered parts, with exact member and part
hashes and a verified reassembly. It includes all current predictions and cell
receipts, phase receipts, largest-case qualification outputs, and both original
and R1 validation sets. Older raw archives remain referenced.

The main evidence paths, relative to this task worktree, are:

- experiments/TRR-0020/canonical_moment_r1/score.json
- experiments/TRR-0020/canonical_moment_r1/summary.json
- experiments/TRR-0020/canonical_moment_r1/canonical_matrix.json
- experiments/TRR-0020/canonical_moment_r1/prediction_freeze.json
- experiments/TRR-0020/canonical_moment_r1/truth_gate.json
- experiments/TRR-0020/canonical_moment_r1/archive.json
- experiments/TRR-0020/RESEARCH_LEDGER.json
- experiments/TRR-0020/CONTINUATION22_PROVENANCE.json
- experiments/TRR-0020/CONTINUATION22_REPRO.md

The detailed chronological report is coordination/results/TRR-0020.md.
The structured task manifest is experiments/TRR-0020/manifest.json.
The owner's stop request is preserved verbatim in
coordination/requests/TRR-0020-STOP-AFTER-CURRENT-BENCHMARK.md.

Benchmark execution and scoring commit:
649abcbd69d64ece05399e12b0d26412137a6286.
Archive and summary implementation/execution commit:
2b9d9834d1903f4593c60da03d9295bdfb78895a.

All work is local on task/TRR-0020. The earlier automatic approval review
blocked public publication pending explicit authorization for that destination.
No alternative upload, push retry or PR publication is attempted in this closeout.
The scientific goal is not marked achieved. The owner-requested benchmark,
recording, consolidation and reflection are complete; further research is stopped.
