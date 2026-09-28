# TRR-0015: equal-budget prefix-fragment comparison

## Request and rationale
The user requests full reconstruction and timing at K256 against the existing A1+A2 K256 baseline. TRR-0014 chose K512 after its opened R1 candidate-only probe found K256 omissions of 12/2032 matched prose and 8/2032 shifted prose, while K512 included all true tokens. This justified a promising-direction operating point but left an equal-budget comparison incomplete.

## Frozen methods
Register prefix_native_fragment256 and prefix_native_fragment512 as active methods before execution. Each uses the same prefix-derived weight metric, embedding top64, intrinsic MLP-only top64, tokenizer proper-suffix expansion in original round-robin order, then native direct-cosine A2 with own reconstructed history. Only the final candidate cap differs (256 or 512); preserve padded logical slots. Intrinsic build256 and projection256 are fixed. No training, additional learned model, target-prefix access, or truth-dependent decisions.
Control: a1_a2_exhaustive_configuration_winner, supplied public Alpaca lens top256, same native direct-cosine A2. No thresholds, abstention, or new routing.
Implementation: parameterize the existing proposer with default budget512; K256 must equal the first256 slots of the K512 proposal on every input. Do not tune after scoring.

## Matrix and ports
- Original TRR-0014 R2: 80 paired observations (matched/LoRA, 32 prose and8 identifier sources), all three methods. Already opened: retrospective ablation, not new confirmation.
- Canonical clean Pile rank4 LoRA:64 records x40 positions.
- Canonical historical Finance generation300:128 right-padded rows, up to128 positions.
All three methods on every observation:272 observations x3 =816 cells, three byte-identical repetitions per cell, median synchronized runtime. Rotate method order per record.
Canonical ports preserve all constants and decision rules. Only opaque identifiers, input serialization, record loop lengths, and removal of right-padding positions change. Each record runs separately using the inherited native candidate helper; label as benchmark-compatible ports, not exact historical batch8 executions.
Carry forward all48 previously completed active canonical cells with exact source hashes in the full registry report, add4 cells for the two new active methods, and show the rerun baseline ports separately. Do not pool benchmarks or compare old timings against current timings.
Use existing sanitized canonical observations hash42fe0e685eed54fd779d678febf646228cb929d7d1bd3f3515cee6e582b30d61. No target capture or new target training is needed.

## Qualification and numerical contract
Use a public synthetic128-position fixture to qualify K512 and bothK256 arms. Check the new K256 proposal is exactly the prefix of K512, K512 output and proposal match the inherited decoder, and the A1 control matches the original native comparator on a compatible16-position fixture. Preserve baseline candidate execution size. Do not introduce numerical batching/microbatching changes.
Run existing focused prefix-metric and fragment tests. Validate complete predictions, source/input/asset/output hashes, finite scores, shape/range, declared argmax rule, exact candidate counts and repeated output identity before truth. Test missing-cell, altered-output, and altered-code rejection.
Original-panel K512 and A1 predictions must reproduce all earlier frozen token and candidate outputs; report any numerical drift and do not silently substitute outputs.

## Resource preflight
Prior identical largest128xK512 run:peak allocated4.897GiB, reserved5.574GiB. Worst persistent tensors:prefix~1.01GB, two derived tables~2.10GB, A1 embedding table~1.05GB, transform~0.017GB; K512 KV expansion~0.27GB, plus native workspaces. New K256 arm shares caches and runs sequentially; it does not enlarge this geometry.
At inspection:RTX5080 16303MiB total,12667MiB free,45C; host~29GiB available. Expected largest reserved<6GiB, retaining>6GiB of the observed free budget; prior full panel617.76s.
Conservative matrix runtime estimate:~35 minutes, watchdog timeout3600s. Qualify before release. Fail closed at freeGPU<2GiB, hostavailable<8GiB, reservedCUDA>6GiB, temperature>=80C or processRSS>10GiB. Isolated process-group watchdog and per-cell create-only resumable receipts. Stop and preserve allocator/driver/thermal anomalies.
Record measured peaks and preflight live snapshots, all timing phases, simulations, preparation/rebuild costs and raw hashes. No historical A1 training cost is invented.

## Scoring
Freeze all816 cells and validate the complete matrix before evaluator reads any source truth. Truth is already opened historically; no fresh-confirmation claim. Report post-BOS token accuracy, exact token inputs, decoded-text exactness, proposal recall, errors despite inclusion, candidate simulations, synchronized phase timing, memory, and paired record bootstrap95% intervals per condition. No pooled score or equivalence claim merely from a nonsignificant difference.
Source-string recovery is unavailable for the historical observations and token-truncated synthetic/book panel; label it unavailable instead of equating it to decoded-token exactness.
Tests use supplied public prefix weights; actual recovery trajectories remain outside this experiment.

## Publication
Prepare and commit local reproducible evidence. The prior public-publication approval block persists; this test request is not permission to publish the previously audited payload. Do not retry the rejected public push or bypass that block.
