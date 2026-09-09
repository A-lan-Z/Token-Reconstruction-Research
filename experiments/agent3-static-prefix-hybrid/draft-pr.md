# Static-prefix B1+A2 K16: quality limits met, 15–18% lower measured time

B1+A2 K16 met the predeclared token/clip quality-loss limits versus both K256 references in all eight domain/snapshot cells. Measured warmed time was 15.4–18.1% lower, so no cell met the declared 20% useful-acceleration criterion against both references. Preserve the bounded static-prefix candidate with that failed acceleration gate; no threshold was relaxed and no recovery/tracking claim is made.

The exact PR25 step 13000 B1 state and fixed public readout are preserved. Five arms share the same observations, numerical settings, static public prefix and native direct-cosine selector. Each hybrid invocation reconstructs from BOS using its own committed tokens/cache. The fresh panel has 32 natural clips/domain, paired across the existing public-base/64/128/256 prefix-changing target snapshots; all P12 and PR25 sources and available fitting/reservation exclusions were excluded. PR25 remains unchanged development evidence.

B1 K16 coverage was 100% at every token and clip. It recovered all 32 Pile clips at every snapshot; Finance exact clips were 32/11/21/29 at base/64/128/256. Finance selection errors despite inclusion were 0/23/11/3. Of those 23 errors at update 64, two followed an earlier wrong commitment; the others were first errors. Static verification was worse than B1 alone at updates 64/128. These findings motivate investigation of verifier/public-prefix mismatch, not automatic construction of a recovery framework. Both K256 arms recovered zero exact Finance clips at 256, versus 29 for B1 K16; larger candidate budgets did not monotonically improve this verifier.

All 40 outputs were frozen and backed up before new evaluator truth opened. Lossless compressed artifacts preserve predictions, candidates, proposal/direct scores and committed-token traces (165,558,559 bytes). Original evaluator label tensors and model weights remain in local/independent actual-byte backups and are not uploaded. Full phase costs, memory, source-paired intervals and failure attribution are retained. Timings include common instrumentation on a shared Windows desktop; NVML omitted the active Finance 128 process, so exclusive GPU access is not established. Windows counters showed the VM workload plus desktop graphics activity. Candidate counts are not speedups, and nonsignificance is not equivalence.

Validation: development qualification passed native-anchor, instrumentation/native, independent-cache and exact-repeat checks. CUDA B1 is a disclosed numerical port of PR25 CPU inference (one of 508 development top1 positions differed); model bytes are identical. All eight confirmation guards passed. An independent scalar audit matched all 40 cells, nested shortlists/B1 rank one, error partitions, timing medians and work counts. Two focused tests passed. All warmup/measured hybrid runs executed 70,746,112 candidate simulations and 524,288 cache-token commitments. No new fitting, P03 access, global STATE change, paid compute, merge or further variant occurred. This bounded matrix does not complete dual-canonical comparison.

- Result: `coordination/results/agent3-static-prefix-hybrid.md`
- Structured evidence: `experiments/agent3-static-prefix-hybrid/manifest.json`
- Full metrics and costs: `experiments/agent3-static-prefix-hybrid/results.json`, `metrics.csv`, `costs.csv`
- Frozen artifacts: `experiments/agent3-static-prefix-hybrid/frozen-candidates/`
- Replay: `experiments/agent3-static-prefix-hybrid/README.md`
- Scientific inference commit: `29b32def8ebafc8afa17e8e8312d1dea48fb4491`

Base: `task/agent3-b1-small-budget-a2`. Head: `task/agent3-static-prefix-hybrid`. Keep this PR unmerged.
