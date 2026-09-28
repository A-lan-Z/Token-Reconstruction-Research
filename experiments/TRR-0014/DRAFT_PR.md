The earlier prefix-only shortlist misses word continuations when the hidden state resembles a completed word. This adds proposals derived from the same prefix's weights and intrinsic MLP responses, expands them into tokenizer suffix pieces, and verifies them with the existing native prefix. The new method uses no separately fitted A1 checkpoint.

On a disjoint 40-clip panel paired across matched and trained prefix-LoRA targets, all 240 cells froze before truth opening. Matched prose: 4063/4064 tokens and 31/32 exact clips, equal to A1+A2 K256. Shifted prose: 4064/4064 and 32/32 exact, versus 4063/4064 and 31/32. Both identifier cells: 312/312 and 8/8 exact.

Reconstruction is 1.61–1.66x the baseline. Prefix-dependent tables take about 1.07 seconds to rebuild; rebuilding every record raises cost to 2.94–2.96x for prose and 5.83–5.90x for short identifiers. Static tokenizer metadata takes 5.70 seconds once. The no-fragment control uses K128, so it is not a budget-matched ablation of the K512 method.

This is a promising component direction, not a replacement claim: genuinely recovered changing prefix weights and the complete canonical dual-benchmark matrix remain untested. The branch is based on the published task/agent4-prefix-only-inversion branch.

Result: `coordination/results/TRR-0014.md`
Manifest: `experiments/TRR-0014/manifest.json`
Reproduction: `experiments/TRR-0014/README_R2.md`
Artifact/member hashes: `experiments/TRR-0014/artifact_index.json`

Validation: seven focused tests; native baseline and proposal-equivalence checks; three identical timing repetitions per cell; complete hash/decision gate with missing-cell, changed-prediction and changed-code rejection tests; 1,688 archived members verified. Failed methods, the memory-guard stop and the excluded non-equivalent batch build are preserved.
