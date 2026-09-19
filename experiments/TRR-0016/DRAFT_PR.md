Original K256 suffix expansion often spends its limited slots on fragments suggested by only one lookup result. This adds a deterministic rule that preserves the base proposals, counts how many distinct lookup candidates suggest each fragment, and prioritizes the repeated suggestions. It reuses the same prefix-derived tables and native A2 at K256, with no separately fitted A1 or new learned state.

On a preregistered fresh paired prose panel, errors fall from 29 to 10 for the matched target and from 25 to 8 for the shifted target, with exact clips increasing from 15/32 to 24/32 and from 18/32 to 25/32. A1+A2 still makes only 2 and 3 errors (30/32 and 29/32 exact), so this is an improvement over the prior method, not a replacement claim. Both canonical token and exact-record scores also improve. Cached inference is about 1.1-3.3% above baseline across fresh prose and canonical setups; table rebuild median is 0.94 seconds.

Validation: 1,408 frozen cells across 352 observations and four methods; three identical repetitions per cell; all 816 unchanged controls reproduced; seven focused tests; largest-cell resource qualification; full hash/argmax/truth gates and negative gate checks. The complete canonical matrix reports 54 cells, including 52 inherited results. Twenty-four evidence archives contain 2,829 hash-verified members.

The unsuccessful context rule, six exploratory orderings, two preparation failures, and one original-K256 timing stall are retained. The stall is not used to claim a speedup. Canonical single-record port caveats remain disclosed. Tests still use a supplied public prefix; an actual evolving recovered-prefix trajectory remains untested. The baseline default and the unfinished research goal are preserved.

Result: `coordination/results/TRR-0016.md`
Manifest: `experiments/TRR-0016/manifest.json`
Evidence index: `experiments/TRR-0016/artifact_index.json`
Reproduction: `experiments/TRR-0016/README.md`

This follow-up is stacked on the previously published result in PR #30.
