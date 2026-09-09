This bounded pilot tests token reconstruction from a fixed public model prefix without a fitted proposer. The original six-clip CPU rounded-RoPE port recovered 80/90 tokens versus 90/90 for A1+A2, at 79.82× reconstruction wall. A two-clip static mismatch port recovered 29/30 versus 30/30.

A later native-loader check identified BF16 rounding of rotary frequencies. The correction exactly matches the independently loaded public checkpoint on the synthetic fixture. A separately reported retrospective GPU check on two existing clips recovers 29/30 tokens in 9.824s versus 30/30 in 1.688s (5.82×); it still misses the <=2× cost target. Search settings are unchanged. Original CPU results and the loader deviation are preserved.

No recovered-prefix or cold-start result is claimed. The canonical matrix remains incomplete. Twenty-five focused tests passed before the loader correction; actual native-loader, GPU gradient, comparator equivalence, resource-guard and independent asset-restore checks passed afterward. No P03 data, new target training, global registry changes or PR merges were used. The GPU was coordinated and released.

Result: `coordination/results/agent4-prefix-only-inversion.md`.
Manifest: `experiments/agent4-prefix-only-inversion/manifest.json`.
Reproduction: `experiments/agent4-prefix-only-inversion/REPRODUCE.md`.

Base: `task/TRR-0012` at `5bbc3bf42a81c814404cf84cb46d55f0d3418667`. Large backup assets remain local and are indexed by the manifest.
