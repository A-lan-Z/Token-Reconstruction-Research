The pilot tests whether a public model prefix can propose and verify tokens without a fitted A1/B1 guesser. On six matched16-position BF16 CPU clips, bounded Adam inversion recovered80/90tokens and1/6clips, versus90/90 and6/6 for the native-policy A1+A2 comparator; reconstruction took1180.69s versus14.79s. On two static Vikhr clips, the result was29/30 versus30/30tokens. The matching result on that identical two-record subset was also29/30.

The report distinguishes missed candidates from an overly strict BF16 stopping gate, so the measured CPU gap is not claimed as a limit on corrected gradient inversion. No recovered-prefix or cold-start tracking result is claimed. Settings and outputs were frozen before scoring, all record denominators were retained,25focused tests passed, and actual model/dependency restores were verified. No GPU, P03 data, new target training, global-registry changes, or PR merges were used.

Result: `coordination/results/agent4-prefix-only-inversion.md`.
Manifest: `experiments/agent4-prefix-only-inversion/manifest.json`.
Reproduction: `experiments/agent4-prefix-only-inversion/REPRODUCE.md`.

Based on published implementation commit5bbc3bf42a81c814404cf84cb46d55f0d3418667; target this PR at `task/TRR-0012` to isolate the pilot. Canonical dual-benchmark comparison remains incomplete. Large actual backup assets remain local and are indexed by the manifest.
