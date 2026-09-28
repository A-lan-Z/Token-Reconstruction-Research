Frozen restored B1 retained nearly unchanged aggregate reconstruction accuracy after one predeclared public prefix-LoRA target trajectory. At 256 updates, Finance errors changed from 22 to 21 and Pile errors from 170 to 171 (16,256 scored tokens per domain). Pile nevertheless had 35 newly broken tokens and 34 improvements; this is bounded transfer evidence, not a formal noninferiority or broad robustness claim.

The target update was meaningful under the declared criterion: held-out next-token NLL fell from 3.4048 to 1.3769 with nonzero effective prefix weight changes. B1 stayed frozen. Signed full-vocabulary boundary analysis explained prediction changes; the early-direction forecast caught many later failures but produced substantial false positives. The limited A1+A2 control lost all exact Finance clips at the final stage, with every first error at position 29.

Evidence and reproducibility:

- Result: `coordination/results/TRR-P12.md`
- Immutable plan: `experiments/TRR-P12/manifest.json`
- Final evidence and raw artifact hashes: `experiments/TRR-P12/final-evidence.json`
- Reproduction: `coordination/results/TRR-P12-reproduction.md`
- Frozen score: `experiments/TRR-P12/score-r1.json`
- Error locations: `experiments/TRR-P12/postfreeze-breakage-r1.json`

Validation includes CPU tests, public numerical qualification, exact full-model/prefix smoke equivalence, repeated A1 predictions, full-vocabulary alignment, strict pretruth freeze checks, source exclusions, resource preflights, independent adapter backups, and root cross-artifact review. Failed attempts and evidence gaps are retained explicitly.

This PR remains draft and unmerged, based on published TRR-P11. P03 was not accessed; no paid compute, decoder rescue, additional trajectory, or PR merge occurred. The canonical dual-benchmark comparison remains incomplete.
