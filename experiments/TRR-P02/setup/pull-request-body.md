P01's early prototype failures remained ambiguous between wiring errors and context-dependent representation changes. P02 diagnoses the matched public model on 46 declared context/token pairs, with 12 targeted full-vocabulary checks, without new fitting or private evaluation truth.

The evidence supports token-dependent deformation and deprioritizing the static/shared-offset no-fit variant. On the same 12 rows, raw boundary prototypes recover 7/12 tokens, frozen-lens projected prototypes 11/12, and historical A1 9/12. Reference subtraction recovers 8/12 overall but worsens the eight non-reference cases from 5/8 to 4/8. Public-panel mean-centering gets 40/40 in the restricted nine-candidate control and 0/12 against the full vocabulary.

The frozen lens improves relative token separation; absolute cross-context L2 variation increases. These are fitted-lens, public teacher-prefix diagnostics, not benchmark scores or evidence of a deployable no-fit correction. A later bounded low-rank compression check could test compactness of the useful fitted geometry; P02 does not perform that check.

Validation: seven focused tests passed, longest-cell batching matched serial outputs exactly, and all measured source and raw artifact hashes were verified. Cached context blocks matched full context blocks; endpoint differences remain, so rank-level cache equivalence is not claimed. The retained CPU run took about 12 seconds and peaked at 6.03 GiB RSS. Three failed executions and one pre-Python launcher failure are preserved and excluded from scientific results.

- Human result: `coordination/results/TRR-P02.md`
- Structured evidence: `experiments/TRR-P02/manifest.json`
- Independent review: `experiments/TRR-P02/review/final-results-audit.md`
- Exact public-case exclusions: `experiments/TRR-P02/setup/public-diagnostic-exclusion.final.json`
- Task-local state: `coordination/parallel/TRR-P02.json`
- Measured source: `470b6f1becfaa6da110048302938feddd7204c30`

This follow-on branch starts at P01 publication head `e3e8a1de020598fb68c1ed8b64c0e155823817f5` and targets `task/TRR-P01`. Existing PRs remain unmerged; global coordination state, benchmark protocols, and the other workstream's checkout remain unchanged.
