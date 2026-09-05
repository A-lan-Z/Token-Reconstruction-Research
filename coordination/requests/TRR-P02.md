# TRR-P02 — Diagnose the representation geometry before proposing another no-fit decoder

## Objective

Continue the independent agent-two workstream. TRR-P01 is a completed negative pilot: neither its static BOS-context dictionary nor its fixed-reference correction should be advanced unchanged.

This round is a bounded mechanism diagnosis, not a prototype-rescue sweep or a new benchmark campaign. Determine whether a simple, checkpoint-derived context treatment has credible support, and identify information that could also help the search-free learned-decoder workstream. A justified decision to stop this particular family is a useful result.

The owner values both (a) reconstruction without offline fitting and (b) a compact publicly trained decoder with no A2 search. Preserve that distinction. Diagnostic use of an existing fitted lens does not make a method fitting-free.

## Starting point and independence

Repository: `A-lan-Z/Token-Reconstruction-Research`.

PR #5 is the completed TRR-P01 pilot. The reviewed publication head is `e3e8a1de020598fb68c1ed8b64c0e155823817f5`, on `task/TRR-P01`, based on `task/TRR-0002`.

Verify current refs and preserve all existing work. Use your separate worktree and a new `task/TRR-P02` branch based on the relevant P01 snapshot. Do not merge earlier PRs, change agent one's checkout or environment, overwrite global coordination state, or wait for TRR-0004 outputs. Use task-local coordination. You may reuse P01's public resources and frozen artifacts read-only.

Read `RESEARCH_CHARTER.md`, the P01 report, and the relevant code. The charter remains authoritative; this brief defines only the scope of this exploratory round. No paid compute or interference with the other agent's running jobs is authorized. Prefer small CPU-compatible probes and existing artifacts; coordinate any shared compute.

## What the evidence already says

On P01's 16-record, 624-token panel:

- Boundary cosine: 255/624 on the matched public target; 243/624 on the shifted full-SFT target.
- Raw embedding cosine: 231/624 on each target.
- Fixed-reference-corrected cosine: 81/624 and 80/624.
- Historical A1 alone: 513/624 and 510/624.
- Historical fixed-K256 A1+A2 geometry port: 613/624 and 15/16 complete records on both targets.

All prototype/reference variants recover the first post-BOS token on all 16 records. Reference correction recovers the second post-BOS token on only 3/16 matched records and 4/16 shifted records. Its early failure therefore cannot be explained entirely by accumulated wrong-prefix tokens. The matched-model failure is large before target-weight mismatch is introduced.

These facts localize the problem; they do not prove a unique cause, rule out all no-fit methods, or establish that an arbitrarily chosen alternative reference would fix it.

## Suggested investigation

Choose the smallest useful set of probes below. You may replace them with a more decisive diagnostic, with a recorded explanation. Do not turn every suggested comparison into a mandatory cross-product.

### 1. Resolve the early failure under controlled public inputs

Use short public sequences such as `[BOS, u, v]`, with the same public forward path and known public diagnostic prefix. Verify the offset sign, position alignment, cache behaviour, and a reference-self-consistency case before attributing failure to representation geometry. This is a small mathematical/wiring check, not another infrastructure audit.

Hold token `v` fixed while varying earlier content at the same sequence length. Separately inspect context length or relative-position effects using explicitly controlled inputs. Do not call an experiment “position-only” when its visible tokens or attention context also change. A uniform translation of all RoPE positions is a numerical/control check, not equivalent to extending the prefix; do not assume a residual activation can be fixed by simply applying an inverse RoPE rotation.

Known prefixes here belong only to public component diagnostics. Any subsequently reported reconstruction method must use charter-permitted information, not a diagnostic teacher prefix.

### 2. Test the shared-offset hypothesis directly

For a modest, declared set of public token IDs and contexts, collect:

`z(C, v) = public_prefix(C + [v])[-1]`

Reuse the P01 baseline prototypes `b_v = z([BOS], v)`.

Inspect whether `z(C, v) - b_v` is approximately shared across tokens, or changes substantially with token identity. Token-pair differences across contexts provide a useful test:

`[z(C, v) - z(C, w)] - [z(C0, v) - z(C0, w)]`.

Measure not only total vector error but whether the error changes the relevant token ranking/margin. A geometrically small error can still matter along a discriminating direction. A small restricted vocabulary is acceptable for diagnosis, but its accuracy is not a full-vocabulary reconstruction result; clearly separate the two.

Do not generate a full 128k-token table for every context. Reuse forward activations across diagnostics and use targeted full-vocabulary checks only where they answer a remaining question.

### 3. Use the frozen historical lens to distinguish geometry from missing information

When the existing public lens is accessible, let `g(h) = W h + b` be its frozen affine projection. On shared public diagnostic queries, compare:

- observed activations against raw boundary prototypes;
- projected activations `g(h)` against projected prototypes `g(b_v)`;
- ordinary historical A1, which compares `g(h)` with public token embeddings.

Use the appropriate consistent normalization and preserve the original implementation as a comparator. Nothing is refit. The projected-prototype result is a **fitted-lens diagnostic**, not a no-fit candidate or a newly invented mechanism.

This comparison asks whether the poor static lookup chiefly reflects its distance geometry, whether the lens suppresses context variation, or whether a fixed prototype remains inadequate even after that transformation. Compare same-token cross-context variation and different-token separation; do not infer a mechanism from an accuracy change alone.

If the lens is unavailable, record that limitation and complete the independent public-context diagnosis rather than blocking the task.

## Scope and stopping

Focus first on the matched public model. Another target sweep is not informative until a proposed mechanism passes its matched diagnostic. Do not train new standalone decoders, repeat agent one's fitting comparisons, retry fixed-point inversion, or run another full P01 evaluation unchanged.

Do not automatically follow a failed correction with more references, more thresholds, or a larger table. A new correction is justified only by a measurable structure identified in these diagnostics. A single small check of such a structure is sufficient for this round; larger implementation and confirmation can be a later task.

Preserve original P01 records as opened development evidence. Use newly generated public diagnostic pairs where helpful, and record any exclusions needed for later confirmation. Label all teacher-prefix, restricted-dictionary, and fitted-lens controls explicitly. Do not expose fresh private evaluation truth for this diagnosis.

## Handoff

Save this assignment to `coordination/requests/TRR-P02.md`, the report to `coordination/results/TRR-P02.md`, structured evidence to `experiments/TRR-P02/manifest.json`, and state to `coordination/parallel/TRR-P02.json`. Commit and publish a clearly labelled follow-on PR against the appropriate parent without merging existing work.

Lead with the research decision, not a list of passing tests:

1. What caused, or most plausibly explains, P01's position-two failure? Which explanations were ruled out and which remain unresolved?
2. Is there evidence for a cheap context/geometry correction, or should this dictionary family be deprioritized?
3. What does the frozen-lens diagnostic imply for a small search-free learned decoder?
4. Is there one concrete next mechanism worth testing? State the evidence and the condition that would make you stop pursuing it.

Provide compact figures/tables or representative cases, commands, identities, compute cost, and limitations. No improvement is required. The goal is to replace an underdetermined negative result with a well-founded next decision, without duplicating agent one's work.

## Source pointers

All repository pointers below refer to P01 publication head `e3e8a1de020598fb68c1ed8b64c0e155823817f5`:

- `coordination/results/TRR-P01.md`
- `experiments/TRR-P01/review/final-results-audit.md`
- `src/token_reconstruction/trr_p01/boundary_prototype.py`
- `reference/strict_bos/round001_teacher.py`

Background for positional controls: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864. Query-dependent context mixing: Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762. These motivate controls; neither paper establishes a working inverse under this project's access model.
