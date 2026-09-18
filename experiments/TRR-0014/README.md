# Reproducing TRR-0014

This worktree is based on the preserved Agent4 prefix-only pilot branch. It adds a new deterministic proposal rule derived only from the supplied prefix's residual-write weights. The project charter remains authoritative.

## Environment and assets
- Ubuntu/WSL; Python3.12.3; torch2.10.0+cu128; transformers5.3.0; RTX5080 16GiB; two CPU threads.
- Public prefix: Llama-3.2-1B-Instruct revision9213176726f574b556790deb65791e0c5aa438b6, embedding+layers0..3, BF16, native FP32 RoPE.
- Actual public-prefix asset SHA256: 10fd7e98719037954a51299eb9cb680544a659400ee2431e8aac333626f5e61c.
- Asset roots are explicit in scripts/trr0014/native.py. They reference preserved sibling worktree backups. No downloads, missing asset substitution, fitting, or external model calls occur.
- The LoRA stage256 adapter is evaluator-only. Reconstruction never imports its loader or reads that asset.

## Execution
Set PYTHONPATH=src, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 from repository root. Every result uses a create-only destination; use a fresh output directory/worktree for replay. Do not overwrite originals.

1. Read PLAN.md, ALTERNATIVES.md and CONFIRMATION_PLAN.md.
2. Run pytest -q tests/test_prefix_weight_metric.py.
3. Run scripts/trr0014/preflight.py under scripts/agent4/watchdog.py.
4. Historical development only: lookup_probe.py then score_probe.py; weight_metric_probe.py then score_weight_metric.py. These negative/positive probes used already-opened development examples; they are not fresh evaluations.
5. Run predict.py --qualify under the watchdog. The first qualifier failed before fresh capture because inherited A1 candidate IDs were on CPU. Retry r2 fixed only explicit device transfer and passed; both receipts are preserved.
6. Freeze the method/selection plan. Run capture.py under the watchdog, retaining source selection and evaluator-only labels separately from sanitized observations.
7. Run predict.py under the watchdog with timeout600. It emits all three methods in both target conditions, three identical repeated outputs per record, and an immutable 144-prediction receipt.
8. Only after successful completion: run score_confirmation.py. It validates matrix completeness, observation/code/prediction hashes, shapes, ranges, declared winner rule, and timing; missing-cell, corrupted-prediction and modified-code negative checks must reject before it reads truth.
9. Rebuild the report from fresh_score.json and timing receipts. Scientific phase commits are recorded in each environment receipt; adding later scoring/report code does not retroactively change execution identity.

The common timed selector uses exactly the inherited A1 ordered proposals/native candidate helper/direct-cosine decision, with whole-array trace transfers. It passed candidate+token equivalence against the inherited comparator. No per-candidate Python scalar conversion is included for any arm. Timings include observation transfer, proposal, candidate verification, scoring, fresh own-cache creation/commitment, output transfer and synchronization. Serialization and hashing are outside this boundary and separate from prefix load/cache construction.

The new metric's cached matrix and transformed embeddings are deterministic temporary calculations, not another learned model. Build uses zero fitting examples and zero optimizer steps. Parameter replacement/in-place update causes an explicit cache invalidation error; rebuild from the supplied prefix before the next record. No prefix update is allowed mid-record.

## Scope
The fresh panel has16 natural128-token clips from two public-domain books and8 generated40-token code/identifier clips, each paired across matched weights and one actual trained prefix-LoRA shift. Both conditions use the identical public reconstruction prefix. Source freshness is local to these studies, not pretraining/corpus disjointness. Natural/stress and target conditions remain separate.

This is an exploratory mechanism study. It does not demonstrate a legitimately recovered prefix, cold-start tracking, broad target robustness, or a completed canonical dual-benchmark replacement. Earlier solvers and all unsuccessful alternatives remain preserved.
