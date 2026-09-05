# TRR-P01 independent CPU test evidence

This review evidence covers the task-local prototype implementation only. It
is not a pinned-checkpoint run and does not open evaluator truth.

Base checkout: `6b618760f50055dc5c8a62e830ab7a9761190cfe`.

Command:

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python3 -m pytest -q tests/test_trr_p01_boundary_prototype.py
```

Observed result on 2026-09-05: `8 passed in 0.92s`.

The tests cover deterministic cosine and raw-L2 lookup, smallest-token-ID
tie resolution, query/prototype chunk equivalence, raw public-embedding
lookup parity, full-vocabulary `[BOS,v]` construction, BF16 table storage,
correct two-token-per-input preparation accounting, create-only, truth-opened, and wrong-geometry metadata table artifact rejection, non-finite input rejection, causal
reference-cache isolation, and exact output invariance across batch sizes on a
fake prefix.

The batching test is an implementation-contract check with a deterministic
fake prefix. The separate pinned-model CPU qualification recorded below is the
equivalence evidence for the planned CPU preparation; a GPU qualification would
be needed only if the execution device or numerics change.

## Independent freeze and historical-control checks

The task-owned CPU integrity suite also covers the completed blind gate and
lightweight historical-control guards:

```
PYTHONPATH=src:scripts/trr_p01 CUDA_VISIBLE_DEVICES='' \
  python3 -m pytest -q \
  tests/test_trr_p01_freeze_score_integrity.py \
  tests/test_trr_p01_historical_comparator.py \
  tests/test_trr_p01_boundary_prototype.py
```

Observed result on 2026-09-05: `20 passed in 1.04s`. The freeze tests create a
synthetic finite public arm at the exact `[16,40,2048]` BF16 geometry and verify
that duplicate/foreign IDs, observation-byte changes, prediction shape/dtype
changes, wrong metadata, tensor/JSONL disagreement, and post-freeze row
reordering fail closed. A monkeypatched private truth loader remained uncalled
when a tampered prediction was rejected, demonstrating that the public
validation gate precedes truth opening. The historical-control tests enforce frozen-lens evaluation and reject
non-pilot geometry before any cache call; the comparator preserves the published
topk proposal order and A2 argmax tie behavior. They do not claim additional
pinned-model numerical equivalence.

The latest CPU qualification receipt records the pinned model revision,
cut-depth 4, 256 fixed probe IDs, BF16 outputs, and `torch.equal` equivalence
between chosen batch 256 and alternate batch 128 (`maximum_absolute_difference`
0.0). This is the required pinned-model CPU qualification for the currently planned
CPU execution; GPU equivalence would be required only if execution later moves
to GPU or changes numerics. The production batch remains 256.

The bounded `scripts/trr_p01/qualify_methods.py` entrypoint is CPU-only and
public-input-only. Its fixed largest cell is an 8-record, length-39 cached prefix
with 2,048 copied-cache K=256 candidate rows at position 39, followed by one
reference-token-220 probe and full-vocabulary cosine/L2 lookup. It records a
preflight estimate, pre/post-model/post-cell host guards, phase timings, peak RSS,
shapes/digests, cache lengths, and source/implementation identities. The
full-40 versus cached-39 prefix comparison is retained as a diagnostic field and
is not a correctness gate because it changes sequence geometry; native cached
outputs still require finite declared shapes and cache-state checks.

The full CPU public table preparation in `runtime/cpu-table-20260905` completed
with 501 forward calls, 256,512 input-token evaluations, BF16 outputs, 44.429
seconds table-build time, 47.088 seconds total build time, and a peak RSS of
3,634,324 KiB. The table tensor has the expected 128,256 x 2,048 BF16 payload
(525,336,576 bytes); the safetensors artifact is 525,337,024 bytes including
metadata. The post-model CPU guard passed before the table build.

## Bounded CPU method-cell qualification

The committed qualifier (`9fb635a7f66866da05a08cb6084da7b1704f13a3`) ran on
2026-09-05 in `/tmp/trr-p01` with `CUDA_VISIBLE_DEVICES=''` and completed with
status `CPU_METHOD_CELL_QUALIFIED` and exit status 0. The exact command and
short log are preserved in `qualifier-cpu-20260905.command.txt` and
`qualifier-cpu-20260905.log` beside this note.

The run used the frozen public plan, the completed BF16 full-vocabulary table,
and the published Alpaca lens (hash
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`). It
constructed eight public probe records of length 40 from the 256 declared
qualification IDs, prepared a native length-39 cache, and evaluated the
predeclared largest candidate cell: 2,048 copied-cache rows (8 records times
K=256) at position 39. It then evaluated reference token 220 for all eight
rows and ran full-vocabulary cosine and raw-L2 lookup for eight position-39
queries.

The public full-40 versus cached-39 diagnostic was exact on this CPU run
(`torch.equal=true`, maximum absolute BF16 difference `0.0`); the qualifier
still treats that geometry-changing comparison as diagnostic-only. The native
cached cell passed finite-value and shape checks, and the persistent cache
remained at length 39 after candidate and reference probes. Counts were 2,048
candidate simulations and cache commits, 312 persistent prefix commits, eight
reference evaluations/commits, four public-prefix call groups, and 2,688 public
prefix token evaluations.

The pre-model, post-model, and post-cell host guards all passed. Required
memory was 5,295,880,560 bytes; live available host memory was respectively
23,381,282,816, 23,260,028,928, and 21,353,287,680 bytes. Peak process RSS
was 5,475,844 KiB. The total measured runtime was 5.414 seconds, with 0.465
seconds for the K=256 candidate cell, 0.017 seconds for the reference probe,
and 0.903 seconds for full-vocabulary lookup. The public output tensor is
11,136,232 bytes and hashes to
`0196d7cd34342ee5ed7fffb9dfe0ea83bb3f50f78e1c58ef03cb2c074deed88a`; the
qualification evidence hashes to
`1592d7aa262eb9c27584a60f78e4cf431d6e31fd62cc90b94ccafa63f4187460`.
All output metadata states `truth_opened=false` and
`source_truth_included=false`; no target or source record was passed to this
process.
