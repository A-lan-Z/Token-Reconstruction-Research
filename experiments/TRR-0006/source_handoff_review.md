# TRR-0006 source handoff review

Review date: 2026-09-06. This is a bounded provenance and exclusion review. It
reads task-local JSON and receipt metadata only. It does not open the private
truth sidecar, read a P04 private ledger or holdout, rerun capture, run a model,
or modify a script.

The handoff is reproducible for the canonical public panel. The remaining
cross-study limitation is real and must stay in the final report: P04 did not
provide per-record target-fit identities, source ranges, or replay sequence
hashes, so the selection is not a complete disjointness certificate for those
rows.

## Immutable handoff bindings

The following hashes were checked against the files currently present in the
TRR-0006 worktree:

| Artifact | Path | SHA-256 |
|---|---|---|
| frozen decision plan | `experiments/TRR-0006/decision_plan.json` | `edeb1bb05a00ad3f580d415f1b9ab632f588b3c50fd1ed9c2f2e055737c766ea` |
| v3 eligibility inventory | `experiments/TRR-0006/eligibility_inventory_1536_projection_v3.json` | `cad1697fc575b27dcd01a80f134d31cc48c6080ebb31706fd87351dd259a67a7` |
| source selection | `experiments/TRR-0006/source_selection.json` | `75909aaf0f9e40176c197d86c09651097010a11519855f1db3dc50fe5e754f43` |
| P04 hash-only exchange copy | `experiments/TRR-0006/coordination/p04_reservation_hashes.json` | `98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904` |
| canonical source panel | `experiments/TRR-0006/panel_capture_v1/panel.json` | `951ded7d07d5be848af9c55ac0eb202b77053b27da7339b9ac43c64bce75b04a` |
| canonical observation manifest | `experiments/TRR-0006/panel_capture_v1/observations.json` | `abeb22804a38947ad67b7cf3a4584fcb569e7dcad7fba48234f64c773511ff0e` |
| canonical capture receipt | `experiments/TRR-0006/panel_capture_v1/capture.json` | `a367deb88bc02466a1de7aff2a9928e1c9d8109a43869e7ab2aabbe2972a2977` |
| canonical capture execution receipt | `experiments/TRR-0006/capture_execution_v1/execution_receipt.json` | `db83f6f65137ff1c12ef48e888e41bfd590876018c7f230f4a5edb9b77d95664` |
| prediction registration | `experiments/TRR-0006/prediction_registration.json` | `097965475e4623f7e884331189751f067e794d88c9358cdce4c7299eaa88207a` |
| complete public freeze v2 | `experiments/TRR-0006/predictions_v1/freeze_receipt_v2.json` | `cd74ec336d10885d120b139acf6353a0f4872e5e659b5a3c1219c770f89c354f` |

The selection and canonical panel both bind method-freeze SHA
`96330c8b935ff33ab3f69600c4456e556f901084ad2958e49287d2d329caa422`, the
published Pile and Finance ranges, and the same source-ID digests:

- Pile: `b76254217d4c173c22639f221b2bda7d6fb2274a8c877f8addb7d560252fabbd`
- Finance: `e7e8e78bf1527252d12888b69f727bd70590d5c14cc2ed242231bee8a934c118`

The registration, prediction run manifest, and freeze v2 all point to
`panel_capture_v1/observations.json`. The duplicate path appears in exclusion
records and diagnostic preflight text only; it is not bound by the registration
or public freeze.

## Population and selection accounting

The frozen population is Pile `[7000,10000)` and Finance `[12000,20000)`, with
selection seed 5005. The clip has 128 tokens including BOS and 127 scored
post-BOS tokens. Public capture uses batches of 8 and 192-token inputs, stores
H128, and the prediction runner processes one source record at a time in
chunks of 8. There are 1,536 unique sources per domain, 3,072 unique sources
total, and 6,144 paired source-target cases per method across the two target
conditions. The target conditions are `public_base` and `public_lora_2601`.

The count-only v3 inventory scanned the full available ranges without selecting
rows or creating truth. It found 2,413 eligible Pile rows (surplus 877 over
1,536) and 6,859 eligible Finance rows (surplus 5,323). The full scan covered
3,000 Pile and 8,000 Finance rows. Its domain accounting was:

| Domain | Eligible | Invalid | Duplicate final sequence | Unique identity-set cardinalities (IDs / hashes / indices; overlaps possible; not row-rejection counts) | P04 opaque exclusions (source / sequence) |
|---|---:|---:|---:|---:|---:|
| Pile | 2,413 | 366 | 1 | 724 / 806 / 272 | 53 / 0 |
| Finance | 6,859 | 2 | 941 | 452 / 544 / 153 | 4 / 12 |

The selector then reapplied the same exclusions in the frozen natural order and
stopped after the first 1,536 eligible rows in each domain. Its diagnostics are
prefix counts for rows encountered before reaching that target, rather than
full-range totals: Pile encountered 128 ID, 6 hash, 18 index, 24 opaque-source,
257 invalid, and 1 duplicate-final-sequence exclusions; Finance encountered
128 ID, 16 hash, 3 opaque-sequence, 0 invalid, and 61 duplicate-final-sequence
exclusions. These two tables have different denominators and must not be added
or compared as if they were the same scan.

## Accessible P04 coverage and limits

The approved hash-only exchange was copied byte-for-byte into the task-owned
coordination directory. The inventory and selector both report
`metadata_consumed_for_row_exclusions: true`; they applied 1,720 available
opaque source hashes and 520 available 129-token sequence hashes. The full
inventory reports 53 Pile and 16 Finance known P04 rows excluded by those
opaque identities. Available field summaries include correction source and
sequence fingerprints (256 each), validation fingerprints (192 each), fresh
panel fingerprints (72 each), and 1,200 fit-replay rendered fingerprints.
Only aggregate counts and derived hashes are retained in this review.

The exchange explicitly marks target-fit per-record IDs, target-fit source
ranges, target-fit sequence fingerprints, and replay sequence hashes
unavailable. No absence is inferred for those rows, and no private P04 ledger
was inspected. The available P04 sequence hashes cover BOS plus 128 post-BOS
tokens (129 tokens), while TRR-0006 deduplicates its 128-token clip (BOS plus
127 scored tokens). The task’s own 128-token source/sequence deduplication is
complete, but a 129-token exchange cannot certify that a different 129th token
did not follow an otherwise matching 128-token prefix. This caveat should
remain next to the target-fit limitation.

## Canonical versus superseded capture

The independently launched `public_observations_v1` capture completed at
05:40:43Z and used the same selection, methods, source ranges, and 8-by-192
geometry. It is retained only as an excluded attempt. The canonical wrapper
launched at 05:41:20Z; the canonical producer ran from 05:41:21Z to 05:41:54Z
(33.576 seconds inner, 35.283 seconds wrapper receipt). The duplicate took
33.921 seconds. The exclusion artifacts are:

- `experiments/TRR-0006/public_observations_v1/superseded_duplicate_attempt.json`, SHA-256 `73367404be5be0538d830781ee73990293d274e070b6f90d4d12485bd706633e`;
- `experiments/TRR-0006/duplicate_capture_exclusion.json`, SHA-256 `a70299526decb83d4008c3d407672077fcf937668d2223aedfd17fceea2551e9`;
- duplicate capture receipt `public_observations_v1/capture.json`, SHA-256 `b05c541a49340e7c168f1f6e4f4b7c9cee956b7261ba1b35743c6057acd03de`.

The exclusion record says the duplicate was not used for registration or
scoring, no truth was opened for the choice, selected records did not change,
and the extra preparation cost is retained. Its recorded chunkwise comparison
is tensor-equal for activations, attention masks, and position IDs in all four
cells, with equal cell metadata; descriptor/file hashes remain distinct, so it
is appropriately excluded from the canonical path. The later registration and
freeze v2 provide the objective binding to `panel_capture_v1`.

## Chronology and failed gate

The inventory ran 05:00:51–05:01:08Z. The one-time 1,024-to-1,536 sample-size
revision was frozen in the decision plan at 05:09:14Z, before source selection
at 05:30:17–05:30:24Z. Capture and prediction then used the fixed selection.
Prediction ran 06:10:32–06:12:05Z at code commit
`33dc6258614188927751ade45a0f0a2efe1f8361` and produced all eight entries.

An initial freeze receipt omitted optional plan and panel bindings. The private
truth preparation failed closed at 06:14:21.942Z with
`truth_created: false` and `truth_opened: false`. Receipt v2 then bound the
same immutable prediction artifacts plus the frozen plan and canonical panel.
The later public receipt records truth preparation outside the reconstruction
root and a post-gate truth open; this review read only that receipt’s metadata,
never the sidecar payload.

## Workspace boundary

The task worktree is on `task/TRR-0006` at prediction commit
`33dc6258614188927751ade45a0f0a2efe1f8361`. There are no tracked-file changes
in the task worktree; current untracked files are task execution artifacts.
The original checkout remains on `task/TRR-0003` at `eab3fc21fdae67fe628a42620029e25829a188b1` with no tracked changes. It currently
contains the two preexisting TRR4 untracked files and an untracked
`coordination/requests/TRR-0006.md` packet copy. That packet has an earlier
mtime (13:44:34 +1000) than the complete packet in the isolated worktree
(14:39:54 +1000), so this review cannot attribute its creation to the isolated
worktree actions; preserve it and do not claim the original checkout has no
untracked task file. The TRR-0005 worktree remains at the published parent
`3a7e8f579e713c3e41d02639237042ca26fd019b`; this review did not inspect or
modify any parallel private study workspace or holdout.

## Issues for final publication

1. `capture_preflight.json` still names
   `experiments/TRR-0006/public_observations_v1/failure.json` and
   `public_observations_v1/timeout_failure.json` as planned failure receipts;
   neither file exists. The canonical execution receipt is present under
   `capture_execution_v1`. This is a documentation gap, not evidence of a
   failed canonical capture, but the final manifest should avoid presenting
   those paths as existing receipts.
2. The preflight metadata records a planned outer timeout of 1,800 seconds,
   while the canonical launch and execution receipts record a 900-second
   timeout. The final execution record should preserve the actual 900-second
   command and distinguish it from the earlier planning value.
3. The first freeze receipt is retained as a failed-gate history artifact; all
   truth/scoring claims should cite `freeze_receipt_v2.json`, whose plan and
   panel bindings are complete.

## Compact metadata safe to archive after scoring

A later public archive can retain the schema/status flags and opaque file
records for the decision plan, source selection, panel, observation manifest,
prediction registration, freeze v2, and the private truth binding. For the
private binding, the safe compact fields are the sidecar byte count and SHA-256,
plan/selection hashes, four observation hashes, two source-order digests, the
two tensor key names and tensor SHA-256 values, geometry, paired-condition
labels, and `truth_opened`/`reconstruction_root_contains_truth` flags. The
binding itself contains no token arrays. Do not copy the sidecar, decoded source, or token IDs into the repository.
Post-freeze aggregate correctness metrics may be retained through the scored
result and its public manifest; this review does not duplicate those metrics
or any private payload.
