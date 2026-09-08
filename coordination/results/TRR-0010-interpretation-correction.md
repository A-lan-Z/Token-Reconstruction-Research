# TRR-0010 interpretation correction

Status: metadata-only correction prepared by TRR-0011. The historical PR20 predictions, scores, selected model states, and decision receipts are preserved. This note changes the evidentiary interpretation of the final panel; it does not rerun, retrain, or reselection any arm.

The frozen TRR-0009 public panel is the checkpoint-selection validation set: 256 Finance records and 128 Pile records. Its selection ledger is [TRR-0009 `selection_v2/source_selection.json](../../experiments/TRR-0009/selection_v2/source_selection.json), and its panel binds that ledger with record-order fingerprints.

The PR20 final panel is [TRR-0010 `source_selection.json`](../../experiments/TRR-0010/evaluation/source_selection.json), with its captured panel at [TRR-0010 `panel.json`](../../experiments/TRR-0010/evaluation/public_capture_watchdog_r5/observations_v1/panel.json). Exact metadata comparison gives:

| Domain | TRR-0009 checkpoint-selection validation | PR20 final panel | Source-ID intersection | Final-sequence intersection | Ordered relationship |
| --- | ---: | ---: | ---: | ---: | --- |
| Finance | 256 | 128 | 128 | 128 | PR20 is the first 128 ordered TRR-0009 Finance records |
| Pile | 128 | 128 | 128 | 128 | PR20 is the identical ordered TRR-0009 Pile panel |
| Total | 384 | 256 | 256 | 256 | Every PR20 final record overlaps checkpoint-selection validation |

Matching used exact per-domain equality of `record_id`, `public_record_sha256`, and `final_sequence_sha256`. Ordered fingerprints use the selector's canonical JSON-list SHA-256 convention. Finance ID fingerprints are `a1b3b5cd4da3a3960b97f90968a5a763d279a24861b7a7a1fd01e6b3a2f81f87` for TRR-0009 and `7f394949247231bd269f66a7529ea570e9403bbbc3fc47d85ded855276581cf1` for PR20. Pile's ID fingerprint is `1f665eaf7415e66525229470a676aaae2f5483e2b31de811ce7594169a9ed520` in both. The corresponding final-sequence fingerprints are Finance `9bb86a3937409b5a585375211ef8e66b701641b869b9c17a0f2fb29b2d935b0f` versus `5f4486af333fe4137f4bce37ae33c625de1adaea3fd44be8a3b2e44f01c8de69`, and Pile `ad3e8dbb4d183cc5083df208395a483f513da4b7c3e0ffa8b23c5186ad47fbba` in both. There were zero ID-to-sequence mismatches.

Therefore, PR20's affected panel should be described as **selection-overlapping development evidence**, not as an independent text-generalization estimate. The overlap is a source-identity and sequence-identity finding. It does not allege that inference read answers or that the records were used for gradient fitting. The historical numerical outputs remain available for reproducibility, with this limitation attached to any claim that treats them as unused-text confirmation. No retraining or checkpoint reselection is authorized by this correction.

The targeted selector defect is concrete. `scripts/trr0010_select_public.py::_extra_exclusion_paths` delegates to `trr0009_select_public.py::_known_exclusion_paths`, whose explicit list ends with TRR-0008 inputs. `select_public` then passes that list, plus the final-B1 ledger, to the recursive identity scanner. The scanner would recognize nested `record_id`, `public_record_sha256`, and `final_sequence_sha256` fields if the TRR-0009 selection ledger were supplied, but the frozen TRR-0010 exclusion union contains no `experiments/TRR-0009/selection_v2/source_selection.json` descriptor. The later registration check validates the already-produced selection and cannot retroactively make it disjoint.

The focused regression should hash-bind the TRR-0009 selection ledger as an explicit TRR-0010 exclusion input, verify that its exact descriptor appears in the generated union, and reject representative Finance and Pile candidates by each of the three identity fields. It should fail closed for an omitted ledger, a changed ledger path/bytes/hash, or a changed identity. With the current ranges, excluding the entire TRR-0009 selection prefix should make the old attempted panel fail closed or require a pre-registered disjoint extension; filtering records after selection is insufficient. The regression must remain identity-only and must not read model outputs, target labels, truth, or scores.

The full machine-readable receipt is [overlap_audit_v1.json](../../experiments/TRR-0011/setup/overlap_audit_v1.json), with the focused recommendation in [overlap_regression_recommendation_v1.json](../../experiments/TRR-0011/setup/overlap_regression_recommendation_v1.json).
