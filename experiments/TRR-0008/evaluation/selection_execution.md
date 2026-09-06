# TRR-0008 create-only selection receipt

The owner-authorized selector ran at frozen commit `98bc91c0bf3a36319bec8528c1dc13da6f2c6387` after validating the frozen decision contract and producer-authored P06 hash-convention receipt. The selector command was the exact command recorded in `experiments/TRR-0008/coordination/root_freeze_ready.md`; its command-harness wall time was 10.3118 s.

The selection completed with status `FROZEN_TRR0008_SOURCE_SELECTION_NO_TRUTH` at `2026-09-06T13:29:40Z`. It selected 1,024 Finance and 384 Pile records. Selection diagnostics were:

- Finance: 1,792 excluded IDs, 179 excluded hashes, 4 P04 H129 sequence exclusions, 7 P06 H128 sequence exclusions, 12 duplicate final sequences, 1,024 selected, 0 invalid.
- Pile: 1,792 excluded IDs, 27 excluded indices, 7 excluded hashes, 41 P04 source-hash exclusions, 1 P06 H128 sequence exclusion, 326 invalid rows, 384 selected.

The create-only ledgers are:

- `experiments/TRR-0008/selection/source_selection.json`: 997,800 bytes, SHA-256 `ea9a7bf2edcc22eee1a8e791a331d423ccd940e0fcab3079b6b43cc456ee8e57`.
- `experiments/TRR-0008/selection/source_exclusions.json`: 18,344 bytes, SHA-256 `acdbabb1a923ed9acccc17076f615201baca0a2d0fb6a40177bf41bb96bd09c4`.

The selector execution receipt records `model_loaded=false`, `target_loaded=false`, `source_text_written=false`, `token_ids_written=false`, `truth_created_or_opened=false`, and `network_used=false`.

The subsequent hash-only reservation export completed at `2026-09-06T13:30:03Z` with status `READY_FOR_TRR0008_CAPTURE_HASH_ONLY`. It contains 1,408 public-record hashes and 1,408 H128 sequence hashes, with no record IDs, source indices, domain/style labels, source text, target labels, token IDs, model weights, or truth. Its output is 216,396 bytes, SHA-256 `0487b9dda91d7eb791c93e1ba704afcea22abfc21f003cad8b99984f523357a4`, reservation digest `a4470a255d5b16e7c51817236e1640b4633460254007a04cae083d68d1a4cabe`, and input selection ledger binding `ea9a7bf2edcc22eee1a8e791a331d423ccd940e0fcab3079b6b43cc456ee8e57`.
