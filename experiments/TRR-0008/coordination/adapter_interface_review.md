# TRR-0008 selector/evaluation adapter interface review

**Status:** synthetic metadata review complete; no source selection, model capture, or truth preparation was run.

The selector-to-evaluation boundary is now aligned around the qualified TRR-0006 source descriptor contract. `scripts/trr0008_select_public.py` emits, for each domain, `dataset_key`, `dataset_id`, `split`, `revision`, Arrow file records, and the exact reserved-holdout partition. It emits the complete tokenizer descriptor from the trusted TRR-0005 producer. This shape is accepted by `trr0006_capture_public._validate_source_descriptors`, so capture and truth will bind to the same dataset revision, file records, tokenizer files, and holdout ranges after owner authorization.

The selector payload also carries `planning_bindings` for the decision contract, identity inventory, and planning status file. `trr0008_eval_capture._validate_planning_bindings` verifies those file records before capture. The metadata remains identity-only: selected row records contain record IDs, source and sequence SHA-256 values, dataset identity, row indices, and token-length fields; they do not contain source text, token IDs, target labels, or truth.

Capture is configured for batches of 8 records with 192 input positions and retains only the first 128 positions in BF16. The geometry estimate is 6,291,456 bytes (6 MiB) for one 8x192x2048 BF16 full-forward batch and 4,194,304 bytes (4 MiB) for its retained 8x128x2048 compact tensor. A full Finance domain is 805,306,368 bytes (768 MiB) before file metadata and a full Pile domain is 301,989,888 bytes (288 MiB); one compact target condition is 512 MiB for Finance and 192 MiB for Pile, or approximately 1.375 GiB for both domains across the two target conditions. These are geometry estimates, not measured capture memory or final file footprints. The adapter qualifies the largest Finance 8x192 batch and invokes the existing TRR-0005 resource guard before each capture; no capture was run in this review.

The qualified producer path is reused at each boundary: capture loads the trusted TRR-0005 public prefix, uses the TRR-0006 row materializer and batch construction, and retains the existing TRR-0004 resource guard. Truth reuses the TRR-0008 selection loader and TRR-0006 materializer, then compares materialized record IDs with the frozen selection order before producing labels. No Arrow row, tokenizer, model, activation, prediction, or truth tensor was opened by these checks.

The synthetic check command was:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python3 -m pytest -q tests/test_trr0008_adapter_interface.py
```

It passed 5 tests. The tests verify exact descriptor acceptance by the qualified capture validator, planning-binding file-record validation, truth reuse of the capture materializer with matching order, and rejection of materialized order drift. The state-dependent selector tests now use explicit temporary draft/pending fixtures rather than relying on the live planning state.

The producer-authored P06 generic hash-construction receipt is copied byte-identically at `experiments/TRR-0008/planning/approved_opaque/p06_hash_construction_receipt.json` (2,339 bytes, SHA-256 `b06ca9ccae7b831318604351ce76f183a8c2745780494e20b263279f146ba92c`). It specifies Pile original source-text UTF-8 bytes without stripping or chat rendering and Finance compact UTF-8 canonical JSON `[system,user,assistant]` with the documented field fallback and stripping rules; these match the trusted local producer. The task-local planning status now records `VERIFIED_P06_PRODUCER_CONFIRMATION`. The approved opaque P06 export and this generic receipt are the only P06 files copied into this task; no P06 source identities, provenance, results, or holdout content was opened.

The decision contract is frozen. Selection remains blocked until root authorizes the create-only selection command. This review does not create selection, exclusion, reservation, capture, prediction, or truth artifacts.
