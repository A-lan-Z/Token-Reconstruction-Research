# Access and cost audit

## What the selected predictor reads
- Frozen sanitized H, record ID/geometry, declaredBOS128000, and its own reconstructed earlier tokens.
- One supplied public Llama prefix as a stand-in for the prefix-recovery interface.
- That prefix's token embeddings, attention output and MLP down matrices for the metric; its own RMSNorm/MLP computation for an intrinsic lookup table.
- The already-required public tokenizer for deterministic suffix metadata.
- A frozen public A1 checkpoint ONLY in the separately labelled comparator arm.

The selected method does not read a trained inverse or token-prediction checkpoint. It does not use auxiliary text examples or optimize any parameters. Prefix-derived caches are checked against parameter identity/version; an update or replacement requires rebuilding. The tokenizer cache is static syntax metadata, not a corpus frequency or learned model.

The unavailable target LoRA is loaded by evaluator capture only. Prediction imports no target update loader. Source truth remains in evaluator_truth.json and is opened only by the scorer after the entire scoped matrix freezes. Opened R1 was used for development and its later probes are explicitly retrospective; R2 selections exclude all R1 paragraph hashes.

## What has and has not been demonstrated
The predictor accepts one frozen prefix and is designed for rebuilding between records. These experiments use a supplied public prefix, not weights obtained from a recovery algorithm. Integration with legitimately recovered changing weights and end-to-end cold-start tracking is not demonstrated. Public target matching is an architectural diagnostic; LoRA target mismatch is paired robustness, not a recovery trajectory.

## Timing boundaries
All R2 methods share synchronized input transfer, proposal, native candidate verification, cosine selection, own-cache commitment and output transfer. Each cell has three identical predictions and a median time. File serialization and hashing are recorded separately. Prefix loading, static tokenizer cache construction, prefix-dependent table rebuild and A1 asset setup are separate preparation phases.

Report both reuse of a fixed prefix and the conservative one-prefix-rebuild-per-record cost. The selected method is not free to rebuild. No unmeasured historical A1 training cost is subtracted to claim a speed win.512 padded logical candidate slots are actually simulated; unique proposals are not silently substituted for cost. Verification still uses the inherited native numerical contract.

A4096-row intrinsic build failed full-vocabulary equivalence to256 (210074604 differing BF16 values, max absolute0.5), so the faster construction is excluded.256 remains the scientific method. One initial qualifier exceeded a conservative reserved-memory cap due diagnostic difference temporaries; it stopped before new capture. Bounded comparison buffers passed without changing the cap, and the failure is preserved.
