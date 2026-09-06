# TRR-0006 private label binding

The final driver consumes one producer binding JSON and one sidecar outside the
reconstruction root. The binding is metadata only; it contains no token or
label arrays.

The binding uses schema
`token-reconstruction.trr0006-private-label-binding.v1`, status
`PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT`, task ID `TRR-0006`, and
`truth_opened: false`, `reconstruction_root_contains_truth: false`. Its
`truth_file` is an absolute `{path, bytes, sha256}` record. It includes file
records `decision_plan` and `source_selection`, `observation_sha256` for all
four cells, `record_ids_sha256` for `pile` and `finance`, and
`truth_tensor_keys: ["finance__token_ids", "pile__token_ids"]` in any order.
The two record-order digests are the SHA-256 of canonical JSON arrays of IDs,
matching `selection_rule.record_ids_sha256` in `source_selection.json`.

The sidecar uses schema
`token-reconstruction.trr0006-private-label-sidecar.v1` and contains exactly
`pile__token_ids` and `finance__token_ids`, each `torch.int64` with shape
`[1536, 128]`. Position zero is BOS `128000`; all 128 values are valid
vocabulary IDs. Each domain tensor is shared by its `public_base` and
`public_lora_2601` target cells. Sidecar metadata records the schema and task;
when present, plan, selection, observation, and record-order digest fields
must agree with the binding. The driver opens this sidecar once, after the
complete public prediction/timing gate, and retains only opaque file/tensor
hashes in task outputs.
