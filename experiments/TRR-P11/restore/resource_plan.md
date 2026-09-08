# TRR-P11 restore resource and execution plan

The bundle is bounded to the two selected state files, the shared public E table,
hash-bound loader/decoder/configuration files, tensor identity sidecars, the
four-row public smoke input (Finance/Pile-prefixed keys), its two-method smoke
output, and the sibling prediction receipt. It does not copy
the training prefix, a tokenizer/source cache, private truth, or the old
temporary P09 states.

The current preflight observed approximately 284.27 GB free on the WSL
filesystem containing the primary package and 19.25 GB free on the Windows
filesystem mounted at /mnt/c. The expected bundle is about 1.2 GB, dominated
by the approximately 1.05 GB public E table. The Windows copy is gated on at
least 10 GB free before and after the copy, leaving room for a clean runtime
and receipts. The secondary copy is persistent NTFS under
/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012 and is recorded with
its observed st_dev and mount root. No physical disk separation is claimed.

The planned sequence is:

1. Agent 1 freezes the new B0/B1 states, public E, code/configuration,
   selection receipt, tensor inventories, four-row smoke inputs, and
   post-selection smoke predictions under the approved package schema.
2. The packager hashes the primary files and copies the same relative bundle
   to the Windows secondary root. The restore gate re-hashes both roots,
   checks path containment/no symlinks, verifies st_dev/mount provenance, and
   compares every file hash.
3. Agent 2 creates an empty persistent clean runtime directory outside both
   bundle roots and outside /tmp/training worktrees. materialize_clean_runtime
   retrieves from the verified secondary Windows copy by default, copies only
   the hash-bound bundle, and records source_boundary=secondary;
   verify_clean_runtime records the actual paths present in that directory. A
   missing file fails closed; nothing is fetched or substituted.
4. The consumer starts from the clean runtime with consumer_environment:
   PYTHONPATH contains only bundled code roots, HF_HUB_OFFLINE,
   HF_DATASETS_OFFLINE, and TRANSFORMERS_OFFLINE are set, and a fresh local
   offline-cache path is recorded. Installed torch/safetensors dependency
   identity is captured in the execution receipt.
5. After root grants the serialized CPU/GPU window, verify_tensor_identity opens
   each restored state and public E sidecar and checks every tensor key, shape,
   dtype, and canonical digest. The bundled fixed-readout adapter then runs
   the four agreed public_base rows for both methods and compare_smoke_prediction_files
   checks exact ordered IDs against the packager receipt. The runner first verifies
   the declared standard or Finance/Pile-prefixed input-key groups and their
   aggregate/per-record digests; it then validates the CLI sibling receipt
   against actual loaded bundle paths, dependency versions, and file hashes.
6. Source eligibility/selection may proceed in parallel after exclusion
   coverage is complete, the evaluation plan is frozen, and the bank identities
   are bound. Scientific prediction and scoring wait for the restore gate.
   This plan itself opens no model tensors, source text, labels, or truth.

The old P09 hashes, pointers, and historical predictions are provenance only.
They do not qualify a new package by themselves. If actual original state bytes
are recovered, complete file and tensor identities may qualify them after
independent verification under the amendment.
