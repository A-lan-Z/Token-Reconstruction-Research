# TRR-P11 restore-gate contract

This contract is the deployment boundary for the rebuilt fixed-readout pair. It
is a recovery gate, not a fitting or evaluation protocol. It contains no source
text, target labels, or evaluation-truth payloads; the selection receipt is only a
hash-bound completion record.

## Required artifacts

The packager publishes one JSON descriptor with schema
token-reconstruction.trr-p11-restore-gate.v1. It binds these roles:

- current_fixed (bank B0) and expanded_fixed (bank B1): the selected
  safetensors state files, with the selected step and model identifier recorded;
- public_readout: the shared public E table;
- loader_code, decoder_code, and package_cli: exact bundled source
  files used by the deployment adapter, including their byte and SHA-256
  bindings;
- package_manifest and frozen_config: the package contract and immutable
  decoder geometry/configuration;
- selection_receipt: the actual post-selection receipt, with selected state
  hashes and explicit smoke/evaluation-truth boundary fields;
- one tensor-identity sidecar for each state and for the public E table;
- smoke_input: the already-opened, truth-free public observations, masks, and
  positions; its explicit key layout is standard or Finance/Pile-prefixed;
- smoke_expected: the prediction IDs emitted by the packager after checkpoint
  selection was complete.

Every file binding has a relative path below a declared copy root, byte count,
and SHA-256. Symlinks and .. escapes are rejected. The descriptor records the
actual st_dev of each copy root and the resolved mount root observed by the
validator; self-assigned boundary labels are not evidence of independence.

## Two-copy requirement

Each required artifact has two copies. The primary is the persistent WSL model
package under the Agent 1 output root. The secondary is the persistent Windows
copy under /mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012. The gate
requires:

1. distinct resolved roots and distinct st_dev values;
2. the secondary root to be under /mnt/c/, with the recorded mount device and
   mount provenance matching what the validator observes;
3. both roots to be real directories, without symlink traversal;
4. every artifact path to remain inside its own root; and
5. equal bytes and SHA-256 across the two copies.

Temporary paths, /tmp, unverified caches, and training-checkout paths are not
valid clean-runtime dependencies. The primary may be the packager's persistent
output while it is being copied; the restore runner must use a separate clean
runtime root. The clean root is create-only and must not be inside a training
worktree or /tmp.

## Tensor identity

The tensor sidecar records the state/readout file binding and, for every tensor,
its key, dtype, shape, and digest of canonical contiguous CPU bytes prefixed by
shape and dtype. The restore adapter verifies the sidecar against the actual
safetensors file after copying. File identity and tensor identity are separate
checks; one cannot substitute for the other.

## Exact public smoke

The smoke is fixed before fitting outcomes are inspected: the first two already
opened TRR-0010 public_base records from Finance followed by the first two from
Pile, in that order, for each of the two new selected methods. The input rows,
masks, positions, geometry, ordered record IDs, opaque source identities, and
per-record activation/mask/position slice hashes are bound before selection.
Record labels alone do not qualify the smoke fixture. The descriptor binds the
exact standard or Finance/Pile-prefixed tensor keys, aggregate tensor digests, and
per-record slice digests. The packager records predictions only after normal
checkpoint selection is complete. The clean Agent 2 runner loads the restored package and
must produce the same ordered prediction IDs and output digest for each method.
The smoke output is a deployment identity check; it is not a score and is not a
new evaluation panel.

Historical P09 hashes, pointers, or old PR20 prediction files alone cannot
satisfy this contract for rebuilt P11 states. If actual original state bytes
are later recovered, their complete file and tensor identities may qualify
under the amendment after independent verification. A P11 package is valid
only once actual selected weights, the tensor inventories, both copies, and the
exact smoke receipt all exist.

## Minimal descriptor shape

Each asset has one relative path and a copy binding for each boundary. A
selection receipt is a required asset, rather than an unbound boolean. The
validator checks that its selected state hashes equal the state files it
hashes.

    {
      "schema": "token-reconstruction.trr-p11-restore-gate.v1",
      "task_id": "TRR-P11",
      "status": "RESTORE_PACKAGE_CANDIDATE",
      "package_id": "trr0012-fixed-pair-<immutable-id>",
      "selection": {
        "complete_before_smoke": true,
        "smoke_used_for_selection": false,
        "independent_evaluation_truth_opened": false,
        "development_labels_used": true,
        "checkpoint_reselection_after_smoke": false
      },
      "boundaries": {
        "primary": {
          "boundary_id": "wsl-primary",
          "kind": "wsl_persistent",
          "root": ".../outputs/TRR-0012/model-package",
          "st_dev": 0,
          "mount_root": "/home"
        },
        "secondary": {
          "boundary_id": "windows-secondary",
          "kind": "windows_persistent",
          "root": "/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012",
          "st_dev": 0,
          "mount_root": "/mnt/c"
        }
      },
      "assets": {
        "current_fixed": {
          "bank": "B0",
          "selected_step": 0,
          "model_id": "new-model-id",
          "relative_path": "states/current_fixed.safetensors",
          "copies": {
            "primary": {"bytes": 0, "sha256": "..."},
            "secondary": {"bytes": 0, "sha256": "..."}
          },
          "tensor_identity": {
            "relative_path": "identity/current_fixed.json",
            "copies": {
              "primary": {"bytes": 0, "sha256": "..."},
              "secondary": {"bytes": 0, "sha256": "..."}
            }
          }
        },
        "expanded_fixed": "...same shape...",
        "public_readout": "...same shape...",
        "loader_code": "...same shape without tensor_identity...",
        "decoder_code": "...same shape without tensor_identity...",
        "package_cli": "...same shape without tensor_identity...",
        "package_manifest": "...same shape without tensor_identity...",
        "frozen_config": "...same shape without tensor_identity...",
        "selection_receipt": "...same shape without tensor_identity...",
        "smoke_input": "...same shape without tensor_identity...",
        "smoke_expected": "...same shape without tensor_identity..."
      },
      "smoke": {
        "truth_free": true,
        "record_order": [
          "finance/public_base/000", "finance/public_base/001",
          "pile/public_base/000", "pile/public_base/001"
        ],
        "methods": ["current_fixed", "expanded_fixed"],
        "record_bindings": [
          {
            "record_id": "finance/public_base/000",
            "source_identity_sha256": "...",
            "activation_slice_sha256": "...",
            "attention_mask_slice_sha256": "...",
            "position_ids_slice_sha256": "..."
          }
        ],
        "input_asset": "smoke_input",
        "expected_asset": "smoke_expected",
        "input_key_layout": "domain_prefixed",
        "input_tensor_keys": {
          "activations": ["finance__activations", "pile__activations"],
          "attention_mask": ["finance__attention_mask", "pile__attention_mask"],
          "position_ids": ["finance__position_ids", "pile__position_ids"]
        },
        "selection_receipt": {
          "complete_before_smoke": true,
          "smoke_used_for_selection": false,
          "independent_evaluation_truth_opened": false
        },
        "input_tensor_digests": {
          "activations": "...",
          "attention_mask": "...",
          "position_ids": "..."
        },
        "prediction_tensor_digests": {
          "current_fixed": "...",
          "expanded_fixed": "..."
        }
      },
      "consumer": {
        "entrypoint_asset": "package_cli",
        "python": "/usr/bin/python3",
        "receipt_schema": "token-reconstruction.trr0012-prediction-receipt.v1",
        "receipt_task_id": "TRR-0012",
        "observation_asset": "smoke_input",
        "output_relative_path": "runtime/smoke_restored.safetensors",
        "receipt_relative_path": "runtime/smoke_restored.receipt.json",
        "package_root_arg": "--package-root",
        "observations_arg": "--observations",
        "output_arg": "--output",
        "device_arg": "--device",
        "device": "cuda",
        "numerical_settings": {
          "device": "cuda",
          "projection_dtype": "float32",
          "argmax_dtype": "float32",
          "preserve_bos": true,
          "vocabulary_size": 128256
        },
        "numerical_settings_sha256": "...",
        "code_relative_roots": ["code"],
        "timeout_seconds": 120,
        "dependency_id": "pinned-installed-dependencies"
      },
      "clean_runtime": {
        "root": ".../trr-p11-restore-runtime",
        "training_worktree": false,
        "temporary": false
      }
    }

The example uses placeholders and is not executable. The validator API in
scripts/trr_p11/restore_gate.py is the executable definition. The consumer
runner is invoked as the bundled entrypoint's predict subcommand with
package-root, observations, output, and device arguments. It must create the
sibling output.receipt.json. The receipt schema is
token-reconstruction.trr0012-prediction-receipt.v1: its runtime object records
the pinned Python executable, dependency versions, device, and imported module
file bindings; methods state_file_binding and readout.file_binding bind the
loaded weights and public E; observations binds the smoke fixture and its
aggregate tensor digests; output binds the generated predictions. The gate
requires the imported loader/decoder paths to resolve inside the hash-bound clean
bundle and rejects training or temporary paths. It runs with shell=False from the
clean root and cannot use a training checkout or fallback download.

## Canonical tensor digest

The selection_receipt JSON must have status
SELECTION_COMPLETE_BEFORE_SMOKE, explicit smoke_used_for_selection=false,
independent_evaluation_truth_opened=false, development_labels_used=true or
false, and selected_methods state_sha256 values equal to the selected state
bindings. Development labels may be used for fitting/checkpoint selection; the
independent evaluation truth boundary remains closed.

The helper scripts/trr_p11/restore_gate.py:tensor_digest first converts a tensor
to a detached contiguous CPU tensor. For the Finance/Pile-prefixed layout, the
runner concatenates each canonical tensor group in the declared key order before
computing the aggregate digest and per-record slices. It hashes the UTF-8 bytes of

    json.dumps(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    )

followed immediately by the tensor flattened as a contiguous uint8 view in C
order. The same digest is required in tensor-identity sidecars and smoke
prediction receipts. The helper is intentionally lazy about importing torch.

`restore_and_run_smoke` returns the same hash-bound metadata needed by the P11
evaluation adapter: `source_boundary="secondary"`, the validated `assets`
object with primary/secondary copies, `boundaries`, `restore_manifest`
(bytes/hash), verified `tensor_identity`, and the normalized
`consumer_receipt`. The consumer receipt includes actual loaded state/readout
file bindings and bundled code paths and records no training-worktree import or
temporary dependency. A downstream evaluation receipt must preserve these
fields rather than replacing them with self-assigned asset IDs.
