# TRR-0011 immutable inference package handoff

This handoff accompanies `inference_package_v2.json` (SHA-256
`a74f687f8287cacfeafa0acc81a97f208672dedf797792c60f88ba93b0318b25`). It
describes how the selected PR20 fixed-readout states can be replayed on a
future, separately approved public panel. It does not authorize source
selection, truth access, model fitting, or a change to any decision rule.

## Frozen replay path

The package validates the frozen registration, public freeze, source files,
state files, public readout table, four saved observation cells, and retained
prediction files before loading model tensors:

```text
validate_package(manifest_path, root)
  -> _load_fixed_method(checked, method_id, device)
  -> _predict_row(model, readout, activation, mask, device)
```

`validate_package` is the first boundary. For each selected fixed method,
`_load_fixed_method` calls the hash-bound
`scripts.trr0010_p09_fixed_loader.py:load_p09_fixed_state` and loads exactly one
public `E` table. `_predict_row` applies the frozen
`projected_hidden`/`logits_from_rows` path, casts the retained `H[128,2048]`
row to FP32, emits full-vocabulary FP32 argmax IDs, and preserves the BOS row.
The frozen model class and loader may be used directly when the same state,
readout, code, and resource bindings are checked.

The existing `trr0011_package.py replay` CLI is deliberately limited to the
four packaged PR20 cells and their retained predictions. It must not be
presented as a new-panel evaluator: it has no new-panel argument and its
expected-output comparison is a replay check. A future new-panel wrapper must
first create and validate a separate panel gate containing the approved panel
descriptor, observation hashes, record order, geometry, method/resource
bindings, and truth boundary. After that gate is independently reviewed, the
wrapper may feed each gated observation through the same loader and row
prediction path above; it must not modify `inference_package_v2.json`, select
sources, or read target labels. Use one isolated process per method/cell when
memory accounting is required.

The capture and decoder geometries remain distinct. The inherited producer
stores BF16 B8x192 public-SDPA activations plus masks and positions, retaining
the first 128 positions. The decoder consumes each retained H row as FP32 and
returns 128 int64 token IDs, with 127 post-BOS positions scored. This package
contains no tokenizer, source text, target labels, truth loader, scorer, or
private panel payload.

## A1+A2 trace boundary

The historical native adapter and its output-only candidate hook are in
`scripts/trr0004_predict_confirmation.py`:

- `_A2Adapter.__call__` invokes the legacy `propose_public_a1` with the frozen
  proposal budget/chunk, then passes the first `DEFAULT_A2_K` candidates to
  `decode_policy`.
- The first measured proposal per record is retained by the existing
  `self._record_proposals.append(...)` branch (`self.calls % 4 == 2`).
- `candidate_tensors(records, sequence_tokens)` stacks those retained IDs and
  scores; `evidence()` reports the proposal and candidate timing counters.

`registration_r5.json` binds `candidate_arrays_persisted: false`, and no
historical A1 candidate arrays or rank traces are available in this package.
The A1+A2 binding is metadata-only. A future output-only trace may use the
existing hook if its code, constants, overhead, and panel gate are bound before
execution; it must not reinterpret a missing historical trace or change the
registered decision rule.

## Replay evidence and cost interpretation

The authoritative bounded replay is
`cuda_replay_matrix_r1.json` (SHA-256
`79fda7559f0e4e426227d5af0046dd35f6b6b1db721d2d25b87895947c9cb234`) plus
`cuda_replay_cost_r1.json` (SHA-256
`7d1c4e6d030995f363736470f54bd3d6e9153ff75337a66332fde76b0777556f`). It
verified two records in each of four cells for both selected fixed methods,
with exact equality to every retained PR20 prediction. The reported 24.457 s
is the sum of full isolated replay-process walls, including package validation,
state/readout loading, and inference. It is not warm throughput and is not a
deployment latency claim.

The sampled maximum device use was 5,441 MiB and the minimum sampled free
memory was 10,537 MiB. These are observed whole-device samples and include the
pre-existing display allocation; they are not an isolated model allocator peak.
The preflight and all eight per-cell receipts are listed in the matrix receipt.

Earlier unversioned CPU receipt paths were reused/overwritten during package
development. They remain diagnostic history only and are not the immutable v2
replay evidence. The versioned CUDA receipts, matrix, and cost receipt are the
authoritative create-only records; no source text, labels, truth, selection, or
decision output was opened during replay.
