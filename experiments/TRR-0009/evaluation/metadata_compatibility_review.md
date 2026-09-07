# TRR-0009 fixed-state metadata compatibility review

Status: proposal only; no gate code, registration, prediction, state, or truth artifact was changed.

The public gate failed before truth access at _validate_state_semantics with:

    state inference semantics changed: continued_fixed_readout

The rejected row is the registered continued_fixed_readout state:

| Field | Bound value |
| --- | --- |
| State path | experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors |
| Bytes | 29,390,492 |
| SHA-256 | 5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14 |
| Header schema | token-reconstruction.trr0007-positionwise.v1 |
| Selected step | "400" |
| Inference contract | current_activation_H_i_only; full_vocabulary_tied_E |
| Geometry | hidden size 2048, context width 128, vocabulary 128256 |
| Loader module/function | token_reconstruction.trr0007_positionwise.load_positionwise_model_state |
| Registration loader flags | current_h_only=true, full_vocabulary=true, history_enabled=false, a2_enabled=false |

The state header contains the exact inference-contract string and geometry, but omits the redundant metadata keys current_H_only and full_vocabulary_cross_entropy. The unchanged anchor contains those keys. The adaptable state contains those keys and additionally binds its support digest. The retained published reference uses a TRR5 schema and is not evaluated by the TRR7-specific header predicate; its registration loader flags and prior loader-equivalence receipt bind the same current-H/full-vocabulary contract.

## Proposed maintenance exception

This document does not authorize implementation. If explicitly approved, a separate maintenance validation step may recognize the fixed row only when all of these conditions hold:

1. The registration row is exactly continued_fixed_readout, with the existing state path, byte count, and SHA-256 above.
2. The state header schema is exactly token-reconstruction.trr0007-positionwise.v1.
3. The state header has the exact inference contract current_activation_H_i_only; full_vocabulary_tied_E.
4. Header geometry is exactly hidden size 2048, context width 128, vocabulary 128256, and selected step "400".
5. The registration loader is exactly the current-H/full-vocabulary positionwise loader with current_h_only=true and full_vocabulary=true; history_enabled=false and a2_enabled=false are required explicitly. A missing flag or any other value fails.
6. The exception applies only to the two absent redundant boolean header keys. Any present value that is false, malformed, or inconsistent must fail.
7. The state file is read metadata-only for this compatibility check; no state tensor is rewritten or supplemented.

The exception must not broaden acceptance for another schema, method, state hash, loader, geometry, inference-contract string, or missing loader flag. The existing gate checks remain unchanged: registration identity, the actual registered source byte hashes, and the historical inference identity; method roles/order/cells; exact state/input/observation/frequency/support bindings; current-H/full-vocabulary loader flags; prediction completeness and hashes; warmup/measured ID equality; timing plan and qualified timing receipt; resource/numerical settings; truth-free flags; and all pre-truth integrity checks. The maintenance validator's source hash and full commit identity are recorded separately; this proposal does not require the current HEAD to equal the historical registration commit.

The maintenance receipt must bind:

- the immutable registration record and its historical inference code commit 55ef2d46413de28be70b4014f596b277c0e1582c;
- the exact fixed-state path, bytes, and SHA-256;
- the metadata-only compatibility result and the two absent-key names;
- the maintenance validator's own full code commit and source hash;
- the existing loader-equivalence PASS receipt used as supporting evidence;
- timestamps, command, environment, and truth/source/label flags all set to unopened/false.

Historical registration provenance and maintenance-validation provenance must remain separate. The maintenance validator cannot rewrite the registration, change the registration's code commit, alter predictions, or silently make a new scientific run appear to have used the maintenance code.

## Required negative tests

Before any implementation review, synthetic metadata tests should fail closed for:

- a changed fixed-state SHA or byte count;
- a non-TRR7 schema;
- a changed inference-contract string;
- hidden/context/vocabulary geometry changes;
- selected step other than "400";
- a present current_H_only=false or full_vocabulary_cross_entropy=false;
- any missing or false registration loader flag;
- a different method ID, role, loader module, loader function, or state path;
- an exception attempted for unchanged, adaptable, or published-reference rows;
- a changed registration, prediction root, observation/capture receipt, source selection, frequency/support binding, timing receipt, or code binding;
- any truth/source-label flag set true.

These tests must run on synthetic metadata or existing receipts only and must not open private truth.

## Residual risk and approval boundary

The absent header keys are redundant with the fixed state's exact inference-contract string and registered loader flags, but accepting metadata by inference rather than by explicit state-header fields leaves a provenance-format weakness. The risk is bounded by exact state hash, schema, geometry, step, loader flags, prior output-equivalence evidence, and unchanged downstream gate checks; it is not eliminated.

The automatic-review rejection was: “Automatic review rejected the proposed compatibility adapter as weakening truth access after a metadata failure.” Therefore this file is a reviewable proposal only. Implementing the exception, rerunning the public gate, or opening fresh truth requires explicit user authorization after review. No implementation or fresh truth access has been performed.

