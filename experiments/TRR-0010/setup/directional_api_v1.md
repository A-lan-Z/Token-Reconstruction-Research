# TRR-0010 directional readout hook v1

This file records the adapter contract for the shared TRR-P09/TRR-0010 runner.
It is code/data agnostic and does not select records, positions, labels, or
source candidates.

## Adapter

Module: `scripts/trr0010_model.py`

Class: `DirectionalTokenReadout`

Method identity:

- `method_id = trr0010_supported_token_directional_readout`
- `readout_mode = supported_token_directional`
- current-H only, full-vocabulary tied-E scores, A2 disabled
- `delta_rows` is `[K,2048]` FP32, zero initialized, with sorted supported IDs
  in `support_ids`; absent IDs have no parameter and remain public-E rows.

The shared decoder calls:

```python
q = adapter.projected_hidden(activation, valid_mask)
logits = adapter.score_rows(
    query_rows=q[record_slots, position_slots],
    logit_scale=decoder.logit_scale,
    embedding=public_E,
    base_logits=optional_fixed_decoder_logits,
)
```

`score_rows` returns a full `[N,V]` matrix through one materialized effective
readout `E_eff = E` with `Delta[S]` inserted. It performs
`q @ E_eff.T * logit_scale`, matching the exported deployment dictionary. The
optional `base_logits` is accepted for protocol geometry validation but is not
used to replace the merged primary path. `q` is the normalized decoder output
from current `H_i`; no labels, targets, position selection, or data routing is
available to the hook. Public E is not renormalized.

The adapter also exposes the protocol methods
`loss_terms(logits, target_ids)`, `optimizer_param_groups(decoder,
base_learning_rate=...)`, and `metadata()`. The anchor must be bound once with
`bind_embedding_statistics(public_E)` before `loss_terms` is called. The
proposed direct-direction learning rate is 1e-4; the common decoder rate is a
shared contract field and is currently proposed as 2e-4.

The registered `score_rows` path uses merged E for training, validation, and
inference. `merged_score_rows` and `merged_forward` expose the same operation
for an already materialized dictionary. Materialization uses one out-of-place
`index_copy` from detached public E plus two support-row buffers, preserving
Delta autograd without mutating E. `sparse_score_rows` and `sparse_forward` are
diagnostic-only paths and require a separate contract amendment before use in
any active method. The mandatory diagnostic comparison records dense-versus-
sparse maximum logit error and argmax agreement.

## State and artifact binding

`save_directional_state` records the task schema, base state hash, fit-manifest
hash, support digest, architecture geometry, and exact tensor state. The dense
`export_effective_embedding` artifact is a separate create-only file with its
own bytes/hash/tensor digest; it replaces public E only in the registered
validation/deployment path. Full E and Delta payloads are not duplicated in
source control or in every checkpoint.
