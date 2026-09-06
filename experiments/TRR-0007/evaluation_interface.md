# TRR-0007 evaluation interface

The evaluation runner should treat the four crossed states as two model
families over two fitting banks:

| descriptor | loader | input contract |
| --- | --- | --- |
| `enriched__current_positionwise` | `load_positionwise_model_state(path, method_id="trr0007_current_positionwise", hidden_size=2048, vocabulary_size=128256)` | current `H_i`, valid mask, normalized public `E` |
| `enriched__residual_mlp512` | same loader with `method_id="trr0007_residual_mlp512"` | current `H_i`, valid mask, normalized public `E` |
| `improved__current_positionwise` | same current loader | current `H_i`, valid mask, normalized public `E` |
| `improved__residual_mlp512` | same extension loader | current `H_i`, valid mask, normalized public `E` |

`src/token_reconstruction/trr0007_positionwise.py` exposes
`load_positionwise_model_state`, `build_current_positionwise`, and
`build_residual_mlp512`.  Both returned models implement
`projected_hidden(activation, valid_mask)`,
`logits_from_rows(projected_hidden, record_slots, position_slots, E)`, and
`forward(activation, valid_mask, E)`.  All methods require floating
`H:[records,positions,2048]`, a boolean right-padded mask with BOS at position
0, and `E:[128256,2048]`.  `forward` emits finite full-vocabulary logits with
zero rows at invalid padding positions.  `logits_from_rows` emits logits only
for the supplied post-BOS row pairs, preserving bounded projection memory.

The current model is a TRR-0005
`affine_trained_diagonal_attention128` architecture.  Each crossed state is
saved under the TRR-0007 schema; the retained selected state is a separate
frozen reference.  The residual model has a nested `base` with
the same architecture and a fixed per-position `layer_norm(H_i)` followed by
`Linear(2048,512)`, GELU, and `Linear(512,2048)`, added before the inherited
normalization and tied projection.  Its final linear weight and bias are
zero-initialized in the neutral training state.  The state serializer records
the method ID, base method, hidden/vocabulary/bottleneck geometry, selected
step, initialization contract, and SHA-256 digests.  The evaluator must use
the serialized state exactly; it should not rebuild the model from a
TRR-0005 state loader or silently drop the nested base keys.

The published frozen reference remains
`experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors`
with SHA-256
`696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2`.
Its loader is the existing `load_decoder_state(...,
method_id="affine_trained_diagonal_attention128", ...)`; it is a separate
reference arm and does not use the TRR-0007 crossed training states.

At fresh evaluation the model receives only the current activation at each
position.  There are no earlier activation vectors, source IDs/token labels,
token history, candidate lists, public-prefix calls, teacher losses, or A2
fallbacks.  The runner should preserve the declared four cells
`pile__public_base`, `pile__public_lora_2601`,
`finance__public_base`, and `finance__public_lora_2601`, and score all 127
