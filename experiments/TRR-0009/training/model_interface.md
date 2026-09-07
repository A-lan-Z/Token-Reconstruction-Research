# TRR-0009 readout interface

The adaptable state is a `safetensors` file with schema
`token-reconstruction.trr0009-supported-readout.v1` and method ID
`trr0009_supported_row_readout`. It binds the published TRR-0007 residual
MLP512 state through `starting_state_sha256`, the public fitting manifest
through `fit_manifest_sha256`, and the sorted supported IDs/counts through
`support_digest`. The frozen published base binding for the current run is
`2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`;
loaders must still verify the state file hash from the receipt rather than
infer it from this prose.

A consumer loads it with:

```python
from pathlib import Path
from trr0009_model import load_supported_readout_state

model = load_supported_readout_state(
    Path("selected.safetensors"),
    base_state_path=Path("trr0007_residual_mlp512/selected.safetensors"),
    support_ids=support_ids,
    support_counts=support_counts,
    hidden_size=2048,
    vocabulary_size=128256,
    context_width=128,
    bottleneck_size=512,
)
```

The prediction contract is current-H only. Given `H` shaped
`[records, positions, 2048]`, a boolean mask with BOS at position zero, and
fixed public normalized `E` shaped `[128256, 2048]`, call
`model(H, mask, E)` for a full logits tensor. For chunked prediction, call
`projected_hidden(H, mask)` once and then `logits_from_rows(hidden,
record_slots, position_slots, E)` for selected rows. Both paths return the
full vocabulary and apply the same bounded supported-row correction. IDs absent
from the support binding remain exactly at the inherited readout. At zero
correction the adapter must be bit-exact with the base decoder; the trainer's
B=1 full-output and B=8 selected-row equivalence receipts record this check.

The adaptable state stores only the two trainable scalars per supported token
plus the complete inherited decoder state. Deployment materialization, when
requested, explicitly constructs a full `[128256, 2048]` effective dictionary
and a full `[128256]` bias vector, and records its preparation time and tensor
hashes.
