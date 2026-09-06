# TRR-0006 bounded frozen-pair fixture preflight

This is a source and input equivalence check for the two already selected
enriched states. It does not select records, fit a state, open truth, load a
public model, or compare timing. The missing tail of the TRR-0006 control
packet still gates any new 1024-record study.

The exact state bindings are:

* `enriched__affine_causal_h_attention128`: TRR-0005
  `joint_fit_qknorm_v1/enriched/affine_causal_h_attention128/selected.safetensors`,
  SHA-256 `ee910b14ad6f282bb933ea44ad24453cb5cce1470c65dbc09d8bcc16f3e8abfd`;
  metadata selects `cosine_scale4` and selected step 1900.
* `enriched__affine_trained_diagonal_attention128`: TRR-0005
  `joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors`,
  SHA-256 `696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2`;
  selected step 1600. Its one-key diagonal path is numerically unchanged by
  the default dot-product score mode.

Both use the retained normalized public E table at
`outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`,
SHA-256
`ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`,
F32 `[128256, 2048]`. The retained raw captures are `[128, 192, 2048]`
BF16, and the stored views are `[128, 128, 2048]` BF16. The check reads only
`activations`, `attention_mask`, and `position_ids`; raw files contain token
IDs for the old public receipt, but the fixture never requests that tensor.

For one row, the raw activation is `192×2048×4 = 1,572,864` bytes after the
retained predictor's BF16-to-F32 device conversion. Raw logits are
`192×128256×4 = 98,500,608` bytes; trimmed logits are `128×128256×4 =
65,667,072` bytes. E is about 0.98 GiB and each state is about 20 MiB.
The retained selected-method receipt reached 1,280,330,240 allocated and
1,302,331,392 reserved bytes. Keep the conservative launch gate at least
8 GiB free and at most 6 GiB reserved, with one model resident at a time;
the utility rechecks the live CUDA gate before E/state load and after each
method. A failure is preserved by the caller and must not be retried with a
changed geometry or dtype.

The retained numerical boundary is used verbatim: CPU F32 E is copied to
CUDA unchanged; CPU BF16 H is copied to CUDA as F32; masks are CUDA bool;
there is no autocast, reduced E, unpadded batch-1 public forward, or altered
matmul recipe; the decoder runs in inference mode and IDs use `argmax`.
The test is limited to the first two retained rows in each of four cells,
for both methods. It requires exact equality of raw-192 first-128 IDs,
trimmed-128 IDs, and the saved TRR-0005 prediction IDs (16 row-method
comparisons total). Input-prefix equality is checked separately and is not
treated as inference proof. Any ID mismatch fails closed.

Run the structural check first:

```text
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006
PYTHONPATH=src python3 experiments/TRR-0006/fixture_equivalence.py \
  --mode input --trr5-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --records 2 \
  --output experiments/TRR-0006/fixture_equivalence/input_prefix.json
```

After the live resource gate is recorded by the preflight owner, run the
already-opened CUDA fixture once:

```text
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006
env PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 experiments/TRR-0006/fixture_equivalence.py \
  --mode cuda --trr5-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --embedding /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors \
  --records 2 \
  --output experiments/TRR-0006/fixture_equivalence/cuda_id_equivalence.json
```

The executable decoder source is the TRR-0005 `da82f6cac45e09ae83452198344c547553cb4433`
implementation carried into the TRR-0006 worktree at the published `3a7e8f5`
tree. The `1dba67a8` maintenance change only removes a legacy `schema` key
before merging timing metadata and adds export/receipt adapters/tests; it
does not alter decoder, E, activation, or argmax computation. The published
commit adds records and audit evidence around that source. This fixture's
new utility is task-local and must not be added to the global TRR-0005
registry or contract.
