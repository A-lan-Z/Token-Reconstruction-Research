# TRR-P09 nested-B1 identity-only exclusion ledger addendum

Status: **PASS_IDENTITY_ONLY_LEDGER_COMPLETE**.

The actual nested B1 fitting bank (the immutable 1,200-row B0 prefix followed
by 10,800 additions) is published as
`experiments/TRR-P09/setup/nested-b1-exclusion-ledger-r1.json`.
Its schema is
`token-reconstruction.trr-p09-nested-b1-exclusion-ledger.v1`, size 7,617,295
bytes, and SHA-256
`48b6656845a18428e1fdd54c194928538cd9aa87dfb010b5238f11df3c33fbb0`.

The ledger contains 12,000 identity-only rows: B0 1,200, B1 additions 10,800,
Alpaca 6,000, Pile 3,600, and Finance 2,400. Each row carries its nested-bank
membership, dataset, stratum, record ID, parent/source record ID, source index,
rendered hash, and any published `public_record_sha256`. The B0 prefix has no
published `public_record_sha256`; its record IDs, source indices, and
`rendered_sha256` are retained, with `rendered_sha256` in the consumer's
identity hash namespace. No source text, token IDs, replacement payloads, or
evaluation results are present in the ledger.

For the 3,500 rows whose actual fitting input has at least 128 active tokens,
`final_sequence_sha256` is computed with the existing producer
`_sequence_digest`: the first 128 actual active int32 IDs, including BOS, in
contiguous little-endian C-order. All 3,500 published sidecar values checked
against the public token tensor with zero mismatches. The remaining 8,500 rows
are explicitly marked
`ineligible_shorter_than_128_active_tokens_no_padding_hash`; no padded hash is
created. The eligible counts are Alpaca natural 780, Pile natural 930, Finance
natural 590, Pile controlled 600, and Finance controlled 600.

The validation receipt is
`experiments/TRR-P09/review/nested-b1-exclusion-ledger-validation-r1.json`,
schema
`token-reconstruction.trr-p09-nested-b1-exclusion-ledger-validation.v1`, size
1,689 bytes, SHA-256
`1d07b0feef44fe5c6a4b4282baf0af70ef95d7e559f33ab83aab646dd6317361`. It ran
the actual `scripts/trr0005_produce_confirmation.py::_collect_exclusions`
against the generated ledger and passed exact equality for every Pile/Finance
record ID, source index, and identity hash set. The collector consumed 18,080
identity/hash values; the Alpaca rows remained represented as dataset metadata
and were not classified as Pile or Finance.

The immutable input bindings are the public metadata records
`/tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/records.json`
(10,767,794 bytes, SHA-256
`367cfba0ffe78f59454861a76f23830f480a8745cc6b35e6b4a0d05eca53638b`) and the
public fitting token/mask tensor
`/tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/inputs.safetensors`
(29,952,264 bytes, SHA-256
`8150184a349bebd31e24b128d23f096702898024a87a405322ee3ba95fa6682e`). The
canonical producer source is `scripts/trr0005_produce_confirmation.py`, whose
SHA-256 at preparation was
`7289ee5db30bc0b2d3c1b50b1d5b6ca1ac235001371f61b92adf10e2dd979e2b`.

The compact preparation manifest is
`experiments/TRR-P09/setup/nested-b1-exclusion-ledger-manifest-r1.json`, size
2,208 bytes, SHA-256
`7934580223520a3ff931ec9d834c908217eca50b1994634e2834df08af59b65e`.
The reproducible CPU command was:

```text
PYTHONPATH=src:. python3 scripts/trr_p09/prepare_nested_exclusion_ledger.py --repository-root /tmp/trr-p09 --records /tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/records.json --inputs /tmp/trr-p09-runtime/stage1-input-preparation-r2-slot-compat-final/inputs.safetensors --ledger experiments/TRR-P09/setup/nested-b1-exclusion-ledger-r1.json --validation experiments/TRR-P09/review/nested-b1-exclusion-ledger-validation-r1.json --manifest experiments/TRR-P09/setup/nested-b1-exclusion-ledger-manifest-r1.json
```

This preparation read only existing public fitting metadata and public fitting
token IDs/masks on CPU. It did not load a model, read activations, access new
public sources, open an evaluation panel or truth, or access P03's holdout.
