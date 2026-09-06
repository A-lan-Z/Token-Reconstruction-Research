# TRR-0006 private truth preparation layout

The task-local producer is `scripts/trr0006_prepare_truth.py`. It accepts the
frozen decision plan, source selection, `panel_capture_v1/panel.json`, and
canonical `panel_capture_v1/observations.json`. It first revalidates the
complete eight-entry public prediction and timing receipt through
`trr0006_freeze_pair.validate_before_truth`. It then reuses the trusted
TRR-0005 tokenizer and row renderer through `trr0006_capture_public` and
materializes the selected public rows in the frozen natural order.

The sidecar is written create-only outside the reconstruction root and uses
schema `token-reconstruction.trr0006-private-label-sidecar.v1`. It contains
exactly two safetensors keys: `pile__token_ids` and `finance__token_ids`, each
`torch.int64` with shape `[records_per_domain, 128]`. Position zero is BOS
`128000`; all values are checked against vocabulary size `128256`. The scorer
maps each domain tensor to both paired target cells (`public_base` and
`public_lora_2601`), so labels are not duplicated into four private tensor
entries. Masks and position IDs remain in the public observation files.

The sidecar metadata binds the decision-plan and source-selection hashes, all
four public observation hashes, and the two ordered source-ID digests. The
create-only binding JSON uses schema
`token-reconstruction.trr0006-private-label-binding.v1`, stores the absolute
sidecar file record and the same hash bindings, and contains no token or label
payload. Both files record `truth_opened: false` and
`reconstruction_root_contains_truth: false`; the binding also records the
per-domain tensor digests and the code-file hashes used for preparation.
