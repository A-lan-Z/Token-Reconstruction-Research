# TRR-0008 selection interface

The task-local adapter is `scripts/trr0008_select_public.py`. Its `select`
command is create-only and refuses to read Arrow rows until all owner gates
pass:

- `experiments/TRR-0008/planning/decision_contract.json` has status
  `FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION` and binds the candidate,
  retained reference, credible alternative, diagnostic, counts 1,024 Finance
  and 384 Pile, ranges, seed 5005, and paired target conditions.
- `experiments/TRR-0007/method_freeze.json` is the reviewed pre-truth freeze.
- `experiments/TRR-0008/planning/identity_inventory_1thread.json` is the
  count-only projection with both requested capacities sufficient.
- `planning_status.json` records explicit P06 source-hash byte confirmation.
  The current status is pending, so real selection remains blocked.

The selection artifact is
`experiments/TRR-0008/selection/source_selection.json` with schema
`token-reconstruction.trr0008-source-selection.v1`. Capture adapters should
bind its file record and consume:

- `records_by_domain`: `{"pile": 384, "finance": 1024}`;
- `source_ranges_half_open`, `sequence_tokens_including_bos` (128),
  `capture_sequence_tokens` (192), and `target_conditions`;
- `selection_rule.records.pile` and `.finance`, each a list of rows with only
  `record_id`, `public_record_sha256`, dataset revision/index metadata,
  `full_token_count`, `post_bos_token_count`, `valid_tokens`, and
  `final_sequence_sha256`;
- `public_sources_frozen` Arrow and tokenizer descriptors.

The same selected identity rows are used for `public_base` and
`public_lora_2601`. The target conditions are paired conditions, not additional
source rows. The capture adapter may transiently render a declared row to
construct activations, but public artifacts must retain only the declared
identity metadata and output tensors.

`source_exclusions.json` uses schema
`token-reconstruction.trr0008-source-exclusions.v1` and contains aggregate
identity counts plus file bindings for the known TRR-0007 ledgers and approved
P04/P06 opaque inputs. It emits no identity values, source text, token IDs,
labels, or truth.

The `reserve` subcommand writes
`token-reconstruction.trr0008-opaque-source-sequence-reservation.v1`. It
contains only sorted `public_record_sha256` and H128 `final_sequence_sha256`
hash sets, counts, and privacy metadata; it intentionally has no record IDs,
indices, domain labels, source text, token IDs, targets, or truth.

This interface binds the existing prospective decision contract by file hash;
it does not duplicate or reinterpret its primary route alpha, safeguard gates,
bootstrap settings, or cost rules.

The selector also requires the contract's explicit numeric bindings: primary
exact component alpha 0.0125 with route alpha 0.025 and practical margin 0.05;
token route practical margin 0.01; safeguard alpha 0.05 with exact/token harm
margins 0.05/0.01; record-bootstrap seed 8008, 10,000 draws, and unit
`source_record`; and the four-cell 1.25 timing gate. The bound
`experiments/TRR-0008/timing/precision40_result.json` receipt must remain the
recorded PASS artifact before selection can proceed.
