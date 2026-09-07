# TRR-0009 selected-method freeze contract

Before any TRR-0009 public source selection, the training/evaluation owner must
create `experiments/TRR-0009/training/method_freeze.json` with schema
`token-reconstruction.trr0009-selected-method-freeze.v1` and status
`FROZEN_TRR0009_METHODS_BEFORE_SOURCE_SELECTION`.

The artifact is separate from the inherited TRR-0007 method freeze and from
the later prediction registration. It must bind the exact TRR-0009 decision
contract (`path`, `bytes`, and `sha256`), the method order
`unchanged_anchor`, `continued_fixed_readout`,
`continued_adaptable_readout`, `retained_reference`, and a `state_bindings`
entry for every method. Each entry contains the role, selected validation step,
loader module/function, and an actual state file record (`path`, `bytes`, and
`sha256`). The selector verifies every state file against its record; missing
or draft state bindings fail closed.

It must also contain a `decision_rules` object with the decision-contract hash,
the exact earliest-maximum public validation style-balanced token-accuracy
rule, validation interval 100, step-zero eligibility, and equal selection
opportunities. `source_code` is a nonempty list of actual file records for the
trainer/model/selection implementation used to produce the states, and
`code_commit` is a full 40-hex commit. `state_selection_frozen` and
`source_selection_started` are respectively true and false. `truth_opened`,
`private_or_truth_payload_read`, and `fresh_evaluation_started` must be false.

The preselection count-only scanner accepts this same artifact and records its
file/state bindings alongside the inherited TRR-0007 freeze. The later timing
registration remains a separate post-observation integrity receipt and cannot
substitute for this preselection freeze.
