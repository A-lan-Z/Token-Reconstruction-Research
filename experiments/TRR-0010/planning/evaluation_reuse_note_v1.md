# TRR-0010 evaluation reuse note (read-only interface review)

This note records the smallest public orchestration reuse found in the TRR-0009/TRR-0005 code and the task-local metadata adapter. The adapter and synthetic tests were implemented without source selection, capture, prediction, GPU execution, or truth access.

## Public observations

Reuse the TRR-0009 capture path in `scripts/trr0009_eval_capture.py`: `capture_public(args)` is the production entrypoint and `capture_from_observations(...)` is only the packaging helper for already-produced tensors. The producer path calls the established `trr0005_produce_confirmation._capture_prefix`, `trr0004_prepare_public_activations._qualify_public_prefix_padding`, and `token_reconstruction.public_activation.capture_public_prefix`. It performs the public full forward at B8x192 and stores the first 128 positions, with truth-free create-only `observations.json`, `panel.json`, and `capture.json` receipts.

TRR-0010 can reuse that producer and geometry, but must use a thin task adapter because the TRR-0009 loader hard-codes the TRR-0009 selection schema/status and four-method contract. The resulting cells remain the shared four-cell order `pile__public_base`, `pile__public_lora_2601`, `finance__public_base`, and `finance__public_lora_2601`; the same source-record order digest is required across both targets.

## Prediction runner

Reuse scripts/trr0010_model.py:load_directional_state only as a training/export helper; production registration must bind its separately exported effective W and a base-only decoder. The prediction adapter does not reconstruct W during inference. Sparse scoring remains diagnostic-only.

The TRR-0009 runner cannot be invoked unchanged: it binds TRR-0009 schemas, four method IDs, and its own timing/registration constants. A TRR-0010 runner should port the loop and use the TRR-0010 gate schema, six-method order, and task output paths. The timing gate contract is already explicit in `scripts/trr0010_eval_gate.py` and `setup/eval_gate_v1.json`: 24 method-cell entries, one warmup plus one measured call per record, exact warmup/measured ID equality, per-record timing and peak-memory fields, and truth/source flags false. The timing owner should preserve those fields while adding the A1+A2 adapter branch; no TRR-0009 timing constants should leak into the TRR-0010 registration.

## Native frozen A1+A2 K256

The minimal native adapter is the retained path in `scripts/trr0004_predict_confirmation.py`: `_load_public_prefix(...)` loads the public P0 prefix, retained A1 lens, and normalized public E; `_A2Adapter.__call__(row_h, row_mask, row_positions)` stages the current-H row to the legacy CPU API, calls `trr0003_footing_compare.propose_public_a1` and `decode_policy`, and returns one normalized prediction. Bind the fixed policy from `scripts/trr0003_footing_compare._fixed_k256_policy()`: direct cosine, fixed schedule `(256,)`, terminal `commit_last_winner`. `scripts/trr0005_run_predictions.py:_load_adapter` shows the existing wiring and resource checks.

The TRR-0010 wrapper must retain the CPU-row staging inside the measured interval and record it in method evidence. It must bind exactly the TRR-0010 gate resources `public_embedding_table`, `retained_a1_lens`, and `public_p0_prefix`; the retained A1 lens identity must be the same for the historical A1+A2 comparator. The actual adapter calls `propose_public_a1(max_k=512, chunk=256)`, retains that proposal in memory, slices the candidate tensor to its first 256 entries for `decode_policy`, and applies the fixed K=256 schedule. This is not a native top-256 proposal. TRR-0010 prediction outputs remain output-only; the wrapper does not persist candidate arrays to disk. The comparator is one method (`frozen_a1_a2_k256`), not separate A1 or A2 arms.

`trr0005_predict_confirmation.run_warmed_prediction` is the reusable 1+1 timing wrapper, but its method-ID validation is TRR-0005-specific. Copy its small loop or add a TRR-0010-local wrapper with the same exact-output check rather than weakening the TRR-0005 constants. Reuse its prediction artifact shape and BOS/right-padding normalization, with TRR-0010 schema and registration hashes.

## Required adapter boundary

The implementation handoff is therefore one task-local runner adapter, not a new capture framework:

1. Feed the TRR-0009 public capture producer through a TRR-0010 selection/receipt adapter.
2. Reuse the TRR-0009 current-H row loop for five full-vocabulary methods, dispatching `load_directional_state`/merged-E for directional states.
3. Dispatch `frozen_a1_a2_k256` to the retained `_A2Adapter` wiring with the three gate-bound public resources and fixed K256 policy.
4. Emit the TRR-0010 prediction/timing schemas and let `scripts/trr0010_eval_gate.py` rehash all 24 entries before any truth caller.

No new source or truth interface is needed for the native comparator; it consumes the same captured H observations and remains an output-only historical comparison.

## Synthetic adapter delivered

scripts/trr0010_eval_runner.py:execute(...) now supplies the task-local
truth-free runner. It validates the frozen registration, streams the four
BF16 observation cells in capture chunk order, and writes all 24 prediction
and timing artifacts through create-only paths before invoking the public gate.
The explicit method_factory argument is test injection only; it is not
available from the production CLI.

For production direct methods, the runner loads exactly one registered
public_embedding_table or effective_readout_w tensor for the method and
releases it with the decoder before the next method. Directional production
loaders must return a base-only decoder and a separately bound effective W;
a loader exposing materialize_effective_embedding is rejected. The only
materialization path is the explicit synthetic test injection, where the
effective readout is materialized once and then held fixed. The native
frozen_a1_a2_k256 branch retains the established CUDA-only comparator and
does not persist candidate arrays.

The remaining final producer work is registration binding of actual
base-only decoder states, direct public E/effective-W tensor records, and
native A1+A2 snapshot/reference descriptors; the timing owner still supplies
the production code/timing receipts, and the separate scorer consumes the
frozen prediction matrix after its truth-gated handoff. No production
capture, prediction, GPU run, or truth access was performed here.

Synthetic verification: python3 -m pytest -q
tests/test_trr0010_eval_gate.py tests/test_trr0010_eval_runner.py -> 26
passed.

## TRR-0010 registration and capture wiring

scripts/trr0010_eval_register.py:build_registration(...) is the task-local
descriptor boundary. It first requires
token-reconstruction.trr0010-final-evaluation-design.v1 with status
FROZEN_TRR0010_FINAL_EVALUATION_DESIGN, all six TRR-0010 contender roles,
explicit frozen decision rules, the 128-per-domain paired panel, the inherited
B8x192-to-first-128 geometry, final-B1 and approved opaque exclusion records,
and exact runner/gate/producer/A1+A2 code bindings. A draft contract therefore
fails before any selection or capture path is opened.

After that freeze, the adapter calls the existing
trr0009_eval_capture.load_selection(..., expected_counts={"finance": 128,
"pile": 128}) only on the identity-only selection ledger. It writes a
TRR-0010 source-selection binding containing ordered record digests and
immutable exclusion records, without copying source rows, IDs, text, token
IDs, answers, or labels. It then consumes the already-produced TRR-0010
panel, observation manifest, and capture receipts and binds all six method
state and resource rows: public E for fixed methods, exported effective W for
directional methods, and public E/lens/P0 plus snapshot/reference descriptors
for the single native A1+A2 comparator.

The lower-level public producer call specification is
trr0009_eval_capture.capture_public(args) through its
_capture_condition_with_producer and
token_reconstruction.public_activation.capture_public_prefix calls. The
future TRR-0010 capture wrapper must pass the frozen identity-only panel and
public model/LoRA inputs, use B8x192 with batch size 8, retain only the first
128 positions, and package each cell under the TRR-0010 observation schema.
The TRR-0009 entrypoint is not called unchanged because its output-root and
metadata schemas are TRR-0009-specific. The wrapper remains truth-free; the
private truth curator is a separate later consumer after the public gate and
final freeze. No selection, source read, model load, capture, GPU run, or
truth access was performed while preparing this adapter.
