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
`finance__public_base`, and `finance__public_lora_2601`, and score all 127 post-BOS positions in every declared record. BOS is fixed by the public interface and is retained only as a diagnostic; exact recovery uses the same 127 post-BOS positions. The bounded A1+A2 adapter runs only on the first 32 public-base records per domain.

## TRR-0007 fresh source and capture entry points

After root freezes `experiments/TRR-0007/evaluation_plan.json` and records the
method-freeze SHA-256, run the count-only eligibility projection first.  The
projection is deliberately task-local even though it reuses the TRR-0006
scanner; the extra ledgers cover the already opened TRR-0006 source panel and
both TRR-0006 public observation panels.  The final broader-bank v5 parent,
source-row, and constructed-sequence ledgers are verified before the
inventory loads the tokenizer or Arrow files and are passed as explicit
identity exclusions.  The P04 file is verified by its approved hash and is
consumed only as an opaque exclusion exchange.

```bash
REPO=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007
cd "$REPO"
TOK=/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
PILE=/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow
FINANCE0=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow
FINANCE1=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow
P04=$REPO/experiments/TRR-0006/coordination/p04_reservation_hashes.json
NEW_BANK_DIR=$REPO/experiments/TRR-0007/support/broader_bank_v5
NEW_BANK_EXCLUSIONS=$NEW_BANK_DIR/public_parent_exclusion_manifest.json
NEW_BANK_PARENTS=$NEW_BANK_DIR/selected_parent_rows.json
NEW_BANK_PLAN=$NEW_BANK_DIR/corpus_plan.json
NEW_BANK_PREFIX=$REPO/experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python3 scripts/trr0006_build_eligibility.py inventory \
  --repository-root "$REPO" --tokenizer "$TOK" --pile-arrow "$PILE" \
  --finance-arrow "$FINANCE0" "$FINANCE1" --requested-per-domain 128 \
  --trr0007-final-bank-exclusion-manifest "$NEW_BANK_EXCLUSIONS" \
  --trr0007-final-bank-parent-ledger "$NEW_BANK_PARENTS" \
  --trr0007-final-bank-corpus-plan "$NEW_BANK_PLAN" \
  --trr0007-public-fitting-prefix-exclusions "$NEW_BANK_PREFIX" \
  --exclude-source experiments/TRR-0006/source_selection.json \
    experiments/TRR-0006/duplicate_capture_exclusion.json \
    experiments/TRR-0006/panel_capture_v1/panel.json \
    experiments/TRR-0006/panel_capture_v1/observations.json \
    experiments/TRR-0006/public_observations_v1/panel.json \
    experiments/TRR-0006/public_observations_v1/observations.json \
    "$NEW_BANK_EXCLUSIONS" "$NEW_BANK_PARENTS" "$NEW_BANK_PLAN" "$NEW_BANK_PREFIX" \
  --p04-exchange "$P04" \
  --output experiments/TRR-0007/selection/eligibility_inventory.json
```

Once the projection confirms at least 128 eligible rows in each declared
range, select the panel with the TRR-0007 adapter.  Set `METHOD_FREEZE` to the
reviewed immutable fit/method ledger.  The selector verifies the ledger and
its four actual state files; it does not accept a bare digest as the authority.
This command materializes public rows transiently through the trusted TRR-0005
renderer, applies all known fit/validation/open-evaluation identities and the
P04 opaque hashes, and writes only identity commitments and selected-row
hashes.

```bash
METHOD_FREEZE=experiments/TRR-0007/method_freeze.json
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python3 scripts/trr0007_eval_select.py \
  --repository-root "$REPO" \
  --plan experiments/TRR-0007/evaluation_plan.json \
  --eligibility-inventory experiments/TRR-0007/selection/eligibility_inventory.json \
  --method-freeze "$METHOD_FREEZE" \
  --final-bank-exclusion-manifest "$NEW_BANK_EXCLUSIONS" \
  --final-bank-parent-ledger "$NEW_BANK_PARENTS" \
  --final-bank-corpus-plan "$NEW_BANK_PLAN" \
  --public-fitting-prefix-exclusions "$NEW_BANK_PREFIX" \
  --tokenizer "$TOK" --pile-arrow "$PILE" \
  --finance-arrow "$FINANCE0" "$FINANCE1" --p04-exchange "$P04" \
  --exclude-source "$NEW_BANK_EXCLUSIONS" "$NEW_BANK_PARENTS" "$NEW_BANK_PLAN" "$NEW_BANK_PREFIX" \
  --exclusions-output experiments/TRR-0007/selection/source_exclusions.json \
  --output experiments/TRR-0007/selection/source_selection.json
```

After the identity-only selection is frozen and the root has marked capture
as permitted, capture all four paired public conditions with the task-local
adapter.  It calls the trusted TRR-0006 batch-8 by 192 public-prefix helper,
retains 128 positions, and stores only BF16 activations, masks, and positions.
The public base and synthetic-LoRA captures are run in one root-serialized GPU
lease; no truth sidecar is opened.

```bash
MODEL=/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
LORA_CONFIG=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/public-calibration/generation.json
LORA_UPDATE=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python3 scripts/trr0007_eval_capture.py capture --execute \
  --repository-root "$REPO" \
  --selection experiments/TRR-0007/selection/source_selection.json \
  --model-snapshot "$MODEL" --lora-config "$LORA_CONFIG" --lora-update "$LORA_UPDATE" \
  --output-root experiments/TRR-0007/evaluation/public_observations \
  --device cuda
```

The capture manifest is then passed to `scripts/trr0007_eval_register.py`,
which binds the four student states, retained TRR-0006 reference, shared
embedding table, public frequency map, and A1/A2 assets.  The four `STATE_*`
paths below are concrete expected fit outputs; the method-freeze ledger is the
final authority and registration fails if any supplied path or hash differs.
The runner writes 20 decoder prediction artifacts plus two A1/A2 anchor
artifacts; the public gate requires all 22 before the metadata-only truth
binding can be inspected.

```bash
REG=experiments/TRR-0007/evaluation/registration.json
OBS=experiments/TRR-0007/evaluation/public_observations/observations.json
CAPTURE=experiments/TRR-0007/evaluation/public_observations/capture.json
FREQ=experiments/TRR-0005/frequency_references_v1.json
A2_REFERENCE=experiments/TRR-0004/evidence/comparators/round001_teacher.py
STATE_CURRENT_DIAG=experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_current_positionwise/selected.safetensors
STATE_CURRENT_RES=experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors
STATE_IMPROVED_DIAG=experiments/TRR-0007/improved_fit_v1/improved_public_bank/trr0007_current_positionwise/selected.safetensors
STATE_IMPROVED_RES=experiments/TRR-0007/improved_fit_v1/improved_public_bank/trr0007_residual_mlp512/selected.safetensors
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python3 scripts/trr0007_eval_register.py \
  --repository-root "$REPO" --plan experiments/TRR-0007/evaluation_plan.json \
  --source-selection experiments/TRR-0007/selection/source_selection.json \
  --exclusions experiments/TRR-0007/selection/source_exclusions.json \
  --observations "$OBS" --capture-receipt "$CAPTURE" \
  --method-freeze "$METHOD_FREEZE" --frequency-reference "$FREQ" \
  --state current_enriched__trained_diagonal="$STATE_CURRENT_DIAG" \
  --state current_enriched__residual_mlp512="$STATE_CURRENT_RES" \
  --state improved_public_bank__trained_diagonal="$STATE_IMPROVED_DIAG" \
  --state improved_public_bank__residual_mlp512="$STATE_IMPROVED_RES" \
  --public-model-snapshot "$MODEL" \
  --lens experiments/TRR-0004/evidence/comparators/public_a1_lens.pt \
  --reference "$A2_REFERENCE" \
  --output-root experiments/TRR-0007/evaluation/predictions \
  --output "$REG"

PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python3 scripts/trr0007_eval_runner.py \
  --repository-root "$REPO" --registration "$REG" --device cuda
```

After the runner has produced every decoder and anchor artifact, create the
public freeze receipt.  The truth preparation command below is intentionally a
separate step: it revalidates that receipt and all 22 outputs, then materializes
the selected public labels into an external sidecar and writes only its
metadata binding under the task root.  The gate reads that header without
opening or stat-ing the sidecar; the scorer performs the single sidecar read.

```bash
REG=experiments/TRR-0007/evaluation/registration.json
RECEIPT=experiments/TRR-0007/evaluation/freeze_receipt.json
PYTHONPATH=.:src:scripts python3 scripts/trr0007_eval_gate.py freeze \
  --repository-root "$REPO" --registration "$REG" --receipt "$RECEIPT"

PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python3 scripts/trr0007_eval_truth.py prepare --execute \
  --repository-root "$REPO" --receipt "$RECEIPT" --registration "$REG" \
  --selection experiments/TRR-0007/selection/source_selection.json \
  --truth-output /tmp/trr0007/private/truth.safetensors \
  --truth-binding experiments/TRR-0007/truth_binding.json

PYTHONPATH=.:src:scripts python3 scripts/trr0007_score.py \
  --repository-root "$REPO" --receipt "$RECEIPT" --registration "$REG" \
  --truth-binding experiments/TRR-0007/truth_binding.json \
  --result experiments/TRR-0007/scored/result.json \
  --manifest experiments/TRR-0007/manifest.json \
  --report coordination/results/TRR-0007.md
```

The truth command must be run only after the public freeze receipt exists.  Its
external sidecar has exactly `pile__token_ids` and `finance__token_ids`, each
128 by 128; it is joined to both paired target conditions.  Exact recovery is
computed over all 127 post-BOS positions, while BOS remains a fixed diagnostic.


The scorer also emits a descriptive same-pass table of token errors by domain,
target, one-based post-BOS prefix bin, and the common TRR-0005 enriched fitting-frequency
bin.  The public frequency map is bound and hashed in registration, the gate
receipt, and the scored result; no improved-bank coverage classification or
additional truth read is performed.
