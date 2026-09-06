# TRR-0005 fresh prediction replay: V1, metadata export, freeze, and score

This is a replay record for the completed fresh evaluation. It is written from
`final_prediction_run_evidence_v1/launch_receipt.json`,
`prediction_contract_export_evidence_v1/export_launch_receipt.json`, and
`freeze_score_attempt_v2/execution_receipt.json`. It does not rerun a command,
open evaluator truth, or change the report/manifest.

The blocks marked **historical transcription** preserve the exact argv recorded
by the completed run. They retain the original worktree and create-only output
paths and are not directly executable after archive restoration. Use the
score-only replay block below, or the optional full reexecution block with new
output roots.

The prediction science was executed from tracked commit
`da82f6cac45e09ae83452198344c547553cb4433`. The later producer schema fix was
applied and tested only after scoring; it is maintenance evidence and is not
the source used for the executed V1 matrix.

## What is being replayed

The fresh panel has four cells and 128 paired records per cell. The prediction
run produced eight methods per cell, for 32 prediction artifacts and 32 timing
receipts. It used one warmup and one measured call per record. The recorded V1
run started at `2026-09-06T00:39:23Z`, ended at `2026-09-06T00:58:09Z`, returned
zero, and recorded `1125.23` seconds from the external timer (`1126.0` seconds
at UTC-second wrapper resolution). Its maximum reserved CUDA memory was
`3546284032` bytes and maximum process RSS was `5816393728` bytes.

V1 is retained as the raw scientific execution record at
`experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/`. The prediction
computation completed with no truth access, but the first strict freeze attempt
was deliberately preserved as a failure: at
`2026-09-06T01:02:31.130226+00:00` it returned 1 before scoring with:

```text
TRR-0005 freeze error: prediction identity changed: finance__public_base/enriched__affine_causal_h_attention128
```

That failure was caused by the legacy row schema
`token-reconstruction.trr0005-prediction-receipt.v1` in the merged JSON rows.
Do not rewrite V1 or treat it as a scored matrix.

The metadata-only exporter created the new root
`experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/`.
It copied all 32 `.safetensors` files byte-for-byte and rebuilt the 32 row
receipts plus the two entry manifests. It changed only the row schema and the
artifact path cross-references; prediction digests, tensor bytes, timing
numbers, and the raw V1 files remain unchanged. The exporter did not load
source text, truth, future activations, model state, or tensor contents. Its
SHA-256 at execution was
`87d6c46ef458edda5adb167977fc7f34c33e87385247e09ac32fefb4488645f9`.

The V2 root does not contain a copied private sidecar. The only truth-related
file added before freeze is the label-free `evaluator_binding.json`, copied
from the already preserved V1 binding. The strict V2 freeze and score then
passed; the scorer reported
`FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE` with 32 prediction
artifacts and 32 timing receipts. The complete phase receipt is
`experiments/TRR-0005/freeze_score_attempt_v2/execution_receipt.json`.

## Restore a replay checkout safely

Use a clean detached checkout at the executable science commit. The
prediction and score receipts bind the scientific source to
`da82f6cac45e09ae83452198344c547553cb4433`. The reviewed post-score helper
handoff is commit `1dba67a8dc75844727866cb4273da28a311df216`; it also contains a
maintenance edit to the tracked prediction driver, so restore only the two
helper files from that commit. The compact state/prediction evidence is preserved in immutable artifact
commit `c88203883038a151ef70e1aba31fab06daf3b65f`; the replay binds
`EVIDENCE_COMMIT` to that commit.

```bash
REPO=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005
SCIENCE_COMMIT=da82f6cac45e09ae83452198344c547553cb4433
HANDOFF_COMMIT=1dba67a8dc75844727866cb4273da28a311df216
EVIDENCE_COMMIT=c88203883038a151ef70e1aba31fab06daf3b65f
REPLAY=/tmp/trr0005-replay-v2

git -C "$REPO" worktree add --detach "$REPLAY" "$SCIENCE_COMMIT"
git -C "$REPLAY" rev-parse HEAD
```

The second command must print `da82f6cac45e09ae83452198344c547553cb4433`.
Restore only the new serialization and truth-alias helpers from the reviewed
handoff commit:

```bash
HANDOFF_TAR=/tmp/trr0005-handoff-helpers.tar

git -C "$REPO" archive --format=tar "$HANDOFF_COMMIT" -- \
  scripts/trr0005_export_prediction_contract.py \
  scripts/trr0005_truth_alias_adapter.py \
  > "$HANDOFF_TAR"
tar -xf "$HANDOFF_TAR" -C "$REPLAY"
```

Restore compact states, configs, receipts, panel metadata, and prediction
bytes from the immutable artifact commit with an explicit experiment-only path.
This path list does not overwrite frozen tracked source:

```bash
EVIDENCE_TAR=/tmp/trr0005-evidence.tar

git -C "$REPO" archive --format=tar "$EVIDENCE_COMMIT" -- \
  experiments/TRR-0005 \
  > "$EVIDENCE_TAR"
tar -xf "$EVIDENCE_TAR" -C "$REPLAY"
```

Do not use `git restore --source "$HANDOFF_COMMIT" -- .` here. In particular,
do not restore `scripts/trr0005_run_predictions.py` from the later maintenance
commit when replaying the historical V1 science. Verify the tracked science
source separately before a prediction replay:

```bash
git -C "$REPLAY" show "$SCIENCE_COMMIT:scripts/trr0005_run_predictions.py" \
  | sha256sum
git -C "$REPLAY" rev-parse HEAD
```

The compact archive carries the panel, observations, selection and method
registration, selected joint decoder states, V1/V2 metadata and prediction
artifacts, and their receipts. The larger raw public H and fitting tensors
remain external where their receipts say so. The corpus, coverage, fitting,
qualification, and asset archive commands are recorded in
[`footing/development_reproduction.md`](footing/development_reproduction.md);
that document is the source for those earlier phases and is not duplicated
here.

## External prerequisites

The exact V1 command depends on the following existing public assets and
frozen metadata:

- the task virtual environment at `$REPO/.venv-trr0005/bin/python`;
- the Llama snapshot at
  `/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`;
- the retained A1 reference
  `experiments/TRR-0004/evidence/comparators/round001_teacher.py` and lens
  `experiments/TRR-0004/evidence/comparators/public_a1_lens.pt`;
- `fresh_confirmation_v1/panel_capture_v2/{panel.json,observations.json}`,
  `selection_plan.json`, and `method_registration.json`;
- the selected states under `experiments/TRR-0005/joint_fit_v1/` and
  `experiments/TRR-0005/joint_fit_qknorm_v1/`, plus the shared public E and
  public A2/P0 assets bound by the registration. The shared normalized E is
  recorded at
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`;
- `experiments/TRR-0005/frequency_references_v1.json` for scoring; and
- the private truth sidecar at `/tmp/trr5/fresh_confirmation_v1.truth.safetensors`
  and its external manifest. Keep both outside the repository and frozen
  prediction root. The sidecar is not a prerequisite for V1, export, or the
  public freeze gate.

When regenerating the external truth sidecar, invoke the narrow alias adapter
rather than calling the raw producer writer directly. It patches only the
producer's serialization symbol, clones shared tensors at the serialization
boundary, and restores the symbol in `finally`. The producer-only command and
its receipt are recorded in the development reproduction document; its output
paths are:

```text
/tmp/trr5/fresh_confirmation_v1.truth.safetensors
/tmp/trr5/fresh_confirmation_v1.truth.manifest.json
```

Do not inspect or copy the private safetensors sidecar during the public
prediction or export phase. Only the manifest's label-free binding descriptor
is copied into the V2 root after V1/export and before freeze.

## Historical V1 prediction transcript (not an executable replay)

The following block is the exact inner `argv` from
`final_prediction_run_evidence_v1/launch_receipt.json`, with its recorded
offline and CPU-thread overrides. It intentionally points at the original
worktree and the original create-only `predictions_v1` root; it is a provenance
transcription, not a directly executable command from `$REPLAY`. The outer
runner used a 1650-second timeout and a 30-second kill grace period; those
values are also preserved in the receipt.

```bash
env HF_HUB_OFFLINE=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts TRANSFORMERS_OFFLINE=1 \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python \
scripts/trr0005_run_predictions.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --panel experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json \
  --selection-plan experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json \
  --registration experiments/TRR-0005/fresh_confirmation_v1/method_registration.json \
  --fit-root experiments/TRR-0005/joint_fit_v1 \
  --causal-fit-root experiments/TRR-0005/joint_fit_qknorm_v1 \
  --output-root experiments/TRR-0005/fresh_confirmation_v1/predictions_v1 \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --reference experiments/TRR-0004/evidence/comparators/round001_teacher.py \
  --lens experiments/TRR-0004/evidence/comparators/public_a1_lens.pt \
  --device cuda \
  --minimum-free-gib 8 \
  --maximum-reserved-gib 6 \
  --maximum-rss-gib 16 \
  --minimum-host-available-gib 10 \
  --max-seconds 1500
```

The V1 root is create-only. A replay that is intended to reproduce the
historical failed contract attempt must preserve the legacy row receipts and
stop at the same strict-freeze failure. Do not apply the later producer patch
before this V1 reproduction.

## Historical V2 exporter transcript (not an executable replay)

The exporter is serialization-only and requires a new destination. The
following block is the exact recorded execution command; its absolute paths
point at the original worktree and completed V1/V2 roots, so do not run it as a
replay command. Its helper hash is recorded in
`prediction_contract_export_evidence_v1/export_launch_receipt.json` and
`export_execution_receipt.json`:

```bash
env PYTHONPATH=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/src \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/scripts/trr0005_export_prediction_contract.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --source-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v1 \
  --output-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export
```

The destination must not exist. The exporter records the source and
 destination file hashes in
`fresh_confirmation_v1/predictions_v2_contract_export/export_provenance.json`.
The exported row schema is
`token-reconstruction.trr0005-fresh-confirmation-prediction.v1`; the two
manifest roots retain their own prediction-manifest and timing-manifest
schemas. The legacy `timing_schema` value remains timing provenance and is not
silently relabelled.

The exporter does not copy `evaluator_binding.json`. Copy the label-free
binding in the next phase, after checking that the destination is absent. The
actual V2 receipt used the following commands:

```bash
test ! -e /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/evaluator_binding.json
cp -- \
  /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/evaluator_binding.json \
  /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/evaluator_binding.json
```

This binding copy is byte-for-byte metadata copying. It does not copy the
private truth tensor.

## Historical V2 freeze transcript (not an executable replay)

The actual V2 freeze phase used the following command, with the recorded
CPU-thread environment. The paths are retained exactly from the receipt and
point at the original create-only root; use the executable replay block below
for a restored checkout:

```bash
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/scripts/trr0005_freeze_confirmation.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --panel /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json \
  --registration /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/method_registration.json \
  --output-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export \
  --plan /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json \
  --receipt /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/freeze_receipt_v2.json
```

This phase validates all 32 prediction/timing rows, artifacts, panel bindings,
registration, and selection plan before any truth loader is called. It writes
the create-only freeze receipt outside the prediction root. The recorded V2
freeze returned zero at `2026-09-06T01:27:59.261162+00:00`.

## Historical V2 score transcript (not an executable replay)

The actual V2 scoring phase used the following command. It is included for
provenance; its paths point at the original worktree and archived output names,
so do not run it as written. Unlike the preceding phases, it opens the
external truth sidecar only after the complete public gate and freeze receipt
pass.

```bash
env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/.venv-trr0005/bin/python \
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/scripts/trr0005_score_confirmation.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005 \
  --panel /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json \
  --registration /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/method_registration.json \
  --selection-plan /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json \
  --observations /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations.json \
  --predictions /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/predictions.json \
  --timings /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/timings.json \
  --receipt /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/freeze_receipt_v2.json \
  --truth /tmp/trr5/fresh_confirmation_v1.truth.safetensors \
  --truth-binding /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/evaluator_binding.json \
  --output-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export \
  --result /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/fresh_confirmation_v1/result.json \
  --frequency-manifest /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005/experiments/TRR-0005/frequency_references_v1.json \
  --bootstrap-draws 10000 \
  --bootstrap-seed 5005
```

The score receipt records return code zero at
`2026-09-06T01:28:11.015844+00:00`, no scorer stderr, and status
`FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE`. The raw V1 root,
metadata-export provenance, V2 freeze receipt, and score execution receipt
remain separate evidence; do not replace the V1 failure with V2 metadata or
rewrite either execution record.

## Executable score-only replay from immutable V2

This is the minimal executable replay after the compact evidence has been
restored into `$REPLAY`. It does not regenerate V1 predictions or run the
metadata exporter. It creates new freeze and result files, leaving the archived
V2 receipt and score result untouched. The panel, registration, selection plan,
observations, V2 manifests, 32 V2 artifacts, and label-free binding must be
present under `$REPLAY`; the external truth sidecar must remain at `$TRUTH`.

```bash
REPLAY=/tmp/trr0005-replay-v2
FRESH="$REPLAY/experiments/TRR-0005/fresh_confirmation_v1"
PANEL="$FRESH/panel_capture_v2/panel.json"
OBSERVATIONS="$FRESH/panel_capture_v2/observations.json"
REGISTRATION="$FRESH/method_registration.json"
PLAN="$FRESH/selection_plan.json"
PRED_ROOT="$FRESH/predictions_v2_contract_export"
PREDICTIONS="$PRED_ROOT/predictions.json"
TIMINGS="$PRED_ROOT/timings.json"
BINDING="$PRED_ROOT/evaluator_binding.json"
TRUTH=/tmp/trr5/fresh_confirmation_v1.truth.safetensors
FREQUENCIES="$REPLAY/experiments/TRR-0005/frequency_references_v1.json"
REPLAY_FREEZE="$FRESH/freeze_receipt_replay_v3.json"
REPLAY_RESULT="$FRESH/result_replay_v3.json"

test -f "$BINDING"
test ! -e "$REPLAY_FREEZE"
test ! -e "$REPLAY_RESULT"

env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$REPLAY/.venv-trr0005/bin/python" "$REPLAY/scripts/trr0005_freeze_confirmation.py" \
  --repository-root "$REPLAY" \
  --panel "$PANEL" \
  --registration "$REGISTRATION" \
  --output-root "$PRED_ROOT" \
  --plan "$PLAN" \
  --receipt "$REPLAY_FREEZE"

env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$REPLAY/.venv-trr0005/bin/python" "$REPLAY/scripts/trr0005_score_confirmation.py" \
  --repository-root "$REPLAY" \
  --panel "$PANEL" \
  --registration "$REGISTRATION" \
  --selection-plan "$PLAN" \
  --observations "$OBSERVATIONS" \
  --predictions "$PREDICTIONS" \
  --timings "$TIMINGS" \
  --receipt "$REPLAY_FREEZE" \
  --truth "$TRUTH" \
  --truth-binding "$BINDING" \
  --output-root "$PRED_ROOT" \
  --result "$REPLAY_RESULT" \
  --frequency-manifest "$FREQUENCIES" \
  --bootstrap-draws 10000 \
  --bootstrap-seed 5005
```

The freeze command remains before the scorer and performs the complete public
check. Only the scorer crosses the truth boundary. Keep the truth sidecar
outside `$REPLAY` and do not inspect it before the freeze command succeeds.

## Optional full reexecution with new create-only roots

A full prediction rerun is unnecessary when the immutable V2 artifacts are
available. If a new scientific execution is explicitly authorized, never point
it at the archived V1 or V2 roots. Use separate roots and regenerate their
manifests and label-free binding:

```bash
REPLAY=/tmp/trr0005-replay-v2
PANEL="$REPLAY/experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json"
PLAN="$REPLAY/experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json"
REGISTRATION="$REPLAY/experiments/TRR-0005/fresh_confirmation_v1/method_registration.json"
REPLAY_FRESH="$REPLAY/experiments/TRR-0005/replay_confirmation_v1"
REPLAY_V1="$REPLAY_FRESH/predictions_v1"
REPLAY_V2="$REPLAY_FRESH/predictions_v2_contract_export"
TRUTH_MANIFEST=/tmp/trr5/fresh_confirmation_v1.truth.manifest.json

test ! -e "$REPLAY_V1"
test ! -e "$REPLAY_V2"

env HF_HUB_OFFLINE=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=.:src:scripts TRANSFORMERS_OFFLINE=1 \
  "$REPLAY/.venv-trr0005/bin/python" "$REPLAY/scripts/trr0005_run_predictions.py" \
  --repository-root "$REPLAY" \
  --panel "$PANEL" \
  --selection-plan "$PLAN" \
  --registration "$REGISTRATION" \
  --fit-root "$REPLAY/experiments/TRR-0005/joint_fit_v1" \
  --causal-fit-root "$REPLAY/experiments/TRR-0005/joint_fit_qknorm_v1" \
  --output-root "$REPLAY_V1" \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --reference "$REPLAY/experiments/TRR-0004/evidence/comparators/round001_teacher.py" \
  --lens "$REPLAY/experiments/TRR-0004/evidence/comparators/public_a1_lens.pt" \
  --device cuda \
  --minimum-free-gib 8 \
  --maximum-reserved-gib 6 \
  --maximum-rss-gib 16 \
  --minimum-host-available-gib 10 \
  --max-seconds 1500

env PYTHONPATH="$REPLAY/src" \
  "$REPLAY/.venv-trr0005/bin/python" "$REPLAY/scripts/trr0005_export_prediction_contract.py" \
  --repository-root "$REPLAY" \
  --source-root "$REPLAY_V1" \
  --output-root "$REPLAY_V2"

test ! -e "$REPLAY_V2/evaluator_binding.json"
cp -- "$TRUTH_MANIFEST" "$REPLAY_V2/evaluator_binding.json"
```

The new V1 and V2 roots have independent create-only manifests, receipts,
and artifact paths. To freeze and score this optional rerun, set `FRESH` and
`PRED_ROOT` in the executable score-only block to `$REPLAY_FRESH` and
`$REPLAY_V2`, choose new receipt/result names under `$REPLAY_FRESH`, and use
`$REPLAY` for every `--repository-root` and executable path.

## Post-score maintenance boundary

After the score completed, the minimal producer maintenance patch removed the
inherited legacy timing `schema` before merging the timing receipt into a
prediction descriptor. It was tested with the real
`trr0005_predict_confirmation.SCHEMA` constant. The targeted driver/export/
truth-alias suite passed 11 tests and the full CPU-only suite passed 302 tests;
receipts are under `experiments/TRR-0005/footing/postscore_tests_v1/`.

That patched source is useful for future runs, but it was not the executable
source of the V1 matrix and must not be substituted when replaying the
historical scientific run. The V1/V2 distinction, byte-identical prediction
artifacts, and pretruth/public gate order above are part of the recorded
experiment.
