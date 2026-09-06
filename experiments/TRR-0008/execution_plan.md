# TRR-0008 end-to-end execution plan

This plan is now an execution record for the completed TRR-0008 run.  The
selection, capture, registration, source-free prediction matrix, metadata
completion, public freeze, truth preparation, and frozen score all completed
under the commands below.  The exact phase commands, receipt hashes,
timestamps, original gate failure, and prediction-file verification are in
`experiments/TRR-0008/execution_receipt_v1.json`.

The only capture source change made while preparing this plan was replacing
the undefined `_resolve(...)` calls with the existing `_resolve_path(...)`
helper.  A synthetic `capture_public(...)` entry smoke exercised every
explicit path resolution and create-only manifest write with all
source/model/GPU loaders stubbed; it did not open public rows or truth.

The sequence is root-authorized and serialized.  The only source inputs are
the public Arrow files, tokenizer/model snapshot, reviewed public LoRA
artifacts, TRR-0007 frozen method states, and the normalized public embedding.
TRR-0007 is read-only.  No command below writes to TRR-0007, P05/P06
workspaces, or the project checkout outside the TRR-0008 worktree.

## Frozen paths and environment

Run from the TRR-0008 worktree, with a fresh version suffix for every retry.
The explicit versioned paths below avoid the adapters' unversioned defaults.

```bash
set -euo pipefail
cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0008

export REPO=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0008
export TRR7=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007
export PROJECT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research

export TOKENIZER=/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
export MODEL_SNAPSHOT="$TOKENIZER"
export PILE_ARROW=/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow
export FINANCE_ARROW_0=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow
export FINANCE_ARROW_1=/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow
export LORA_CONFIG="$PROJECT/outputs/TRR-0002/public-calibration/generation.json"
export LORA_UPDATE="$PROJECT/outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors"

export DECISION=experiments/TRR-0008/planning/decision_contract.json
export PLANNING_STATUS=experiments/TRR-0008/coordination/planning_status.json
export INVENTORY=experiments/TRR-0008/planning/identity_inventory_1thread.json
export METHOD_FREEZE=experiments/TRR-0007/method_freeze.json
export TIMING_RECEIPT=experiments/TRR-0008/timing/precision40_result.json
# Registration requires a timing plan and final receipt as a pair.  The
# The post-run serialization adapter materialized this JSON plan before
# registration.  It binds the canonical final precision40 receipt below; do
# not substitute a Markdown plan or timing/result.json.
export TIMING_PLAN=experiments/TRR-0008/timing/plan.json

export SELECTION_ROOT=experiments/TRR-0008/selection
export SELECTION="$SELECTION_ROOT/source_selection.json"
export EXCLUSIONS="$SELECTION_ROOT/source_exclusions.json"
export RESERVATION="$SELECTION_ROOT/opaque_source_sequence_reservation.json"
export CAPTURE_ROOT=experiments/TRR-0008/evaluation/public_observations_v1
export OBSERVATIONS="$CAPTURE_ROOT/observations.json"
export PRED_ROOT=experiments/TRR-0008/evaluation/predictions_v1
export REGISTRATION=experiments/TRR-0008/evaluation/registration_v1.json
# The runner created run_manifest.json.  The first public gate correctly
# rejected that immutable file because its registration binding lacked the
# required byte count.  The approved metadata-only repair created this
# separate manifest for gate/truth/scoring; the original remains preserved.
export ORIGINAL_RUN_MANIFEST="$PRED_ROOT/run_manifest.json"
export RUN_MANIFEST="$PRED_ROOT/run_manifest.metadata_completed.json"
export RUN_MANIFEST_REPAIR_RECEIPT="$PRED_ROOT/run_manifest.metadata_completed.receipt.json"
export PUBLIC_FREEZE=experiments/TRR-0008/evaluation/public_freeze_v1.json
export TRUTH=/tmp/trr0008_truth_v1.safetensors
export TRUTH_BINDING=experiments/TRR-0008/evaluation/truth_binding_v1.json
export SCORE=experiments/TRR-0008/evaluation/score_v1.json

export PYTHONPATH=.:src:scripts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The reviewed LoRA paths above are the paths recorded by the public TRR-0007
capture interface.  The current filesystem has all six public input paths
above.  The inventory binds the Arrow byte counts and hashes as follows:
Pile `61270696 / 77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113`;
Finance shard 0 `503571864 / b49ca0980a0b02fecbef2220eee0ef5d3c3c893ae42b4e1910edec993c3d164e`;
Finance shard 1 `53756664 / ce4b0786646cd68561da736f145fd5df7ba2f4e754e0caa3ae646d6be9900bd3`.
The normalized public embedding is bound by the method freeze to
`ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
The final timing receipt is bound to 4,076,989 bytes and
`a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`.
Registration and all later gates recheck these bindings.

## Read-only asset and workspace preflight

Run this before selection and again before any GPU stage.  It must fail closed
if an input is missing, a required file hash changes, a task output already
exists, the timing plan has not been materialized, or an output resolves into
another workspace.  This command only stats and hashes declared public or
frozen artifacts; it does not load Arrow rows, token IDs, model weights, or
truth.

```bash
python3 - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["REPO"]).resolve()
trr7 = Path(os.environ["TRR7"]).resolve()
if root.name != "TRR-0008" or root == trr7:
    raise SystemExit(f"wrong or shared worktree: repo={root} trr7={trr7}")
if root.is_symlink() or trr7.is_symlink():
    raise SystemExit("worktree root must not be a symlink")

files = {
    "tokenizer": Path(os.environ["TOKENIZER"]),
    "pile_arrow": Path(os.environ["PILE_ARROW"]),
    "finance_arrow_0": Path(os.environ["FINANCE_ARROW_0"]),
    "finance_arrow_1": Path(os.environ["FINANCE_ARROW_1"]),
    "lora_config": Path(os.environ["LORA_CONFIG"]),
    "lora_update": Path(os.environ["LORA_UPDATE"]),
    "decision": root / os.environ["DECISION"],
    "planning_status": root / os.environ["PLANNING_STATUS"],
    "inventory": root / os.environ["INVENTORY"],
    "method_freeze": trr7 / os.environ["METHOD_FREEZE"],
    "timing_receipt": root / os.environ["TIMING_RECEIPT"],
    "timing_plan": root / os.environ["TIMING_PLAN"],
}
for label, path in files.items():
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} is unavailable or symlink: {path}")

model = Path(os.environ["MODEL_SNAPSHOT"]).resolve()
if model.is_symlink() or not model.is_dir():
    raise SystemExit(f"model snapshot is unavailable or symlink: {model}")

expected = {
    "pile_arrow": (61270696, "77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113"),
    "finance_arrow_0": (503571864, "b49ca0980a0b02fecbef2220eee0ef5d3c3c893ae42b4e1910edec993c3d164e"),
    "finance_arrow_1": (53756664, "ce4b0786646cd68561da736f145fd5df7ba2f4e754e0caa3ae646d6be9900bd3"),
    "timing_receipt": (4076989, "a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02"),
}
for label, (size, digest) in expected.items():
    path = files[label]
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    actual = (path.stat().st_size, h.hexdigest())
    if actual != (size, digest):
        raise SystemExit(f"{label} binding changed: {actual} != {(size, digest)}")

# Every write target is task-local, except the deliberately external truth
# sidecar.  Existing targets are never removed or reused.
task = root / "experiments" / "TRR-0008"
for env_name in (
    "CAPTURE_ROOT", "PRED_ROOT", "REGISTRATION",
    "PUBLIC_FREEZE", "TRUTH_BINDING", "SCORE",
):
    raw = Path(os.environ[env_name])
    path = (root / raw if not raw.is_absolute() else raw).resolve()
    try:
        path.relative_to(task)
    except ValueError as exc:
        raise SystemExit(f"{env_name} escapes TRR-0008: {path}") from exc
    if path.exists() or path.is_symlink():
        raise SystemExit(f"create-only target already exists: {path}")
truth = Path(os.environ["TRUTH"]).resolve()
if truth.is_symlink() or truth.exists():
    raise SystemExit(f"truth sidecar target already exists: {truth}")
try:
    truth.relative_to(root)
except ValueError:
    pass
else:
    raise SystemExit(f"truth sidecar must be outside repository: {truth}")
print("asset/workspace preflight: PASS")
PY
```

The `TIMING_PLAN` check was intentionally fail-closed.  The post-run
serialization adapter subsequently created `timing/plan.json`, which was
bound together with the canonical `precision40_result.json` by registration.
No alternative timing artifact was used.

## Identity-only selection prerequisite

The identity-only selection is already frozen at the actual task paths
`experiments/TRR-0008/selection/source_selection.json`,
`experiments/TRR-0008/selection/source_exclusions.json`, and
`experiments/TRR-0008/selection/opaque_source_sequence_reservation.json`.
Verify those files before capture; do not rerun the create-only selector when
they exist.  For a clean unreleased checkout, the exact owner-authorized
selection command would be the following.  It reads public identity rows and
writes only metadata; it does not load the model or create truth.

```bash
python3 scripts/trr0008_select_public.py select \
  --repository-root "$REPO" \
  --decision-contract "$DECISION" \
  --planning-status "$PLANNING_STATUS" \
  --inventory "$INVENTORY" \
  --method-freeze "$METHOD_FREEZE" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --output "$SELECTION" \
  --exclusions-output "$EXCLUSIONS"

python3 scripts/trr0008_select_public.py reserve \
  --repository-root "$REPO" \
  --selection "$SELECTION" \
  --output "$RESERVATION"
```

The selector and reservation are create-only.  A selection failure leaves its
failure/log evidence for review; do not delete it or replace the source panel
with a new sample.  Verify the resulting selection status is
`FROZEN_TRR0008_SOURCE_SELECTION_NO_TRUTH`, counts are Finance 1024/Pile 384,
and all source/truth flags remain false before capture.

## Capture: largest-cell qualification and full public observations

Capture uses the task-local adapter with the trusted B8 x 192 producer and
retains only positions 0..127.  It first executes the actual largest
representative Finance batch for each target condition.  The producer's
qualification must report `primary_geometry.active_output_bit_exact=true`
with `maximum_absolute_difference=0.0` for both `public_base` and
`public_lora_2601`; otherwise stop before treating any observations as valid.
The full run then writes four BF16 observation files, masks, positions, and
metadata-only receipts.

```bash
python3 scripts/trr0008_eval_capture.py capture --execute \
  --repository-root "$REPO" \
  --selection "$SELECTION" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --lora-config "$LORA_CONFIG" \
  --lora-update "$LORA_UPDATE" \
  --output-root "$CAPTURE_ROOT" \
  --device cuda
```

Use one exclusive GPU lease.  Before launch, require no foreign CUDA compute
process, at least 8 GiB free GPU memory, at least 10 GiB host available
memory, and a clean task commit.  The capture producer enforces 8 GiB maximum
reserved GPU memory and 16 GiB maximum RSS while capturing.  Preserve the
launch commit and SHA-256 of the capture/trusted producer source files before
launch; the capture receipt records the final source bindings as well.

The retained activation payload is deterministic in size:
`2 * (1024 + 384) * 128 * 2048 * 2 = 1,476,395,008` bytes (1.375 GiB) of
BF16 values, plus sidecars and safetensors metadata.  The temporary full
192-token payload for one complete target/domain cell is at most 2.0625 GiB
if materialized at once, while the producer frees each cell before the next.
TRR-0007's public capture receipt measured 10.73 seconds for its smaller
four-cell panel and 2.27 GB maximum RSS; the larger panel is expected to be
well within the 600-second serialized lease when the live guard remains
healthy.  Treat these as forecasts, not a relaxation of the guard.

After success, verify `capture.json`, `observations.json`, `panel.json`, all
four observation files, exact `[finance=1024,pile=384]` counts, BF16
`[records,128,2048]` activation geometry, all-one masks, positions 0..127,
and every truth/source flag false.  A capture `failure.json`, nonfinite value,
geometry mismatch, or guard anomaly ends the lease and is preserved.

## Registration and source-free prediction

Registration binds the frozen method states, normalized public E, captured
observation manifest, four scientific methods, and the canonical 40-block
runtime receipt.  It writes no predictions and opens no source or truth.
`DECISION` is the owner-frozen decision contract and remains the scoring
authority; registration binds this same contract rather than the earlier power
artifact.

```bash
python3 scripts/trr0008_eval_register.py \
  --repository-root "$REPO" \
  --method-freeze "$METHOD_FREEZE" \
  --observations "$OBSERVATIONS" \
  --output-root "$PRED_ROOT" \
  --plan "$DECISION" \
  --timing-plan "$TIMING_PLAN" \
  --timing-receipt "$TIMING_RECEIPT" \
  --output "$REGISTRATION"
```

The registration must contain exactly the four methods
`trr6__enriched_trained_diagonal_attention128`,
`current_enriched__residual_mlp512`,
`improved_public_bank__residual_mlp512`, and
`improved_public_bank__trained_diagonal`; A2 is absent.  Before runner launch,
verify its `code_commit` is the current clean commit and its output root is
new and below TRR-0008.

The existing source-free runner qualifier has already passed the 16-method/cell
fixture with exact ID equivalence.  If runner, contract, positionwise loader,
or numerical settings change after that receipt, rerun the qualifier before
this full matrix; the capture-only resolver/test change does not change that
runner path.  For the authorized full run:

```bash
python3 scripts/trr0008_eval_runner.py \
  --repository-root "$REPO" \
  --registration "$REGISTRATION" \
  --device cuda
```

The runner uses the registration output root and writes exactly 16 prediction
artifacts plus 16 per-cell timing artifacts and `run_manifest.json`.  It uses
only current H, frozen E, and registered decoder states.  Its numerical
settings are applied and checked at runtime; keep the environment above and
do not change TF32, dtype, or thread settings.

The approved row-wise forecast is 11,264 prediction rows, each with one
warmup and one measured call (22,528 decoder calls) and about 11,266 guard
checks.  The current forecast is approximately 395 seconds mean and 407
seconds at the p95 estimate, below the 600-second runner guard.  Peak memory
is dominated by the 0.9785 GiB float32 E table, 20.99/29.39 MB decoder
states, and approximately 62.1 MiB of one-row full-vocabulary logits; the
runner processes eight records per chunk and does not retain all logits.  The
registration guard is 6 GiB maximum reserved GPU, 8 GiB minimum free GPU,
16 GiB maximum RSS, and 10 GiB minimum host available memory.  Stop on any
foreign process, guard breach, nonfinite output, warmup/measured ID mismatch,
OOM, or partial-output anomaly.

## Public gate, truth preparation, and scoring

The public gate is mandatory before any label read.  It verifies all 16
prediction hashes and timing files, the registration/observation/code
bindings, and the final timing receipt.  It writes a create-only freeze
receipt.

```bash
python3 scripts/trr0008_eval_gate.py \
  --repository-root "$REPO" \
  --registration "$REGISTRATION" \
  --run-manifest "$RUN_MANIFEST" \
  --timing-receipt "$TIMING_RECEIPT" \
  --output "$PUBLIC_FREEZE"
```

The runner's original manifest is retained at `$ORIGINAL_RUN_MANIFEST`.  The
first gate invocation against that file failed closed before truth because
its `registration` object had the path and SHA-256 but omitted `bytes`.  The
approved create-only repair added exactly that one metadata field and wrote
`$RUN_MANIFEST` plus `$RUN_MANIFEST_REPAIR_RECEIPT`; it did not alter the
original manifest, predictions, registration, code commit, or scientific
outputs.  The gate command above was then repeated with `$RUN_MANIFEST` and
returned `PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH`.  See the phase receipt for
the exact failed command/message and both manifest hashes.

Only after the gate reports `PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH` may root
prepare the external truth sidecar.  The truth adapter calls the public gate
again before materializing selected rows and invokes the metadata-only
pre-truth gate after writing its header.  Its sidecar is outside the
repository and prediction root; the header is task-local and contains only
identity/hash metadata.

```bash
python3 scripts/trr0008_eval_truth.py prepare --execute \
  --repository-root "$REPO" \
  --receipt "$PUBLIC_FREEZE" \
  --run-manifest "$RUN_MANIFEST" \
  --registration "$REGISTRATION" \
  --selection "$SELECTION" \
  --tokenizer "$TOKENIZER" \
  --pile-arrow "$PILE_ARROW" \
  --finance-arrow "$FINANCE_ARROW_0" "$FINANCE_ARROW_1" \
  --truth-output "$TRUTH" \
  --truth-binding "$TRUTH_BINDING"
```

Truth preparation is CPU-only and is the first authorized source-row/label
operation.  If the public gate or header identity check fails, preserve the
failure and do not inspect or reuse a sidecar.  The scorer independently
revalidates the full public gate and truth-sidecar binding before its sole
`safetensors` truth read:

```bash
python3 scripts/trr0008_score.py \
  --repository-root "$REPO" \
  --predictions-root "$PRED_ROOT" \
  --truth "$TRUTH" \
  --registration "$REGISTRATION" \
  --run-manifest "$RUN_MANIFEST" \
  --freeze-receipt "$PUBLIC_FREEZE" \
  --truth-binding "$TRUTH_BINDING" \
  --result "$SCORE" \
  --decision-contract "$DECISION" \
  --timing-receipt "$TIMING_RECEIPT" \
  --bootstrap-draws 10000
```

The score must serialize as JSON before its create-only write.  It reports
record-level exact CP gain-minus-loss bounds, source-record bootstrap token
bounds, all four safeguard cells, candidate versus current residual, candidate
versus same-bank diagonal, and the canonical timing decision.  A missing,
malformed, inconclusive, or mismatched timing receipt cannot pass the cost
route.

## Restart and failure handling

Each stage is create-only.  Never `rm`, overwrite, symlink, alias, or patch a
partial output.  Preserve `failure.json` and any partial artifact list.  A
reviewed retry uses a new suffix for every dependent root and sidecar, for
example `v2` for capture, predictions, freeze, truth binding, truth sidecar,
and score; it also creates a new registration binding to that root.  A retry
never silently mixes artifacts from two suffixes.

The safe order is selection/reservation -> capture -> registration -> runner ->
public gate -> truth binding/sidecar -> scorer.  Truth cannot be opened by a
failed or incomplete public matrix.  A source/code/contract change after
registration invalidates registration and any qualifier receipt; root must
make a new commit and restart from registration or earlier.  Release the GPU
lease after capture, and after runner if no further GPU diagnostics are
needed.  Record commands, commit/source hashes, input/state bindings,
start/end times, preparation/adaptation/inference/I/O times, peak memory,
failure receipts, and artifact SHA-256 values in the generated receipts.

## Completed execution handoff

The authoritative completed-manifest path for the post-run gate, truth
adapter, and scorer is
`experiments/TRR-0008/evaluation/predictions_v1/run_manifest.metadata_completed.json`.
The runner output remains separately available at
`experiments/TRR-0008/evaluation/predictions_v1/run_manifest.json`.  The
task-local execution receipt records the exact commands and all phase
evidence; no private truth payload is stored in this repository.
