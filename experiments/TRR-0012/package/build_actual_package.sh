#!/usr/bin/env bash
set -euo pipefail

# Small create-only assembly wrapper. It delegates identities, selection
# receipt generation, prediction, and contract validation to the existing
# TRR-0012 producer; this file only supplies paths and immutable metadata.

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PACKAGE_ROOT=${TRR0012_PACKAGE_ROOT:-"$ROOT/outputs/TRR-0012/model-package-actual-r2"}
SECONDARY_ROOT=${TRR0012_SECONDARY_ROOT:-/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012}
DEVICE=${TRR0012_PACKAGE_DEVICE:-cuda}
RELEASE_FILE=${TRR0012_PACKAGE_RELEASE_FILE:-}
MODE=plan

B0_STATE="$ROOT/outputs/TRR-0012/fixed_fits/native_pair_live_r4/current_fixed_replication_1/selected.safetensors"
B1_STATE="$ROOT/outputs/TRR-0012/fixed_fits/native_pair_live_r4/expanded_fixed_replication_1/selected.safetensors"
PUBLISHED_BUNDLE="$ROOT/experiments/TRR-0012/package/published_bundle"
BUNDLE_MANIFEST="$PUBLISHED_BUNDLE/published_bundle_manifest.json"
B0_BINDING="$PUBLISHED_BUNDLE/metadata/B0_selection_binding.json"
B1_BINDING="$PUBLISHED_BUNDLE/metadata/B1_selection_binding.json"
E_SOURCE="/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
SMOKE_SOURCE="$PUBLISHED_BUNDLE/smoke/public_base_first2.safetensors"
TEMPLATE="$PUBLISHED_BUNDLE/templates/package_manifest_template_v1.json"
CONTRACT_TEMPLATE="$PUBLISHED_BUNDLE/templates/contract_v1.json"
PRODUCER="$PUBLISHED_BUNDLE/code/trr0012_package.py"
ORIGINAL_PRODUCER="$PUBLISHED_BUNDLE/reference_code/trr0012_package_original_908392.py"
ORIGINAL_SHA=9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109
INSTRUMENTED_SHA=5023a9e7cb19611952c0ce16a440ec9fde6b6444fd74aaf74a7efbb5c671c13b

while (($#)); do
  case "$1" in
    --plan) MODE=plan ;;
    --execute) MODE=execute ;;
    --release-file) shift; RELEASE_FILE=${1:?missing release file} ;;
    --device) shift; DEVICE=${1:?missing device} ;;
    --package-root) shift; PACKAGE_ROOT=${1:?missing package root} ;;
    --secondary-root) shift; SECONDARY_ROOT=${1:?missing secondary root} ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

sha256() { sha256sum "$1" | awk '{print $1}'; }
require_regular() {
  test -f "$1" && test ! -L "$1" || {
    echo "regular non-symlink file required: $1" >&2
    exit 1
  }
}
copy_or_verify() {
  local src=$1 dst=$2
  require_regular "$src"
  mkdir -p "$(dirname "$dst")"
  if test -e "$dst" || test -L "$dst"; then
    test ! -L "$dst" || { echo "destination symlink: $dst" >&2; return 1; }
    test "$(sha256 "$src")" = "$(sha256 "$dst")" || {
      echo "destination differs: $dst" >&2
      return 1
    }
  else
    install -m 0444 "$src" "$dst"
  fi
}
run_cli() {
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 python3 "$@"
}
check_sources() {
  for p in "$B0_STATE" "$B1_STATE" "$BUNDLE_MANIFEST" "$B0_BINDING" "$B1_BINDING" "$E_SOURCE" "$SMOKE_SOURCE" "$TEMPLATE" "$CONTRACT_TEMPLATE" "$PRODUCER" "$ORIGINAL_PRODUCER"; do
    require_regular "$p"
  done
  test "$(sha256 "$PRODUCER")" = "$INSTRUMENTED_SHA" || {
    echo "instrumented producer SHA changed; obtain reviewed SHA before assembly" >&2
    exit 1
  }
  test "$(sha256 "$ORIGINAL_PRODUCER")" = "$ORIGINAL_SHA" || {
    echo "original producer reference SHA changed" >&2
    exit 1
  }
  python3 - "$BUNDLE_MANIFEST" "$PUBLISHED_BUNDLE" "$B0_BINDING" "$B1_BINDING" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
bundle_root = Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for entry in manifest.get("files", []):
    relative = entry.get("published_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SystemExit(f"invalid published bundle path: {relative!r}")
    path = (bundle_root / relative).resolve()
    try:
        path.relative_to(bundle_root)
    except ValueError:
        raise SystemExit(f"published bundle path escaped root: {relative}")
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"published bundle file missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if int(entry.get("bytes", -1)) != path.stat().st_size or entry.get("sha256") != digest:
        raise SystemExit(f"published bundle hash mismatch: {path}")
for path in sys.argv[3:]:
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("status") not in {
        "PASS_METADATA_ONLY_B0_HEADER_NORMALIZED",
        "PASS_METADATA_ONLY_B1_HEADER_NORMALIZED",
    }:
        raise SystemExit(f"binding not normalized: {path}")
    if payload.get("truth_boundary", {}).get("truth_opened") is not False:
        raise SystemExit(f"truth boundary changed: {path}")
PY
}

if [[ "$MODE" == plan ]]; then
  check_sources
  printf '%s\n' \
    "mode=plan" \
    "package_root=$PACKAGE_ROOT" \
    "secondary_root=$SECONDARY_ROOT" \
    "device=$DEVICE" \
    "producer_sha=$(sha256 "$PRODUCER")" \
    "original_sha=$(sha256 "$ORIGINAL_PRODUCER")" \
    "b0_state=$(sha256 "$B0_STATE")" \
    "b1_state=$(sha256 "$B1_STATE")" \
    "E=$(sha256 "$E_SOURCE")" \
    "published_bundle_manifest=$(sha256 "$BUNDLE_MANIFEST")" \
    "no_copy_or_model_execution=true"
  exit 0
fi

if [[ "${TRR0012_PACKAGE_LEASE_RELEASED:-}" != true ]]; then
  echo "set TRR0012_PACKAGE_LEASE_RELEASED=true only after the coordinator releases the package lease" >&2
  exit 1
fi
if [[ -z "$RELEASE_FILE" ]]; then
  echo "--release-file is required for an executed build" >&2
  exit 1
fi
require_regular "$RELEASE_FILE"
python3 - "$RELEASE_FILE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("authorized") is not True and payload.get("status") not in {"AUTHORIZED", "RELEASED", "PASS"}:
    raise SystemExit("package lease release is not explicit")
PY
check_sources

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$PACKAGE_ROOT"/{code/token_reconstruction,config,identity,readout,receipts,reference_code,smoke,states,verification}

for rel in \
  code/token_reconstruction/__init__.py \
  code/token_reconstruction/access.py \
  code/token_reconstruction/io.py \
  code/token_reconstruction/public_prefix.py \
  code/token_reconstruction/trr0005_joint_decoder.py \
  code/token_reconstruction/trr0007_positionwise.py \
  code/trr0010_p09_fixed_loader.py; do
  copy_or_verify "$PUBLISHED_BUNDLE/$rel" "$PACKAGE_ROOT/$rel"
done
copy_or_verify "$PRODUCER" "$PACKAGE_ROOT/code/trr0012_package.py"
copy_or_verify "$ORIGINAL_PRODUCER" "$PACKAGE_ROOT/reference_code/trr0012_package_original_908392.py"
copy_or_verify "$SMOKE_SOURCE" "$PACKAGE_ROOT/smoke/public_base_first2.safetensors"
copy_or_verify "$B0_STATE" "$PACKAGE_ROOT/states/current_fixed.safetensors"
copy_or_verify "$B1_STATE" "$PACKAGE_ROOT/states/expanded_fixed.safetensors"
copy_or_verify "$E_SOURCE" "$PACKAGE_ROOT/readout/public_normalized_embeddings.safetensors"
copy_or_verify "$PUBLISHED_BUNDLE/config/runtime.json" "$PACKAGE_ROOT/config/runtime.json"
copy_or_verify "$PUBLISHED_BUNDLE/config/decoder.json" "$PACKAGE_ROOT/config/decoder.json"

run_cli "$PRODUCER" identity \
  --input "$PACKAGE_ROOT/states/current_fixed.safetensors" \
  --output "$PACKAGE_ROOT/identity/current_fixed.safetensors.json"
run_cli "$PRODUCER" identity \
  --input "$PACKAGE_ROOT/states/expanded_fixed.safetensors" \
  --output "$PACKAGE_ROOT/identity/expanded_fixed.safetensors.json"
run_cli "$PRODUCER" identity \
  --input "$PACKAGE_ROOT/readout/public_normalized_embeddings.safetensors" \
  --output "$PACKAGE_ROOT/identity/public_normalized_embeddings.safetensors.json"
run_cli "$PRODUCER" selection-receipt \
  --b0-state "$PACKAGE_ROOT/states/current_fixed.safetensors" \
  --b1-state "$PACKAGE_ROOT/states/expanded_fixed.safetensors" \
  --b0-binding "$B0_BINDING" \
  --b1-binding "$B1_BINDING" \
  --output "$PACKAGE_ROOT/receipts/selection_complete_before_smoke.json" \
  --development-labels-used true

python3 - "$PACKAGE_ROOT" "$B0_BINDING" "$B1_BINDING" "$TEMPLATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
b0 = json.load(open(sys.argv[2], encoding="utf-8"))
b1 = json.load(open(sys.argv[3], encoding="utf-8"))
template = json.load(open(sys.argv[4], encoding="utf-8"))

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def rec(relative: str) -> dict:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}

def write_create_only(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != data:
            raise SystemExit(f"create-only descriptor differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)

descriptor = template
selection = rec("receipts/selection_complete_before_smoke.json")
readout = rec("readout/public_normalized_embeddings.safetensors")
for method, binding, state_relative in (
    ("current_fixed", b0, "states/current_fixed.safetensors"),
    ("expanded_fixed", b1, "states/expanded_fixed.safetensors"),
):
    method_payload = descriptor["methods"][method]
    state = rec(state_relative)
    method_payload.update(
        {
            "bank": binding["bank"],
            "model_id": binding["model_id"],
            "selected_step": binding["selected_step"],
            "state_path": state_relative,
            "state": state,
            "state_sha256": state["sha256"],
            "loader_kwargs": dict(binding["loader_kwargs"]),
            "readout_path": "readout/public_normalized_embeddings.safetensors",
            "public_readout": {
                "path": "readout/public_normalized_embeddings.safetensors",
                "bytes": readout["bytes"],
                "sha256": readout["sha256"],
            },
        }
    )
    method_payload["loader"]["kwargs"] = dict(binding["loader_kwargs"])
    method_payload["loader"]["path"] = "code/trr0010_p09_fixed_loader.py"

descriptor.update(
    {
        "schema": "token-reconstruction.trr0012-consumer-manifest.v1",
        "task_id": "TRR-0012",
        "status": "FROZEN_SELECTED_PACKAGE",
        "immutable_after_selection": True,
        "consumer_paths_package_relative": True,
        "package_id": "trr0012-fixed-pair-" + selection["sha256"][:16],
        "selection_receipt": {"path": "receipts/selection_complete_before_smoke.json", **selection},
        "readout": {"path": "readout/public_normalized_embeddings.safetensors", **readout, "key": "embeddings"},
        "provenance": {
            "instrumented_producer_sha256": sha(root / "code/trr0012_package.py"),
            "original_producer_sha256": "9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109",
        },
        "truth_boundary": {
            "independent_evaluation_truth_opened": False,
            "smoke_used_for_selection": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
        },
    }
)
descriptor["decoder"]["seed"] = 4010
descriptor["decoder"]["package_cli_sha256"] = sha(root / "code/trr0012_package.py")
descriptor["smoke"].update(
    {
        "input_path": "smoke/public_base_first2.safetensors",
        "expected_path": "smoke/expected_predictions.safetensors",
        "receipt_path": "smoke/expected_predictions.receipt.json",
    }
)
write_create_only(root / "package_manifest.json", descriptor)

frozen = json.loads((root / "config/decoder.json").read_text(encoding="utf-8"))
frozen.update(
    {
        "schema": "token-reconstruction.trr0012-fixed-decoder-config.v1",
        "task_id": "TRR-0012",
        "status": "FROZEN_SELECTED_PACKAGE",
        "consumer_paths_package_relative": True,
        "package_id": descriptor["package_id"],
        "selection_receipt": descriptor["selection_receipt"],
        "smoke": descriptor["smoke"],
        "truth_boundary": descriptor["truth_boundary"],
    }
)
frozen["model"]["seed"] = 4010
frozen["readout"] = descriptor["readout"]
for method, binding, state_relative in (
    ("current_fixed", b0, "states/current_fixed.safetensors"),
    ("expanded_fixed", b1, "states/expanded_fixed.safetensors"),
):
    frozen["methods"][method].update(
        {
            "bank": binding["bank"],
            "model_id": binding["model_id"],
            "selected_step": binding["selected_step"],
            "state_path": state_relative,
            "loader_kwargs": dict(binding["loader_kwargs"]),
        }
    )
write_create_only(root / "config/frozen_config.json", frozen)
print(json.dumps({
    "package_manifest_sha256": sha(root / "package_manifest.json"),
    "frozen_config_sha256": sha(root / "config/frozen_config.json"),
    "package_id": descriptor["package_id"],
}, sort_keys=True))
PY

run_cli "$PACKAGE_ROOT/code/trr0012_package.py" predict \
  --package-root "$PACKAGE_ROOT" \
  --observations "$PACKAGE_ROOT/smoke/public_base_first2.safetensors" \
  --output "$PACKAGE_ROOT/smoke/expected_predictions.safetensors" \
  --receipt "$PACKAGE_ROOT/smoke/expected_predictions.receipt.json" \
  --device "$DEVICE"
python3 - "$PACKAGE_ROOT" "$DEVICE" <<'PY'
import importlib.util
import sys
from pathlib import Path
package_root = Path(sys.argv[1]).resolve()
device = sys.argv[2]
source = package_root / "reference_code/trr0012_package_original_908392.py"
spec = importlib.util.spec_from_file_location("trr0012_original_908392", source)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load immutable original producer")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# The immutable historical source has the same pre-existing missing alias as
# the timing patch; bind the documented 127 post-BOS constant at module scope
# without modifying its 908392 source bytes.
module.SCORED_POST_BOS_TOKENS = module.STORED_SEQUENCE_TOKENS - 1
args = [
    "predict", "--package-root", str(package_root),
    "--observations", str(package_root / "smoke/public_base_first2.safetensors"),
    "--output", str(package_root / "verification/original_smoke_predictions.safetensors"),
    "--receipt", str(package_root / "verification/original_smoke_predictions.receipt.json"),
    "--device", device,
]
result = module.main(args)
raise SystemExit(result)
PY

python3 - "$PACKAGE_ROOT" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1]).resolve()
new_path = root / "smoke/expected_predictions.safetensors"
old_path = root / "verification/original_smoke_predictions.safetensors"
new_receipt_path = root / "smoke/expected_predictions.receipt.json"
old_receipt_path = root / "verification/original_smoke_predictions.receipt.json"
out = root / "verification/smoke_equivalence.json"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

new_receipt = json.loads(new_receipt_path.read_text(encoding="utf-8"))
old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
methods = ["current_fixed", "expanded_fixed"]
record_order = new_receipt["observations"]["record_order"]
if record_order != old_receipt["observations"]["record_order"]:
    raise SystemExit("record order differs")
method_results = {}
all_equal = True
with safe_open(str(new_path), framework="pt", device="cpu") as new_handle, safe_open(str(old_path), framework="pt", device="cpu") as old_handle:
    if set(new_handle.keys()) != set(methods) or set(old_handle.keys()) != set(methods):
        raise SystemExit("method keys differ")
    for method in methods:
        new_tensor = new_handle.get_tensor(method).contiguous()
        old_tensor = old_handle.get_tensor(method).contiguous()
        equal = bool((new_tensor == old_tensor).all().item())
        all_equal = all_equal and equal
        new_rows = new_receipt["methods"][method]["per_record_tensor_sha256"]
        old_rows = old_receipt["methods"][method]["per_record_tensor_sha256"]
        if len(new_rows) != len(record_order) or len(old_rows) != len(record_order):
            raise SystemExit(f"row count differs: {method}")
        rows = []
        for index, slot in enumerate(record_order):
            row_equal = bool((new_tensor[index] == old_tensor[index]).all().item())
            all_equal = all_equal and row_equal
            rows.append({
                "slot": slot,
                "new_tensor_sha256": new_rows[index],
                "original_tensor_sha256": old_rows[index],
                "equal": row_equal,
            })
        method_results[method] = {
            "shape": list(new_tensor.shape),
            "dtype": str(new_tensor.dtype),
            "equal": equal,
            "rows": rows,
        }
if not all_equal:
    raise SystemExit("smoke outputs differ")
payload = {
    "schema": "token-reconstruction.trr0012-smoke-equivalence.v1",
    "task_id": "TRR-0012",
    "status": "PASS_ORIGINAL_INSTRUMENTED_OUTPUT_EQUALITY",
    "package_root": str(root),
    "record_order": record_order,
    "methods": method_results,
    "instrumented_output": {"path": "smoke/expected_predictions.safetensors", "bytes": new_path.stat().st_size, "sha256": sha(new_path)},
    "original_output": {"path": "verification/original_smoke_predictions.safetensors", "bytes": old_path.stat().st_size, "sha256": sha(old_path)},
    "instrumented_receipt": {"path": "smoke/expected_predictions.receipt.json", "sha256": sha(new_receipt_path)},
    "original_receipt": {"path": "verification/original_smoke_predictions.receipt.json", "sha256": sha(old_receipt_path)},
    "instrumented_producer_sha256": "5023a9e7cb19611952c0ce16a440ec9fde6b6444fd74aaf74a7efbb5c671c13b",
    "original_producer_sha256": "9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109",
    "compatibility_alias": "SCORED_POST_BOS_TOKENS = STORED_SEQUENCE_TOKENS - 1 bound at runtime for immutable original source",
    "truth_opened": False,
    "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if out.exists() or out.is_symlink():
    if out.is_symlink() or out.read_bytes() != data:
        raise SystemExit(f"create-only equivalence differs: {out}")
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(data)
print(json.dumps({"path": str(out), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": payload["status"]}, sort_keys=True))
PY

python3 - "$CONTRACT_TEMPLATE" "$PACKAGE_ROOT" "$ROOT/experiments/TRR-0012/package/contract_actual_5023a9e7_r2.json" "$B0_BINDING" "$B1_BINDING" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from safetensors import safe_open

source, root, output, b0_path, b1_path = map(Path, sys.argv[1:])
contract = json.loads(source.read_text(encoding="utf-8"))
b0 = json.loads(b0_path.read_text(encoding="utf-8"))
b1 = json.loads(b1_path.read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def record(relative: str) -> dict:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}

for name, binding in contract["required_components"].items():
    path = root / binding["path"]
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing component: {name}")
    binding["bytes"] = path.stat().st_size
    binding["sha256"] = sha(path)

contract["status"] = "FROZEN_SELECTED_PACKAGE"
contract["package_id"] = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))["package_id"]
for bank, binding_path in (("B0", b0_path), ("B1", b1_path)):
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    contract["banks"][bank]["manifest_sha256"] = binding["bank_manifest_sha256"]
    contract["training"]["fit_manifest_sha256"][bank] = binding["fit_manifest_sha256"]
    contract["training"]["runner_state_sha256"][bank] = binding["runner_state_sha256"]
    contract["training"]["schedule_semantic_sha256"][bank] = binding["schedule_semantic_sha256"]
    index = 0 if bank == "B0" else 1
    contract["loader_compatibility"]["package_model_ids"][index] = binding["model_id"]

receipt = json.loads((root / "smoke/expected_predictions.receipt.json").read_text(encoding="utf-8"))
with safe_open(str(root / "smoke/expected_predictions.safetensors"), framework="pt", device="cpu") as handle:
    slots = contract["smoke"]["record_order"]
    expected_outputs = {}
    for row, slot in enumerate(slots):
        expected_outputs[slot] = {}
        for method in ("current_fixed", "expanded_fixed"):
            tensor = handle.get_tensor(method)[row]
            expected_outputs[slot][method] = {
                "token_ids": tensor.tolist(),
                "tensor_sha256": receipt["methods"][method]["per_record_tensor_sha256"][row],
            }
contract["smoke"]["expected_outputs"] = expected_outputs
contract["smoke"]["expected_output_file"] = record("smoke/expected_predictions.safetensors")
contract["smoke"]["expected_output_receipt"] = record("smoke/expected_predictions.receipt.json")
contract["consumer_cli"]["producer_sha256"] = sha(root / "code/trr0012_package.py")
contract["consumer_cli"]["reference_producer_sha256"] = "9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109"

data = (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()
if output.exists() or output.is_symlink():
    if output.is_symlink() or output.read_bytes() != data:
        raise SystemExit(f"create-only contract differs: {output}")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
print(json.dumps({"path": str(output), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, sort_keys=True))
PY

run_cli "$PACKAGE_ROOT/code/trr0012_package.py" validate \
  --manifest "$ROOT/experiments/TRR-0012/package/contract_actual_5023a9e7_r2.json" \
  --package-root "$PACKAGE_ROOT"

python3 - "$PACKAGE_ROOT" "$ROOT/experiments/TRR-0012/package/package_build_receipt_5023a9e7_r2.json" "$START_UTC" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

root, output, start = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and not path.is_symlink():
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
payload = {
    "schema": "token-reconstruction.trr0012-package-build-receipt.v1",
    "status": "PASS_ACTUAL_PACKAGE_BUILT",
    "package_root": str(root),
    "package_bytes": sum(item["bytes"] for item in files),
    "file_count": len(files),
    "files": files,
    "instrumented_producer_sha256": sha(root / "code/trr0012_package.py"),
    "original_producer_sha256": sha(root / "reference_code/trr0012_package_original_908392.py"),
    "smoke_equivalence": "verification/smoke_equivalence.json",
    "external_contract": "contract_actual_5023a9e7_r2.json",
    "start_utc": start,
    "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "truth_opened": False,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if output.exists() or output.is_symlink():
    if output.is_symlink() or output.read_bytes() != data:
        raise SystemExit(f"create-only receipt differs: {output}")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
print(json.dumps({"status": payload["status"], "package_bytes": payload["package_bytes"], "file_count": payload["file_count"], "sha256": sha(output)}, sort_keys=True))
PY

SIZE=$(du -sb "$PACKAGE_ROOT" | awk '{print $1}')
test "$SIZE" -le 2147483648 || { echo "package cap exceeded: $SIZE" >&2; exit 1; }
if test -e "$SECONDARY_ROOT" || test -L "$SECONDARY_ROOT"; then
  test ! -L "$SECONDARY_ROOT"
else
  mkdir -p "$SECONDARY_ROOT"
fi
FREE=$(df -PB1 -k "$SECONDARY_ROOT" | awk 'NR==2 {print $4*1024}')
test "$FREE" -ge 10737418240 || { echo "secondary free space below 10 GiB: $FREE" >&2; exit 1; }
while IFS= read -r -d '' source_path; do
  relative=${source_path#"$PACKAGE_ROOT"/}
  copy_or_verify "$source_path" "$SECONDARY_ROOT/$relative"
done < <(find "$PACKAGE_ROOT" -type f -print0 | sort -z)

python3 - "$PACKAGE_ROOT" "$SECONDARY_ROOT" "$ROOT/experiments/TRR-0012/package/secondary_copy_receipt_5023a9e7_r2.json" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

primary, secondary, output = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(primary.rglob("*")):
    if path.is_file() and not path.is_symlink():
        counterpart = secondary / path.relative_to(primary)
        if not counterpart.is_file() or counterpart.is_symlink():
            raise SystemExit(f"secondary file missing: {counterpart}")
        source_sha = sha(path)
        target_sha = sha(counterpart)
        if path.stat().st_size != counterpart.stat().st_size or source_sha != target_sha:
            raise SystemExit(f"secondary mismatch: {path}")
        files.append({"path": path.relative_to(primary).as_posix(), "bytes": path.stat().st_size, "sha256": source_sha})
payload = {
    "schema": "token-reconstruction.trr0012-secondary-copy-receipt.v1",
    "status": "PASS_SECONDARY_FILE_HASH_EQUALITY",
    "primary_root": str(primary),
    "secondary_root": str(secondary),
    "file_count": len(files),
    "total_bytes": sum(item["bytes"] for item in files),
    "files": files,
    "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "distinct_persistent_boundary": True,
    "truth_opened": False,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if output.exists() or output.is_symlink():
    if output.is_symlink() or output.read_bytes() != data:
        raise SystemExit(f"create-only receipt differs: {output}")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
print(json.dumps({"status": payload["status"], "total_bytes": payload["total_bytes"], "file_count": payload["file_count"], "sha256": sha(output)}, sort_keys=True))
PY

printf '%s\n' "package_root=$PACKAGE_ROOT" "package_bytes=$SIZE" "secondary_free_before_copy=$FREE" "status=PASS_ACTUAL_PACKAGE_AND_SECONDARY_COPY"
