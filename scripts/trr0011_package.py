"""TRR-0011 immutable frozen-inference package and replay checker.

This package loads only the selected TRR-0010 fixed-readout decoder states and
public readout table, consumes already-opened public H observations, and
compares token IDs with the retained pre-truth prediction artifacts.  It does
not import a tokenizer, source text, truth labels, scorer, trainer, or panel
selector.  External resources are hash-bound and read-only; replay outputs are
create-only receipts.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from safetensors import safe_open

# Running by file path puts ``scripts/`` rather than the repository root at
# sys.path[0].  Pin both package roots explicitly before importing the frozen
# loader below.
_PACKAGE_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_PACKAGE_REPO_ROOT, _PACKAGE_REPO_ROOT / "src"):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


PACKAGE_SCHEMA = "token-reconstruction.trr0011-inference-package.v1"
PACKAGE_TASK = "TRR-0011"
OBSERVATION_KEYS = {"activations", "attention_mask", "position_ids"}
PREDICTION_KEY = "predictions"
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000


class PackageError(RuntimeError):
    """Raised when a package binding or replay check fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"JSON object required: {path}")
    return value


def _resolve_binding(binding: Mapping[str, Any], *, root: Path, description: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise PackageError(f"{description} binding is malformed")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PackageError(f"{description} path is absent")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink():
        raise PackageError(f"{description} must not resolve through a symlink")
    if not path.is_file():
        raise PackageError(f"{description} is unavailable: {path}")
    try:
        expected_bytes = int(binding["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageError(f"{description} byte binding is malformed") from exc
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise PackageError(f"{description} SHA-256 binding is malformed")
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise PackageError(
            f"{description} changed: bytes {actual_bytes}/{expected_bytes}, "
            f"sha {actual_sha}/{expected_sha}"
        )
    return path, {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}


def _require_readonly(binding: Mapping[str, Any], *, description: str) -> None:
    if binding.get("readonly") is not True:
        raise PackageError(f"{description} must be explicitly read-only")


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError("cannot resolve package repository HEAD") from exc


def _check_source_binding(binding: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    _path, record = _resolve_binding(binding, root=root, description=description)
    return record


def validate_package(manifest_path: Path, *, root: Path) -> dict[str, Any]:
    """Validate every compact package binding without loading model tensors."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("task_id") != PACKAGE_TASK:
        raise PackageError("package schema or task identity changed")
    frozen = manifest.get("frozen_source")
    if not isinstance(frozen, Mapping):
        raise PackageError("frozen source bindings are absent")
    source_commit = frozen.get("code_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise PackageError("frozen source commit is malformed")
    # The package is a successor wrapper, so the current package HEAD need not
    # equal the historical TRR-0010 commit.  Every executed TRR-0010 source
    # file is still checked by bytes and hash below.
    source_files = frozen.get("code_bindings")
    if not isinstance(source_files, Mapping) or not source_files:
        raise PackageError("frozen source code bindings are absent")
    checked_sources = {
        str(name): _check_source_binding(binding, root=root, description=f"source {name}")
        for name, binding in source_files.items()
    }
    for name, binding in source_files.items():
        if binding.get("frozen_code_binding") is not True:
            raise PackageError(f"source binding is not marked frozen: {name}")
    package_source = manifest.get("package_source_binding")
    package_test = manifest.get("package_test_binding")
    if not isinstance(package_source, Mapping) or not isinstance(package_test, Mapping):
        raise PackageError("package source/test bindings are absent")
    _check_source_binding(package_source, root=root, description="TRR-0011 package wrapper")
    _check_source_binding(package_test, root=root, description="TRR-0011 package tests")
    registration_path, registration_record = _resolve_binding(
        frozen.get("registration"), root=root, description="TRR-0010 registration"
    )
    freeze_path, freeze_record = _resolve_binding(
        frozen.get("public_freeze"), root=root, description="TRR-0010 public freeze"
    )
    registration = _load_json(registration_path)
    freeze = _load_json(freeze_path)
    if registration.get("code_commit") != source_commit or freeze.get("code_commit") != source_commit:
        raise PackageError("registration/freeze code commit differs from package binding")
    if registration.get("truth_opened") is not False or freeze.get("truth_opened") is not False:
        raise PackageError("replay package cannot bind post-truth registration/freeze")
    if registration.get("source_text_loaded") is not False or freeze.get("source_text_loaded") is not False:
        raise PackageError("replay package source-text boundary is not closed")
    geometry = manifest.get("capture_geometry")
    if not isinstance(geometry, Mapping):
        raise PackageError("capture geometry is absent")
    expected_shape = [128, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]
    if geometry.get("activation_shape_per_cell") != expected_shape:
        raise PackageError("decoder input geometry is not [128,128,2048]")
    if geometry.get("sequence_tokens_captured") != 192 or geometry.get("retained_first_tokens") != 128:
        raise PackageError("capture/decode sequence geometry is not B8x192 -> first128")
    if geometry.get("activation_dtype") != "torch.bfloat16":
        raise PackageError("activation dtype binding changed")
    if geometry.get("truth_free") is not True:
        raise PackageError("package is not truth-free")

    cells = manifest.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != {
        "pile__public_base", "pile__public_lora_2601", "finance__public_base", "finance__public_lora_2601"
    }:
        raise PackageError("package cell set changed")
    checked_cells: dict[str, Any] = {}
    for cell_id, cell in cells.items():
        if not isinstance(cell, Mapping) or int(cell.get("records", -1)) != 128:
            raise PackageError(f"cell record count changed: {cell_id}")
        if cell.get("capture_cell_geometry") != expected_shape:
            raise PackageError(f"cell geometry changed: {cell_id}")
        obs_binding = cell.get("observation")
        _require_readonly(obs_binding, description=f"observation {cell_id}")
        obs_path, obs_record = _resolve_binding(obs_binding, root=root, description=f"observation {cell_id}")
        checked_cells[cell_id] = {"observation": obs_record, "path": str(obs_path), "records": 128}

    methods = manifest.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {"current_fixed", "expanded_fixed"}:
        raise PackageError("selected package methods changed")
    checked_methods: dict[str, Any] = {}
    registration_rows = {str(row.get("id")): row for row in registration.get("methods", []) if isinstance(row, Mapping)}
    for method_id, method in methods.items():
        if not isinstance(method, Mapping) or method_id not in registration_rows:
            raise PackageError(f"method binding is absent: {method_id}")
        row = registration_rows[method_id]
        if row.get("role") != method.get("role"):
            raise PackageError(f"method role changed: {method_id}")
        state_binding = method.get("state")
        readout_binding = method.get("public_embedding_table")
        _require_readonly(state_binding, description=f"state {method_id}")
        _require_readonly(readout_binding, description=f"readout {method_id}")
        state_path, state_record = _resolve_binding(state_binding, root=root, description=f"state {method_id}")
        readout_path, readout_record = _resolve_binding(readout_binding, root=root, description=f"readout {method_id}")
        if row.get("state", {}).get("sha256") != state_record["sha256"]:
            raise PackageError(f"method state differs from registration: {method_id}")
        expected_outputs = method.get("outputs")
        if not isinstance(expected_outputs, Mapping) or set(expected_outputs) != set(cells):
            raise PackageError(f"expected output cells changed: {method_id}")
        checked_outputs = {}
        for cell_id, out_binding in expected_outputs.items():
            _require_readonly(out_binding, description=f"output {method_id}/{cell_id}")
            out_path, out_record = _resolve_binding(out_binding, root=root, description=f"output {method_id}/{cell_id}")
            checked_outputs[cell_id] = {"path": str(out_path), **out_record, "prediction_sha256": out_binding.get("prediction_sha256")}
        checked_methods[method_id] = {
            "state": {"path": str(state_path), **state_record},
            "readout": {"path": str(readout_path), **readout_record},
            "outputs": checked_outputs,
        }
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_source": {"code_commit": source_commit, "registration": registration_record, "public_freeze": freeze_record, "sources": checked_sources},
        "cells": checked_cells,
        "methods": checked_methods,
        "current_package_head": _git_head(root),
    }


def _read_single_tensor(path: Path, *, expected_key: str, description: str) -> torch.Tensor:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if list(handle.keys()) != [expected_key]:
                raise PackageError(f"{description} tensor key changed")
            return handle.get_tensor(expected_key).detach().contiguous()
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"cannot load {description}") from exc


def _load_observation_row(cell: Mapping[str, Any], row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(str(cell["path"]))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != OBSERVATION_KEYS:
            raise PackageError("observation tensor keys changed")
        activations = handle.get_slice("activations")
        masks = handle.get_slice("attention_mask")
        positions = handle.get_slice("position_ids")
        if tuple(activations.get_shape()) != (128, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
            raise PackageError("observation activation geometry changed")
        if tuple(masks.get_shape()) != (128, STORED_SEQUENCE_TOKENS) or tuple(positions.get_shape()) != (128, STORED_SEQUENCE_TOKENS):
            raise PackageError("observation sidecar geometry changed")
        activation = activations[row : row + 1].contiguous()[0]
        mask_raw = masks[row : row + 1].contiguous()[0]
        position = positions[row : row + 1].contiguous()[0].to(torch.long)
    if activation.dtype != torch.bfloat16 or not torch.isfinite(activation.float()).all().item():
        raise PackageError("observation activation dtype/finiteness changed")
    mask = mask_raw.to(torch.bool).contiguous()
    expected_positions = torch.arange(STORED_SEQUENCE_TOKENS, dtype=torch.long)
    if not bool(mask.all().item()) or not torch.equal(position, expected_positions):
        raise PackageError("observation mask/positions changed")
    return activation, mask, position


def _load_fixed_method(checked: Mapping[str, Any], method_id: str, *, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    method = checked["methods"][method_id]
    manifest_method = checked["manifest"]["methods"][method_id]
    loader_kwargs = dict(manifest_method["loader"]["kwargs"])
    state_path = Path(method["state"]["path"])
    readout_path = Path(method["readout"]["path"])
    try:
        from scripts.trr0010_p09_fixed_loader import load_p09_fixed_state
        model = load_p09_fixed_state(state_path, **loader_kwargs)
    except Exception as exc:
        raise PackageError(f"fixed decoder state failed strict load: {method_id}") from exc
    readout = _read_single_tensor(readout_path, expected_key="embeddings", description=f"readout {method_id}")
    if tuple(readout.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or not readout.dtype.is_floating_point:
        raise PackageError(f"readout geometry changed: {method_id}")
    model = model.to(device=device).eval()
    readout = readout.to(device=device).contiguous()
    return model, readout


def _predict_row(model: torch.nn.Module, readout: torch.Tensor, activation: torch.Tensor, mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
        staged_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
        try:
            projected = model.projected_hidden(staged, staged_mask)
            positions = torch.arange(1, STORED_SEQUENCE_TOKENS, device=device, dtype=torch.long)
            rows = torch.zeros_like(positions)
            logits = model.logits_from_rows(projected, rows, positions, readout)
        except AttributeError:
            logits_full = model(staged, staged_mask, readout)
            if tuple(logits_full.shape) != (1, STORED_SEQUENCE_TOKENS, VOCABULARY_SIZE):
                raise PackageError("decoder logits geometry changed")
            logits = logits_full[0, 1:]
        if tuple(logits.shape) != (SCORED_POST_BOS_TOKENS, VOCABULARY_SIZE) or not torch.isfinite(logits).all().item():
            raise PackageError("decoder logits shape/finiteness changed")
        output = torch.empty(STORED_SEQUENCE_TOKENS, dtype=torch.long)
        output[0] = BOS_TOKEN_ID
        output[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
        return output


def replay(
    manifest_path: Path,
    *,
    root: Path,
    method_id: str,
    cell_id: str,
    max_records: int | None,
    device_name: str,
    output_path: Path | None,
) -> dict[str, Any]:
    checked = validate_package(manifest_path, root=root)
    if method_id not in checked["methods"]:
        raise PackageError(f"replay method is not packaged: {method_id}")
    if cell_id not in checked["cells"]:
        raise PackageError(f"replay cell is not packaged: {cell_id}")
    records = checked["cells"][cell_id]["records"]
    limit = records if max_records is None else int(max_records)
    if limit <= 0 or limit > records:
        raise PackageError(f"max_records must be in [1,{records}]")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise PackageError("CUDA replay requested but CUDA is unavailable")
    device = torch.device(device_name)
    source_root = root / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    model, readout = _load_fixed_method(checked, method_id, device=device)
    expected_path = Path(checked["methods"][method_id]["outputs"][cell_id]["path"])
    expected = _read_single_tensor(expected_path, expected_key="predictions", description="expected prediction")
    if tuple(expected.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise PackageError("expected prediction geometry changed")
    row_records=[]
    for row in range(limit):
        activation, mask, _positions = _load_observation_row(checked["cells"][cell_id], row)
        predicted = _predict_row(model, readout, activation, mask, device=device)
        expected_row = expected[row].to(dtype=torch.long, device="cpu").contiguous()
        if not torch.equal(predicted, expected_row):
            mismatch = torch.nonzero(predicted != expected_row, as_tuple=False).reshape(-1).tolist()
            raise PackageError(f"replay mismatch at {method_id}/{cell_id}/record{row}: positions={mismatch[:8]}")
        row_records.append({"row":row,"prediction_sha256":tensor_digest(predicted),"exact_match":True})
    result={
      "schema":"token-reconstruction.trr0011-package-replay.v1","task_id":PACKAGE_TASK,
      "status":"PASS_TRUTH_FREE_REPLAY","manifest_sha256":checked["manifest_sha256"],
      "source_code_commit":checked["frozen_source"]["code_commit"],"package_runtime_head":checked["current_package_head"],
      "method_id":method_id,"cell_id":cell_id,"device":str(device),"records_verified":limit,"records_available":records,
      "capture_geometry":checked["manifest"]["capture_geometry"],"expected_prediction_file":checked["methods"][method_id]["outputs"][cell_id],
      "rows":row_records,"truth_opened":False,"source_text_loaded":False,"target_labels_loaded":False,
    }
    if output_path is not None:
        output_path=Path(output_path).expanduser().resolve()
        if output_path.exists() or output_path.is_symlink(): raise PackageError(f"replay receipt is create-only: {output_path}")
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        result['receipt_path']=str(output_path)
        result['receipt_bytes']=output_path.stat().st_size
        result['receipt_sha256']=sha256_file(output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--manifest',type=Path,required=True)
    common.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    validate_parser=sub.add_parser('validate',parents=[common])
    validate_parser.set_defaults(handler='validate')
    replay_parser=sub.add_parser('replay',parents=[common])
    replay_parser.add_argument('--method',choices=['current_fixed','expanded_fixed'],required=True)
    replay_parser.add_argument('--cell',choices=['pile__public_base','pile__public_lora_2601','finance__public_base','finance__public_lora_2601'],required=True)
    replay_parser.add_argument('--max-records',type=int,default=1)
    replay_parser.add_argument('--device',default='cpu')
    replay_parser.add_argument('--output',type=Path,required=True)
    replay_parser.set_defaults(handler='replay')
    return parser


def main(argv: list[str] | None = None) -> int:
    args=_parser().parse_args(argv)
    try:
        if args.handler=='validate':
            checked=validate_package(args.manifest,root=args.root.resolve())
            print(json.dumps({'status':'PASS_PACKAGE_BINDINGS','manifest_sha256':checked['manifest_sha256'],'methods':sorted(checked['methods']),'cells':sorted(checked['cells'])},sort_keys=True))
        else:
            result=replay(args.manifest,root=args.root.resolve(),method_id=args.method_id if hasattr(args,'method_id') else args.method,cell_id=args.cell,max_records=args.max_records,device_name=args.device,output_path=args.output)
            print(json.dumps(result,sort_keys=True))
    except PackageError as exc:
        print(f"TRR0011 package error: {exc}",file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
