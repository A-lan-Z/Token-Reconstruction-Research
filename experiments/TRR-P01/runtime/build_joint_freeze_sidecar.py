#!/usr/bin/env python3
"""Build and validate the TRR-P01 joint pre-truth freeze sidecar."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ARMS = ("arm-000", "arm-001")
METHODS = (
    "boundary.cosine",
    "boundary.l2",
    "raw_embedding.cosine",
    "raw_embedding.l2",
    "reference_corrected.cosine",
    "reference_corrected.l2",
    "historical_a1.cosine",
    "historical_a1_a2_port.cosine",
)
COMMIT = "e43a595d0f4300d5db8f93c86881b455dfa30ea4"
JOINT_SCHEMA = "token-reconstruction.trr-p01-joint-freeze.v1"
VALIDATION_SCHEMA = "token-reconstruction.trr-p01-joint-freeze-validation.v1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def offset_records(diagnostics_path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with safe_open(diagnostics_path, framework="pt", device="cpu") as handle:
        keys = sorted(key for key in handle.keys() if key.endswith(".offsets"))
        expected = ["reference_corrected.cosine.offsets", "reference_corrected.l2.offsets"]
        if keys != expected:
            raise RuntimeError(f"raw correction offset keys changed: {keys}")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tuple(tensor.shape) != (16, 40, 2048) or tensor.dtype != torch.float32:
                raise RuntimeError(f"raw correction offset geometry changed: {key}")
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(f"raw correction offsets are non-finite: {key}")
            output.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "digest": tensor_digest(tensor),
                }
            )
    return output


def build(sidecar_path: Path, validation_path: Path, runtime_root: Path) -> None:
    if sidecar_path.exists() or sidecar_path.is_symlink():
        raise RuntimeError(f"sidecar already exists: {sidecar_path}")
    if validation_path.exists() or validation_path.is_symlink():
        raise RuntimeError(f"validation already exists: {validation_path}")
    arms: list[dict[str, Any]] = []
    prototype = runtime_root / "cpu-table-20260905" / "boundary_prototypes.safetensors"
    if not prototype.is_file():
        raise RuntimeError("prototype artifact is missing")
    for arm in ARMS:
        run_root = runtime_root / f"reconstruct-final-r2-{arm}"
        public_root = runtime_root / "evaluator-final-20260905" / "public" / arm
        evidence = load_json(run_root / "reconstructor_evidence.json")
        finish = load_json(run_root / "finish_receipt.json")
        freeze = load_json(run_root / "freeze_receipt.json")
        route = load_json(run_root / "route.json")
        if evidence.get("truth_opened") is not False or evidence.get("implementation_commit") != COMMIT:
            raise RuntimeError(f"evidence identity/truth state changed: {arm}")
        if evidence.get("methods") != list(METHODS):
            raise RuntimeError(f"declared methods changed: {arm}")
        if finish.get("truth_opened") is not False or finish.get("implementation_commit") != COMMIT:
            raise RuntimeError(f"finish identity/truth state changed: {arm}")
        if freeze.get("truth_opened") is not False or freeze.get("status") != "FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN":
            raise RuntimeError(f"freeze state changed: {arm}")
        if route.get("truth_opened") is not False or route.get("methods") != list(METHODS):
            raise RuntimeError(f"route state changed: {arm}")
        diagnostics = run_root / "lookup_diagnostics.safetensors"
        artifacts = {
            "public_config": record(public_root / "sanitized_config.json"),
            "public_index": record(public_root / "observation_index.json"),
            "public_observations": record(public_root / "observations.safetensors"),
            "predictions": record(run_root / "predictions.safetensors"),
            "prediction_rows": record(run_root / "predictions.jsonl"),
            "lookup_diagnostics": record(diagnostics),
            "reconstructor_evidence": record(run_root / "reconstructor_evidence.json"),
            "finish_receipt": record(run_root / "finish_receipt.json"),
            "phase_progress": record(run_root / "phase_progress.jsonl"),
            "freeze_receipt": record(run_root / "freeze_receipt.json"),
            "route": record(run_root / "route.json"),
        }
        if finish.get("prediction_sha256") != artifacts["predictions"]["sha256"]:
            raise RuntimeError(f"finish prediction hash mismatch: {arm}")
        if finish.get("prediction_rows_sha256") != artifacts["prediction_rows"]["sha256"]:
            raise RuntimeError(f"finish JSONL hash mismatch: {arm}")
        if finish.get("lookup_diagnostics_sha256") != artifacts["lookup_diagnostics"]["sha256"]:
            raise RuntimeError(f"finish diagnostics hash mismatch: {arm}")
        if finish.get("phase_progress_sha256") != artifacts["phase_progress"]["sha256"]:
            raise RuntimeError(f"finish progress hash mismatch: {arm}")
        if finish.get("evidence_sha256") != artifacts["reconstructor_evidence"]["sha256"]:
            raise RuntimeError(f"finish evidence hash mismatch: {arm}")
        if freeze.get("evidence", {}).get("sha256") != artifacts["reconstructor_evidence"]["sha256"]:
            raise RuntimeError(f"freeze evidence hash mismatch: {arm}")
        if freeze.get("prediction", {}).get("tensor", {}).get("sha256") != artifacts["predictions"]["sha256"]:
            raise RuntimeError(f"freeze prediction hash mismatch: {arm}")
        offsets = offset_records(diagnostics)
        arms.append(
            {
                "arm": arm,
                "truth_opened": False,
                "implementation_commit": COMMIT,
                "methods": list(METHODS),
                "artifacts": artifacts,
                "raw_correction_offsets": {
                    "source_artifact": artifacts["lookup_diagnostics"],
                    "tensors": offsets,
                },
                "finish_status": finish.get("status"),
                "freeze_status": freeze.get("status"),
            }
        )
    payload = {
        "schema": JOINT_SCHEMA,
        "task_id": "TRR-P01",
        "created_utc": utc_now(),
        "status": "JOINT_MATRIX_FROZEN_BEFORE_TRUTH_OPEN",
        "truth_opened": False,
        "truth_opened_before_joint_freeze": False,
        "implementation_commit": COMMIT,
        "execution_order": list(ARMS),
        "methods": list(METHODS),
        "prototype": record(prototype),
        "public_model_provenance": record(runtime_root / "public-model-provenance-20260905.json"),
        "arms": arms,
        "scoring_gate": "both arms regular freeze validation and this sidecar hash validation must pass before private truth is opened",
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    validate(sidecar_path, validation_path)


def validate(sidecar_path: Path, validation_path: Path) -> None:
    if validation_path.exists() or validation_path.is_symlink():
        raise RuntimeError(f"validation already exists: {validation_path}")
    sidecar = load_json(sidecar_path)
    if sidecar.get("schema") != JOINT_SCHEMA or sidecar.get("status") != "JOINT_MATRIX_FROZEN_BEFORE_TRUTH_OPEN":
        raise RuntimeError("joint sidecar status/schema changed")
    if sidecar.get("truth_opened") is not False or sidecar.get("truth_opened_before_joint_freeze") is not False:
        raise RuntimeError("joint sidecar truth state changed")
    if sidecar.get("implementation_commit") != COMMIT or sidecar.get("execution_order") != list(ARMS):
        raise RuntimeError("joint sidecar identity/order changed")
    checks = 0
    for arm in sidecar.get("arms", []):
        for name, item in arm.get("artifacts", {}).items():
            path = Path(item["path"])
            if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"joint hash mismatch: {arm.get('arm')}:{name}")
            checks += 1
        diag = Path(arm["raw_correction_offsets"]["source_artifact"]["path"])
        actual = {item["key"]: item for item in offset_records(diag)}
        expected = {item["key"]: item for item in arm["raw_correction_offsets"]["tensors"]}
        if actual != expected:
            raise RuntimeError(f"raw offset digest mismatch: {arm.get('arm')}")
        checks += len(actual)
    for item in (sidecar.get("prototype"), sidecar.get("public_model_provenance")):
        path = Path(item["path"])
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"joint global hash mismatch: {path}")
        checks += 1
    result = {
        "schema": VALIDATION_SCHEMA,
        "task_id": "TRR-P01",
        "validated_utc": utc_now(),
        "status": "JOINT_HASH_VALIDATION_PASS_BEFORE_TRUTH_OPEN",
        "truth_opened": False,
        "truth_opened_before_validation": False,
        "sidecar": record(sidecar_path),
        "validated_file_records": checks,
        "validated_arms": list(ARMS),
        "validated_methods": list(METHODS),
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.sidecar.resolve(), args.validation.resolve())
    else:
        build(args.sidecar.resolve(), args.validation.resolve(), args.runtime_root.resolve())
    print({"status": "JOINT_HASH_VALIDATION_PASS_BEFORE_TRUTH_OPEN", "truth_opened": False, "sidecar": str(args.sidecar), "validation": str(args.validation)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
