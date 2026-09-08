"""Create-only TRR-P12 B1 prediction/projected-feature runner.

The runner consumes a sanitized activation bundle accepted by the restored
package producer.  It loads the persistent B1 binding once, then invokes the
package math one record at a time.  Outputs are predictions and projected
hidden rows; full logits are ephemeral.  No source text, truth, token IDs, or
target weights are loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

# Keep imports package-local and avoid adding the repository's model source
# tree.  The frozen package itself validates its vendored decoder namespace.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from safetensors.torch import save_file

from scripts.trr_p12.adapter import (
    AdapterError,
    B1PersistentAdapter,
    B1_STATE_SHA256,
    READOUT_SHA256,
)


TASK_ID = "TRR-P12"
SCHEMA = "token-reconstruction.trr-p12-b1-runner-receipt.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p12-b1-predictions.v1"
PROJECTED_SCHEMA = "token-reconstruction.trr-p12-b1-projected-hidden.v1"
GEOMETRY_SCHEMA = "token-reconstruction.trr-p12-b1-compact-geometry.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_binding(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"{label} must be a regular file: {path}")
    return {
        "label": label,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def create_only_path(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise AdapterError(f"{label} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def current_rss_bytes() -> int | None:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_bytes() -> int | None:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "rss_bytes": current_rss_bytes(),
        "process_peak_rss_bytes": peak_rss_bytes(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def phase_begin(device: torch.device) -> tuple[float, dict[str, int | None]]:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter(), memory_snapshot(device)


def phase_end(device: torch.device, started: float, before: dict[str, int | None]) -> dict[str, Any]:
    synchronize(device)
    after = memory_snapshot(device)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "before": before,
        "after": after,
        "cuda_peak": {
            key: after[key]
            for key in ("cuda_max_allocated_bytes", "cuda_max_reserved_bytes")
            if key in after
        },
    }


def _relative_or_external(root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "external_cli_input"


def _module_bindings(package_root: Path, package: Any) -> list[dict[str, Any]]:
    records = package._loaded_decoder_modules(package_root)
    for record in records:
        path = Path(record["loaded_path"]).resolve()
        try:
            path.relative_to((package_root / "code").resolve())
        except ValueError as exc:
            raise AdapterError("loaded decoder module escaped frozen package code root") from exc
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--equivalence-receipt", type=Path, default=None)
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "TRR-P12" / "b1_predictions.safetensors",
    )
    parser.add_argument(
        "--projected-output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "TRR-P12" / "b1_projected_hidden.safetensors",
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "TRR-P12" / "b1_compact_geometry.json",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "TRR-P12" / "b1_run.receipt.json",
    )
    parser.add_argument("--gpu-cost-per-hour", type=float, default=None)
    parser.add_argument("--check-only", action="store_true", help="validate create-only paths and exit before loading the model")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root).expanduser().resolve()
    observations_arg = Path(args.observations).expanduser()
    prediction_output = create_only_path(Path(args.prediction_output), label="prediction output")
    projected_output = create_only_path(Path(args.projected_output), label="projected output")
    geometry_output = create_only_path(Path(args.geometry_output), label="geometry output")
    receipt_output = create_only_path(Path(args.receipt_output), label="runner receipt")
    if args.gpu_cost_per_hour is not None and float(args.gpu_cost_per_hour) < 0:
        raise AdapterError("GPU cost rate must be nonnegative")
    if args.device == "cuda:0" and args.equivalence_receipt is None:
        raise AdapterError("cuda:0 requires --equivalence-receipt")
    if args.check_only:
        payload = {
            "schema": "token-reconstruction.trr-p12-b1-runner-check.v1",
            "package_root": str(package_root),
            "observations": str(observations_arg),
            "device": args.device,
            "outputs": [str(prediction_output), str(projected_output), str(geometry_output), str(receipt_output)],
            "status": "CHECK_ONLY_NO_MODEL_LOAD",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    if not package_root.is_dir() or package_root.is_symlink():
        raise AdapterError(f"package root is unavailable: {package_root}")
    device = torch.device(args.device)
    wall_started_utc = utc_now()
    wall_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    package_phase_started, package_phase_before = phase_begin(device)
    adapter = B1PersistentAdapter(
        package_root,
        device=args.device,
        equivalence_receipt=args.equivalence_receipt,
    )
    package_phase = phase_end(device, package_phase_started, package_phase_before)

    observation_phase_started, observation_phase_before = phase_begin(device)
    observation_path, batch = adapter.load_observation_batch(observations_arg)
    activations, masks, positions, slots = batch
    observation_phase = phase_end(device, observation_phase_started, observation_phase_before)

    predictions: list[torch.Tensor] = []
    projected: list[torch.Tensor] = []
    geometry_rows: list[dict[str, Any]] = []
    record_phases: list[dict[str, Any]] = []
    infer_phase_started, infer_phase_before = phase_begin(device)
    iterator = adapter.iter_loaded_observations(batch)
    for index in range(int(activations.shape[0])):
        record_started, record_before = phase_begin(device)
        result = next(iterator)
        record_phase = phase_end(device, record_started, record_before)
        record_phase["record_index"] = index
        record_phase["slot"] = result.slot
        record_phases.append(record_phase)
        predictions.append(result.prediction)
        projected.append(result.projected_hidden)
        geometry = result.geometry()
        geometry.update(
            {
                "record_index": index,
                "prediction_sha256": tensor_digest(result.prediction),
                "projected_hidden_sha256": tensor_digest(result.projected_hidden),
                "valid_token_count": int(result.valid_mask.sum().item()),
                "position_id_first_last": [int(result.position_ids[0]), int(result.position_ids[-1])],
            }
        )
        geometry_rows.append(geometry)
        del result
    infer_phase = phase_end(device, infer_phase_started, infer_phase_before)

    prediction_tensor = torch.stack(predictions, dim=0).to(dtype=torch.long, device="cpu").contiguous()
    projected_tensor = torch.stack(projected, dim=0).to(dtype=torch.float32, device="cpu").contiguous()
    if tuple(prediction_tensor.shape) != (len(slots), 128):
        raise AdapterError("prediction output geometry changed")
    if tuple(projected_tensor.shape) != (len(slots), 127, 2048):
        raise AdapterError("projected output geometry changed")

    output_phase_started, output_phase_before = phase_begin(device)
    save_file(
        {"expanded_fixed": prediction_tensor},
        str(prediction_output),
        metadata={
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "method_id": "expanded_fixed",
            "compute_dtype": "torch.float32",
            "truth_opened": "false",
            "source_text_loaded": "false",
            "token_ids_loaded": "false",
        },
    )
    save_file(
        {"projected_hidden": projected_tensor},
        str(projected_output),
        metadata={
            "schema": PROJECTED_SCHEMA,
            "task_id": TASK_ID,
            "method_id": "expanded_fixed",
            "compute_dtype": "torch.float32",
            "scored_positions": "1:128",
            "truth_opened": "false",
            "source_text_loaded": "false",
            "token_ids_loaded": "false",
        },
    )
    geometry_payload = {
        "schema": GEOMETRY_SCHEMA,
        "task_id": TASK_ID,
        "method_id": "expanded_fixed",
        "package_root": str(package_root),
        "observations": str(observation_path),
        "record_count": len(slots),
        "record_order": list(slots),
        "rows": geometry_rows,
        "prediction_output_geometry": list(prediction_tensor.shape),
        "projected_output_geometry": list(projected_tensor.shape),
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
    }
    geometry_output.write_text(json.dumps(geometry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_phase = phase_end(device, output_phase_started, output_phase_before)

    package = adapter._package
    descriptor, descriptor_path = package._load_package_descriptor(package_root)
    loaded_modules = _module_bindings(package_root, package)
    package_binding = {
        "root": str(package_root),
        "descriptor": file_binding(descriptor_path, label="package descriptor"),
        "b1_state": file_binding(adapter.state_path, label="expanded_fixed B1 state"),
        "public_readout": file_binding(adapter.readout_path, label="public normalized readout"),
        "method_id": "expanded_fixed",
        "state_sha256": B1_STATE_SHA256,
        "readout_sha256": READOUT_SHA256,
    }
    observation_binding = {
        "path": str(observation_path),
        "relative_to_package": _relative_or_external(package_root, observation_path),
        "bytes": observation_path.stat().st_size,
        "sha256": sha256_file(observation_path),
        "tensor_bindings": {
            "activations": {"shape": list(activations.shape), "dtype": str(activations.dtype), "sha256": tensor_digest(activations)},
            "attention_mask": {"shape": list(masks.shape), "dtype": str(masks.dtype), "sha256": tensor_digest(masks)},
            "position_ids": {"shape": list(positions.shape), "dtype": str(positions.dtype), "sha256": tensor_digest(positions)},
        },
        "record_order": list(slots),
    }
    wall_seconds = time.perf_counter() - wall_started
    gpu_hours = wall_seconds / 3600.0 if device.type == "cuda" else 0.0
    rate = float(args.gpu_cost_per_hour) if args.gpu_cost_per_hour is not None else None
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PREDICTIONS_AND_PROJECTED_FEATURES_GENERATED",
        "created_utc": utc_now(),
        "wall_started_utc": wall_started_utc,
        "package": package_binding,
        "observations": observation_binding,
        "code": {
            "runner": file_binding(Path(__file__), label="P12 runner code"),
            "adapter": file_binding(Path(__file__).with_name("adapter.py"), label="P12 adapter code"),
            "loaded_decoder_modules": loaded_modules,
            "import_policy": "frozen package producer and vendored package/code decoder modules only; no repository model-source prefix imports",
        },
        "outputs": {
            "predictions": file_binding(prediction_output, label="prediction output"),
            "projected_hidden": file_binding(projected_output, label="projected hidden output"),
            "compact_geometry": file_binding(geometry_output, label="compact geometry output"),
            "prediction_tensor_sha256": tensor_digest(prediction_tensor),
            "projected_hidden_tensor_sha256": tensor_digest(projected_tensor),
        },
        "geometry": {
            "record_count": len(slots),
            "prediction_shape": list(prediction_tensor.shape),
            "projected_hidden_shape": list(projected_tensor.shape),
            "vocabulary_size": 128256,
            "hidden_size": 2048,
            "stored_tokens": 128,
            "scored_positions": "1:128",
            "logits_retained": False,
        },
        "device": str(device),
        "numeric_profile": "FP32 staged activation, package readout/model path",
        "timing": {
            "clock": "time.perf_counter",
            "package_model_load": package_phase,
            "observation_load": observation_phase,
            "sequential_record_inference": infer_phase,
            "records": record_phases,
            "output_write": output_phase,
            "wall_seconds": wall_seconds,
            "memory_after_run": memory_snapshot(device),
        },
        "resource": {
            "process_rss_bytes": current_rss_bytes(),
            "process_peak_rss_bytes": peak_rss_bytes(),
            "cuda_peak_scope": "run phases with per-phase reset; null for CPU",
        },
        "cost": {
            "cpu_hours": wall_seconds / 3600.0,
            "gpu_hours": gpu_hours,
            "gpu_rate_per_hour": rate,
            "estimated_gpu_cost": None if rate is None else gpu_hours * rate,
            "currency": None,
        },
        "access_boundary": {
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
            "target_weights_loaded": False,
            "target_prefix_queried": False,
        },
        "record_batching": "one record per model projection/logit call; no numerical batching or microbatch substitution",
    }
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = sha256_file(receipt_output)
    receipt["receipt_path"] = str(receipt_output)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run(args)
    except (AdapterError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 runner error: {exc}", file=sys.stderr)
        return 2
    if not args.check_only:
        print(json.dumps({"status": payload["status"], "receipt_path": payload.get("receipt_path")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
