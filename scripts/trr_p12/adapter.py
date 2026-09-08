"""Persistent FP32 B1 inference over the restored TRR-0012 package.

The package producer and its vendored loader remain the only decoder
implementation.  ``B1PersistentAdapter`` loads the hash-bound public readout
and ``expanded_fixed`` B1 state once per process, then streams one record at a
time with the package's exact projection/logit path.  CPU is always available;
``cuda:0`` is fail-closed until a hash-bound CPU/GPU fixture-equivalence
receipt, made by :func:`compare_cpu_gpu_fixture`, is supplied.

This module writes no artifact by itself and never loads source text, truth,
or target weights.  The runner owns create-only output paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator
import sys

import torch
from safetensors import safe_open


SCHEMA = "token-reconstruction.trr-p12-b1-one-record-adapter.v2"
OBSERVATION_PATH_SCHEMA = "token-reconstruction.trr-p12-sanitized-observation-path.v1"
EQUIVALENCE_SCHEMA = "token-reconstruction.trr-p12-cpu-gpu-fixture-equivalence.v1"
PACKAGE_METHOD_ID = "expanded_fixed"
B1_STATE_SHA256 = "088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706"
READOUT_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
STORED_SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000


class AdapterError(RuntimeError):
    """Raised when the frozen package cannot be consumed fail-closed."""


@dataclass(frozen=True)
class B1OneRecordResult:
    slot: str
    activation_shape: tuple[int, ...]
    activation_dtype: str
    valid_mask: torch.Tensor
    position_ids: torch.Tensor
    projected_hidden: torch.Tensor
    logits: torch.Tensor
    prediction: torch.Tensor
    state_sha256: str
    readout_sha256: str
    compute_device: str = "cpu"

    def geometry(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "slot": self.slot,
            "activation_shape": list(self.activation_shape),
            "activation_dtype": self.activation_dtype,
            "valid_mask_shape": list(self.valid_mask.shape),
            "position_ids_shape": list(self.position_ids.shape),
            "projected_hidden_shape": list(self.projected_hidden.shape),
            "projected_hidden_dtype": str(self.projected_hidden.dtype),
            "logits_shape": list(self.logits.shape),
            "logits_dtype": str(self.logits.dtype),
            "prediction_shape": list(self.prediction.shape),
            "prediction_dtype": str(self.prediction.dtype),
            "compute_device": self.compute_device,
            "bos_token_id": BOS_TOKEN_ID,
            "scored_positions": "1:128",
            "state_sha256": self.state_sha256,
            "readout_sha256": self.readout_sha256,
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _code_digest() -> str:
    return _sha256_file(Path(__file__).resolve())


def _package_module(package_root: Path) -> Any:
    root = Path(package_root).expanduser().resolve()
    path = root / "code" / "trr0012_package.py"
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"frozen package producer is unavailable: {path}")
    module_name = f"_trr_p12_frozen_package_{abs(hash(str(root)))}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AdapterError("cannot construct frozen package import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AdapterError("cannot import frozen package producer") from exc
    return module


def _regular_json(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object")
    return value


def _validate_equivalence_receipt(package_root: Path, package: Any, receipt_path: Path) -> dict[str, Any]:
    receipt = _regular_json(receipt_path, label="CPU/GPU equivalence receipt")
    if receipt.get("schema") != EQUIVALENCE_SCHEMA or receipt.get("status") != "CPU_GPU_FIXTURE_EQUAL":
        raise AdapterError("CUDA requires a completed CPU/GPU fixture-equivalence receipt")
    if receipt.get("method_id") != PACKAGE_METHOD_ID or receipt.get("prediction_equal") is not True:
        raise AdapterError("CUDA equivalence receipt does not certify expanded_fixed exact equality")
    if receipt.get("cpu_prediction_sha256") != receipt.get("gpu_prediction_sha256"):
        raise AdapterError("CUDA equivalence receipt has different CPU/GPU prediction digests")
    if not isinstance(receipt.get("gpu_device"), str) or not receipt["gpu_device"].startswith("cuda"):
        raise AdapterError("CUDA equivalence receipt has no GPU device binding")
    descriptor, descriptor_path = package._load_package_descriptor(package_root)
    if receipt.get("package_descriptor_sha256") != _sha256_file(descriptor_path):
        raise AdapterError("CUDA equivalence receipt package descriptor changed")
    if receipt.get("b1_state_sha256") != B1_STATE_SHA256 or receipt.get("readout_sha256") != READOUT_SHA256:
        raise AdapterError("CUDA equivalence receipt frozen bindings changed")
    if receipt.get("adapter_code_sha256") != _code_digest():
        raise AdapterError("CUDA equivalence receipt adapter code changed")
    if receipt.get("truth_opened") is not False or receipt.get("target_weights_loaded") is not False:
        raise AdapterError("CUDA equivalence receipt violates the access boundary")
    return receipt


class B1PersistentAdapter:
    """Persistent package binding for record-streamed FP32 B1 inference."""

    def __init__(
        self,
        package_root: Path,
        *,
        device: str = "cpu",
        equivalence_receipt: Path | None = None,
        _allow_unqualified_cuda: bool = False,
    ) -> None:
        requested_device = torch.device(str(device))
        if requested_device.type not in {"cpu", "cuda"}:
            raise AdapterError("device must be cpu or cuda:0")
        if requested_device.type == "cuda":
            if str(requested_device) != "cuda:0":
                raise AdapterError("only cuda:0 is registered for P12")
            if not torch.cuda.is_available():
                raise AdapterError("CUDA inference requested but CUDA is unavailable")
        root = Path(package_root).expanduser().resolve()
        package = _package_module(root)
        try:
            descriptor, descriptor_path = package._load_package_descriptor(root)
            if requested_device.type == "cuda" and not _allow_unqualified_cuda:
                if equivalence_receipt is None:
                    raise AdapterError("cuda:0 requires a CPU/GPU fixture-equivalence receipt")
                _validate_equivalence_receipt(root, package, Path(equivalence_receipt))
            readout, readout_path = package._load_readout(root, descriptor)
            if package.sha256_file(readout_path) != READOUT_SHA256:
                raise AdapterError("frozen public readout SHA-256 changed")
            readout = readout.to(device=requested_device, dtype=torch.float32).contiguous()
            model, state_sha, state_path = package._load_package_method(
                root,
                descriptor,
                PACKAGE_METHOD_ID,
                device=requested_device,
                readout=readout,
            )
            if state_sha != B1_STATE_SHA256 or package.sha256_file(state_path) != B1_STATE_SHA256:
                raise AdapterError("frozen B1 state SHA-256 changed")
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("cannot load persistent frozen B1 binding") from exc
        self.package_root = root
        self._package = package
        self._descriptor = descriptor
        self._descriptor_path = descriptor_path
        self._readout = readout
        self._model = model
        self.device = requested_device
        self.state_sha256 = str(state_sha)
        self.readout_sha256 = str(package.sha256_file(readout_path))
        self.state_path = Path(state_path)
        self.readout_path = Path(readout_path)
        self.equivalence_receipt = Path(equivalence_receipt).expanduser().resolve() if equivalence_receipt else None

    def _observation_path(self, observations: Path | None) -> Path:
        """Resolve package smoke or a P12-owned sanitized bundle without copying it.

        The restored package remains read-only.  External input is accepted only
        below this worktree's ``outputs/TRR-P12`` directory, and its safetensors
        key set is checked before the unchanged package observation loader reads
        tensor values.
        """

        if observations is None:
            path = self.package_root / "smoke" / "public_base_first2.safetensors"
        else:
            path = Path(observations).expanduser()
            if not path.is_absolute():
                path = self.package_root / path
        path = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise AdapterError(f"observations must be a regular file: {path}")
        package_smoke_root = (self.package_root / "smoke").resolve()
        p12_root = (Path(__file__).resolve().parents[2] / "outputs" / "TRR-P12").resolve()
        try:
            path.relative_to(package_smoke_root)
            package_smoke = True
        except ValueError:
            package_smoke = False
        if not package_smoke:
            try:
                path.relative_to(p12_root)
            except ValueError as exc:
                raise AdapterError(
                    "external observations must be a P12-owned sanitized file under outputs/TRR-P12"
                ) from exc
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                        raise AdapterError(
                            "P12 sanitized observations must contain exactly activations, attention_mask, position_ids"
                        )
            except AdapterError:
                raise
            except Exception as exc:
                raise AdapterError("P12 sanitized observation metadata is unreadable") from exc
        return path

    def _infer_loaded_record(
        self,
        activation: torch.Tensor,
        mask: torch.Tensor,
        position: torch.Tensor,
        slot: str,
    ) -> B1OneRecordResult:
        try:
            mask = mask.to(dtype=torch.bool, device="cpu").contiguous()
            position = position.to(dtype=torch.long, device="cpu").contiguous()
            with torch.inference_mode():
                staged = activation.to(device=self.device, dtype=torch.float32).unsqueeze(0)
                staged_mask = mask.to(device=self.device).unsqueeze(0)
                staged_positions = position.to(device=self.device)
                projected_full = self._model.projected_hidden(staged, staged_mask)
                projected = projected_full[0, 1:].to(device="cpu", dtype=torch.float32).contiguous()
                rows = torch.zeros_like(staged_positions[1:])
                logits = self._model.logits_from_rows(
                    projected_full,
                    rows,
                    staged_positions[1:],
                    self._readout,
                ).to(device="cpu", dtype=torch.float32).contiguous()
            if tuple(projected.shape) != (STORED_SEQUENCE_TOKENS - 1, HIDDEN_SIZE):
                raise AdapterError("B1 projected hidden geometry changed")
            if tuple(logits.shape) != (STORED_SEQUENCE_TOKENS - 1, VOCABULARY_SIZE):
                raise AdapterError("B1 full-vocabulary logit geometry changed")
            if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(projected).all().item()):
                raise AdapterError("B1 output contains non-finite values")
            prediction = torch.empty((STORED_SEQUENCE_TOKENS,), dtype=torch.long)
            prediction[0] = BOS_TOKEN_ID
            prediction[1:] = torch.argmax(logits, dim=-1).to(dtype=torch.long)
            return B1OneRecordResult(
                slot=str(slot),
                activation_shape=tuple(int(item) for item in activation.shape),
                activation_dtype=str(activation.dtype),
                valid_mask=mask,
                position_ids=position,
                projected_hidden=projected,
                logits=logits,
                prediction=prediction,
                state_sha256=self.state_sha256,
                readout_sha256=self.readout_sha256,
                compute_device=str(self.device),
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("frozen B1 one-record inference failed") from exc

    def load_observation_batch(
        self, observations: Path | None = None
    ) -> tuple[Path, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
        """Load and validate one sanitized observation bundle exactly once."""

        try:
            observation_path = self._observation_path(observations)
            batch = self._package._observation_batch(observation_path)
            return observation_path, batch
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("frozen B1 observation batch could not be loaded") from exc

    def iter_loaded_observations(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    ) -> Iterator[B1OneRecordResult]:
        """Infer each loaded row sequentially; no numerical record batching."""

        try:
            activations, masks, positions, slots = batch
            for index in range(int(activations.shape[0])):
                yield self._infer_loaded_record(activations[index], masks[index], positions[index], slots[index])
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("frozen B1 observation-stream inference failed") from exc

    def iter_observations(self, observations: Path | None = None) -> Iterator[B1OneRecordResult]:
        """Yield one result per row after loading the observation tensor once."""

        _observation_path, batch = self.load_observation_batch(observations)
        yield from self.iter_loaded_observations(batch)

    def infer_observations(self, observations: Path | None = None) -> list[B1OneRecordResult]:
        """Materialize streamed results for small fixture qualification only."""

        return list(self.iter_observations(observations))

    def infer_one_record(self, observations: Path | None = None, *, record_index: int = 0) -> B1OneRecordResult:
        """Run one FP32 B1 record using the persistent package binding."""

        try:
            _observation_path, batch = self.load_observation_batch(observations)
            activations, masks, positions, slots = batch
            index = int(record_index)
            if index < 0 or index >= int(activations.shape[0]):
                raise AdapterError(f"record index {index} is outside observation batch")
            return self._infer_loaded_record(activations[index], masks[index], positions[index], slots[index])
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("frozen B1 one-record inference failed") from exc


class B1CPUAdapter(B1PersistentAdapter):
    """Backwards-compatible CPU-only facade used by the initial smoke test."""

    def __init__(self, package_root: Path, *, device: str = "cpu") -> None:
        if str(device) != "cpu":
            raise AdapterError("P12 CPU facade is CPU-only")
        super().__init__(package_root, device="cpu")


def infer_b1_one_record(
    package_root: Path,
    observations: Path | None = None,
    *,
    record_index: int = 0,
    device: str = "cpu",
) -> B1OneRecordResult:
    """Construct one CPU facade and run one B1 record."""

    adapter = B1CPUAdapter(package_root, device=device)
    return adapter.infer_one_record(observations, record_index=record_index)


def compare_cpu_gpu_fixture(
    package_root: Path,
    observations: Path,
    *,
    gpu_device: str = "cuda:0",
    receipt_output: Path | None = None,
) -> dict[str, Any]:
    """Qualify exact CPU/GPU predictions on one public fixture.

    The internal unqualified CUDA construction is reachable only here.  The
    returned receipt is the sole authorization accepted by a later public
    ``cuda:0`` adapter construction.
    """

    if str(gpu_device) != "cuda:0":
        raise AdapterError("fixture qualification is registered only for cuda:0")
    if not torch.cuda.is_available():
        raise AdapterError("CUDA fixture qualification requested but CUDA is unavailable")
    root = Path(package_root).expanduser().resolve()
    fixture = Path(observations).expanduser().resolve()
    package = _package_module(root)
    cpu = B1PersistentAdapter(root, device="cpu")
    observation_path, _batch = cpu.load_observation_batch(fixture)
    gpu = B1PersistentAdapter(root, device="cuda:0", _allow_unqualified_cuda=True)
    cpu_predictions: list[torch.Tensor] = []
    gpu_predictions: list[torch.Tensor] = []
    cpu_projected: list[torch.Tensor] = []
    gpu_projected: list[torch.Tensor] = []
    cpu_slots: list[str] = []
    gpu_slots: list[str] = []
    for result in cpu.iter_observations(observation_path):
        cpu_predictions.append(result.prediction)
        cpu_projected.append(result.projected_hidden)
        cpu_slots.append(result.slot)
    for result in gpu.iter_observations(observation_path):
        gpu_predictions.append(result.prediction)
        gpu_projected.append(result.projected_hidden)
        gpu_slots.append(result.slot)
    cpu_prediction = torch.stack(cpu_predictions, dim=0).contiguous()
    gpu_prediction = torch.stack(gpu_predictions, dim=0).contiguous()
    if cpu_slots != gpu_slots or tuple(cpu_prediction.shape) != tuple(gpu_prediction.shape):
        raise AdapterError("CPU/GPU fixture record order or prediction geometry differs")
    equal = bool(torch.equal(cpu_prediction, gpu_prediction))
    if not equal:
        raise AdapterError("CPU/GPU fixture predictions differ exactly")
    projected_max_abs = 0.0
    for left, right in zip(cpu_projected, gpu_projected):
        projected_max_abs = max(projected_max_abs, float(torch.max(torch.abs(left - right)).item()))
    descriptor, descriptor_path = package._load_package_descriptor(root)
    payload: dict[str, Any] = {
        "schema": EQUIVALENCE_SCHEMA,
        "task_id": "TRR-P12",
        "status": "CPU_GPU_FIXTURE_EQUAL",
        "method_id": PACKAGE_METHOD_ID,
        "package_root": str(root),
        "package_descriptor": str(descriptor_path),
        "package_descriptor_sha256": _sha256_file(descriptor_path),
        "b1_state_sha256": B1_STATE_SHA256,
        "readout_sha256": READOUT_SHA256,
        "adapter_code_sha256": _code_digest(),
        "observation_path_adapter": {
            "schema": OBSERVATION_PATH_SCHEMA,
            "external_root": str((Path(__file__).resolve().parents[2] / "outputs" / "TRR-P12").resolve()),
            "package_smoke_root": str((root / "smoke").resolve()),
            "external_required_keys": ["activations", "attention_mask", "position_ids"],
            "package_loader_unchanged": True,
        },
        "observations": {
            "path": str(observation_path),
            "sha256": _sha256_file(observation_path),
            "record_count": len(cpu_slots),
            "slots": cpu_slots,
        },
        "cpu_device": "cpu",
        "gpu_device": str(gpu.device),
        "cpu_prediction_sha256": _tensor_digest(cpu_prediction),
        "gpu_prediction_sha256": _tensor_digest(gpu_prediction),
        "prediction_shape": list(cpu_prediction.shape),
        "prediction_equal": equal,
        "projected_hidden_max_abs_difference": projected_max_abs,
        "truth_opened": False,
        "target_weights_loaded": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
    }
    if receipt_output is not None:
        output = Path(receipt_output).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise AdapterError(f"equivalence receipt is create-only: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["receipt_path"] = str(output)
        payload["receipt_sha256"] = _sha256_file(output)
    return payload


__all__ = [
    "AdapterError",
    "B1CPUAdapter",
    "B1OneRecordResult",
    "B1PersistentAdapter",
    "B1_STATE_SHA256",
    "EQUIVALENCE_SCHEMA",
    "OBSERVATION_PATH_SCHEMA",
    "READOUT_SHA256",
    "SCHEMA",
    "compare_cpu_gpu_fixture",
    "infer_b1_one_record",
]
