"""Truth-blind TRR-0010 public prediction runner.

This adapter reuses the TRR-0009 observation-row loop and the retained
TRR-0005/TRR-0004 A1+A2 implementation, while emitting the TRR-0010 six-method
and four-cell artifact contract.  It validates frozen registration/assets
before loading a model, writes create-only predictions/timings, and invokes the
TRR-0010 public gate before returning.  It has no tokenizer, source reader,
truth loader, or scorer.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts import trr0010_eval_gate as gate


class RunnerError(RuntimeError):
    """Raised when a frozen public prediction run fails closed."""


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    activations: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class LoadedMethod:
    method_id: str
    adapter: Any
    evidence: dict[str, Any]


METHOD_FACTORY = Callable[[str, Mapping[str, Any], torch.Tensor, torch.device], Any]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RunnerError(f"repository root unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("cannot resolve runner code commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RunnerError("runner code commit is not a full hash")
    return value


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform != "darwin" else value
    except (AttributeError, OSError, ValueError):
        return None


def _host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    _synchronize(device)
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _write_json_create(path: Path, payload: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise RunnerError(f"refusing to overwrite create-only JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise RunnerError(f"could not write JSON artifact: {path}") from exc
    return gate.file_record(path, root=root)


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return gate._record(  # noqa: SLF001 - task gate is the binding authority
            {"path": str(Path(path).expanduser().resolve()), "bytes": Path(path).stat().st_size, "sha256": gate.sha256_file(Path(path))},
            root=root,
            description=description,
        )
    except (OSError, gate.GateError) as exc:
        raise RunnerError(str(exc)) from exc


def _load_registration(path: Path, *, root: Path, require_current_head: bool, require_runtime_sources: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    try:
        registration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunnerError("registration is invalid JSON") from exc
    if not isinstance(registration, Mapping):
        raise RunnerError("registration must be a JSON object")
    try:
        checked = gate._validate_registration(registration, root=root)  # noqa: SLF001
    except gate.GateError as exc:
        raise RunnerError(f"registration failed TRR-0010 public gate: {exc}") from exc
    if require_current_head and _git_head(root) != str(registration["code_commit"]):
        raise RunnerError("registration code commit is not current")
    if require_runtime_sources:
        _verify_imported_module_source(
            gate,
            code_bindings=checked["code_bindings"],
            root=root,
            description="TRR-0010 gate module",
        )
        _verify_source_path_bound(
            Path(__file__).resolve(),
            code_bindings=checked["code_bindings"],
            root=root,
            description="TRR-0010 prediction runner",
        )
    registration_record = _record(path, root=root, description="registration")
    return dict(registration), checked, registration_record


def _runtime_embedding_binding(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = registration.get("runtime_embedding")
    inputs = registration.get("input_bindings")
    nested = inputs.get("runtime_embedding") if isinstance(inputs, Mapping) else None
    public = inputs.get("public_embedding") if isinstance(inputs, Mapping) else None
    candidates = [value for value in (direct, nested, public) if isinstance(value, Mapping)]
    if len(candidates) != 1:
        raise RunnerError("registration must bind exactly one runtime_embedding/public_embedding asset")
    return candidates[0]


def _load_bound_readout(
    binding: Mapping[str, Any],
    *,
    root: Path,
    device: torch.device,
    description: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    checked = gate._record(binding, root=root, description=description)  # noqa: SLF001
    started = time.perf_counter()
    try:
        with safe_open(checked["path"], framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if len(keys) != 1:
                raise RunnerError(f"{description} must contain exactly one tensor")
            table_cpu = handle.get_tensor(keys[0]).detach().contiguous()
        if (
            table_cpu.ndim != 2
            or int(table_cpu.shape[0]) != gate.VOCABULARY_SIZE
            or not table_cpu.dtype.is_floating_point
            or not torch.isfinite(table_cpu.float()).all().item()
        ):
            raise RunnerError(f"{description} vocabulary geometry or dtype changed")
        table = table_cpu.to(device=device).contiguous()
        _synchronize(device)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"{description} could not be loaded") from exc
    return table, {
        "binding": checked,
        "shape": [int(value) for value in table.shape],
        "dtype": str(table.dtype),
        "load_seconds": float(time.perf_counter() - started),
    }


def _load_embedding(
    registration: Mapping[str, Any],
    *,
    root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    return _load_bound_readout(
        _runtime_embedding_binding(registration),
        root=root,
        device=device,
        description="runtime embedding",
    )


def _iter_observation_chunks(cell: Mapping[str, Any], *, records: int, hidden_size: int) -> Iterator[Chunk]:
    observation = cell.get("observation", cell)
    path = Path(str(observation.get("path", ""))).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"observation unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != gate.OBSERVATION_KEYS:
                raise RunnerError("observation tensor keys changed")
            activation_slice = handle.get_slice("activations")
            mask_slice = handle.get_slice("attention_mask")
            position_slice = handle.get_slice("position_ids")
            expected_shape = (records, gate.STORED_SEQUENCE_TOKENS, hidden_size)
            if tuple(activation_slice.get_shape()) != expected_shape:
                raise RunnerError("observation activation geometry changed")
            expected_sidecar = (records, gate.STORED_SEQUENCE_TOKENS)
            if tuple(mask_slice.get_shape()) != expected_sidecar or tuple(position_slice.get_shape()) != expected_sidecar:
                raise RunnerError("observation sidecar geometry changed")
            if records <= 0 or records % 8:
                raise RunnerError("record count is not divisible by capture chunk size")
            expected_positions = torch.arange(gate.STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0)
            for start in range(0, records, 8):
                stop = start + 8
                activations = activation_slice[start:stop].contiguous()
                mask_raw = mask_slice[start:stop]
                positions = position_slice[start:stop]
                if activations.dtype != torch.bfloat16 or mask_raw.dtype not in (torch.bool, torch.uint8, torch.int64):
                    raise RunnerError("observation dtype changed")
                if not torch.isfinite(activations.float()).all().item():
                    raise RunnerError("observation contains non-finite values")
                mask = mask_raw.to(torch.bool).contiguous()
                if not mask.all().item() or not torch.equal(positions.to(torch.long), expected_positions.expand_as(positions)):
                    raise RunnerError("observation mask or positions changed")
                yield Chunk(start, stop, activations, mask, positions.to(torch.long).contiguous())
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"observation read failed: {path}") from exc


def _iter_rows(cell: Mapping[str, Any], *, records: int, hidden_size: int) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    seen = 0
    for chunk in _iter_observation_chunks(cell, records=records, hidden_size=hidden_size):
        for offset in range(chunk.stop - chunk.start):
            row = chunk.start + offset
            yield row, chunk.activations[offset], chunk.mask[offset], chunk.positions[offset]
            seen += 1
    if seen != records:
        raise RunnerError(f"observation rows cover {seen}, expected {records}")


def _normalize_prediction(raw: Any, mask: torch.Tensor, *, method_id: str) -> torch.Tensor:
    try:
        values = torch.as_tensor(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RunnerError(f"{method_id} emitted a non-tensor prediction") from exc
    if values.ndim == 2:
        if int(values.shape[-1]) != gate.VOCABULARY_SIZE or not values.dtype.is_floating_point:
            raise RunnerError(f"{method_id} logits geometry or dtype changed")
        values = values.argmax(dim=-1)
    elif values.ndim == 1:
        if values.dtype.is_floating_point or values.dtype.is_complex or values.dtype == torch.bool:
            raise RunnerError(f"{method_id} token IDs must be signed integers")
    else:
        raise RunnerError(f"{method_id} prediction geometry changed")
    if values.ndim != 1 or int(values.shape[0]) != gate.STORED_SEQUENCE_TOKENS:
        raise RunnerError(f"{method_id} prediction geometry changed")
    values = values.to(dtype=torch.long, device="cpu").contiguous()
    valid = mask.to(dtype=torch.bool, device="cpu")
    if not bool(valid[0].item()):
        raise RunnerError(f"{method_id} observation has no BOS")
    output = torch.full((gate.STORED_SEQUENCE_TOKENS,), gate.INVALID_TOKEN_ID, dtype=torch.long)
    output[valid] = values[valid]
    output[0] = gate.BOS_TOKEN_ID
    active = output[valid]
    if active.lt(0).any().item() or active.ge(gate.VOCABULARY_SIZE).any().item():
        raise RunnerError(f"{method_id} emitted an invalid vocabulary ID")
    # The TRR-0010 final panel uses full 128-position clips.  Right-padding is
    # retained for interface compatibility but the gate will reject -1 outputs.
    return output


class _MergedCurrentHAdapter:
    """Inference-only current-H adapter with one fixed effective readout."""

    def __init__(self, model: torch.nn.Module, embedding: torch.Tensor, *, device: torch.device, method_id: str, allow_materialization: bool = False) -> None:
        self.method_id = method_id
        self.device = device
        model.requires_grad_(False)
        model = model.to(device=device).eval()
        self.materialized_once = False
        if allow_materialization and hasattr(model, "materialize_effective_embedding"):
            try:
                effective = model.materialize_effective_embedding(embedding).detach().contiguous()
                base = getattr(model, "base", None)
                if not isinstance(base, torch.nn.Module):
                    raise RunnerError(f"{method_id} directional model has no base decoder")
                base.requires_grad_(False)
                self.model = base.eval()
                self.embedding = effective
                self.materialized_once = True
                del model
            except RunnerError:
                raise
            except Exception as exc:
                raise RunnerError(f"{method_id} effective readout materialization failed") from exc
        else:
            self.model = model
            self.embedding = embedding
        self._cell_calls = 0
        self.calls = 0

    def begin_cell(self) -> None:
        self._cell_calls = self.calls

    @torch.inference_mode()
    def __call__(self, activation: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        self.calls += 1
        staged = activation.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        staged_mask = mask.to(device=self.device, dtype=torch.bool).unsqueeze(0)
        try:
            projected = self.model.projected_hidden(staged, staged_mask)
            positions = torch.arange(1, gate.STORED_SEQUENCE_TOKENS, device=self.device, dtype=torch.long)
            rows = torch.zeros_like(positions)
            logits = self.model.logits_from_rows(projected, rows, positions, self.embedding)
        except AttributeError:
            logits_full = self.model(staged, staged_mask, self.embedding)
            if logits_full.ndim != 3 or tuple(logits_full.shape[:2]) != (1, gate.STORED_SEQUENCE_TOKENS):
                raise RunnerError(f"{self.method_id} decoder returned unexpected logits geometry")
            logits = logits_full[0, 1:]
        if tuple(logits.shape) != (gate.SCORED_POST_BOS_TOKENS, gate.VOCABULARY_SIZE) or not torch.isfinite(logits).all().item():
            raise RunnerError(f"{self.method_id} logits geometry or finiteness changed")
        output = torch.full((gate.STORED_SEQUENCE_TOKENS,), gate.INVALID_TOKEN_ID, dtype=torch.long)
        output[0] = gate.BOS_TOKEN_ID
        output[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
        return output

    def evidence(self) -> dict[str, Any]:
        return {
            "calls": self.calls - self._cell_calls,
            "inference_path": "base decoder plus one fixed effective readout W",
            "effective_readout_materialized_once_at_load": self.materialized_once,
            "effective_readout_binding_mode": (
                "synthetic_materialized_once" if self.materialized_once else "registered_fixed_tensor"
            ),
            "trainable_delta_retained": False,
            "rematerialized_per_record": False,
        }


def _load_tensor_binding(binding: Mapping[str, Any], *, root: Path, description: str) -> torch.Tensor:
    checked = gate._record(binding, root=root, description=description)  # noqa: SLF001
    path = Path(checked["path"])
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if len(keys) != 1:
                raise RunnerError(f"{description} must contain one tensor")
            return handle.get_tensor(keys[0]).detach().contiguous()
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"{description} tensor could not be loaded") from exc


def _loader_path(binding: Any, *, root: Path, description: str) -> Path:
    if not isinstance(binding, Mapping):
        raise RunnerError(f"{description} binding is malformed")
    return Path(gate._record(binding, root=root, description=description)["path"])  # noqa: SLF001


def _verify_source_path_bound(
    source_path: Path,
    *,
    code_bindings: Mapping[str, Any],
    root: Path,
    description: str,
) -> dict[str, Any]:
    source_path = Path(source_path).expanduser().resolve()
    for name, binding in code_bindings.items():
        checked = gate._record(binding, root=root, description=f"code {name}")  # noqa: SLF001
        if Path(checked["path"]).resolve() == source_path:
            return {
                "module": description,
                "path": str(source_path),
                "binding": checked,
            }
    raise RunnerError(f"{description} source is not one of the frozen code bindings")


def _verify_imported_module_source(
    module: Any,
    *,
    code_bindings: Mapping[str, Any],
    root: Path,
    description: str,
) -> dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RunnerError(f"{description} has no source file")
    module_path = Path(raw_path).expanduser().resolve()
    if module_path.suffix == ".pyc":
        source_path = module_path.with_suffix(".py")
        if source_path.is_file():
            module_path = source_path
    return _verify_source_path_bound(
        module_path,
        code_bindings=code_bindings,
        root=root,
        description=description,
    )


def _dynamic_loader(row: Mapping[str, Any], *, method_id: str, root: Path, device: torch.device, embedding: torch.Tensor, code_bindings: Mapping[str, Any], allow_materialization: bool = False) -> LoadedMethod:
    loader = row.get("loader")
    if not isinstance(loader, Mapping) or not isinstance(loader.get("module"), str) or not isinstance(loader.get("function"), str):
        raise RunnerError(f"{method_id} loader module/function is absent")
    state_record = gate._record(row["state"], root=root, description=f"state {method_id}")  # noqa: SLF001
    kwargs = dict(loader.get("kwargs", {})) if isinstance(loader.get("kwargs", {}), Mapping) else None
    if kwargs is None:
        raise RunnerError(f"{method_id} loader kwargs are malformed")
    for name, binding in dict(loader.get("path_args", {})).items():
        kwargs[str(name)] = _loader_path(binding, root=root, description=f"{method_id} path argument {name}")
    for name, binding in dict(loader.get("tensor_args", {})).items():
        kwargs[str(name)] = _load_tensor_binding(binding, root=root, description=f"{method_id} tensor argument {name}")
    started = time.perf_counter()
    try:
        module = importlib.import_module(str(loader["module"]))
        module_evidence = _verify_imported_module_source(
            module,
            code_bindings=code_bindings,
            root=root,
            description=f"{method_id} loader module {loader['module']}",
        )
        function = getattr(module, str(loader["function"]))
        model = function(Path(state_record["path"]), **kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("registered loader did not return torch.nn.Module")
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"{method_id} registered loader failed") from exc
    if not allow_materialization and hasattr(model, "materialize_effective_embedding"):
        raise RunnerError(
            f"{method_id} loader returned a trainable directional module; "
            "production registration requires a base-only decoder plus bound effective W"
        )
    adapter = _MergedCurrentHAdapter(model, embedding, device=device, method_id=method_id, allow_materialization=allow_materialization)
    return LoadedMethod(
        method_id,
        adapter,
        {
            "loader": f"{loader['module']}.{loader['function']}",
            "loader_module": module_evidence,
            "state": state_record,
            "model_preparation_seconds": time.perf_counter() - started,
            **adapter.evidence(),
        },
    )


def _verify_a1_nested_assets(
    p0_record: Mapping[str, Any],
    *,
    path_args: Mapping[str, Any],
    root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    descriptor_path = Path(str(p0_record["path"]))
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunnerError("A1+A2 public P0 descriptor is not valid JSON") from exc
    if not isinstance(descriptor, Mapping):
        raise RunnerError("A1+A2 public P0 descriptor must be an object")
    model_snapshot = descriptor.get("model_snapshot")
    if not isinstance(model_snapshot, Mapping):
        model_snapshot = {}
    files = model_snapshot.get("files") or descriptor.get("snapshot_files")
    if not isinstance(files, Mapping) or not files:
        raise RunnerError("A1+A2 P0 descriptor lacks nested snapshot file bindings")
    snapshot_arg = path_args.get("snapshot") or path_args.get("model_snapshot")
    if isinstance(snapshot_arg, Mapping):
        snapshot_arg = snapshot_arg.get("path")
    snapshot_arg = snapshot_arg or model_snapshot.get("path") or descriptor.get("snapshot")
    if not isinstance(snapshot_arg, str) or not snapshot_arg:
        raise RunnerError("A1+A2 snapshot directory binding is absent")
    snapshot_path = Path(snapshot_arg).expanduser().resolve()
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise RunnerError("A1+A2 snapshot directory is unavailable")
    checked_files: dict[str, Any] = {}
    for name, binding in files.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise RunnerError("A1+A2 nested snapshot binding is malformed")
        row = dict(binding)
        actual_snapshot_path = row.get("snapshot_path") or str(snapshot_path / name)
        row["path"] = actual_snapshot_path
        checked_files[name] = gate._record(
            row,
            root=root,
            description=f"A1+A2 snapshot file {name}",
        )  # noqa: SLF001
    reference_binding = (
        path_args.get("reference_path")
        if isinstance(path_args.get("reference_path"), Mapping)
        else path_args.get("reference")
        if isinstance(path_args.get("reference"), Mapping)
        else descriptor.get("reference_binding")
        if isinstance(descriptor.get("reference_binding"), Mapping)
        else descriptor.get("reference_source")
        if isinstance(descriptor.get("reference_source"), Mapping)
        else descriptor.get("reference_path")
        if isinstance(descriptor.get("reference_path"), Mapping)
        else None
    )
    if not isinstance(reference_binding, Mapping):
        raise RunnerError("A1+A2 reference source binding is absent")
    reference_record = gate._record(
        reference_binding,
        root=root,
        description="A1+A2 reference source",
    )  # noqa: SLF001
    return snapshot_path, Path(reference_record["path"]), {
        "descriptor": gate._record(
            p0_record,
            root=root,
            description="A1+A2 public P0 descriptor",
        ),  # noqa: SLF001
        "snapshot_files": checked_files,
        "reference_source": reference_record,
    }

def _load_native_a1_a2(row: Mapping[str, Any], *, root: Path, device: torch.device, embedding: torch.Tensor | None = None, code_bindings: Mapping[str, Any] | None = None) -> LoadedMethod:
    """Load the retained A1+A2 comparator through the native fixed-K256 path.

    The final registration producer must provide loader path arguments
    ``snapshot``/``reference_path``.  A public-P0 descriptor JSON may provide
    those two paths when the resources themselves remain hash-only bindings.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RunnerError("native A1+A2 K256 requires CUDA")
    resources = row.get("resources")
    loader = row.get("loader")
    if not isinstance(resources, Mapping) or not isinstance(loader, Mapping):
        raise RunnerError("A1+A2 resources/loader are absent")
    e_record = gate._record(resources["public_embedding_table"], root=root, description="A1+A2 public embedding")  # noqa: SLF001
    lens_record = gate._record(resources["retained_a1_lens"], root=root, description="A1+A2 retained lens")  # noqa: SLF001
    p0_record = gate._record(resources["public_p0_prefix"], root=root, description="A1+A2 public P0 prefix")  # noqa: SLF001
    e_table = _load_tensor_binding(e_record, root=root, description="A1+A2 public embedding")
    if embedding is not None and (
        tuple(e_table.shape) != tuple(embedding.detach().cpu().shape)
        or not torch.equal(e_table, embedding.detach().cpu())
    ):
        raise RunnerError("A1+A2 public embedding differs from injected runtime embedding")
    path_args = loader.get("path_args", {})
    if not isinstance(path_args, Mapping):
        raise RunnerError("A1+A2 loader path_args are malformed")
    snapshot_path, reference_path, nested_assets = _verify_a1_nested_assets(
        p0_record,
        path_args=path_args,
        root=root,
    )
    started = time.perf_counter()
    try:
        if code_bindings is None:
            raise RunnerError("A1+A2 loader code bindings are absent")
        legacy = importlib.import_module("scripts.trr0004_predict_confirmation")
        legacy_evidence = _verify_imported_module_source(
            legacy,
            code_bindings=code_bindings,
            root=root,
            description="A1+A2 loader module",
        )
        precut, lens, embeddings, public_evidence = legacy._load_public_prefix(  # noqa: SLF001
            snapshot=snapshot_path,
            reference_path=reference_path,
            lens_path=Path(lens_record["path"]),
            embedding_path=Path(e_record["path"]),
            device=device,
        )
        footing = importlib.import_module("scripts.trr0003_footing_compare")
        footing_evidence = _verify_imported_module_source(
            footing,
            code_bindings=code_bindings,
            root=root,
            description="A1+A2 footing module",
        )
        adapter = legacy._A2Adapter(  # noqa: SLF001
            precut=precut,
            lens=lens,
            embeddings=embeddings,
            device=device,
            policy=footing._fixed_k256_policy(),  # noqa: SLF001
        )
        adapter.method_id = gate.A1_A2_METHOD_ID
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("native A1+A2 K256 loader failed") from exc
    return LoadedMethod(gate.A1_A2_METHOD_ID, adapter, {
        "loader": "scripts.trr0004_predict_confirmation._A2Adapter",
        "resources": {"public_embedding_table": e_record, "retained_a1_lens": lens_record, "public_p0_prefix": p0_record},
        "loader_modules": {"a1_a2": legacy_evidence, "footing": footing_evidence},
        "nested_assets": nested_assets,
        "public_prefix": public_evidence,
        "fixed_policy": (
            "propose_public_a1(max_k=512, chunk=256) -> "
            "decode_policy(candidates[:,:,:256], fixed K=256, commit_last_winner)"
        ),
        "candidate_proposal_budget": int(getattr(legacy, "DEFAULT_A2_PROPOSAL_K", 512)),
        "candidate_proposal_chunk": int(getattr(legacy, "DEFAULT_A1_CHUNK", 256)),
        "candidate_budget": int(getattr(legacy, "DEFAULT_A2_K", 256)),
        "candidate_arrays_retained_in_memory": True,
        "candidate_arrays_persisted_to_disk": False,
        "model_preparation_seconds": time.perf_counter() - started,
    })


def _load_method(method_id: str, row: Mapping[str, Any], *, root: Path, device: torch.device, embedding: torch.Tensor, method_factory: METHOD_FACTORY | None, code_bindings: Mapping[str, Any] | None = None, allow_materialization: bool = False) -> LoadedMethod:
    if method_factory is not None:
        started = time.perf_counter()
        adapter_or_model = method_factory(method_id, row, embedding, device)
        if isinstance(adapter_or_model, torch.nn.Module):
            adapter: Any = _MergedCurrentHAdapter(adapter_or_model, embedding, device=device, method_id=method_id, allow_materialization=allow_materialization)
            evidence = {"loader": "synthetic_or_injected_module", **adapter.evidence()}
        elif callable(adapter_or_model):
            adapter = adapter_or_model
            evidence = {"loader": "synthetic_or_injected_adapter", "trainable_delta_retained": False, "rematerialized_per_record": False}
        else:
            raise RunnerError(f"method factory returned a non-callable adapter: {method_id}")
        evidence.setdefault("model_preparation_seconds", time.perf_counter() - started)
        return LoadedMethod(method_id, adapter, evidence)
    if method_id == gate.A1_A2_METHOD_ID:
        return _load_native_a1_a2(row, root=root, device=device, embedding=embedding, code_bindings=code_bindings)
    if code_bindings is None:
        raise RunnerError(f"{method_id} loader code bindings are absent")
    return _dynamic_loader(
        row,
        method_id=method_id,
        root=root,
        device=device,
        embedding=embedding,
        code_bindings=code_bindings,
        allow_materialization=allow_materialization,
    )


def _revalidate_runtime_bindings(
    registration: Mapping[str, Any],
    *,
    checked: Mapping[str, Any],
    root: Path,
    method_factory: METHOD_FACTORY | None,
) -> dict[str, Any]:
    if method_factory is not None:
        return {"mode": "synthetic_injection", "production_loader_recheck": False}
    code_bindings = checked.get("code_bindings")
    if not isinstance(code_bindings, Mapping):
        raise RunnerError("runtime code bindings are absent")
    for name, binding in code_bindings.items():
        gate._record(binding, root=root, description=f"runtime code {name}")  # noqa: SLF001
    rows = {str(row["id"]): row for row in registration["methods"]}
    checked_runtime: dict[str, Any] = {"mode": "production", "methods": {}}
    for method_id in gate.METHOD_ORDER:
        row = rows[method_id]
        state = gate._record(row["state"], root=root, description=f"runtime state {method_id}")  # noqa: SLF001
        resources = row.get("resources")
        if not isinstance(resources, Mapping):
            raise RunnerError(f"runtime resources are absent: {method_id}")
        resource_records = {
            str(name): gate._record(binding, root=root, description=f"runtime {method_id} resource {name}")  # noqa: SLF001
            for name, binding in resources.items()
        }
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise RunnerError(f"runtime loader is absent: {method_id}")
        module_records: dict[str, Any] = {}
        if method_id == gate.A1_A2_METHOD_ID:
            p0_record = resource_records["public_p0_prefix"]
            path_args = loader.get("path_args", {})
            if not isinstance(path_args, Mapping):
                raise RunnerError(f"runtime A1+A2 path args are malformed: {method_id}")
            _, _, nested = _verify_a1_nested_assets(p0_record, path_args=path_args, root=root)
            module_records["nested_assets"] = nested
            legacy = importlib.import_module("scripts.trr0004_predict_confirmation")
            footing = importlib.import_module("scripts.trr0003_footing_compare")
            module_records["loader"] = _verify_imported_module_source(
                legacy,
                code_bindings=code_bindings,
                root=root,
                description="runtime A1+A2 loader module",
            )
            module_records["footing"] = _verify_imported_module_source(
                footing,
                code_bindings=code_bindings,
                root=root,
                description="runtime A1+A2 footing module",
            )
        else:
            module_name = loader.get("module")
            if not isinstance(module_name, str):
                raise RunnerError(f"runtime loader module is absent: {method_id}")
            module = importlib.import_module(module_name)
            module_records["loader"] = _verify_imported_module_source(
                module,
                code_bindings=code_bindings,
                root=root,
                description=f"runtime {method_id} loader module",
            )
        checked_runtime["methods"][method_id] = {
            "state": state,
            "resources": resource_records,
            "modules": module_records,
        }
    return checked_runtime

def _prediction_artifact(
    path: Path,
    *,
    values: torch.Tensor,
    registration_sha: str,
    method_id: str,
    cell_id: str,
    root: Path,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RunnerError(f"prediction artifact is not create-only: {path}")
    checked = values.to(dtype=torch.long, device="cpu").contiguous()
    if tuple(checked.shape) != (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS):
        raise RunnerError("prediction artifact geometry changed")
    metadata = {
        "schema": gate.PREDICTION_SCHEMA,
        "task_id": gate.TASK_ID,
        "registration_sha256": registration_sha,
        "method_id": method_id,
        "cell_id": cell_id,
        "records": str(gate.RECORDS_PER_CELL),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
        "geometry_json": json.dumps({"records": gate.RECORDS_PER_CELL, **gate.STATIC_GEOMETRY}, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": checked}, str(path), metadata=metadata)
    record = gate.file_record(path, root=root)
    record["prediction_sha256"] = gate.tensor_digest(checked)
    return record


def _run_cell(*, adapter: Any, cell: Mapping[str, Any], records: int, hidden_size: int, device: torch.device, method_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    begin_cell = getattr(adapter, "begin_cell", None)
    if callable(begin_cell):
        begin_cell()
    values = torch.empty((records, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long)
    warm_sum = 0.0
    measured_sum = 0.0
    per_record: list[float] = []
    for index, activation, mask, positions in _iter_rows(cell, records=records, hidden_size=hidden_size):
        _synchronize(device)
        started = time.perf_counter()
        warm = _normalize_prediction(adapter(activation, mask, positions), mask, method_id=method_id)
        _synchronize(device)
        warm_elapsed = time.perf_counter() - started
        warm_sum += warm_elapsed
        started = time.perf_counter()
        measured = _normalize_prediction(adapter(activation, mask, positions), mask, method_id=method_id)
        _synchronize(device)
        measured_elapsed = time.perf_counter() - started
        measured_sum += measured_elapsed
        per_record.append(float(measured_elapsed))
        if not torch.equal(warm, measured):
            raise RunnerError(f"warmup/measured IDs differ: {method_id}/{index}")
        values[index] = measured
    evidence_fn = getattr(adapter, "evidence", None)
    adapter_evidence: dict[str, Any] = {}
    if callable(evidence_fn):
        evidence = evidence_fn()
        if not isinstance(evidence, Mapping):
            raise RunnerError(f"{method_id} adapter evidence is malformed")
        adapter_evidence = dict(evidence)
    return values, {
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_seconds_sum": float(warm_sum),
        "measured_seconds_sum": float(measured_sum),
        "per_record_measured_seconds": per_record,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
        "adapter_evidence": adapter_evidence,
    }


def execute(*, registration_path: Path, repository_root: Path, device_name: str = "cuda", require_current_head: bool = True, method_factory: METHOD_FACTORY | None = None) -> dict[str, Any]:
    """Run the complete truth-free TRR-0010 matrix.

    ``method_factory`` is a synthetic-test dependency injection point only; the
    production CLI never supplies it and therefore always loads the hashed
    registration resources.  It makes the registration/gate and artifact
    integration testable without public assets or GPU execution.
    """

    root = _root(repository_root)
    started_utc = _utc_now()
    started = time.perf_counter()
    registration_verification_started = time.perf_counter()
    registration, checked, registration_record = _load_registration(
        Path(registration_path),
        root=root,
        require_current_head=require_current_head,
        require_runtime_sources=method_factory is None,
    )
    registration_hash_verification_seconds = float(time.perf_counter() - registration_verification_started)
    registration_path = Path(registration_path).expanduser().resolve()
    output_root = Path(checked["output_root"]).resolve()
    run_path = output_root / "run_manifest.json"
    if run_path.exists() or run_path.is_symlink():
        raise RunnerError(f"run manifest is not create-only: {run_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA is unavailable")
    try:
        synthetic_embedding: torch.Tensor | None = None
        embedding_evidence: dict[str, Any] = {
            "mode": "per_method_registered_readout",
            "global_runtime_embedding_loaded": False,
        }
        if method_factory is not None:
            synthetic_embedding, runtime_evidence = _load_embedding(registration, root=root, device=device)
            embedding_evidence = {
                "mode": "synthetic_injected_runtime_embedding",
                **runtime_evidence,
            }
        common_preparation_seconds = float(time.perf_counter() - started)
        rows_by_id = {str(row["id"]): row for row in registration["methods"]}
        predictions: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        model_startup: dict[str, Any] = {}
        for method_id in gate.METHOD_ORDER:
            row = rows_by_id[method_id]
            _reset_cuda_peak(device)
            preparation_started = time.perf_counter()
            method_embedding: torch.Tensor | None = synthetic_embedding
            readout_evidence: dict[str, Any] = {}
            if method_factory is None and method_id == gate.A1_A2_METHOD_ID:
                loaded = _load_method(
                    method_id,
                    row,
                    root=root,
                    device=device,
                    embedding=None,
                    method_factory=None,
                    code_bindings=checked["code_bindings"],
                    allow_materialization=False,
                )
            else:
                if method_factory is None:
                    resource_name = (
                        "effective_readout_w"
                        if method_id in gate.DIRECTIONAL_METHOD_IDS
                        else "public_embedding_table"
                    )
                    resource_binding = row.get("resources", {}).get(resource_name)
                    if not isinstance(resource_binding, Mapping):
                        raise RunnerError(f"{method_id} registered {resource_name} binding is absent")
                    method_embedding, readout_evidence = _load_bound_readout(
                        resource_binding,
                        root=root,
                        device=device,
                        description=f"{method_id} deployed readout {resource_name}",
                    )
                if method_embedding is None:
                    raise RunnerError(f"{method_id} has no deployed readout tensor")
                loaded = _load_method(
                    method_id,
                    row,
                    root=root,
                    device=device,
                    embedding=method_embedding,
                    method_factory=method_factory,
                    code_bindings=checked["code_bindings"],
                    allow_materialization=method_factory is not None,
                )
            _synchronize(device)
            cold_peak = _cuda_peak(device)
            prep_seconds = float(time.perf_counter() - preparation_started)
            preparation_accounting = {
                "seconds": prep_seconds,
                "method_loads": 1,
                "cells_covered": len(gate.CELL_ORDER),
                "shared_across_cells": True,
                "charge_count": 1,
                "scope": "one readout/decoder preparation per method, shared across all four cells; not four loads",
            }
            loaded.evidence["model_preparation_seconds"] = prep_seconds
            loaded.evidence["model_preparation_accounting"] = preparation_accounting
            loaded.evidence["cuda_cold_peak"] = dict(cold_peak)
            loaded.evidence["cuda_cold_peak_scope"] = (
                "CUDA peak counters reset immediately before this method load; "
                "the cold peak includes the resident method after preparation"
            )
            if readout_evidence:
                loaded.evidence["deployed_readout"] = readout_evidence
            model_startup[method_id] = dict(loaded.evidence)
            hidden_size = gate.OBSERVATION_HIDDEN_SIZE
            if method_embedding is not None:
                hidden_size = int(method_embedding.shape[1])
            for cell_id in gate.CELL_ORDER:
                cell = checked["observation_bindings"][cell_id]
                _reset_cuda_peak(device)
                values, timing = _run_cell(adapter=loaded.adapter, cell=cell, records=gate.RECORDS_PER_CELL, hidden_size=hidden_size, device=device, method_id=method_id)
                steady_peak = _cuda_peak(device)
                style, condition = cell_id.split("__", 1)
                prediction_path = output_root / "predictions" / style / condition / f"{method_id}.safetensors"
                prediction_record = _prediction_artifact(
                    prediction_path,
                    values=values,
                    registration_sha=registration_record["sha256"],
                    method_id=method_id,
                    cell_id=cell_id,
                    root=root,
                )
                timing_payload = {
                    "schema": gate.TIMING_SCHEMA,
                    "task_id": gate.TASK_ID,
                    "method_id": method_id,
                    "cell_id": cell_id,
                    "records": gate.RECORDS_PER_CELL,
                    "registration_sha256": registration_record["sha256"],
                    **timing,
                    "timed_interval": "synchronized row staging -> decoder/A1+A2 adapter -> full-vocabulary argmax -> CPU IDs",
                    "model_preparation_seconds": prep_seconds,
                    "model_preparation_accounting": dict(preparation_accounting),
                    "peak_memory_scope": (
                        "host process_max_rss_bytes is ru_maxrss high-water for this runner invocation; "
                        "it accumulates across methods and cells and is not an isolated per-method deployment measurement"
                    ),
                    "cuda_peak_memory_scope": (
                        "cuda_peak_* fields are reset immediately before this cell; "
                        "the steady peak covers this cell while the current method remains resident"
                    ),
                    "peak_memory": {
                        "process_max_rss_bytes": _rss_bytes(),
                        "host_available_bytes": _host_available_bytes(),
                        **steady_peak,
                        "cuda_cold_peak_allocated_bytes": cold_peak["cuda_peak_allocated_bytes"],
                        "cuda_cold_peak_reserved_bytes": cold_peak["cuda_peak_reserved_bytes"],
                    },
                    "prediction_artifact": prediction_record,
                    "truth_opened": False,
                    "source_text_written": False,
                    "source_text_loaded": False,
                    "token_ids_written": False,
                    "target_labels_loaded": False,
                    "candidate_arrays_persisted": False,
                }
                timing_path = output_root / "timings" / style / condition / f"{method_id}.run.json"
                _write_json_create(timing_path, timing_payload, root=root)
                timing_record = gate.file_record(timing_path, root=root)
                key = f"{method_id}::{cell_id}"
                predictions[key] = prediction_record
                timings[key] = timing_record
            del loaded
            if method_embedding is not synthetic_embedding:
                del method_embedding
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        run_payload = {
            "schema": gate.RUN_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": gate.RUN_STATUS,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "common_preparation": {
                "seconds": common_preparation_seconds,
                "registration_hash_verification_seconds": registration_hash_verification_seconds,
                "included_in_elapsed_seconds": True,
                "scope": (
                    "once per run before the method loop; includes registration parsing, "
                    "code/input/observation hash verification, path setup, and shared synthetic setup"
                ),
            },
            "code_commit": registration["code_commit"],
            "registration": registration_record,
            "input_bindings": checked["input_bindings"],
            "observation_bindings": checked["observation_bindings"],
            "code_bindings": checked["code_bindings"],
            "timing_plan": checked["timing_plan"],
            "runtime_embedding": embedding_evidence,
            "model_startup": model_startup,
            "predictions": predictions,
            "timings": timings,
            "truth_opened": False,
            "source_text_written": False,
            "source_text_loaded": False,
            "token_ids_written": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        }
        run_payload["runtime_recheck"] = _revalidate_runtime_bindings(
            registration,
            checked=checked,
            root=root,
            method_factory=method_factory,
        )
        _write_json_create(run_path, run_payload, root=root)
        try:
            freeze = gate.validate_public_outputs(registration_path=registration_path, run_manifest_path=run_path, repository_root=root, require_current_head=False)
        except gate.GateError as exc:
            raise RunnerError(f"TRR-0010 public output gate failed: {exc}") from exc
        return {"status": run_payload["status"], "run_manifest": gate.file_record(run_path, root=root), "freeze": freeze, "elapsed_seconds": run_payload["elapsed_seconds"]}
    except RunnerError:
        raise
    except Exception as exc:
        failure = output_root / "run_manifest.failure.json"
        if not failure.exists() and not failure.is_symlink():
            try:
                _write_json_create(failure, {"schema": "token-reconstruction.trr0010-run-failure.v1", "task_id": gate.TASK_ID, "status": "PUBLIC_EVALUATION_FAILED_CLOSED", "truth_opened": False, "source_text_written": False, "source_text_loaded": False, "token_ids_written": False, "target_labels_loaded": False, "candidate_arrays_persisted": False, "started_utc": started_utc, "ended_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)}, root=root)
            except Exception:
                pass
        raise RunnerError("TRR-0010 public evaluation failed closed") from exc
    finally:
        gc.collect()


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-stale-head", action="store_true", help="synthetic/replay use only; do not require registration.code_commit == HEAD")
    args = parser.parse_args()
    result = execute(registration_path=args.registration, repository_root=args.repository_root, device_name=args.device, require_current_head=not args.allow_stale_head)
    print(json.dumps({"status": result["status"], "run_manifest": result["run_manifest"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
