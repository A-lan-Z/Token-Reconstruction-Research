"""Run the frozen TRR-0007 public prediction matrix.

The runner consumes only the source-free observation manifest and registered
public/model state.  It runs every decoder method on all four paired cells and
the bounded P0 A1+A2 adapter on the first 32 public-base records per domain.
Each record receives one warmup and one measured invocation; only the measured
IDs are serialized after exact repeat comparison.  No source tokens, target
labels, truth sidecars, or candidate arrays are read or written.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts import trr0007_eval_contract as contract


class RunnerError(contract.ContractError):
    """Raised when a prediction run cannot proceed safely."""


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    activations: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root(value: Path) -> Path:
    result = Path(value).expanduser().resolve()
    if result.is_symlink() or not result.is_dir():
        raise RunnerError(f"repository root is unavailable: {result}")
    return result


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * (1024 if platform.system() != "Darwin" else 1)


def _host_available_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError as exc:
        raise RunnerError("host available-memory guard unavailable") from exc
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except ValueError:
                    break
    raise RunnerError("host available-memory guard unavailable")


def _nvidia_compute_apps() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or not fields[0]:
            continue
        rows.append({"pid": fields[0], "process_name": fields[1], "used_memory": fields[2]})
    return rows


def _guard(
    *,
    device: torch.device,
    guard: Mapping[str, Any],
    started: float,
    stage: str,
) -> dict[str, Any]:
    try:
        max_seconds = int(guard["maximum_seconds"])
        max_rss = int(guard["maximum_rss_bytes"])
        min_host = int(guard["minimum_host_available_bytes"])
        min_gpu = int(guard["minimum_free_gpu_bytes"])
        max_reserved = int(guard["maximum_reserved_gpu_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("resource guard is malformed") from exc
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise RunnerError(f"wall-time guard expired at {stage}")
    rss = _rss_bytes()
    if rss > max_rss:
        raise RunnerError(f"host RSS guard failed at {stage}: {rss} bytes")
    available = _host_available_bytes()
    if available < min_host:
        raise RunnerError(f"host available-memory guard failed at {stage}: {available} bytes")
    result: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": float(elapsed),
        "host_rss_bytes": rss,
        "host_available_bytes": available,
        "device": str(device),
        "status": "PASS",
    }
    if device.type != "cuda":
        return result
    if not torch.cuda.is_available():
        raise RunnerError("CUDA requested but unavailable")
    free, total = torch.cuda.mem_get_info(device)
    reserved = int(torch.cuda.memory_reserved(device))
    apps = [row for row in _nvidia_compute_apps() if row["pid"] != str(os.getpid())]
    if apps:
        raise RunnerError(f"GPU is not exclusive at {stage}: {apps!r}")
    if int(free) < min_gpu:
        raise RunnerError(f"GPU free-memory guard failed at {stage}: {int(free)} bytes")
    if reserved > max_reserved:
        raise RunnerError(f"GPU reservation guard failed at {stage}: {reserved} bytes")
    result.update(
        {
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_reserved_bytes": reserved,
            "compute_apps": apps,
        }
    )
    return result


def _configure_numerics(settings: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract.NUMERICAL_SETTINGS
    if dict(settings) != expected:
        raise RunnerError("numerical settings changed")
    try:
        torch.set_num_threads(int(expected["cpu_intraop_threads"]))
        torch.set_num_interop_threads(int(expected["cpu_interop_threads"]))
    except RuntimeError as exc:
        raise RunnerError("unable to set CPU numerical settings before execution") from exc
    torch.backends.cuda.matmul.allow_tf32 = bool(expected["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(expected["cuda_cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(expected["float32_matmul_precision"]))
    return dict(expected)


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("cannot resolve executable source commit") from exc
    if not contract._COMMIT.fullmatch(value):
        raise RunnerError("executable source commit is not a full hash")
    return value


def _verify_code_bindings(registration: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = registration.get("code_bindings")
    if not isinstance(rows, list) or len(rows) != len(contract.CODE_BINDING_SPECS):
        raise RunnerError("code bindings are incomplete")
    result: list[dict[str, Any]] = []
    for index, (role, relative) in enumerate(contract.CODE_BINDING_SPECS):
        row = rows[index]
        if not isinstance(row, Mapping) or row.get("role") != role or row.get("path") != relative:
            raise RunnerError(f"code binding {role} is missing or reordered")
        try:
            checked = contract.validate_file_record(
                row,
                repository_root=root,
                description=f"code binding {role}",
                verify=True,
            )
        except contract.ContractError as exc:
            raise RunnerError(str(exc)) from exc
        result.append(checked | {"role": role})
    return result


def _load_embedding(
    registration: Mapping[str, Any],
    *,
    root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    binding = registration["runtime_assets"]["normalized_public_E"]
    try:
        record = contract.validate_file_record(
            binding, repository_root=root, description="normalized public E", verify=True
        )
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc
    started = time.perf_counter()
    try:
        values = load_file(record["path"], device="cpu")
        if set(values) != {"embeddings"}:
            raise RunnerError("normalized public E must contain only embeddings")
        table_cpu = values["embeddings"].detach().contiguous()
        if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (contract.VOCAB_SIZE, contract.HIDDEN_SIZE):
            raise RunnerError("normalized public E geometry or dtype changed")
        table = table_cpu.to(device=device).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("normalized public E could not be loaded") from exc
    finally:
        if "values" in locals():
            del values
        if "table_cpu" in locals():
            del table_cpu
        gc.collect()
    return table, {
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
        "dtype": "torch.float32",
        "load_seconds": float(time.perf_counter() - started),
    }


def _iter_observation_chunks(
    cell: Mapping[str, Any],
    *,
    records: int,
    chunk_records: int,
) -> Iterator[Chunk]:
    observation = cell["observation"]
    path = Path(str(observation["path"])).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"observation is unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"activations", "attention_mask", "position_ids"}
            if keys != required:
                raise RunnerError(f"observation tensor keys changed for {cell['cell_id']}: {sorted(keys)}")
            activation_slice = handle.get_slice("activations")
            mask_slice = handle.get_slice("attention_mask")
            position_slice = handle.get_slice("position_ids")
            expected_h = (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
            expected_side = (records, contract.STORED_SEQUENCE_TOKENS)
            if tuple(activation_slice.get_shape()) != expected_h:
                raise RunnerError(f"observation activation geometry changed for {cell['cell_id']}")
            if tuple(mask_slice.get_shape()) != expected_side or tuple(position_slice.get_shape()) != expected_side:
                raise RunnerError(f"observation sidecar geometry changed for {cell['cell_id']}")
            if records % chunk_records:
                raise RunnerError("record count is not divisible by observation chunk size")
            for start in range(0, records, chunk_records):
                stop = start + chunk_records
                activations = activation_slice[start:stop]
                mask_raw = mask_slice[start:stop]
                positions = position_slice[start:stop]
                if activations.dtype != torch.bfloat16:
                    raise RunnerError(f"observation activations are not BF16: {cell['cell_id']}")
                if mask_raw.dtype not in (torch.bool, torch.uint8):
                    raise RunnerError(f"observation mask dtype changed: {cell['cell_id']}")
                if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise RunnerError(f"observation position dtype changed: {cell['cell_id']}")
                if not torch.isfinite(activations.float()).all().item():
                    raise RunnerError(f"observation contains non-finite values: {cell['cell_id']}")
                if mask_raw.dtype == torch.uint8 and ((mask_raw != 0) & (mask_raw != 1)).any().item():
                    raise RunnerError(f"observation mask is not binary: {cell['cell_id']}")
                mask = mask_raw.to(torch.bool).contiguous()
                if not mask.all().item():
                    raise RunnerError(f"observation clip is not fully valid: {cell['cell_id']}")
                expected_positions = torch.arange(
                    contract.STORED_SEQUENCE_TOKENS, dtype=torch.long
                ).unsqueeze(0).expand_as(positions)
                if not torch.equal(positions.to(torch.long), expected_positions):
                    raise RunnerError(f"observation positions are not 0..127: {cell['cell_id']}")
                yield Chunk(
                    start=start,
                    stop=stop,
                    activations=activations.contiguous(),
                    mask=mask,
                    positions=positions.to(torch.long).contiguous(),
                )
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"observation chunk read failed for {cell['cell_id']}") from exc


def _iter_rows(cell: Mapping[str, Any], *, records: int) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if records <= 0 or records > contract.RECORDS_PER_DOMAIN:
        raise RunnerError("requested row count is invalid")
    seen = 0
    for chunk in _iter_observation_chunks(
        cell,
        records=contract.RECORDS_PER_DOMAIN,
        chunk_records=contract.CAPTURE_BATCH_RECORDS,
    ):
        for offset in range(chunk.stop - chunk.start):
            row = chunk.start + offset
            if row >= records:
                return
            yield row, chunk.activations[offset], chunk.mask[offset], chunk.positions[offset]
            seen += 1
    if seen != records:
        raise RunnerError(f"observation rows cover {seen}, expected {records}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _predict_decoder_row(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = valid_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    try:
        projected = model.projected_hidden(staged, mask)
        positions = torch.arange(1, contract.STORED_SEQUENCE_TOKENS, device=device, dtype=torch.long)
        rows = torch.zeros(positions.shape, device=device, dtype=torch.long)
        logits = model.logits_from_rows(projected, rows, positions, embedding)
    except AttributeError:
        logits_full = model(staged, mask, embedding)
        if logits_full.ndim != 3 or tuple(logits_full.shape[:2]) != (1, contract.STORED_SEQUENCE_TOKENS):
            raise RunnerError(f"decoder returned unexpected geometry: {tuple(logits_full.shape)}")
        logits = logits_full[0, 1:]
    if logits.ndim != 2 or tuple(logits.shape) != (contract.SCORED_POST_BOS_TOKENS, contract.VOCAB_SIZE):
        raise RunnerError(f"decoder row logits geometry changed: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise RunnerError("decoder emitted non-finite logits")
    values = torch.full((contract.STORED_SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
    values[0] = contract.BOS_TOKEN_ID
    values[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
    return contract.normalize_prediction(values, valid_mask)


class _A2Adapter:
    """Output-only adapter around the retained public P0 A1+A2 implementation."""

    def __init__(
        self,
        *,
        precut: torch.nn.Module,
        lens: torch.nn.Module,
        embeddings: torch.Tensor,
        device: torch.device,
        policy: Any,
    ) -> None:
        self.precut = precut
        self.lens = lens
        self.embeddings = embeddings
        self.device = device
        self.policy = policy
        self.calls = 0
        self.proposal_seconds = 0.0
        self.candidate_simulations = 0
        self.executed_candidate_simulations = 0
        self.prefix_commit_tokens = 0
        self.prefix_calls = 0
        self.last_a1_prediction: torch.Tensor | None = None
        self._baseline: dict[str, Any] = {}

    def begin_cell(self) -> None:
        self._baseline = {
            "calls": self.calls,
            "proposal_seconds": self.proposal_seconds,
            "candidate_simulations": self.candidate_simulations,
            "executed_candidate_simulations": self.executed_candidate_simulations,
            "prefix_commit_tokens": self.prefix_commit_tokens,
            "prefix_calls": self.prefix_calls,
        }

    @torch.inference_mode()
    def predict(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        observations = activation.detach().to(device="cpu").view(1, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
        attention_mask = valid_mask.detach().to(device="cpu", dtype=torch.long).view(1, contract.STORED_SEQUENCE_TOKENS)
        position_ids = positions.detach().to(device="cpu", dtype=torch.long).view(1, contract.STORED_SEQUENCE_TOKENS)
        try:
            legacy = importlib.import_module("scripts.trr0003_footing_compare")
        except ModuleNotFoundError:
            legacy = importlib.import_module("trr0003_footing_compare")
        before = int(getattr(self.precut, "checked_cache_transitions", 0))
        proposal = legacy.propose_public_a1(
            observations=observations,
            attention_mask=attention_mask,
            lens=self.lens,
            normalized_embeddings=self.embeddings,
            max_k=contract.A2_PROPOSAL_K,
            chunk=256,
        )
        decoded = legacy.decode_policy(
            observations=observations,
            attention_mask=attention_mask,
            position_ids=position_ids,
            candidates=proposal.candidates[:, :, : contract.A2_K].contiguous(),
            a1_confidence=proposal.top1_confidence,
            precut=self.precut,
            device=self.device,
            policy=self.policy,
            record_batch_size=1,
        )
        after = int(getattr(self.precut, "checked_cache_transitions", 0))
        if after < before:
            raise RunnerError("public-prefix transition counter moved backwards")
        self.calls += 1
        self.proposal_seconds += float(proposal.elapsed_seconds)
        self.candidate_simulations += int(decoded.candidate_simulations)
        self.executed_candidate_simulations += int(decoded.executed_candidate_simulations)
        self.prefix_commit_tokens += int(decoded.prefix_commit_tokens)
        self.prefix_calls += after - before
        a1_raw = proposal.candidates[0, :, 0].to(device="cpu", dtype=torch.long)
        self.last_a1_prediction = contract.normalize_prediction(a1_raw, valid_mask)
        raw = decoded.predictions[0].to(device="cpu", dtype=torch.long)
        return contract.normalize_prediction(raw, valid_mask)

    def evidence(self) -> dict[str, Any]:
        return {
            "calls": self.calls - int(self._baseline.get("calls", 0)),
            "proposal_seconds_sum": self.proposal_seconds - float(self._baseline.get("proposal_seconds", 0.0)),
            "proposal_budget": contract.A2_PROPOSAL_K,
            "candidate_budget": contract.A2_K,
            "candidate_simulations": self.candidate_simulations - int(self._baseline.get("candidate_simulations", 0)),
            "executed_candidate_simulations": self.executed_candidate_simulations - int(self._baseline.get("executed_candidate_simulations", 0)),
            "prefix_commit_tokens": self.prefix_commit_tokens - int(self._baseline.get("prefix_commit_tokens", 0)),
            "public_prefix_calls": self.prefix_calls - int(self._baseline.get("prefix_calls", 0)),
            "a2_fallback": False,
            "candidate_output": "output_only; no candidate tensors persisted",
        }


def _import_reference(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"public prefix reference is unavailable: {path}")
    name = "trr0007_public_prefix_reference"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RunnerError("cannot import public prefix reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RunnerError("public prefix reference import failed") from exc
    return module


def _load_a2_adapter(
    registration: Mapping[str, Any],
    *,
    root: Path,
    embedding: torch.Tensor,
    device: torch.device,
) -> tuple[_A2Adapter, dict[str, Any]]:
    assets = registration["runtime_assets"]["a1_a2"]
    snapshot_value = assets["public_model_snapshot"]
    snapshot = Path(str(snapshot_value["path"])).expanduser().resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RunnerError(f"public model snapshot is unavailable: {snapshot}")
    try:
        lens_record = contract.validate_file_record(
            assets["lens"], repository_root=root, description="A1 lens", verify=True
        )
        reference_record = contract.validate_file_record(
            assets["reference"], repository_root=root, description="A1+A2 reference", verify=True
        )
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc
    reference = _import_reference(Path(reference_record["path"]))
    try:
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        full.requires_grad_(False)
        if int(full.config.hidden_size) != contract.HIDDEN_SIZE or int(full.config.vocab_size) != contract.VOCAB_SIZE:
            raise RunnerError("public model geometry changed")
        precut = reference.PublicP0Precut(full, (0, 1, 2, 3)).to(device).eval()
        lens = reference.load_frozen_lens(Path(lens_record["path"]), device=device)
        lens.eval()
        lens.requires_grad_(False)
        try:
            legacy = importlib.import_module("scripts.trr0003_footing_compare")
        except ModuleNotFoundError:
            legacy = importlib.import_module("trr0003_footing_compare")
        policy = legacy._fixed_k256_policy()
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("public P0 A1+A2 assets could not be loaded") from exc
    del full
    _synchronize(device)
    return _A2Adapter(
        precut=precut,
        lens=lens,
        embeddings=embedding,
        device=device,
        policy=policy,
    ), {
        "snapshot": str(snapshot),
        "lens": dict(lens_record),
        "reference": dict(reference_record),
        "loader": "TRR3 public P0 + frozen public Alpaca A1 lens + fixed K256 direct-cosine policy",
        "proposal_budget": contract.A2_PROPOSAL_K,
        "candidate_budget": contract.A2_K,
    }


def _load_decoder(
    row: Mapping[str, Any],
    *,
    root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    method_id = str(row["id"])
    try:
        state = contract.validate_file_record(
            row["state"], repository_root=root, description=f"{method_id} state", verify=True
        )
        loader_desc = row["loader"]
        module = importlib.import_module(str(loader_desc["module"]))
        function = getattr(module, str(loader_desc["function"]))
        kwargs = dict(loader_desc.get("kwargs", {}))
        model = function(Path(state["path"]), **kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("loader did not return a torch.nn.Module")
        model.requires_grad_(False)
        model = model.to(device=device).eval()
        _synchronize(device)
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc
    except Exception as exc:
        raise RunnerError(f"decoder {method_id} could not be loaded") from exc
    hidden_size = int(getattr(model, "hidden_size", -1))
    vocabulary_size = int(getattr(model, "vocabulary_size", -1))
    if hidden_size != contract.HIDDEN_SIZE or vocabulary_size != contract.VOCAB_SIZE:
        raise RunnerError(f"decoder {method_id} geometry changed")
    return model, {
        "method_id": method_id,
        "state": dict(state),
        "loader": f"{loader_desc['module']}.{loader_desc['function']}",
        "load_seconds": None,
        "parameter_count": int(sum(int(p.numel()) for p in model.parameters())),
    }


def _prediction_path(output_root: Path, cell_id: str, method_id: str) -> Path:
    style, condition = cell_id.split("__", 1)
    return output_root / style / condition / f"{method_id}.safetensors"


def _write_prediction(
    path: Path,
    *,
    prediction: torch.Tensor,
    registration: Mapping[str, Any],
    cell: Mapping[str, Any],
    method_id: str,
    records: int,
    a1_prediction: torch.Tensor | None = None,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RunnerError(f"prediction artifact is not create-only: {path}")
    checked = contract.validate_prediction_tensor(prediction, records=records)
    tensors: dict[str, torch.Tensor] = {"predictions": checked}
    if method_id == contract.ANCHOR_METHOD_ID:
        if a1_prediction is None:
            raise RunnerError("A1 diagnostic is required for the bounded anchor")
        checked_a1 = contract.validate_prediction_tensor(a1_prediction, records=records)
        tensors["a1_predictions"] = checked_a1
    elif a1_prediction is not None:
        raise RunnerError("A1 diagnostic is only valid for the bounded anchor")
    metadata = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration_sha256": registration["registration_sha256"],
        "observation_sha256": cell["observation"]["sha256"],
        "cell_id": str(cell["cell_id"]),
        "method_id": method_id,
        "records": str(records),
        "geometry_json": json.dumps(
            {
                "records": records,
                "sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
                "hidden_size": contract.HIDDEN_SIZE,
                "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
            },
            sort_keys=True,
        ),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    artifact = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": contract.sha256_file(path)}
    return {
        "path": str(path),
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
        "prediction_sha256": contract.tensor_digest(checked),
        "a1_prediction_sha256": (
            contract.tensor_digest(tensors["a1_predictions"])
            if "a1_predictions" in tensors else None
        ),
        "records": records,
    }


def _run_decoder_method(
    *,
    row: Mapping[str, Any],
    registration: Mapping[str, Any],
    observations: Mapping[str, Any],
    embedding: torch.Tensor,
    output_root: Path,
    root: Path,
    device: torch.device,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    load_started = time.perf_counter()
    model, state_evidence = _load_decoder(row, root=root, device=device)
    state_evidence["load_seconds"] = float(time.perf_counter() - load_started)
    _guard(
        device=device,
        guard=registration["resource_guard"],
        started=started,
        stage=f"after_{row['id']}_load",
    )
    prediction_entries: dict[str, Any] = {}
    timing_entries: dict[str, Any] = {}
    try:
        for cell_id in contract.CELL_ORDER:
            cell = observations["cells"][cell_id]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            values = torch.empty(
                (contract.RECORDS_PER_DOMAIN, contract.STORED_SEQUENCE_TOKENS), dtype=torch.long
            )
            warmup_sum = 0.0
            measured_sum = 0.0
            per_record: list[float] = []
            rows_seen = 0
            for row_index, activation, mask, _positions in _iter_rows(cell, records=contract.RECORDS_PER_DOMAIN):
                _guard(
                    device=device,
                    guard=registration["resource_guard"],
                    started=started,
                    stage=f"before_{row['id']}_{cell_id}_row_{row_index}",
                )
                t0 = time.perf_counter()
                warm = _predict_decoder_row(model, embedding, activation, mask, device=device)
                _synchronize(device)
                warm_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                measured = _predict_decoder_row(model, embedding, activation, mask, device=device)
                _synchronize(device)
                measured_seconds = time.perf_counter() - t1
                if not torch.equal(warm, measured):
                    raise RunnerError(f"warmup and measured predictions differ: {row['id']}/{cell_id}/{row_index}")
                values[row_index] = measured
                warmup_sum += warm_seconds
                measured_sum += measured_seconds
                per_record.append(float(measured_seconds))
                rows_seen += 1
            if rows_seen != contract.RECORDS_PER_DOMAIN:
                raise RunnerError(f"incomplete rows for {row['id']}/{cell_id}: {rows_seen}")
            artifact_path = _prediction_path(output_root, cell_id, str(row["id"]))
            artifact = _write_prediction(
                artifact_path,
                prediction=values,
                registration=registration,
                cell=cell,
                method_id=str(row["id"]),
                records=contract.RECORDS_PER_DOMAIN,
            )
            timing = {
                "schema": contract.TIMING_SCHEMA,
                "task_id": contract.TASK_ID,
                "method_id": str(row["id"]),
                "cell_id": cell_id,
                "records": contract.RECORDS_PER_DOMAIN,
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_seconds_sum": float(warmup_sum),
                "measured_seconds_sum": float(measured_sum),
                "per_record_measured_seconds": per_record,
                "warmup_output_exact_match_measured": True,
                "measured_output_selected": True,
                "steady_interval": "CPU BF16 activation -> FP32 current-H decoder -> full public-vocabulary argmax -> CPU IDs",
                "model_preparation_seconds": state_evidence["load_seconds"],
                "fit_cost": row.get("fit_cost"),
                "peak_memory": {
                    "process_max_rss_bytes": _rss_bytes(),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
                },
                "prediction_artifact": artifact,
                "truth_opened": False,
                "candidate_arrays_persisted": False,
            }
            contract.write_create_only(
                artifact_path.with_suffix(".run.json"),
                timing,
            )
            prediction_entries[cell_id] = {
                "method_id": str(row["id"]),
                "cell_id": cell_id,
                "records": contract.RECORDS_PER_DOMAIN,
                "prediction_artifact": artifact,
                "prediction_sha256": artifact["prediction_sha256"],
                "state": state_evidence,
                "observation_sha256": cell["observation"]["sha256"],
                "truth_opened": False,
                "candidate_arrays_persisted": False,
            }
            timing_entries[f"{str(row['id'])}::{cell_id}"] = timing
            del values
            gc.collect()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prediction_entries, timing_entries


def _run_anchor(
    *,
    row: Mapping[str, Any],
    registration: Mapping[str, Any],
    observations: Mapping[str, Any],
    embedding: torch.Tensor,
    output_root: Path,
    root: Path,
    device: torch.device,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    load_started = time.perf_counter()
    adapter, a2_evidence = _load_a2_adapter(registration, root=root, embedding=embedding, device=device)
    a2_evidence["load_seconds"] = float(time.perf_counter() - load_started)
    _guard(device=device, guard=registration["resource_guard"], started=started, stage="after_anchor_load")
    prediction_entries: dict[str, Any] = {}
    timing_entries: dict[str, Any] = {}
    for cell_id in contract.BASE_CELL_ORDER:
        cell = observations["cells"][cell_id]
        adapter.begin_cell()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        values = torch.empty((contract.ANCHOR_RECORDS_PER_DOMAIN, contract.STORED_SEQUENCE_TOKENS), dtype=torch.long)
        a1_values = torch.empty_like(values)
        warmup_sum = 0.0
        measured_sum = 0.0
        per_record: list[float] = []
        rows_seen = 0
        for row_index, activation, mask, positions in _iter_rows(
            cell, records=contract.ANCHOR_RECORDS_PER_DOMAIN
        ):
            t0 = time.perf_counter()
            warm = adapter.predict(activation, mask, positions)
            _synchronize(device)
            warm_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            measured = adapter.predict(activation, mask, positions)
            _synchronize(device)
            measured_seconds = time.perf_counter() - t1
            if not torch.equal(warm, measured):
                raise RunnerError(f"warmup and measured A2 predictions differ: {cell_id}/{row_index}")
            if adapter.last_a1_prediction is None:
                raise RunnerError(f"A1 diagnostic is missing: {cell_id}/{row_index}")
            a1_values[row_index] = adapter.last_a1_prediction
            values[row_index] = measured
            warmup_sum += warm_seconds
            measured_sum += measured_seconds
            per_record.append(float(measured_seconds))
            rows_seen += 1
            _guard(
                device=device,
                guard=registration["resource_guard"],
                started=started,
                stage=f"after_{cell_id}_anchor_row_{row_index}",
            )
        if rows_seen != contract.ANCHOR_RECORDS_PER_DOMAIN:
            raise RunnerError(f"incomplete anchor rows for {cell_id}: {rows_seen}")
        artifact_path = _prediction_path(output_root, cell_id, contract.ANCHOR_METHOD_ID)
        artifact = _write_prediction(
            artifact_path,
            prediction=values,
            registration=registration,
            cell=cell,
            method_id=contract.ANCHOR_METHOD_ID,
            records=contract.ANCHOR_RECORDS_PER_DOMAIN,
            a1_prediction=a1_values,
        )
        evidence = a2_evidence | adapter.evidence()
        timing = {
            "schema": contract.TIMING_SCHEMA,
            "task_id": contract.TASK_ID,
            "method_id": contract.ANCHOR_METHOD_ID,
            "cell_id": cell_id,
            "records": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "warmup_seconds_sum": float(warmup_sum),
            "measured_seconds_sum": float(measured_sum),
            "per_record_measured_seconds": per_record,
            "warmup_output_exact_match_measured": True,
            "measured_output_selected": True,
            "steady_interval": "CPU activation staging -> public A1 proposal -> P0 prefix K256 A2 -> CPU IDs",
            "model_preparation_seconds": a2_evidence["load_seconds"],
            "a2": evidence,
            "a1_prediction_sha256": artifact["a1_prediction_sha256"],
            "peak_memory": {
                "process_max_rss_bytes": _rss_bytes(),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            },
            "prediction_artifact": artifact,
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        }
        contract.write_create_only(artifact_path.with_suffix(".run.json"), timing)
        prediction_entries[cell_id] = {
            "method_id": contract.ANCHOR_METHOD_ID,
            "cell_id": cell_id,
            "records": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "prediction_artifact": artifact,
            "prediction_sha256": artifact["prediction_sha256"],
            "a1_prediction_sha256": artifact["a1_prediction_sha256"],
            "observation_sha256": cell["observation"]["sha256"],
            "timing_a2": evidence,
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        }
        timing_entries[f"{contract.ANCHOR_METHOD_ID}::{cell_id}"] = timing
        del values
        gc.collect()
    del adapter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction_entries, timing_entries


def _output_root(registration: Mapping[str, Any], root: Path) -> Path:
    raw = Path(str(registration["output_root"])).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    output = raw.resolve()
    task_root = (root / "experiments" / "TRR-0007").resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RunnerError(f"output root must be task-owned below {task_root}") from exc
    if output.is_symlink():
        raise RunnerError(f"output root is a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def execute(
    *,
    registration_path: Path,
    repository_root: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    registration = contract.load_registration(registration_path)
    registration["registration_sha256"] = contract.sha256_file(registration_path)
    if _git_head(root) != registration["code_commit"]:
        raise RunnerError("registration code_commit does not match executable HEAD")
    code_evidence = _verify_code_bindings(registration, root)
    _configure_numerics(registration["numerical_settings"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA is unavailable")
    started = time.perf_counter()
    _guard(device=device, guard=registration["resource_guard"], started=started, stage="before_public_inputs")
    _manifest, observations, observation_record = contract.load_observation_manifest(
        registration, repository_root=root, verify_assets=True
    )
    embedding, embedding_evidence = _load_embedding(registration, root=root, device=device)
    _guard(device=device, guard=registration["resource_guard"], started=started, stage="after_public_embedding")
    output_root = _output_root(registration, root)
    if (output_root / "registration.json").exists() or (output_root / "registration.json").is_symlink():
        raise RunnerError("output root already contains registration.json")
    # Preserve an exact registration copy in the run root for later audit.
    (output_root / "registration.json").write_bytes(registration_path.read_bytes())
    all_predictions: dict[str, Any] = {}
    all_timings: dict[str, Any] = {}
    rows = registration["methods"]
    try:
        for row in rows:
            method_id = str(row["id"])
            if method_id == contract.ANCHOR_METHOD_ID:
                predictions, timings = _run_anchor(
                    row=row,
                    registration=registration,
                    observations=observations,
                    embedding=embedding,
                    output_root=output_root,
                    root=root,
                    device=device,
                    started=started,
                )
            else:
                predictions, timings = _run_decoder_method(
                    row=row,
                    registration=registration,
                    observations=observations,
                    embedding=embedding,
                    output_root=output_root,
                    root=root,
                    device=device,
                    started=started,
                )
            all_predictions.update({f"{method_id}::{cell}": value for cell, value in predictions.items()})
            all_timings.update(timings)
        expected = sum(len(contract.expected_method_cells(method_id)) for method_id in contract.METHOD_ORDER)
        if len(all_predictions) != expected or len(all_timings) != expected:
            raise RunnerError(f"prediction/timing matrix incomplete: {len(all_predictions)}/{expected}")
        run_manifest = {
            "schema": contract.RUN_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "COMPLETE_PUBLIC_PREDICTIONS_NO_TRUTH",
            "registration": {
                "path": str(registration_path),
                "bytes": int(registration_path.stat().st_size),
                "sha256": registration["registration_sha256"],
            },
            "code_commit": registration["code_commit"],
            "code_bindings": code_evidence,
            "observation_manifest": observation_record,
            "runtime_embedding": embedding_evidence,
            "device": str(device),
            "numerical_settings": dict(registration["numerical_settings"]),
            "predictions_complete": True,
            "timing_complete": True,
            "prediction_count": len(all_predictions),
            "timing_count": len(all_timings),
            "predictions": all_predictions,
            "timings": all_timings,
            "truth_opened": False,
            "candidate_arrays_persisted": False,
            "started_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        contract.write_create_only(output_root / "predictions.json", {
            "schema": contract.PREDICTION_SCHEMA,
            "task_id": contract.TASK_ID,
            "registration_sha256": registration["registration_sha256"],
            "entries": all_predictions,
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        })
        contract.write_create_only(output_root / "timings.json", {
            "schema": contract.TIMING_SCHEMA,
            "task_id": contract.TASK_ID,
            "registration_sha256": registration["registration_sha256"],
            "entries": all_timings,
            "truth_opened": False,
        })
        contract.write_create_only(output_root / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0007-prediction-failure.v1",
            "task_id": contract.TASK_ID,
            "status": "FAILED_NO_TRUTH",
            "registration_sha256": registration["registration_sha256"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "truth_opened": False,
            "candidate_arrays_persisted": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        try:
            contract.write_create_only(output_root / "failure.json", failure)
        except Exception:
            pass
        if isinstance(exc, RunnerError):
            raise
        raise RunnerError("prediction run failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute(
            registration_path=args.registration,
            repository_root=args.repository_root,
            device_name=args.device,
        )
    except (RunnerError, contract.ContractError) as exc:
        print(f"TRR-0007 prediction run failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
