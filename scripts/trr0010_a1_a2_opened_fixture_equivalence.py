#!/usr/bin/env python3
"""One-row-per-domain native A1+A2 versus TRR-0010 wrapper diagnostic.

This truth-free development diagnostic uses the already-opened TRR-0009
public-base H observations. It compares a direct call to the retained native
proposal/decode helpers with the existing TRR-0010 _load_native_a1_a2 adapter
on the same H row, public E, retained lens, public prefix, and fixed K=256
policy. The two paths intentionally share one loaded public model; the result
therefore verifies wrapper input staging, causal-state transitions, and output
normalization rather than independent algorithm accuracy.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import time
import traceback
from typing import Any

import torch

from scripts import trr0004_predict_confirmation as legacy
from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_runner as runner


SCHEMA = "token-reconstruction.trr0010-a1-a2-opened-fixture-equivalence.v2"
FAILURE_SCHEMA = "token-reconstruction.trr0010-a1-a2-opened-fixture-equivalence-failure.v2"
DEFAULT_DESCRIPTOR = Path("experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json")
DEFAULT_OBSERVATIONS = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0009/experiments/TRR-0009/evaluation/public_observations_v2/observations.json"
)
CELL_ORDER = ("pile__public_base", "finance__public_base")
MINIMUM_FREE_GIB = 8.0
MAXIMUM_RESERVED_GIB = 6.0
MAXIMUM_RSS_GIB = 16.0
MINIMUM_DISK_GIB = 20.0


class DiagnosticError(RuntimeError):
    """Raised when this development diagnostic fails closed."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise DiagnosticError(f"repository root unavailable: {root}")
    return root


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{description} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{description} must be an object")
    return value


def _record(path: Path, *, root: Path, description: str, readonly: bool = False) -> dict[str, Any]:
    try:
        return gate.file_record(path, root=root, readonly=readonly)
    except gate.GateError as exc:
        raise DiagnosticError(f"{description}: {exc}") from exc


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(left.get(key)) != str(right.get(key)):
            raise DiagnosticError(f"{description} {key} changed")


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DiagnosticError("cannot resolve executable commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise DiagnosticError("executable commit is malformed")
    return value


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    return 0


def _host_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    raise DiagnosticError("host MemAvailable is unavailable")


def _guard_args(max_seconds: float) -> argparse.Namespace:
    return argparse.Namespace(
        minimum_free_gib=MINIMUM_FREE_GIB,
        maximum_reserved_gib=MAXIMUM_RESERVED_GIB,
        maximum_rss_gib=MAXIMUM_RSS_GIB,
        max_seconds=max_seconds,
    )


def _resource_guard(
    *,
    args: argparse.Namespace,
    device: torch.device,
    started: float,
    stage: str,
) -> dict[str, Any]:
    if time.perf_counter() - started > args.max_seconds:
        raise DiagnosticError(f"wall-time guard expired at {stage}")
    if _rss_bytes() > int(MAXIMUM_RSS_GIB * 2**30):
        raise DiagnosticError(f"host RSS guard failed at {stage}")
    if _host_available_bytes() < int(10 * 2**30):
        raise DiagnosticError(f"host MemAvailable guard failed at {stage}")
    disk_path = Path(args.output_root)
    while not disk_path.exists():
        parent = disk_path.parent
        if parent == disk_path:
            raise DiagnosticError(f"disk free-space guard failed at {stage}: no existing ancestor")
        disk_path = parent
    if not disk_path.is_dir():
        disk_path = disk_path.parent
    try:
        disk_free = int(shutil.disk_usage(disk_path).free)
    except OSError as exc:
        raise DiagnosticError(f"disk free-space guard failed at {stage}") from exc
    if disk_free < int(MINIMUM_DISK_GIB * 2**30):
        raise DiagnosticError(f"disk free-space guard failed at {stage}: {disk_free}")
    try:
        result = legacy._resource_preflight(  # noqa: SLF001 - retained guard
            _guard_args(args.max_seconds),
            device,
            stage=stage,
            started=started,
        )
    except Exception as exc:
        raise DiagnosticError(f"GPU guard failed at {stage}: {exc}") from exc
    return result


def _descriptor_and_bindings(
    descriptor_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    descriptor = _load_json(descriptor_path, description="A1+A2 descriptor")
    if descriptor.get("schema") != "token-reconstruction.trr0010-a1-a2-public-runtime-descriptor.v1":
        raise DiagnosticError("A1+A2 descriptor schema changed")
    if descriptor.get("status") != "PUBLIC_READONLY_BINDING_ONLY_NO_INFERENCE":
        raise DiagnosticError("A1+A2 descriptor is not preparation-only")
    for key in ("public_embedding_table", "retained_a1_lens", "public_reference"):
        if not isinstance(descriptor.get(key), Mapping):
            raise DiagnosticError(f"descriptor binding missing: {key}")
    descriptor_record = _record(descriptor_path, root=root, description="A1+A2 descriptor")
    e_binding = descriptor["public_embedding_table"]
    lens_binding = descriptor["retained_a1_lens"]
    reference_binding = descriptor["public_reference"]
    e_record = _record(
        Path(str(e_binding["path"])),
        root=root,
        description="public normalized E",
        readonly=e_binding.get("readonly") is True,
    )
    lens_record = _record(
        Path(str(lens_binding["path"])),
        root=root,
        description="retained A1 lens",
        readonly=lens_binding.get("readonly") is True,
    )
    reference_record = _record(
        Path(str(reference_binding["path"])),
        root=root,
        description="public reference helper",
        readonly=reference_binding.get("readonly") is True,
    )
    expected_code = descriptor.get("native_code_bindings")
    if not isinstance(expected_code, Mapping):
        raise DiagnosticError("descriptor native code bindings are absent")
    code_records: dict[str, Any] = {}
    for name, binding in expected_code.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
            raise DiagnosticError(f"descriptor code binding malformed: {name}")
        actual = _record(
            Path(str(binding["path"])),
            root=root,
            description=f"code {name}",
            readonly=binding.get("readonly") is True,
        )
        for key in ("bytes", "sha256"):
            expected = binding.get(key)
            if expected is not None and str(actual[key]) != str(expected):
                raise DiagnosticError(f"descriptor code binding changed: {name}/{key}")
        code_records[str(name)] = actual
    code_records["gate"] = _record(Path(gate.__file__), root=root, description="TRR-0010 gate")
    code_records["diagnostic"] = _record(Path(__file__), root=root, description="TRR-0010 diagnostic")
    return descriptor, descriptor_record, e_record, lens_record, reference_record, code_records


def _make_row(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    descriptor_path: Path,
    e_record: Mapping[str, Any],
    lens_record: Mapping[str, Any],
) -> dict[str, Any]:
    p0_record = _record(descriptor_path, root=root, description="public P0 descriptor")
    model_snapshot = descriptor.get("model_snapshot")
    reference_binding = descriptor.get("reference_binding")
    if not isinstance(model_snapshot, Mapping) or not isinstance(reference_binding, Mapping):
        raise DiagnosticError("descriptor model_snapshot/reference_binding aliases are absent")
    snapshot = model_snapshot.get("path")
    if not isinstance(snapshot, str):
        raise DiagnosticError("descriptor public snapshot path is absent")
    return {
        "id": gate.A1_A2_METHOD_ID,
        "resources": {
            "public_embedding_table": dict(e_record),
            "retained_a1_lens": dict(lens_record),
            "public_p0_prefix": dict(p0_record),
        },
        "loader": {
            "interface": gate.A1_A2_LOADER_INTERFACE,
            "current_h_only": False,
            "full_vocabulary": True,
            "history_enabled": False,
            "a2_enabled": True,
            "reconstructed_prefix": True,
            "candidate_k": 256,
            "path_args": {
                "snapshot": {"path": snapshot, "readonly": True},
                "reference_path": dict(reference_binding),
            },
        },
    }


def _opened_cells(manifest: Mapping[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != "token-reconstruction.trr0009-public-observation-manifest.v1":
        raise DiagnosticError("TRR9 observation manifest schema changed")
    if manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise DiagnosticError("TRR9 observation manifest is not public-only")
    for key in ("truth_opened", "source_text_loaded", "target_labels_loaded", "token_ids_written", "candidate_arrays_persisted"):
        if manifest.get(key) is not False:
            raise DiagnosticError(f"TRR9 manifest truth flag changed: {key}")
    rows = {
        str(row.get("cell_id")): row
        for row in manifest.get("cells", [])
        if isinstance(row, Mapping)
    }
    result: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_ORDER:
        row = rows.get(cell_id)
        if not isinstance(row, Mapping) or not isinstance(row.get("observation"), Mapping):
            raise DiagnosticError(f"TRR9 public-base cell missing: {cell_id}")
        binding = dict(row["observation"])
        binding["readonly"] = True
        observation_record = _record(
            Path(str(binding["path"])),
            root=root,
            description=f"TRR9 observation {cell_id}",
            readonly=True,
        )
        _same_binding(observation_record, row["observation"], description=f"TRR9 observation {cell_id}")
        result[cell_id] = {
            "cell_id": cell_id,
            "records": int(row.get("records", 0)),
            "observation": observation_record,
        }
    return result


def _direct_native(
    adapter: Any,
    activation: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    observations = activation.detach().to(device="cpu").view(1, activation.shape[0], activation.shape[1])
    attention_mask = mask.detach().to(device="cpu", dtype=torch.long).view(1, mask.shape[0])
    position_ids = positions.detach().to(device="cpu", dtype=torch.long).view(1, positions.shape[0])
    before = int(getattr(adapter.precut, "checked_cache_transitions", 0))
    native = importlib.import_module("trr0003_footing_compare")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    proposal = native.propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=adapter.lens,
        normalized_embeddings=adapter.embeddings,
        max_k=legacy.DEFAULT_A2_PROPOSAL_K,
        chunk=legacy.DEFAULT_A1_CHUNK,
    )
    decoded = native.decode_policy(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        candidates=proposal.candidates[:, :, :legacy.DEFAULT_A2_K].contiguous(),
        a1_confidence=proposal.top1_confidence,
        precut=adapter.precut,
        device=device,
        policy=adapter.policy,
        record_batch_size=legacy.DEFAULT_RECORD_BATCH_SIZE,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    after = int(getattr(adapter.precut, "checked_cache_transitions", 0))
    if after < before:
        raise DiagnosticError("native causal-state transition counter moved backwards")
    return decoded.predictions[0].to(device=activation.device, dtype=torch.long), {
        "seconds": float(elapsed),
        "prefix_transition_delta": after - before,
        "candidate_simulations": int(decoded.executed_candidate_simulations),
        "prefix_commit_tokens": int(decoded.prefix_commit_tokens),
        "proposal_candidates_shape": [int(x) for x in proposal.candidates.shape],
    }


def _run(
    *,
    args: argparse.Namespace,
    root: Path,
    descriptor_path: Path,
    descriptor: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    started: float,
    e_record: Mapping[str, Any],
    lens_record: Mapping[str, Any],
    code_records: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise DiagnosticError("this A1+A2 diagnostic requires CUDA")
    device = torch.device("cuda")
    code_bindings = {
        name: value
        for name, value in code_records.items()
        if name not in {"gate", "diagnostic"}
    }
    row = _make_row(
        descriptor,
        root=root,
        descriptor_path=descriptor_path,
        e_record=e_record,
        lens_record=lens_record,
    )
    _resource_guard(args=args, device=device, started=started, stage="before_load")
    loaded = runner._load_native_a1_a2(  # noqa: SLF001 - existing path under test
        row,
        root=root,
        device=device,
        embedding=None,
        code_bindings=code_bindings,
    )
    adapter = loaded.adapter
    results: dict[str, Any] = {}
    try:
        for cell_id in CELL_ORDER:
            cell = observations[cell_id]
            seen = 0
            for index, activation, mask, positions in runner._iter_rows(  # noqa: SLF001
                cell,
                records=int(cell["records"]),
                hidden_size=gate.OBSERVATION_HIDDEN_SIZE,
            ):
                if index != 0:
                    break
                _resource_guard(args=args, device=device, started=started, stage=f"{cell_id}/before_direct")
                input_digest = {
                    "activation": gate.tensor_digest(activation),
                    "mask": gate.tensor_digest(mask),
                    "positions": gate.tensor_digest(positions),
                }
                direct, direct_cost = _direct_native(
                    adapter,
                    activation,
                    mask,
                    positions,
                    device=device,
                )
                _resource_guard(args=args, device=device, started=started, stage=f"{cell_id}/before_wrapper")
                before_wrapper = int(getattr(adapter.precut, "checked_cache_transitions", 0))
                torch.cuda.synchronize(device)
                wrapper_started = time.perf_counter()
                wrapped = adapter(activation, mask, positions)
                torch.cuda.synchronize(device)
                wrapper_elapsed = time.perf_counter() - wrapper_started
                after_wrapper = int(getattr(adapter.precut, "checked_cache_transitions", 0))
                direct_normalized = runner._normalize_prediction(  # noqa: SLF001
                    direct,
                    mask,
                    method_id=gate.A1_A2_METHOD_ID,
                )
                wrapped_normalized = runner._normalize_prediction(  # noqa: SLF001
                    wrapped,
                    mask,
                    method_id=gate.A1_A2_METHOD_ID,
                )
                if not torch.equal(direct_normalized, wrapped_normalized):
                    raise DiagnosticError(f"native/wrapper IDs differ: {cell_id}")
                _resource_guard(args=args, device=device, started=started, stage=f"{cell_id}/after_wrapper")
                results[cell_id] = {
                    "records_compared": 1,
                    "token_positions_compared": int(direct_normalized.numel()),
                    "exact_token_equality": True,
                    "prediction_digest_sha256": gate.tensor_digest(wrapped_normalized),
                    "input_digest": input_digest,
                    "direct_native": direct_cost,
                    "wrapper_seconds": float(wrapper_elapsed),
                    "wrapper_prefix_transition_delta": after_wrapper - before_wrapper,
                    "shared_public_model_instance": True,
                    "comparison_scope": (
                        "same loaded public P0/lens/E and same retained native "
                        "proposal/decode helpers; wrapper staging and normalization checked"
                    ),
                    "native_helpers": [
                        "trr0003_footing_compare.propose_public_a1",
                        "trr0003_footing_compare.decode_policy",
                    ],
                }
                seen = 1
                break
            if seen != 1:
                raise DiagnosticError(f"opened observation had no row: {cell_id}")
        source_end = _descriptor_and_bindings(descriptor_path, root=root)[-1]
        return results, source_end
    finally:
        del adapter, loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": gate.sha256_file(path)}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--observation-manifest", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    root = _root(args.repository_root)
    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = root / args.output_root
    args.output_root = args.output_root.resolve()
    task_root = (root / "experiments" / gate.TASK_ID).resolve()
    try:
        args.output_root.relative_to(task_root)
    except ValueError as exc:
        raise SystemExit(f"output root must be under {task_root}") from exc
    started_utc = _utc_now()
    started = time.perf_counter()
    descriptor_path = args.descriptor.expanduser()
    if not descriptor_path.is_absolute():
        descriptor_path = root / descriptor_path
    descriptor_path = descriptor_path.resolve()
    observations_path = args.observation_manifest.expanduser()
    if not observations_path.is_absolute():
        observations_path = root / observations_path
    observations_path = observations_path.resolve()
    result_path = args.output_root / "result.json"
    failure_path = args.output_root.parent / f"{args.output_root.name}_failure" / "failure.json"
    try:
        if result_path.exists() or result_path.is_symlink() or failure_path.exists() or failure_path.is_symlink():
            raise DiagnosticError("diagnostic output is create-only")
        descriptor, descriptor_record, e_record, lens_record, reference_record, source_start = _descriptor_and_bindings(
            descriptor_path, root=root
        )
        manifest = _load_json(observations_path, description="TRR9 observations")
        observation_manifest_record = _record(
            observations_path,
            root=root,
            description="TRR9 observation manifest",
            readonly=True,
        )
        observations = _opened_cells(manifest, root=root)
        commit_start = _git_head(root)
        results, source_end = _run(
            args=args,
            root=root,
            descriptor_path=descriptor_path,
            descriptor=descriptor,
            observations=observations,
            started=started,
            e_record=e_record,
            lens_record=lens_record,
            code_records=source_start,
        )
        commit_end = _git_head(root)
        if commit_end != commit_start:
            raise DiagnosticError(f"git HEAD changed during diagnostic: {commit_start} -> {commit_end}")
        for name, before in source_start.items():
            after = source_end.get(name)
            if not isinstance(after, Mapping):
                raise DiagnosticError(f"source binding disappeared: {name}")
            _same_binding(before, after, description=f"source {name}")
        _same_binding(
            descriptor_record,
            _record(descriptor_path, root=root, description="A1+A2 descriptor"),
            description="descriptor",
        )
        _same_binding(
            observation_manifest_record,
            _record(observations_path, root=root, description="TRR9 observation manifest", readonly=True),
            description="observation manifest",
        )
        payload = {
            "schema": SCHEMA,
            "task_id": gate.TASK_ID,
            "status": "A1_A2_NATIVE_WRAPPER_EQUIVALENCE_PASS",
            "claim_scope": (
                "one public-base H row per domain; direct retained native "
                "proposal/decode call versus existing TRR-0010 wrapper; "
                "shared public model, no independent accuracy oracle"
            ),
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "command": {"argv": [str(x) for x in sys.argv], "cwd": str(Path.cwd())},
            "git": {"commit_start": commit_start, "commit_end": commit_end, "source_files_unchanged": True},
            "inputs": {
                "descriptor": descriptor_record,
                "observation_manifest": observation_manifest_record,
                "resources": {
                    "public_embedding_table": dict(e_record),
                    "retained_a1_lens": dict(lens_record),
                    "public_reference": dict(reference_record),
                    "public_p0_descriptor": dict(descriptor_record),
                },
                "cells": {cell_id: observations[cell_id]["observation"] for cell_id in CELL_ORDER},
                "records_per_cell": 1,
                "truth_opened": False,
                "source_text_loaded": False,
                "target_labels_loaded": False,
                "candidate_arrays_persisted": False,
            },
            "policy": descriptor["policy"],
            "results": results,
            "code_bindings": source_end,
            "resource_guard": {
                "minimum_free_gib": MINIMUM_FREE_GIB,
                "maximum_reserved_gib": MAXIMUM_RESERVED_GIB,
                "maximum_rss_gib": MAXIMUM_RSS_GIB,
                "minimum_host_available_gib": 10.0,
                "minimum_disk_free_gib": MINIMUM_DISK_GIB,
                "maximum_seconds": args.max_seconds,
                "guard_points": "before load, before each direct/wrapper call, after each wrapper call",
            },
        }
        output_record = _write_create_only(result_path, payload)
        print(json.dumps({"status": payload["status"], "result": output_record}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": "A1_A2_NATIVE_WRAPPER_EQUIVALENCE_FAILED_CLOSED",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "command": {"argv": [str(x) for x in sys.argv], "cwd": str(Path.cwd())},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        }
        try:
            _write_create_only(failure_path, failure)
        except Exception:
            pass
        print(json.dumps({"status": failure["status"], "failure": str(failure_path), "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
