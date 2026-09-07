"""Assemble the frozen TRR-0010 method rows before registration.

This is an identity-only assembly helper.  It promotes the four method rows
whose public state is already available (unchanged, current fixed, expanded
fixed, and the retained A1+A2 comparator), and accepts exactly two later
directional descriptor files.  It does not select records, read activation or
label payloads, run a model, capture observations, open truth, or call the
registration/prediction runners.

The directional descriptor contract is deliberately narrow.  A descriptor is
either a JSON object containing ``method_row`` or the method-row object itself;
the row must already contain the exact gate fields ``id``, ``role``, ``state``,
``loader``, ``resources``, ``cells``, and ``records_per_cell``.  The helper
rehashes the bound state/readout files and checks the current-H/full-vocabulary
semantics before emitting a complete six-row method map.  Until both
descriptors are supplied, ``--emit-pending`` writes an explicit incomplete
assembly for inspection and cannot be passed to registration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register


TEMPLATE = ROOT / "experiments/TRR-0010/setup/final_evaluation_design_template_v1.json"
B1_STATE = Path("/tmp/trr-p09-runtime/fixed-control-b1-r3/states/checkpoint_step_013000.safetensors")
B1_STATE_SHA256 = "5757506003f0a4b82cb1cd6de2a82f11345f2dbcbce13c8cf18c4f515de3ec3a"
B1_SELECTED_STEP = 13000
B1_BANK_MANIFEST_SHA256 = "aefaa5f47aa1042dffa7210767ed4a0be1fd848c373f8cd6de1ad4c739035a31"
B1_FIT_MANIFEST_SHA256 = B1_BANK_MANIFEST_SHA256
B1_SCHEDULE_SHA256 = "8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5"
B1_RUNNER_STATE_SHA256 = "209155048df936359870a1419803800d7bad4ab72a46e09bf3287648079964da"
BASE_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
METHOD_ORDER = list(gate.METHOD_ORDER)
REQUIRED_METHOD_FIELDS = {"id", "role", "cells", "records_per_cell", "state", "loader", "resources"}
FALSE_FLAGS = (
    "truth_opened",
    "source_text_loaded",
    "target_labels_loaded",
    "source_text_written",
    "token_ids_written",
    "candidate_arrays_persisted",
)


class AssemblyError(ValueError):
    """Raised when a method assembly is incomplete or does not bind safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path_value: Any, *, description: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AssemblyError(f"{description} path is absent")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink():
        raise AssemblyError(f"{description} must be a direct file, not a symlink")
    path = path.resolve()
    if not path.is_file():
        raise AssemblyError(f"{description} is unavailable: {path}")
    return path


def _bound_record(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{description} record is absent")
    path = _resolve(value.get("path"), description=description)
    expected_bytes = value.get("bytes")
    expected_sha = value.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise AssemblyError(f"{description} byte count is malformed")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise AssemblyError(f"{description} SHA-256 is malformed")
    actual_bytes = path.stat().st_size
    actual_sha = _sha256(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise AssemblyError(f"{description} bytes or SHA-256 changed")
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }
    # Gate/registration permit public files outside this worktree only when
    # they are explicitly marked shared read-only.
    try:
        path.relative_to(ROOT)
    except ValueError:
        record["readonly"] = True
    else:
        if value.get("readonly") is True:
            record["readonly"] = True
    return record


def _truth_free(value: Mapping[str, Any], *, description: str) -> None:
    for key in FALSE_FLAGS:
        raw = value.get(key)
        if raw is True or (isinstance(raw, str) and raw.lower() == "true"):
            raise AssemblyError(f"{description} records forbidden truth/source access: {key}")


def _records_equal(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(left.get(key)) != str(right.get(key)):
            raise AssemblyError(f"{description} {key} differs")


def _normalise_known_row(row: Mapping[str, Any], *, method_id: str) -> dict[str, Any]:
    # The historical design template predates the common cells/counts fields;
    # those are filled from the shared gate below.  Directional descriptors
    # are checked for the complete row shape by _load_directional first.
    required_core = REQUIRED_METHOD_FIELDS - {"cells", "records_per_cell"}
    missing = sorted(required_core - set(row))
    if missing:
        raise AssemblyError(f"known method {method_id} is missing fields: {missing}")
    if row.get("id") != method_id or row.get("role") != gate.METHOD_ROLES[method_id]:
        raise AssemblyError(f"known method {method_id} identity/role changed")
    if "cells" in row and list(row.get("cells", ())) != list(gate.CELL_ORDER):
        raise AssemblyError(f"known method {method_id} cell order changed")
    expected_counts = {cell: gate.RECORDS_PER_CELL for cell in gate.CELL_ORDER}
    if "records_per_cell" in row and (not isinstance(row.get("records_per_cell"), Mapping) or dict(row["records_per_cell"]) != expected_counts):
        raise AssemblyError(f"known method {method_id} records-per-cell changed")
    if not isinstance(row.get("loader"), Mapping) or not isinstance(row.get("resources"), Mapping):
        raise AssemblyError(f"known method {method_id} loader/resources are absent")
    _truth_free(row, description=f"known method {method_id}")
    state = _bound_record(row["state"], description=f"known state {method_id}")
    resources: dict[str, Any] = {}
    for name in gate.METHOD_RESOURCE_REQUIREMENTS[method_id]:
        if name not in row["resources"]:
            raise AssemblyError(f"known method {method_id} resource is absent: {name}")
        resources[name] = _bound_record(row["resources"][name], description=f"known {method_id} resource {name}")
    if "base_decoder_state" in resources:
        _records_equal(resources["base_decoder_state"], state, description=f"known {method_id} base/state")
    loader = deepcopy(dict(row["loader"]))
    if method_id != gate.A1_A2_METHOD_ID:
        expected = {
            "interface": gate.LOADER_INTERFACE,
            "current_h_only": True,
            "full_vocabulary": True,
            "history_enabled": False,
            "a2_enabled": False,
        }
        for key, value in expected.items():
            actual = loader.get(key)
            matches = actual is value if isinstance(value, bool) else actual == value
            if not matches:
                raise AssemblyError(f"known method {method_id} loader semantics changed: {key}")
        if not isinstance(loader.get("module"), str) or not isinstance(loader.get("function"), str):
            raise AssemblyError(f"known method {method_id} loader module/function is absent")
    else:
        if loader.get("interface") != gate.A1_A2_LOADER_INTERFACE or loader.get("candidate_k") != 256:
            raise AssemblyError("A1+A2 loader semantics changed")
    return {
        "id": method_id,
        "role": gate.METHOD_ROLES[method_id],
        "cells": list(gate.CELL_ORDER),
        "records_per_cell": expected_counts,
        "state": state,
        "loader": loader,
        "resources": resources,
    }


def _b1_row(template_row: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(template_row))
    b1 = _bound_record({"path": str(B1_STATE), "bytes": B1_STATE.stat().st_size, "sha256": B1_STATE_SHA256}, description="expanded fixed B1 state")
    embedding = row["resources"]["public_embedding_table"]
    embedding = _bound_record(embedding, description="expanded fixed public E")
    row.update({
        "id": gate.EXPANDED_FIXED_METHOD_ID,
        "role": gate.METHOD_ROLES[gate.EXPANDED_FIXED_METHOD_ID],
        "cells": list(gate.CELL_ORDER),
        "records_per_cell": {cell: gate.RECORDS_PER_CELL for cell in gate.CELL_ORDER},
        "state": b1,
        "resources": {"base_decoder_state": b1, "public_embedding_table": embedding},
        "freeze_status": "BOUND_KNOWN_B1_RECEIPT; FINAL_DESIGN_NOT_FROZEN",
        "loader": {
            "interface": gate.LOADER_INTERFACE,
            "module": "scripts.trr0010_p09_fixed_loader",
            "function": "load_p09_fixed_state",
            "current_h_only": True,
            "full_vocabulary": True,
            "history_enabled": False,
            "a2_enabled": False,
            "kwargs": {
                "hidden_size": 2048,
                "vocabulary_size": gate.VOCABULARY_SIZE,
                "context_width": 128,
                "bottleneck_size": 512,
                "expected_state_sha256": B1_STATE_SHA256,
                "expected_selected_step": B1_SELECTED_STEP,
                "expected_bank_manifest_sha256": B1_BANK_MANIFEST_SHA256,
                "expected_fit_manifest_sha256": B1_FIT_MANIFEST_SHA256,
                "expected_schedule_semantic_sha256": B1_SCHEDULE_SHA256,
                "expected_base_state_sha256": BASE_STATE_SHA256,
                "expected_embedding_sha256": EMBEDDING_SHA256,
                "expected_runner_state_sha256": B1_RUNNER_STATE_SHA256,
            },
            "path_args": {},
            "tensor_args": {},
        },
    })
    return _normalise_known_row(row, method_id=gate.EXPANDED_FIXED_METHOD_ID)


def _a1a2_path_args(row: dict[str, Any]) -> None:
    """Upgrade the template's snapshot string to the registration descriptor."""
    descriptor_path = ROOT / "experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    snapshot = descriptor["model_snapshot"]
    reference = descriptor["reference_binding"]
    row["loader"]["path_args"] = {
        "snapshot": {
            "path": snapshot["path"],
            "readonly": True,
            "files": snapshot["files"],
        },
        "reference_path": _bound_record(reference, description="A1+A2 reference helper"),
    }


def _load_directional(path: Path, *, method_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"directional descriptor is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise AssemblyError(f"directional descriptor must be an object: {path}")
    row = payload.get("method_row", payload)
    if not isinstance(row, Mapping):
        raise AssemblyError(f"directional descriptor method_row is absent: {path}")
    if row.get("id") != method_id or row.get("role") != gate.METHOD_ROLES[method_id]:
        raise AssemblyError(f"directional descriptor identity/role changed: {path}")
    if payload.get("status") not in ("SELECTED_DIRECTIONAL_DEPLOYMENT", "SELECTED_DIRECTIONAL_METHOD"):
        raise AssemblyError(f"directional descriptor is not a selected deployment: {path}")
    _truth_free(payload, description=f"directional descriptor {path}")
    missing = sorted(REQUIRED_METHOD_FIELDS - set(row))
    if missing:
        raise AssemblyError(f"directional descriptor is missing fields {missing}: {path}")
    result = _normalise_known_row(row, method_id=method_id)
    if set(result["resources"]) != {"effective_readout_w", "base_decoder_state"}:
        raise AssemblyError(f"directional descriptor resources are incomplete: {path}")
    return result


def _cli_contract() -> dict[str, Any]:
    required = list(inspect.signature(register.build_registration).parameters)
    expected = [
        "repository_root", "design_path", "selection_path", "panel_path",
        "observation_manifest_path", "capture_path", "timing_plan_path", "method_rows",
    ]
    if required[: len(expected)] != expected:
        raise AssemblyError(f"registration signature changed: {required}")
    help_commands = [
        ["python3", "scripts/trr0010_eval_register.py", "--help"],
        ["python3", "scripts/trr0010_eval_runner.py", "--help"],
    ]
    help_checks = []
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:src:scripts"
    for command in help_commands:
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssemblyError(f"CLI help failed: {' '.join(command)}: {completed.stderr.strip()}")
        help_checks.append({
            "command": "env PYTHONPATH=.:src:scripts " + " ".join(command),
            "status": "PASS",
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        })
    return {
        "registration_help": "env PYTHONPATH=.:src:scripts python3 scripts/trr0010_eval_register.py --help",
        "registration_invocation": "env PYTHONPATH=.:src:scripts python3 scripts/trr0010_eval_register.py --payload <registration_payload.json>",
        "registration_required_build_registration_parameters": expected,
        "registration_optional_build_registration_parameters": required[len(expected):],
        "prediction_help": "env PYTHONPATH=.:src:scripts python3 scripts/trr0010_eval_runner.py --help",
        "prediction_invocation": "env PYTHONPATH=.:src:scripts python3 scripts/trr0010_eval_runner.py --repository-root <repo> --registration <registration.json> --device cuda",
        "prediction_parser_flags": ["--repository-root", "--registration", "--device", "--allow-stale-head"],
        "help_checks": help_checks,
        "registration_payload_warning": "The assembly output is not a registration payload; selection, panel, observations, capture, timing, frequency references, and all six frozen rows are required before registration.",
    }


def assemble(*, current_directional: Path | None, expanded_directional: Path | None, output: Path, emit_pending: bool) -> dict[str, Any]:
    if (current_directional is None) != (expanded_directional is None) and not emit_pending:
        raise AssemblyError("both directional descriptors are required together")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    methods = template.get("methods")
    if not isinstance(methods, Mapping):
        raise AssemblyError("design template method map is absent")
    known = {
        gate.UNCHANGED_METHOD_ID: _normalise_known_row(methods[gate.UNCHANGED_METHOD_ID], method_id=gate.UNCHANGED_METHOD_ID),
        gate.CURRENT_FIXED_METHOD_ID: _normalise_known_row(methods[gate.CURRENT_FIXED_METHOD_ID], method_id=gate.CURRENT_FIXED_METHOD_ID),
        gate.A1_A2_METHOD_ID: _normalise_known_row(methods[gate.A1_A2_METHOD_ID], method_id=gate.A1_A2_METHOD_ID),
    }
    _a1a2_path_args(known[gate.A1_A2_METHOD_ID])
    known[gate.EXPANDED_FIXED_METHOD_ID] = _b1_row(methods[gate.EXPANDED_FIXED_METHOD_ID])
    pending = [gate.CURRENT_DIRECTIONAL_METHOD_ID, gate.EXPANDED_DIRECTIONAL_METHOD_ID]
    assembled = None
    if current_directional is not None and expanded_directional is not None:
        assembled = {
            gate.CURRENT_DIRECTIONAL_METHOD_ID: _load_directional(current_directional, method_id=gate.CURRENT_DIRECTIONAL_METHOD_ID),
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: _load_directional(expanded_directional, method_id=gate.EXPANDED_DIRECTIONAL_METHOD_ID),
        }
        pending = []
    rows = {**known, **(assembled or {})}
    payload: dict[str, Any] = {
        "schema": "token-reconstruction.trr0010-method-row-assembly.v1",
        "task_id": gate.TASK_ID,
        "status": "READY_FOR_DESIGN_BINDING" if not pending else "PENDING_DIRECTIONAL_SELECTED_DESCRIPTORS",
        "registration_ready": not pending,
        "truth_opened": False,
        "source_selection_run": False,
        "capture_run": False,
        "prediction_run": False,
        "rows_are_registration_inputs": False,
        "design_template": {
            "path": str(TEMPLATE),
            "sha256": _sha256(TEMPLATE),
        },
        "method_order": METHOD_ORDER,
        "known_method_ids": [gate.UNCHANGED_METHOD_ID, gate.CURRENT_FIXED_METHOD_ID, gate.EXPANDED_FIXED_METHOD_ID, gate.A1_A2_METHOD_ID],
        "pending_method_ids": pending,
        "methods": rows,
        "directional_descriptor_contract": {
            "required_input": "JSON object with method_row or a method-row object",
            "required_ids": [gate.CURRENT_DIRECTIONAL_METHOD_ID, gate.EXPANDED_DIRECTIONAL_METHOD_ID],
            "required_loader_interface": gate.LOADER_INTERFACE,
            "required_flags": {"current_h_only": True, "full_vocabulary": True, "history_enabled": False, "a2_enabled": False},
            "required_resources": ["base_decoder_state", "effective_readout_w"],
            "state_and_base_must_match": True,
            "all_bound_files_rehashed": True,
            "truth_free": True,
        },
        "cli_contract": _cli_contract(),
        "known_b1_header_receipt": {
            "path": "experiments/TRR-0010/setup/final_evaluation_b1_loader_cpu_v1.json",
            "status": "PASS_CPU_STRICT_P09_HEADER",
            "checkpoint_sha256": B1_STATE_SHA256,
        },
        "a2_publication": {
            "publication_commit": "02fa090140cc3c55172225122eb4d7649e995951",
            "final_manifest_sha256_prefix": "6371426c",
            "final_manifest_sha256_complete": False,
        },
        "exclusions": [
            "No source-selection or source-text read",
            "No activation capture or model forward",
            "No truth materialization/opening/scoring",
            "No registration/prediction runner invocation",
        ],
    }
    output = output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise AssemblyError(f"assembly output is create-only: {output}")
    output.relative_to((ROOT / "experiments" / gate.TASK_ID).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-directional", type=Path)
    parser.add_argument("--expanded-directional", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-pending", action="store_true", help="emit the four-known-row assembly while both directional rows remain pending")
    args = parser.parse_args(argv)
    try:
        payload = assemble(
            current_directional=args.current_directional,
            expanded_directional=args.expanded_directional,
            output=args.output,
            emit_pending=args.emit_pending,
        )
    except (AssemblyError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": payload["status"], "output": str(args.output), "pending_method_ids": payload["pending_method_ids"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
