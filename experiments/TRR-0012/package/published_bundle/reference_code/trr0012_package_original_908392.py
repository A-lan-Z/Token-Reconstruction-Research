"""Build and validate the small, portable TRR-0012 inference package.

The producer side can read an already-opened public observation file to make a
two-record smoke fixture.  It never reads source text or token IDs.  The
consumer-side package is validated through relative, hash-bound descriptors;
the selected decoder states and expected smoke outputs are deliberately left
unbound until the new fit has selected its checkpoints.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


TASK_ID = "TRR-0012"
CONTRACT_SCHEMA = "token-reconstruction.trr0012-fixed-readout-package.v1"
SMOKE_SCHEMA = "token-reconstruction.trr0012-public-smoke.v1"
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
STORED_SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
SMOKE_ROWS = 2
SMOKE_DOMAINS = ("finance", "pile")
SMOKE_KEYS = ("activations", "attention_mask", "position_ids")
PACKAGE_METHODS = ("current_fixed", "expanded_fixed")
PACKAGE_COMPONENTS = frozenset(
    {
        "current_fixed",
        "expanded_fixed",
        "public_readout",
        "namespace_init",
        "decoder_dependency",
        "decoder_support_access",
        "decoder_support_io",
        "decoder_support_public_prefix",
        "decoder_code",
        "loader_code",
        "package_cli",
        "frozen_config",
        "package_manifest",
        "selection_receipt",
        "smoke_input",
        "smoke_expected",
        "identity_current_fixed",
        "identity_expanded_fixed",
        "identity_public_readout",
    }
)
SMOKE_RECORD_ORDER = tuple(
    f"{domain}/public_base/{row:03d}"
    for domain in ("finance", "pile")
    for row in range(SMOKE_ROWS)
)
SMOKE_RECORD_IDENTITIES = {
    "finance/public_base/000": {
        "record_id": "Josephgflowers/Finance-Instruct-500k/train@583a98fb0ec14d904e9423b671d9d0fea88891b6:row-012278",
        "public_record_sha256": "57fa99cb426d5d0f6d47881bd377a08ac25a69696494652279e6fc7725427ea5",
        "tensor_digests": {
            "activation": "0dbf9982aad7befb717e71a73e396a8e875a581718017709cfaa29853e0614d8",
            "attention_mask": "ad2111cafc0f27f11f264b746fa13cff0f6977bcc1181f396a041bb698767fb1",
            "position_ids": "1487c05e9daf222f65b5e32ca1f838c7312f6ded9d970613e26737cdeb659aed",
        },
    },
    "finance/public_base/001": {
        "record_id": "Josephgflowers/Finance-Instruct-500k/train@583a98fb0ec14d904e9423b671d9d0fea88891b6:row-014057",
        "public_record_sha256": "5b7929fd81f0ff5fbec708e1841a393babe33e58ece8efa4cfdfee7b71314f66",
        "tensor_digests": {
            "activation": "8cf446f5ac1bb325dc045b9eb2cb30ec523bf475331caa5928df08295641e2d1",
            "attention_mask": "ad2111cafc0f27f11f264b746fa13cff0f6977bcc1181f396a041bb698767fb1",
            "position_ids": "1487c05e9daf222f65b5e32ca1f838c7312f6ded9d970613e26737cdeb659aed",
        },
    },
    "pile/public_base/000": {
        "record_id": "NeelNanda/pile-10k/train@127bfedcd5047750df5ccf3a12979a47bfa0bafa:row-007721",
        "public_record_sha256": "d01a69e865d706dfc667e10b624021d10279087f63a9a4f8655e3f448063a5e6",
        "tensor_digests": {
            "activation": "9a053be4cf92a1a3542da1c485d457ebd977cf7610b1b4595e39e498251e51ed",
            "attention_mask": "ad2111cafc0f27f11f264b746fa13cff0f6977bcc1181f396a041bb698767fb1",
            "position_ids": "1487c05e9daf222f65b5e32ca1f838c7312f6ded9d970613e26737cdeb659aed",
        },
    },
    "pile/public_base/001": {
        "record_id": "NeelNanda/pile-10k/train@127bfedcd5047750df5ccf3a12979a47bfa0bafa:row-008703",
        "public_record_sha256": "993a8875128088946c08a1be5c90ea48c32e9487782ecd0c7bc6567dc1eca399",
        "tensor_digests": {
            "activation": "0bd4bdcbf8b2a37ced41fab7beb9521be3c0986e71a6563048c7182172779bb9",
            "attention_mask": "ad2111cafc0f27f11f264b746fa13cff0f6977bcc1181f396a041bb698767fb1",
            "position_ids": "1487c05e9daf222f65b5e32ca1f838c7312f6ded9d970613e26737cdeb659aed",
        },
    },
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PackageError(RuntimeError):
    """Raised when a package contract is malformed or fails closed."""


def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"regular file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    """Digest tensor metadata followed by contiguous CPU bytes."""

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


def _file_record(path: Path, *, package_root: Path) -> dict[str, Any]:
    package_root = Path(package_root).resolve()
    raw = Path(path)
    try:
        relative = raw.relative_to(package_root)
    except ValueError as exc:
        raise PackageError(f"file is outside package root: {raw}") from exc
    current = package_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PackageError(f"package file traverses a symlink: {current}")
    if raw.is_symlink() or not raw.is_file():
        raise PackageError(f"package file must be a regular file: {relative}")
    return {
        "path": relative.as_posix(),
        "bytes": raw.stat().st_size,
        "sha256": sha256_file(raw),
        "readonly": True,
    }


def _safe_relative(path: str, *, label: str) -> Path:
    if not isinstance(path, str) or not path:
        raise PackageError(f"{label} path is missing")
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PackageError(f"{label} path must be package-relative")
    if any(part.lower() in {"tmp", "temp", "temporary"} for part in candidate.parts):
        raise PackageError(f"{label} path may not reference temporary storage")
    return candidate


def _binding(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise PackageError(f"{key} binding is missing")
    return value


def _validate_token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != STORED_SEQUENCE_TOKENS:
        raise PackageError(f"{label} must contain exactly 128 token IDs")
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= VOCABULARY_SIZE:
            raise PackageError(f"{label}[{index}] is outside vocabulary range")
        result.append(int(token))
    if result[0] != BOS_TOKEN_ID:
        raise PackageError(f"{label}[0] must be the fixed BOS token")
    return result


def _validate_smoke_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(SMOKE_RECORD_ORDER):
        raise PackageError("smoke record count changed")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping) or item.get("slot") != SMOKE_RECORD_ORDER[index]:
            raise PackageError("smoke order must be Finance rows 0-1 then Pile rows 0-1")
        expected = SMOKE_RECORD_IDENTITIES[SMOKE_RECORD_ORDER[index]]
        if item.get("record_id") != expected["record_id"] or item.get("public_record_sha256") != expected["public_record_sha256"]:
            raise PackageError(f"smoke source identity changed: {SMOKE_RECORD_ORDER[index]}")
        digests = item.get("tensor_digests")
        if digests != expected["tensor_digests"]:
            raise PackageError(f"smoke tensor slice identity changed: {SMOKE_RECORD_ORDER[index]}")
        normalized.append(dict(item))
    return normalized


def _validate_selection_receipt(
    package_root: Path,
    binding: Mapping[str, Any],
    checked_components: Mapping[str, Mapping[str, Any]],
    *,
    required: bool,
) -> dict[str, Any] | None:
    path = package_root / _safe_relative(binding.get("path"), label="selection receipt")
    if not path.is_file() or path.is_symlink():
        if required:
            raise PackageError("selection receipt is unavailable")
        return None
    receipt = _json_object(path, label="selection receipt")
    if receipt.get("status") != "SELECTION_COMPLETE_BEFORE_SMOKE":
        raise PackageError("selection receipt is not complete-before-smoke")
    for key, expected in {
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "checkpoint_reselection_after_smoke": False,
    }.items():
        if receipt.get(key) is not expected:
            raise PackageError(f"selection receipt boundary changed: {key}")
    if not isinstance(receipt.get("development_labels_used"), bool):
        raise PackageError("selection receipt development_labels_used is missing")
    selected_methods = receipt.get("selected_methods")
    if not isinstance(selected_methods, Mapping) or set(selected_methods) != set(PACKAGE_METHODS):
        raise PackageError("selection receipt must bind both new methods")
    for method_id in PACKAGE_METHODS:
        selected = selected_methods[method_id]
        if not isinstance(selected, Mapping):
            raise PackageError(f"selection receipt method is malformed: {method_id}")
        if not isinstance(selected.get("model_id"), str) or not selected["model_id"]:
            raise PackageError(f"selection receipt model id is missing: {method_id}")
        if not selected["model_id"].startswith("TRR-0012/"):
            raise PackageError(f"selection receipt model id is not a new TRR-0012 identity: {method_id}")
        if isinstance(selected.get("selected_step"), bool) or not isinstance(selected.get("selected_step"), int) or selected["selected_step"] < 0:
            raise PackageError(f"selection receipt selected step is malformed: {method_id}")
        expected_sha = checked_components[method_id].get("sha256")
        if expected_sha is not None and selected.get("state_sha256") != expected_sha:
            raise PackageError(f"selection receipt state hash differs: {method_id}")
    return receipt


def validate_manifest(manifest_path: Path, *, package_root: Path) -> dict[str, Any]:
    """Validate the portable package without importing a model or loading tensors."""

    manifest_path = Path(manifest_path).resolve()
    package_root = Path(package_root).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid package manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise PackageError("package manifest must be an object")
    if manifest.get("schema") != CONTRACT_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise PackageError("package schema or task identity changed")
    status = manifest.get("status")
    if status not in {"DRAFT_PRE_SELECTION", "FROZEN_SELECTED_PACKAGE"}:
        raise PackageError("unknown package status")
    if manifest.get("immutable_after_selection") is not True:
        raise PackageError("package immutability flag is absent")
    if manifest.get("consumer_paths_package_relative") is not True:
        raise PackageError("consumer path boundary is absent")

    required = manifest.get("required_components")
    if not isinstance(required, Mapping) or set(required) != set(PACKAGE_COMPONENTS):
        missing = sorted(set(PACKAGE_COMPONENTS) - set(required or {}))
        extra = sorted(set(required or {}) - set(PACKAGE_COMPONENTS))
        raise PackageError(f"required component inventory changed; missing={missing}, extra={extra}")
    checked: dict[str, Any] = {}
    for name in sorted(PACKAGE_COMPONENTS):
        binding = required[name]
        if not isinstance(binding, Mapping):
            raise PackageError(f"component binding is malformed: {name}")
        path = _safe_relative(binding.get("path"), label=f"component {name}")
        if binding.get("readonly") is not True:
            raise PackageError(f"component is not marked readonly: {name}")
        if name == "current_fixed" and binding.get("bank") != "B0":
            raise PackageError("current_fixed must bind bank B0")
        if name == "expanded_fixed" and binding.get("bank") != "B1":
            raise PackageError("expanded_fixed must bind bank B1")
        must_bind = status == "FROZEN_SELECTED_PACKAGE" or binding.get("required_before_selection") is True or binding.get("bound_before_selection") is True
        resolved = package_root / path
        if must_bind:
            if not resolved.is_file() or resolved.is_symlink():
                raise PackageError(f"component unavailable: {name}")
            actual = _file_record(resolved, package_root=package_root)
            for field in ("bytes", "sha256"):
                if binding.get(field) != actual[field]:
                    raise PackageError(f"component {name} {field} changed")
            checked[name] = actual
        else:
            if binding.get("bytes") is not None or binding.get("sha256") is not None:
                raise PackageError(f"unselected component has premature identity: {name}")
            checked[name] = {"path": path.as_posix(), "status": "PENDING_SELECTION", "bytes": None, "sha256": None}

    smoke = manifest.get("smoke")
    if not isinstance(smoke, Mapping) or smoke.get("schema") != SMOKE_SCHEMA:
        raise PackageError("smoke contract is absent or changed")
    if smoke.get("truth_free") is not True or smoke.get("source_text_loaded") is not False or smoke.get("token_ids_loaded") is not False:
        raise PackageError("smoke truth boundary is not closed")
    if smoke.get("smoke_used_for_selection") is not False or smoke.get("independent_evaluation_truth_opened") is not False:
        raise PackageError("smoke selection/evaluation boundary is not closed")
    if list(smoke.get("record_order", ())) != list(SMOKE_RECORD_ORDER):
        raise PackageError("smoke record order changed")
    if list(smoke.get("methods", ())) != list(PACKAGE_METHODS):
        raise PackageError("smoke method order changed")
    fixture = _binding(smoke, "fixture")
    fixture_path = _safe_relative(fixture.get("path"), label="smoke fixture")
    fixture_file = package_root / fixture_path
    if not fixture_file.is_file() or fixture_file.is_symlink():
        raise PackageError("smoke fixture is unavailable")
    fixture_actual = _file_record(fixture_file, package_root=package_root)
    if fixture.get("bytes") != fixture_actual["bytes"] or fixture.get("sha256") != fixture_actual["sha256"]:
        raise PackageError("smoke fixture binding changed")
    records = _validate_smoke_records(smoke.get("records"))

    expected_outputs = smoke.get("expected_outputs")
    if status == "FROZEN_SELECTED_PACKAGE" and expected_outputs is None:
        raise PackageError("selected package expected outputs are missing")
    if expected_outputs is not None:
        if not isinstance(expected_outputs, Mapping) or set(expected_outputs) != set(SMOKE_RECORD_ORDER):
            raise PackageError("smoke expected outputs must cover all four ordered records")
        for slot in SMOKE_RECORD_ORDER:
            slot_outputs = expected_outputs[slot]
            if not isinstance(slot_outputs, Mapping) or set(slot_outputs) != set(PACKAGE_METHODS):
                raise PackageError(f"smoke outputs must include both methods: {slot}")
            for method_id in PACKAGE_METHODS:
                output = slot_outputs[method_id]
                if not isinstance(output, Mapping):
                    raise PackageError(f"smoke output is malformed: {slot}/{method_id}")
                _validate_token_ids(output.get("token_ids"), label=f"smoke output {slot}/{method_id}")
                if _SHA256.fullmatch(str(output.get("tensor_sha256"))) is None:
                    raise PackageError(f"smoke output digest is missing: {slot}/{method_id}")

    selection = manifest.get("selection_boundary")
    if not isinstance(selection, Mapping):
        raise PackageError("selection boundary is absent")
    if status == "FROZEN_SELECTED_PACKAGE":
        for key, expected in {
            "complete_before_smoke": True,
            "smoke_used_for_selection": False,
            "independent_evaluation_truth_opened": False,
            "checkpoint_reselection_after_smoke": False,
        }.items():
            if selection.get(key) is not expected:
                raise PackageError(f"selection boundary changed: {key}")
        if not isinstance(selection.get("development_labels_used"), bool):
            raise PackageError("selection development_labels_used must be explicit")
        receipt = _validate_selection_receipt(package_root, required["selection_receipt"], checked, required=True)
    else:
        receipt = _validate_selection_receipt(package_root, required["selection_receipt"], checked, required=False)

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "components": checked,
        "smoke_fixture": fixture_actual,
        "smoke_records": records,
        "selection_receipt": receipt,
        "status": status,
    }


def _load_selection_records(selection_path: Path) -> dict[str, list[dict[str, str]]]:
    """Read only identity metadata from a frozen public selection ledger."""

    try:
        payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read public selection metadata: {selection_path}") from exc
    records = payload.get("selection_rule", {}).get("records", {})
    result: dict[str, list[dict[str, str]]] = {}
    for domain in SMOKE_DOMAINS:
        rows = records.get(domain)
        if not isinstance(rows, list) or len(rows) < SMOKE_ROWS:
            raise PackageError(f"selection metadata lacks two {domain} rows")
        result[domain] = []
        for row in rows[:SMOKE_ROWS]:
            if not isinstance(row, Mapping):
                raise PackageError("selection record is malformed")
            record_id = row.get("record_id")
            public_record_sha = row.get("public_record_sha256")
            if not isinstance(record_id, str) or not isinstance(public_record_sha, str) or _SHA256.fullmatch(public_record_sha) is None:
                raise PackageError(f"selection identity is malformed: {domain}")
            result[domain].append(
                {"record_id": record_id, "public_record_sha256": public_record_sha}
            )
    return result


def extract_smoke(
    *,
    source_dir: Path,
    selection_path: Path,
    output_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Copy two rows per public-base domain into a small fixture.

    This function reads only H, attention-mask, and position-ID slices from
    the already-opened public observation files.  It does not read token IDs,
    source text, labels, or target weights.
    """

    source_dir = Path(source_dir).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise PackageError(f"smoke fixture is create-only: {output_path}")
    if _SHA256.fullmatch(source_manifest_sha256) is None:
        raise PackageError("source manifest SHA-256 is malformed")
    identities = _load_selection_records(selection_path)
    tensors: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    source_files: dict[str, Any] = {}
    expected_positions = torch.arange(STORED_SEQUENCE_TOKENS, dtype=torch.long)
    for domain in SMOKE_DOMAINS:
        source = source_dir / f"{domain}__public_base.safetensors"
        source_files[domain] = {
            "label": f"TRR-0010 public_base {domain}",
            "bytes": source.stat().st_size if source.is_file() else None,
            "sha256": sha256_file(source),
        }
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(SMOKE_KEYS):
                raise PackageError(f"{domain} observation keys changed: {sorted(keys)}")
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in SMOKE_KEYS}
            if shapes["activations"] != (128, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
                raise PackageError(f"{domain} activation geometry changed")
            if shapes["attention_mask"] != (128, STORED_SEQUENCE_TOKENS) or shapes["position_ids"] != (128, STORED_SEQUENCE_TOKENS):
                raise PackageError(f"{domain} sidecar geometry changed")
            activation = handle.get_slice("activations")[:SMOKE_ROWS].contiguous()
            mask = handle.get_slice("attention_mask")[:SMOKE_ROWS].contiguous()
            positions = handle.get_slice("position_ids")[:SMOKE_ROWS].to(torch.long).contiguous()
        if activation.dtype != torch.bfloat16 or not torch.isfinite(activation.float()).all().item():
            raise PackageError(f"{domain} activation dtype/finiteness changed")
        mask_bool = mask.to(torch.bool).contiguous()
        if not bool(mask_bool.all().item()) or not torch.equal(positions, expected_positions.repeat(SMOKE_ROWS, 1)):
            raise PackageError(f"{domain} smoke mask or positions changed")
        tensors[f"{domain}__activations"] = activation
        tensors[f"{domain}__attention_mask"] = mask_bool
        tensors[f"{domain}__position_ids"] = positions
        for row in range(SMOKE_ROWS):
            records.append(
                {
                    "slot": f"{domain}/public_base/{row:03d}",
                    "domain": domain,
                    "row": row,
                    "record_id": identities[domain][row]["record_id"],
                    "public_record_sha256": identities[domain][row]["public_record_sha256"],
                    "tensor_digests": {
                        "activation": tensor_digest(activation[row]),
                        "attention_mask": tensor_digest(mask_bool[row]),
                        "position_ids": tensor_digest(positions[row]),
                    },
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_path),
        metadata={
            "schema": SMOKE_SCHEMA,
            "task_id": TASK_ID,
            "source_manifest_sha256": source_manifest_sha256,
            "source_capture": "TRR-0010/public_capture_watchdog_r5/observations_v1",
            "truth_opened": "false",
            "source_text_loaded": "false",
            "token_ids_loaded": "false",
        },
    )
    return {
        "schema": SMOKE_SCHEMA,
        "task_id": TASK_ID,
        "status": "SMOKE_FIXTURE_EXTRACTED",
        "fixture": {
            "path": output_path.as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "source_manifest_sha256": source_manifest_sha256,
        "source_files": source_files,
        "records": records,
        "geometry": {
            "domains": list(SMOKE_DOMAINS),
            "rows_per_domain": SMOKE_ROWS,
            "activation_shape_per_row": [STORED_SEQUENCE_TOKENS, HIDDEN_SIZE],
            "activation_dtype": "torch.bfloat16",
            "attention_mask_shape_per_row": [STORED_SEQUENCE_TOKENS],
            "position_ids_shape_per_row": [STORED_SEQUENCE_TOKENS],
            "bos_token_id": BOS_TOKEN_ID,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
    }



PACKAGE_OBSERVATION_KEYS = frozenset(("activations", "attention_mask", "position_ids"))
PACKAGE_STATE_TENSOR_KEYS = frozenset(
    {
        "base.W",
        "base.b",
        "base.key.bias",
        "base.key.weight",
        "base.output.bias",
        "base.output.weight",
        "base.query.bias",
        "base.query.weight",
        "base.s",
        "base.value.bias",
        "base.value.weight",
        "down.bias",
        "down.weight",
        "up.bias",
        "up.weight",
    }
)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be a JSON object: {path}")
    return value


def _regular_package_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"{label} is not a regular file: {path}")
    return path


def _reject_symlink_ancestors(path: Path, *, root: Path, label: str) -> None:
    root = Path(root)
    current = root
    try:
        relative = Path(path).relative_to(root)
    except ValueError as exc:
        raise PackageError(f"{label} escapes its root") from exc
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PackageError(f"{label} traverses a symlink: {current}")


def _package_relative_file(root: Path, value: Any, *, label: str) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path", value.get("relative_path"))
    relative = _safe_relative(value, label=label)
    root = Path(root).resolve()
    path = root / relative
    _reject_symlink_ancestors(path, root=root, label=label)
    if not path.is_file():
        raise PackageError(f"{label} is not a regular file: {path}")
    return _regular_package_file(path, label=label)


def _descriptor_path(root: Path) -> Path:
    candidates = (
        Path(root) / "package_manifest.json",
        Path(root) / "config" / "frozen_config.json",
        Path(root) / "config" / "decoder.json",
    )
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise PackageError("package descriptor is missing (package_manifest.json or config/decoder.json)")


def _load_package_descriptor(root: Path) -> tuple[dict[str, Any], Path]:
    root = Path(root).resolve()
    descriptor_path = _descriptor_path(root)
    descriptor = _json_object(descriptor_path, label="package descriptor")
    if descriptor.get("task_id") not in {None, TASK_ID}:
        raise PackageError("package descriptor task identity changed")
    if descriptor.get("consumer_paths_package_relative") is False:
        raise PackageError("package descriptor permits non-relative consumer paths")
    return descriptor, descriptor_path


def _method_descriptor(descriptor: Mapping[str, Any], method_id: str) -> tuple[Any, dict[str, Any], Any]:
    methods = descriptor.get("methods")
    if not isinstance(methods, Mapping) or set(PACKAGE_METHODS) - set(methods):
        raise PackageError("package descriptor does not bind both current_fixed and expanded_fixed")
    method = methods.get(method_id)
    if not isinstance(method, Mapping):
        raise PackageError(f"package descriptor method is malformed: {method_id}")
    state_value = method.get("state_path", method.get("state"))
    loader_value = method.get("loader_kwargs")
    if loader_value is None and isinstance(method.get("loader"), Mapping):
        loader_value = method["loader"].get("kwargs")
    if not isinstance(loader_value, Mapping):
        raise PackageError(f"package descriptor loader kwargs are missing: {method_id}")
    # Readout is shared in the package, but accepting a method-local binding
    # keeps this compatible with the final restore descriptor.
    readout_value = method.get("readout_path", method.get("public_readout"))
    if readout_value is None:
        readout_value = descriptor.get("readout", {}).get("path") if isinstance(descriptor.get("readout"), Mapping) else None
    if state_value is None or readout_value is None:
        raise PackageError(f"package descriptor paths are missing: {method_id}")
    return state_value, dict(loader_value), readout_value


def _check_optional_file_binding(path: Path, binding: Mapping[str, Any] | None, *, label: str) -> None:
    if not isinstance(binding, Mapping):
        return
    if binding.get("bytes") is not None and int(binding["bytes"]) != path.stat().st_size:
        raise PackageError(f"{label} byte binding changed")
    if binding.get("sha256") is not None and str(binding["sha256"]) != sha256_file(path):
        raise PackageError(f"{label} SHA-256 binding changed")


def _resolve_cli_input(root: Path, value: Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path(root) / _safe_relative(raw.as_posix(), label=label)
    resolved = raw.resolve()
    if any(part.lower() in {"tmp", "temp", "temporary"} for part in resolved.parts):
        raise PackageError(f"{label} may not use temporary storage")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise PackageError(f"{label} traverses a symlink: {current}")
    return _regular_package_file(raw, label=label)


def _observation_batch(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Load standard N-row observations or the bundled two-domain smoke fixture."""

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if PACKAGE_OBSERVATION_KEYS.issubset(keys):
                activations = handle.get_tensor("activations").contiguous()
                masks = handle.get_tensor("attention_mask").contiguous()
                positions = handle.get_tensor("position_ids").to(torch.long).contiguous()
                slots = [f"record/{index:03d}" for index in range(int(activations.shape[0]))]
            else:
                expected = {
                    f"{domain}__{key}"
                    for domain in SMOKE_DOMAINS
                    for key in SMOKE_KEYS
                }
                if keys != expected:
                    raise PackageError(
                        "observations must contain activations/attention_mask/position_ids "
                        "or the bundled finance/pile smoke keys"
                    )
                values = []
                mask_values = []
                position_values = []
                slots = []
                for domain in SMOKE_DOMAINS:
                    values.append(handle.get_tensor(f"{domain}__activations").contiguous())
                    mask_values.append(handle.get_tensor(f"{domain}__attention_mask").contiguous())
                    position_values.append(handle.get_tensor(f"{domain}__position_ids").to(torch.long).contiguous())
                    slots.extend(f"{domain}/public_base/{row:03d}" for row in range(int(values[-1].shape[0])))
                activations = torch.cat(values, dim=0)
                masks = torch.cat(mask_values, dim=0)
                positions = torch.cat(position_values, dim=0)
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"cannot read observation tensors: {path}") from exc

    if activations.ndim != 3 or tuple(activations.shape[1:]) != (STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise PackageError("observation activation geometry must be [N,128,2048]")
    if masks.ndim != 2 or tuple(masks.shape) != (int(activations.shape[0]), STORED_SEQUENCE_TOKENS):
        raise PackageError("observation mask geometry must be [N,128]")
    if positions.ndim != 2 or tuple(positions.shape) != (int(activations.shape[0]), STORED_SEQUENCE_TOKENS):
        raise PackageError("observation position geometry must be [N,128]")
    if int(activations.shape[0]) <= 0:
        raise PackageError("observation batch is empty")
    if activations.dtype not in {torch.bfloat16, torch.float32}:
        raise PackageError(f"observation activation dtype changed: {activations.dtype}")
    if not bool(torch.isfinite(activations.float()).all().item()):
        raise PackageError("observation activations contain non-finite values")
    masks = masks.to(torch.bool).contiguous()
    if not bool(masks.any(dim=1).all().item()):
        raise PackageError("an observation has no valid positions")
    expected_positions = torch.arange(STORED_SEQUENCE_TOKENS, dtype=torch.long).expand(int(activations.shape[0]), -1)
    if not torch.equal(positions, expected_positions):
        raise PackageError("fixed-readout position IDs changed from 0..127")
    return activations, masks, positions, slots


def _package_loader(package_root: Path) -> Any:
    code_root = (Path(package_root).resolve() / "code").resolve()
    loader_path = _regular_package_file(code_root / "trr0010_p09_fixed_loader.py", label="package loader")
    existing_namespace = sys.modules.get("token_reconstruction")
    if existing_namespace is not None:
        existing_file = getattr(existing_namespace, "__file__", None)
        if existing_file is None:
            raise PackageError("package-relative decoder namespace has no source path")
        try:
            Path(existing_file).resolve().relative_to(code_root)
        except ValueError as exc:
            raise PackageError("decoder namespace is shadowed by a non-package import") from exc
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    module_name = "_trr0012_vendored_p09_loader"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, loader_path)
        if spec is None or spec.loader is None:
            raise PackageError("cannot construct package loader import")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise PackageError("cannot import package-relative fixed loader") from exc
    load = getattr(module, "load_p09_fixed_state", None)
    if load is None:
        raise PackageError("vendored loader has no load_p09_fixed_state")
    return load


def _load_readout(package_root: Path, descriptor: Mapping[str, Any]) -> tuple[torch.Tensor, Path]:
    readout = descriptor.get("readout")
    if not isinstance(readout, Mapping):
        readout = {}
    value = readout.get("path", "readout/public_normalized_embeddings.safetensors")
    path = _package_relative_file(package_root, value, label="public readout")
    _check_optional_file_binding(path, readout, label="public readout")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embeddings"}:
                raise PackageError("public readout tensor key changed")
            table = handle.get_tensor("embeddings").contiguous()
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError("cannot load public readout") from exc
    if tuple(table.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or table.dtype != torch.float32:
        raise PackageError("public readout geometry/dtype changed")
    if not bool(torch.isfinite(table).all().item()):
        raise PackageError("public readout contains non-finite values")
    return table, path


def _load_package_method(
    package_root: Path,
    descriptor: Mapping[str, Any],
    method_id: str,
    *,
    device: torch.device,
    readout: torch.Tensor,
) -> tuple[torch.nn.Module, str, Path]:
    state_value, loader_kwargs, _readout_value = _method_descriptor(descriptor, method_id)
    state_path = _package_relative_file(package_root, state_value, label=f"{method_id} state")
    method = descriptor["methods"][method_id]
    state_binding = method.get("state") if isinstance(method, Mapping) else None
    _check_optional_file_binding(state_path, state_binding if isinstance(state_binding, Mapping) else None, label=f"{method_id} state")
    expected_state_sha = loader_kwargs.get("expected_state_sha256")
    if expected_state_sha is None:
        loader_kwargs["expected_state_sha256"] = sha256_file(state_path)
    try:
        model = _package_loader(package_root)(state_path, **loader_kwargs)
    except Exception as exc:
        raise PackageError(f"strict package load failed: {method_id}") from exc
    model = model.to(device=device).eval()
    expected_sha = str(loader_kwargs["expected_state_sha256"])
    return model, expected_sha, state_path


def _predict_row_package(
    model: torch.nn.Module,
    readout: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    position: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
        staged_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
        staged_positions = position.to(device=device, dtype=torch.long)
        try:
            projected = model.projected_hidden(staged, staged_mask)
            rows = torch.zeros_like(staged_positions[1:])
            logits = model.logits_from_rows(projected, rows, staged_positions[1:], readout)
        except AttributeError:
            logits_full = model(staged, staged_mask, readout)
            if tuple(logits_full.shape) != (1, STORED_SEQUENCE_TOKENS, VOCABULARY_SIZE):
                raise PackageError("decoder logits geometry changed")
            logits = logits_full[0, 1:]
        if tuple(logits.shape) != (SCORED_POST_BOS_TOKENS, VOCABULARY_SIZE) or not bool(torch.isfinite(logits).all().item()):
            raise PackageError("decoder logits shape/finiteness changed")
        output = torch.empty(STORED_SEQUENCE_TOKENS, dtype=torch.long)
        output[0] = BOS_TOKEN_ID
        output[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
        if bool((output < 0).any().item()) or bool((output >= VOCABULARY_SIZE).any().item()):
            raise PackageError("decoder emitted an out-of-range token id")
        return output


def _ensure_create_only(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise PackageError(f"{label} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _path_label(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "external_cli_input"


def _runtime_dependency_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise PackageError(f"required runtime distribution is unavailable: {distribution}") from exc


def _loaded_file_record(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_package_file(Path(path), label=label).resolve()
    _reject_symlink_ancestors(path, root=Path(root).resolve(), label=label)
    return {
        "relative_path": _path_label(root, path),
        "loaded_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _loaded_decoder_modules(package_root: Path) -> list[dict[str, Any]]:
    """Report every local module imported by the vendored strict decoder."""

    names = (
        "_trr0012_vendored_p09_loader",
        "token_reconstruction",
        "token_reconstruction.access",
        "token_reconstruction.io",
        "token_reconstruction.public_prefix",
        "token_reconstruction.trr0005_joint_decoder",
        "token_reconstruction.trr0007_positionwise",
    )
    records: list[dict[str, Any]] = []
    code_root = Path(package_root).resolve() / "code"
    for name in names:
        module = sys.modules.get(name)
        module_path = getattr(module, "__file__", None) if module is not None else None
        if module_path is None:
            raise PackageError(f"vendored decoder module was not imported: {name}")
        path = Path(module_path)
        try:
            path.resolve().relative_to(code_root.resolve())
        except ValueError as exc:
            raise PackageError(f"vendored decoder module escaped package code root: {name}") from exc
        record = _loaded_file_record(package_root, path, label=f"decoder module {name}")
        record["module"] = name
        records.append(record)
    return records


def _runtime_provenance(package_root: Path, *, device: torch.device, imported_modules: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_descriptor = package_root / "config" / "runtime.json"
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "dependencies": {
            "numpy": _runtime_dependency_version("numpy"),
            "safetensors": _runtime_dependency_version("safetensors"),
            "torch": _runtime_dependency_version("torch"),
        },
        "device": str(device),
        "numeric_profile": "qualified FP32 decoder",
        "imported_modules": imported_modules,
    }
    if runtime_descriptor.is_file() and not runtime_descriptor.is_symlink():
        result["runtime_descriptor"] = _loaded_file_record(
            package_root, runtime_descriptor, label="runtime descriptor"
        )
    return result


TENSOR_IDENTITY_SCHEMA = "token-reconstruction.trr-p11-tensor-identity.v1"


def write_tensor_identity(*, input_path: Path, output_path: Path) -> dict[str, Any]:
    """Create a canonical tensor identity sidecar without importing a model."""

    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_symlink() or not input_path.is_file():
        raise PackageError(f"tensor identity input is not a regular file: {input_path}")
    output_path = _ensure_create_only(Path(output_path).expanduser().resolve(), label="tensor identity sidecar")
    tensors: dict[str, Any] = {}
    try:
        with safe_open(str(input_path), framework="pt", device="cpu") as handle:
            keys = sorted(str(key) for key in handle.keys())
            for key in keys:
                value = handle.get_tensor(key).detach().cpu().contiguous()
                tensors[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": tensor_digest(value),
                }
    except Exception as exc:
        raise PackageError(f"cannot create tensor identity sidecar: {input_path}") from exc
    payload = {
        "schema": TENSOR_IDENTITY_SCHEMA,
        "task_id": TASK_ID,
        "file_sha256": sha256_file(input_path),
        "tensor_keys_sorted": keys,
        "tensors": tensors,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "tensor_keys_sorted": keys,
        "file_sha256": payload["file_sha256"],
    }


def _digest_binding(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PackageError(f"{label} must be a lowercase SHA-256 digest")
    return value


def write_selection_receipt(
    *,
    b0_state: Path,
    b1_state: Path,
    b0_binding: Path,
    b1_binding: Path,
    output_path: Path,
    package_relative_b0: str = "states/current_fixed.safetensors",
    package_relative_b1: str = "states/expanded_fixed.safetensors",
    development_labels_used: bool | None = None,
) -> dict[str, Any]:
    """Bind newly selected state files to an explicit pre-smoke receipt."""

    if development_labels_used is None:
        raise PackageError("development_labels_used must be explicitly supplied")
    states = {
        "current_fixed": (Path(b0_state).expanduser().resolve(), Path(b0_binding).expanduser().resolve(), "B0", package_relative_b0),
        "expanded_fixed": (Path(b1_state).expanduser().resolve(), Path(b1_binding).expanduser().resolve(), "B1", package_relative_b1),
    }
    selected: dict[str, Any] = {}
    for method_id, (state_path, binding_path, bank, package_path) in states.items():
        if state_path.is_symlink() or not state_path.is_file():
            raise PackageError(f"selected {method_id} state is not a regular file")
        binding = _json_object(binding_path, label=f"{method_id} selection binding")
        model_id = binding.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id.startswith("continued_"):
            raise PackageError(f"{method_id} model_id must identify the new replication")
        step = binding.get("selected_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise PackageError(f"{method_id} selected_step is malformed")
        actual_state_sha = sha256_file(state_path)
        declared_state_sha = binding.get("state_sha256")
        if declared_state_sha is not None and _digest_binding(declared_state_sha, label=f"{method_id}.state_sha256") != actual_state_sha:
            raise PackageError(f"{method_id} state hash differs from selection binding")
        selected[method_id] = {
            "bank": bank,
            "model_id": model_id,
            "selected_step": step,
            "state_path": package_path,
            "state_sha256": actual_state_sha,
            "state_bytes": state_path.stat().st_size,
            "bank_manifest_sha256": _digest_binding(binding.get("bank_manifest_sha256"), label=f"{method_id}.bank_manifest_sha256"),
            "fit_manifest_sha256": _digest_binding(binding.get("fit_manifest_sha256"), label=f"{method_id}.fit_manifest_sha256"),
            "schedule_semantic_sha256": _digest_binding(binding.get("schedule_semantic_sha256"), label=f"{method_id}.schedule_semantic_sha256"),
            "runner_state_sha256": _digest_binding(binding.get("runner_state_sha256"), label=f"{method_id}.runner_state_sha256"),
        }
    output_path = _ensure_create_only(Path(output_path).expanduser().resolve(), label="selection receipt")
    payload = {
        "schema": "token-reconstruction.trr0012-selection-receipt.v1",
        "task_id": TASK_ID,
        "status": "SELECTION_COMPLETE_BEFORE_SMOKE",
        "selected_methods": selected,
        "selection_rule": "earliest strict maximum of equal-domain public validation token accuracy; step 0 eligible",
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "checkpoint_reselection_after_smoke": False,
        "development_labels_used": bool(development_labels_used),
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path), "selected_methods": selected}

def predict_package(
    *,
    package_root: Path,
    observations: Path,
    output_path: Path,
    device_name: str = "cpu",
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Run both packaged fixed methods on arbitrary N x 128 x 2048 observations.

    The same vendored loader and FP32 row path are used for the four-record
    smoke and later public panels.  This command emits one int64 prediction
    tensor per method, each [N,128], plus a create-only hash-bound receipt.
    """

    package_root = Path(package_root).expanduser().resolve()
    descriptor, descriptor_path = _load_package_descriptor(package_root)
    selection_path = package_root / "receipts" / "selection_complete_before_smoke.json"
    if not selection_path.is_file() or selection_path.is_symlink():
        raise PackageError("selection receipt must exist before prediction generation")
    selection = _json_object(selection_path, label="selection receipt")
    for key, expected in {
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "checkpoint_reselection_after_smoke": False,
    }.items():
        if selection.get(key) is not expected:
            raise PackageError(f"selection receipt boundary is not frozen: {key}")
    if not isinstance(selection.get("development_labels_used"), bool):
        raise PackageError("selection receipt development_labels_used is missing")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise PackageError("CUDA prediction requested but CUDA is unavailable")
    device = torch.device(device_name)
    observation_path = _resolve_cli_input(package_root, observations, label="observations")
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = package_root / _safe_relative(output_path.as_posix(), label="prediction output")
    output_path = _ensure_create_only(output_path.resolve(), label="prediction output")
    if receipt_path is None:
        receipt_path = output_path.with_suffix(".receipt.json")
    receipt_path = _ensure_create_only(Path(receipt_path).expanduser().resolve(), label="prediction receipt")

    activations, masks, positions, slots = _observation_batch(observation_path)
    readout, readout_path = _load_readout(package_root, descriptor)
    readout_device = readout.to(device=device).contiguous()
    predictions: dict[str, torch.Tensor] = {}
    method_receipts: dict[str, Any] = {}
    for method_id in PACKAGE_METHODS:
        model, state_sha, state_path = _load_package_method(
            package_root, descriptor, method_id, device=device, readout=readout_device
        )
        rows = []
        for row in range(int(activations.shape[0])):
            rows.append(
                _predict_row_package(
                    model,
                    readout_device,
                    activations[row],
                    masks[row],
                    positions[row],
                    device=device,
                )
            )
        prediction = torch.stack(rows, dim=0).to(dtype=torch.long, device="cpu").contiguous()
        predictions[method_id] = prediction
        state_record = _loaded_file_record(
            package_root, state_path, label=f"loaded {method_id} state"
        )
        method_receipts[method_id] = {
            # Keep flat fields for the restore worker, while the nested binding
            # records the exact file loaded by this clean consumer process.
            "state_path": state_record["relative_path"],
            "loaded_state_path": state_record["loaded_path"],
            "state_bytes": state_record["bytes"],
            "state_sha256": state_record["sha256"],
            "state_file_binding": state_record,
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "tensor_sha256": tensor_digest(prediction),
            "per_record_tensor_sha256": [tensor_digest(value) for value in prediction],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_file(
        predictions,
        str(output_path),
        metadata={
            "schema": "token-reconstruction.trr0012-predictions.v1",
            "task_id": TASK_ID,
            "methods": ",".join(PACKAGE_METHODS),
            "shape": json.dumps(list(next(iter(predictions.values())).shape), separators=(",", ":")),
            "compute_dtype": "torch.float32",
            "truth_opened": "false",
            "source_text_loaded": "false",
            "token_ids_loaded": "false",
        },
    )
    imported_modules = _loaded_decoder_modules(package_root)
    readout_record = _loaded_file_record(package_root, readout_path, label="loaded public readout")
    result = {
        "schema": "token-reconstruction.trr0012-prediction-receipt.v1",
        "task_id": TASK_ID,
        "status": "PREDICTIONS_GENERATED_AFTER_SELECTION",
        "package_descriptor": _path_label(package_root, descriptor_path),
        "package_descriptor_sha256": sha256_file(descriptor_path),
        "selection_receipt": _path_label(package_root, selection_path),
        "selection_receipt_sha256": sha256_file(selection_path),
        "observations": {
            "path": _path_label(package_root, observation_path),
            "bytes": observation_path.stat().st_size,
            "sha256": sha256_file(observation_path),
            "tensor_sha256": {
                "activations": tensor_digest(activations),
                "attention_mask": tensor_digest(masks),
                "position_ids": tensor_digest(positions),
            },
            "shape": [int(value) for value in activations.shape],
            "record_order": slots,
        },
        "output": {
            "path": _path_label(package_root, output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "readout": {
            "path": readout_record["relative_path"],
            "loaded_path": readout_record["loaded_path"],
            "bytes": readout_record["bytes"],
            "sha256": readout_record["sha256"],
            "tensor_key": "embeddings",
            "shape": list(readout.shape),
            "dtype": str(readout.dtype),
            "file_binding": readout_record,
        },
        "methods": method_receipts,
        "device": str(device),
        "numeric_profile": "qualified FP32 decoder",
        "runtime": _runtime_provenance(
            package_root, device=device, imported_modules=imported_modules
        ),
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
        "truth_opened": False,
    }
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["receipt"] = {
        "path": _path_label(package_root, receipt_path),
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256_file(receipt_path),
    }
    return result

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-smoke")
    extract.add_argument("--source-dir", type=Path, required=True)
    extract.add_argument("--selection", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--source-manifest-sha256", required=True)
    extract.add_argument("--receipt", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--package-root", type=Path, required=True)
    predict.add_argument("--observations", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--receipt", type=Path)
    predict.add_argument("--device", default="cpu")
    identity = sub.add_parser("identity")
    identity.add_argument("--input", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    selection_receipt = sub.add_parser("selection-receipt")
    selection_receipt.add_argument("--b0-state", type=Path, required=True)
    selection_receipt.add_argument("--b1-state", type=Path, required=True)
    selection_receipt.add_argument("--b0-binding", type=Path, required=True)
    selection_receipt.add_argument("--b1-binding", type=Path, required=True)
    selection_receipt.add_argument("--output", type=Path, required=True)
    selection_receipt.add_argument("--development-labels-used", choices=("true", "false"), required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--package-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract-smoke":
            receipt = extract_smoke(
                source_dir=args.source_dir,
                selection_path=args.selection,
                output_path=args.output,
                source_manifest_sha256=args.source_manifest_sha256,
            )
            receipt_path = Path(args.receipt).resolve()
            if receipt_path.exists() or receipt_path.is_symlink():
                raise PackageError(f"receipt is create-only: {receipt_path}")
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": receipt["status"], "fixture": receipt["fixture"], "receipt": str(receipt_path)}, sort_keys=True))
        elif args.command == "predict":
            result = predict_package(
                package_root=args.package_root,
                observations=args.observations,
                output_path=args.output,
                device_name=args.device,
                receipt_path=args.receipt,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "identity":
            result = write_tensor_identity(input_path=args.input, output_path=args.output)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "selection-receipt":
            result = write_selection_receipt(
                b0_state=args.b0_state,
                b1_state=args.b1_state,
                b0_binding=args.b0_binding,
                b1_binding=args.b1_binding,
                output_path=args.output,
                development_labels_used=args.development_labels_used == "true",
            )
            print(json.dumps(result, sort_keys=True))
        else:
            checked = validate_manifest(args.manifest, package_root=args.package_root)
            print(json.dumps({"status": "PASS_PACKAGE_CONTRACT", "manifest_sha256": checked["manifest_sha256"], "package_status": checked["status"]}, sort_keys=True))
    except PackageError as exc:
        print(f"TRR0012 package error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
