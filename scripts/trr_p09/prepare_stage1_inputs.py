#!/usr/bin/env python3
"""Compile the signed TRR-P09 public fitting input bank on CPU.

This is the data-only stage between the signed stage-1 plan and a later public
forward capture.  It reuses the published TRR-0005 renderer/scanner and the
TRR-0007 controlled replacement helper.  It never loads the model, creates H,
opens evaluator truth, or writes to another worktree.  The output directory is
create-only and contains token/mask/position inputs plus identity metadata; a
later capture owns activation shards.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Permit direct execution from the repository root or from scripts/.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.trr0005_prepare_public_corpus import (  # noqa: E402
    PreparationError,
    _Deadline,
    _load_public_dataset,
    _load_tokenizer,
    _scan_fit_candidates,
    _special_token_ids,
    _candidate,
    _Candidate,
)
from scripts.trr0007_support_diagnostics import (  # noqa: E402
    _planned_replacement_positions,
)
from token_reconstruction.trr0005_public_corpus import (  # noqa: E402
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SOURCE_PARTITIONS,
    apply_replacements,
)

TASK_ID = "TRR-P09"
PLAN_SHA256 = "bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c"
PLAN_BYTES = 26127
COUNTERSIGN_COMMIT = "5bfed9ec6a7bb29a988ec0a4b1343b7745d8b81b"
SEQUENCE_WIDTH = 192
B0_ROWS = 1200
TARGET_ROWS = 12000
ADDITION_ROWS = 10800
CONTROLLED_IDS = 3600
REPLACEMENTS_PER_ROW = 30
DIAGNOSTIC_SEED = 4010
STABLE_SEED = 7007

# These are immutable published inputs.  The repository-local P09 plan binds
# their metadata; the external root is explicit because published worktrees
# are intentionally not copied into this task worktree.
DEFAULT_TRR0007_ROOT = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0007"
)
DEFAULT_TOKENIZER = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)

STRATA = (
    ("alpaca_natural", "alpaca", False, (1390, 2260, 1610, 740), (1251, 2034, 1449, 666)),
    ("pile_natural", "pile", False, (620, 890, 570, 920), (558, 801, 513, 828)),
    ("finance_natural", "finance", False, (360, 570, 300, 570), (324, 513, 270, 513)),
    ("pile_controlled", "pile", True, (0, 0, 0, 600), (0, 0, 0, 540)),
    ("finance_controlled", "finance", True, (0, 0, 0, 600), (0, 0, 0, 540)),
)
STRATUM_ORDER = tuple(item[0] for item in STRATA)
BIN_RANGES = ((40, 63), (64, 95), (96, 127), (128, 191))
DIAGNOSTIC_QUOTAS = {
    "alpaca_natural": 32,
    "pile_natural": 16,
    "finance_natural": 10,
    "pile_controlled": 3,
    "finance_controlled": 3,
}

class PreparationErrorLocal(RuntimeError):
    pass


@dataclass(frozen=True)
class InputRow:
    """A transient row; token IDs are emitted only in the input tensor."""

    record_id: str
    source_record_id: str
    dataset_key: str
    stratum: str
    source_row_index: int | None
    rendered_sha256: str
    source_full_token_count: int
    target_post_bos_token_count: int
    token_ids: tuple[int, ...]
    public_record_sha256: str | None = None
    synthetic: bool = False
    parent_h128_sha256: str | None = None
    sequence_h128_sha256: str | None = None
    replacement_positions: tuple[int, ...] = ()
    replacement_token_ids: tuple[int, ...] = ()

    def sidecar(self, global_row: int) -> dict[str, Any]:
        value: dict[str, Any] = {
            "global_row": int(global_row),
            "capture_local_row": (int(global_row) - B0_ROWS) if int(global_row) >= B0_ROWS else None,
            "b0_prefix_row": int(global_row) if int(global_row) < B0_ROWS else None,
            "record_id": self.record_id,
            "source_record_id": self.source_record_id,
            "dataset_key": self.dataset_key,
            "stratum": self.stratum,
            "synthetic": bool(self.synthetic),
            "source_row_index": self.source_row_index,
            "rendered_sha256": self.rendered_sha256,
            "source_full_token_count": int(self.source_full_token_count),
            "target_post_bos_token_count": int(self.target_post_bos_token_count),
            "target_full_token_count": int(self.target_post_bos_token_count + 1),
            "sequence_h128_sha256": self.sequence_h128_sha256,
        }
        if self.public_record_sha256 is not None:
            value["public_record_sha256"] = self.public_record_sha256
        if self.parent_h128_sha256 is not None:
            value["parent_h128_sha256"] = self.parent_h128_sha256
        if self.synthetic:
            value.update(
                {
                    "replacement_count": len(self.replacement_positions),
                    "replacement_positions_after_bos": list(self.replacement_positions),
                    "replacement_positions_one_based": [int(x) + 1 for x in self.replacement_positions],
                    "replacement_token_ids": list(self.replacement_token_ids),
                }
            )
        return value


@dataclass(frozen=True)
class _Stage1Candidate:
    """Candidate with rendered and public-record identities kept separately."""

    dataset_key: str
    dataset_id: str
    split: str
    revision: str
    row_index: int
    record_id: str
    domain: str
    rendered_sha256: str
    public_record_sha256: str
    token_ids: tuple[int, ...]

    @property
    def full_token_count(self) -> int:
        return len(self.token_ids)


def _public_record_sha256(dataset_key: str, row: Mapping[str, Any], rendered_text_digest: str) -> str:
    """Compute the approved public-record identity, separate from rendering."""
    if dataset_key == "pile":
        value = row.get("text", "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise PreparationErrorLocal("Pile public-record field is not text")
        return digest_bytes(value.encode("utf-8"))
    if dataset_key == "finance":
        def normalized(key: str) -> str:
            value = row.get(key, "")
            if value is None:
                return ""
            if not isinstance(value, str):
                raise PreparationErrorLocal(f"Finance public-record field {key!r} is not text")
            return value.strip()

        system = normalized("system") or None
        user = normalized("user")
        assistant = normalized("assistant")
        if not user:
            instruction = normalized("instruction")
            input_text = normalized("input")
            user = instruction + (("\n\n" + input_text) if input_text else "")
        if not assistant:
            assistant = normalized("output")
        if not user or not assistant:
            raise PreparationErrorLocal("Finance public-record fields are empty")
        payload = json.dumps(
            [system, user, assistant], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return digest_bytes(payload)
    # No separate historical canonicalizer is bound for Alpaca in the signed
    # P09 input contract.  The established Alpaca renderer is therefore the
    # declared public-record representation, but it remains a distinct field
    # and comparison namespace throughout selection.
    return str(rendered_text_digest)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def h128_digest(ids: Sequence[int]) -> str | None:
    if len(ids) < 128:
        return None
    # This is the public H128 convention: BOS plus first 127 active IDs,
    # signed int32 little-endian/native C-order.
    tensor = torch.tensor(list(ids[:128]), dtype=torch.int32).contiguous()
    return digest_bytes(tensor.numpy().tobytes(order="C"))


def stable_source_key(dataset_id: str, split: str, revision: str, row_index: int) -> str:
    return digest_bytes(
        f"TRR-0007|7007|{dataset_id}|{split}|{revision}|row:{int(row_index)}".encode()
    )


def validate_plan_object(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != "token-reconstruction.trr-p09-stage1-public-bank-plan.v1":
        raise PreparationErrorLocal("stage1 plan schema changed")
    if plan.get("task_id") != TASK_ID or plan.get("status") != "PROPOSED_STAGE1_PLAN_NO_SELECTION_NO_FREEZE":
        raise PreparationErrorLocal("stage1 plan is not the signed preselection plan")
    boundary = plan.get("access_boundary")
    if not isinstance(boundary, Mapping) or any(bool(boundary.get(k)) for k in (
        "model_loaded", "public_forward_started", "activation_bank_created", "fit_started", "truth_opened", "final_panel_selected"
    )):
        raise PreparationErrorLocal("plan access boundary already contains a scientific run")
    recipe = plan.get("stage1_deterministic_recipe")
    if not isinstance(recipe, Mapping):
        raise PreparationErrorLocal("stage1 recipe missing")
    strata = recipe.get("strata")
    if not isinstance(strata, list) or len(strata) != len(STRATA):
        raise PreparationErrorLocal("stage1 stratum table changed")
    expected = {name: (domain, controlled, target, additions) for name, domain, controlled, target, additions in STRATA}
    for item in strata:
        if not isinstance(item, Mapping) or item.get("id") not in expected:
            raise PreparationErrorLocal("unexpected stage1 stratum")
        name = str(item["id"])
        _, _, target, additions = expected[name]
        if int(item.get("target_rows", -1)) != sum(target) or int(item.get("addition_rows", -1)) != sum(additions):
            raise PreparationErrorLocal(f"{name} quota total changed")
        if list(item.get("target_bin_counts_40_63_64_95_96_127_128_191", ())) != list(target):
            raise PreparationErrorLocal(f"{name} target bins changed")
        if list(item.get("addition_bin_counts_40_63_64_95_96_127_128_191", ())) != list(additions):
            raise PreparationErrorLocal(f"{name} addition bins changed")
    expanded = recipe.get("expanded_bank", {})
    if int(expanded.get("B0_records", -1)) != B0_ROWS or int(expanded.get("new_records", -1)) != ADDITION_ROWS or int(expanded.get("target_records", -1)) != TARGET_ROWS:
        raise PreparationErrorLocal("expanded row counts changed")
    controlled = recipe.get("controlled_replacement", {})
    if int(controlled.get("occurrences_per_record", -1)) != REPLACEMENTS_PER_ROW:
        raise PreparationErrorLocal("controlled replacement count changed")
    if controlled.get("support_list_sha256") != "0e84f625e24fb4a2038972cf417923dc20a8cc2fd85e55ac6e9b7a846f5ca576":
        raise PreparationErrorLocal("controlled support-list digest changed")
    diag = recipe.get("fixed_diagnostic", {})
    if int(diag.get("per_bank_record_count", -1)) != 64 or int(diag.get("seed", -1)) != DIAGNOSTIC_SEED:
        raise PreparationErrorLocal("diagnostic rule changed")
    schedule = recipe.get("training_schedule", {})
    if schedule.get("formula") != "N = max(12000, 1000 * ceil((5 * actual_B1_valid_positions / 512) / 1000))":
        raise PreparationErrorLocal("exposure formula changed")


def load_and_verify_plan(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise PreparationErrorLocal(f"plan unavailable: {path}")
    payload = path.read_bytes()
    if len(payload) != PLAN_BYTES or digest_bytes(payload) != PLAN_SHA256:
        raise PreparationErrorLocal("signed stage1 plan bytes do not match the root-approved plan")
    plan = json.loads(payload)
    if not isinstance(plan, dict):
        raise PreparationErrorLocal("plan must be an object")
    validate_plan_object(plan)
    return plan


def verify_countersignature(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    plan = value.get("plan", {})
    if plan.get("sha256") != PLAN_SHA256 or int(plan.get("bytes", -1)) != PLAN_BYTES or plan.get("commit") != COUNTERSIGN_COMMIT:
        raise PreparationErrorLocal("countersignature does not bind signed plan")
    attestation = value.get("attestation", {})
    if attestation.get("root") != "COUNTERSIGNED" or attestation.get("agent1") != "COUNTERSIGNED":
        raise PreparationErrorLocal("both plan countersignatures are required")
    if attestation.get("model_loaded") or attestation.get("gpu_used") or attestation.get("final_evaluation_truth_opened") or attestation.get("fitting_started"):
        raise PreparationErrorLocal("countersignature scope is too broad")
    return value


def _json_records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("records")
    if not isinstance(value, list):
        raise PreparationErrorLocal(f"{label} has no record list")
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _regular_hash(path: Path, expected: str, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
        raise PreparationErrorLocal(f"{label} is missing or hash-mismatched")
    return path


def b0_paths(published_root: Path) -> tuple[Path, Path, Path]:
    base = Path(published_root).expanduser().resolve() / "experiments/TRR-0007/support/broader_capture_v2"
    return base / "enriched_fit_records.json", base / "enriched_fit_cut4.safetensors", base / "enriched_manifest.json"


def load_b0(published_root: Path) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    records_path, tensor_path, manifest_path = b0_paths(published_root)
    _regular_hash(records_path, "808cfd0f95ea3ee66c1d4094f3c10f1e346ba986a6d77138a61b8f8f13c4a738", label="B0 records")
    _regular_hash(tensor_path, "a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d", label="B0 tensor payload")
    _regular_hash(manifest_path, "c7a857e545a2f252ce8b3ab71bb2336e552fd212c7c434e39cef66f233b77a08", label="B0 manifest")
    records = _json_records(json.loads(records_path.read_text(encoding="utf-8")), label="B0 records")
    if len(records) != B0_ROWS or any(int(row.get("slot", -1)) != i for i, row in enumerate(records)):
        raise PreparationErrorLocal("B0 records are not the immutable 1,200-row prefix")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        required = {"token_ids", "attention_mask", "position_ids"}
        if not required.issubset(set(handle.keys())):
            raise PreparationErrorLocal("B0 token/mask/position tensors are incomplete")
        token_ids = handle.get_tensor("token_ids").contiguous()
        attention = handle.get_tensor("attention_mask").contiguous()
        positions = handle.get_tensor("position_ids").contiguous()
    if tuple(token_ids.shape) != (B0_ROWS, SEQUENCE_WIDTH) or tuple(attention.shape) != tuple(token_ids.shape) or tuple(positions.shape) != tuple(token_ids.shape):
        raise PreparationErrorLocal("B0 tensor geometry changed")
    if int(token_ids[0, 0]) != BOS_TOKEN_ID or not bool(token_ids[:, 0].eq(BOS_TOKEN_ID).all()):
        raise PreparationErrorLocal("B0 BOS prefix changed")
    for index in range(B0_ROWS):
        active = int(attention[index].sum().item())
        expected = torch.zeros(SEQUENCE_WIDTH, dtype=positions.dtype)
        expected[:active] = torch.arange(active, dtype=positions.dtype)
        if not bool(positions[index].eq(expected).all()):
            raise PreparationErrorLocal("B0 position IDs changed")
    if not bool(attention[:, 0].eq(1).all()):
        raise PreparationErrorLocal("B0 attention masks lost BOS")
    return records, token_ids, attention, positions, {
        "records": {"path": str(records_path), "bytes": records_path.stat().st_size, "sha256": digest_file(records_path)},
        "tensor": {"path": str(tensor_path), "bytes": tensor_path.stat().st_size, "sha256": digest_file(tensor_path)},
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": digest_file(manifest_path)},
    }


def load_identity_ids(published_root: Path) -> list[int]:
    recipe_path = Path(published_root).expanduser().resolve() / "experiments/TRR-0007/support/broader_recipe_v2/improved_public_sampling_recipe.json"
    _regular_hash(recipe_path, "602530bb3e0d530b0f08c6be2f05b9833f0e149d45f658bf456c6b0cb6dc9478", label="TRR-0007 replacement recipe")
    value = json.loads(recipe_path.read_text(encoding="utf-8"))
    values = value.get("controlled_component", {}).get("identity_pool", {}).get("selected_token_ids")
    if not isinstance(values, list) or len(values) != CONTROLLED_IDS or len(set(values)) != CONTROLLED_IDS:
        raise PreparationErrorLocal("frozen controlled-ID list changed")
    # The published recipe's digest is newline-delimited integer bytes, as in
    # the support constructor; verify that convention explicitly.
    h = hashlib.sha256()
    for value in values:
        h.update(f"{int(value)}\n".encode())
    if h.hexdigest() != "0e84f625e24fb4a2038972cf417923dc20a8cc2fd85e55ac6e9b7a846f5ca576":
        raise PreparationErrorLocal("frozen controlled-ID digest changed")
    return [int(x) for x in values]


def collect_identity_exclusions(plan: Mapping[str, Any], root: Path, published_root: Path) -> dict[str, Any]:
    """Compile namespace-separated hash/ID sets from bound public metadata only."""
    ids: set[str] = set()
    rendered: set[str] = set()
    public_record: set[str] = set()
    h128: set[str] = set()
    source_indices: dict[str, set[int]] = defaultdict(set)
    source_files: list[dict[str, Any]] = []
    # B0 record IDs and rendered hashes are explicitly public identity metadata.
    b0_records, _, _, _, b0_files = load_b0(published_root)
    source_files.extend(b0_files.values())
    for row in b0_records:
        for key, target in (("record_id", ids), ("source_record_id", ids), ("rendered_sha256", rendered)):
            value = row.get(key)
            if isinstance(value, str) and value:
                target.add(value)
    # The plan's approved opaque ledgers are hash-only files.  Their values are
    # never printed or serialized; only namespaces declared by the plan are read.
    ledger_specs = plan["already_bound_inputs"]["exclusion_metadata"]["approved_opaque_ledgers"]
    for spec in ledger_specs:
        rel = Path(str(spec["path"]))
        path = (root / rel).resolve() if not rel.is_absolute() else rel
        if not path.is_file() or path.is_symlink():
            # Published P08/0008 ledgers are external to this worktree; try
            # the repository's published worktree root without guessing data.
            path = (DEFAULT_TRR0007_ROOT.parent.parent / rel).resolve()
        if not path.is_file() or digest_file(path) != str(spec["sha256"]):
            raise PreparationErrorLocal(f"approved exclusion ledger unavailable: {rel}")
        source_files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path), "id": spec.get("id")})
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PreparationErrorLocal(f"approved exclusion ledger is not readable JSON: {path}") from exc
        _scan_declared_hashes(value, ids=ids, rendered=rendered, public_record=public_record, h128=h128, source_indices=source_indices, allowed=set(spec.get("namespace", ())))
    # The opened TRR-0009 public-base source selection is task-local metadata.
    selection = root / "experiments/TRR-0009/selection_v2/source_selection.json"
    if not selection.is_file() or selection.is_symlink() or digest_file(selection) != "c2e996514f7f45e55d7bfadc27fc8048bdb07a2c979b208de8716972d8d1def2":
        raise PreparationErrorLocal("required TRR-0009 public-base selection metadata is unavailable or changed")
    if selection.is_file():
        source_files.append({"path": str(selection), "bytes": selection.stat().st_size, "sha256": digest_file(selection)})
        _scan_declared_hashes(json.loads(selection.read_text()), ids=ids, rendered=rendered, public_record=public_record, h128=h128, source_indices=source_indices, allowed={"identity_ids", "source_indices"})
    exclusions = {
        "record_ids": ids,
        "rendered_sha256": rendered,
        "public_record_sha256": public_record,
        "h128_sha256": h128,
        "source_indices": source_indices,
        "source_files": source_files,
    }
    exclusions["summary"] = {
        "record_id_count": len(ids),
        "rendered_hash_count": len(rendered),
        "public_record_hash_count": len(public_record),
        "h128_count": len(h128),
        "source_index_count": sum(len(values) for values in source_indices.values()),
        "source_file_count": len(source_files),
        "canonical_digest": digest_bytes(canonical_bytes({
            "record_ids": sorted(ids),
            "rendered_sha256": sorted(rendered),
            "public_record_sha256": sorted(public_record),
            "h128_sha256": sorted(h128),
            "source_indices": {key: sorted(values) for key, values in sorted(source_indices.items())},
        })),
    }
    return exclusions


def _scan_declared_hashes(
    value: Any,
    *,
    ids: set[str],
    rendered: set[str],
    public_record: set[str],
    h128: set[str],
    source_indices: dict[str, set[int]],
    allowed: set[str],
    field: str | None = None,
    dataset_key: str | None = None,
) -> None:
    """Read only declared identity namespaces, retaining dataset/index scope."""
    known_hash = {"public_record_sha256", "rendered_sha256", "public_rendered_sha256", "final_sequence_sha256", "h128_sha256", "sequence128_sha256"}
    known_id = {"record_id", "source_record_id", "public_record_id"}
    known_index = {"source_index", "row_index", "raw_index", "dataset_index", "index"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            next_field = lowered if lowered in known_hash | known_id | known_index else field
            next_dataset = lowered if lowered in {"alpaca", "pile", "finance"} else dataset_key
            _scan_declared_hashes(
                child, ids=ids, rendered=rendered, public_record=public_record, h128=h128,
                source_indices=source_indices, allowed=allowed, field=next_field, dataset_key=next_dataset,
            )
    elif isinstance(value, list):
        for child in value:
            _scan_declared_hashes(
                child, ids=ids, rendered=rendered, public_record=public_record, h128=h128,
                source_indices=source_indices, allowed=allowed, field=field, dataset_key=dataset_key,
            )
    elif isinstance(value, str):
        if len(value) == 64 and field in {"rendered_sha256", "public_rendered_sha256"} and "rendered_sha256" in allowed:
            rendered.add(value)
        elif len(value) == 64 and field == "public_record_sha256" and "public_record_sha256" in allowed:
            public_record.add(value)
        elif len(value) == 64 and field in {"final_sequence_sha256", "h128_sha256", "sequence128_sha256"} and "final_sequence_sha256" in allowed:
            h128.add(value)
        elif field in known_id and ("identity_ids" in allowed or not allowed):
            ids.add(value)
    elif isinstance(value, int) and not isinstance(value, bool) and field in known_index and "source_indices" in allowed and dataset_key is not None:
        source_indices.setdefault(dataset_key, set()).add(int(value))


def load_b0_control_parents(published_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(published_root).expanduser().resolve() / "experiments/TRR-0007/support/broader_bank_v5/selected_parent_rows.json"
    expected = "7de2ce7fca1d9489f652e0229b7cbfeacd625ca6ebd37c56d4405962da79f547"
    _regular_hash(path, expected, label="B0 controlled-parent metadata")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 120:
        raise PreparationErrorLocal("B0 controlled-parent metadata changed")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PreparationErrorLocal("B0 controlled-parent metadata is malformed")
    return [dict(row) for row in rows], {"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)}


def replacement_cycle_audit(observed_b0: Sequence[int], signed_identity_ids: Sequence[int]) -> dict[str, Any]:
    """Describe the published B0 prefix plus nine signed B1 cycles.

    B0 is immutable published evidence, so its observed replacement sequence is
    retained even when it predates the current signed identity recipe.  The
    returned digests bind the two namespaces separately and then bind their
    concatenated 36,000-occurrence runtime sequence.
    """
    b0_values = tuple(int(value) for value in observed_b0)
    signed_values = tuple(int(value) for value in signed_identity_ids)
    if len(b0_values) != CONTROLLED_IDS:
        raise PreparationErrorLocal("B0 controlled cycle does not contain 3,600 occurrences")
    if len(signed_values) != CONTROLLED_IDS or len(set(signed_values)) != CONTROLLED_IDS:
        raise PreparationErrorLocal("signed controlled-ID cycle is not exactly 3,600 unique IDs")
    b1_values = signed_values * 9
    full_values = b0_values + b1_values
    return {
        "b0_replacement_occurrences": len(b0_values),
        "b0_replacement_token_digest": digest_bytes(canonical_bytes(b0_values)),
        "b0_actual_replacement_token_digest": digest_bytes(canonical_bytes(b0_values)),
        "b0_actual_replacement_token_count": len(b0_values),
        "b0_identity_cycle_matches_signed_recipe": b0_values == signed_values,
        "b0_provenance_status": "PUBLISHED_B0_ACTUAL_IDS_RETAINED_SIGNED_B1_SEPARATE",
        "signed_b1_cycle_count": 9,
        "signed_b1_replacement_token_digest": digest_bytes(canonical_bytes(b1_values)),
        "full_ten_cycle_identity_digest": digest_bytes(canonical_bytes(full_values)),
        "full_ten_cycle_composition": "published_b0_actual_3600_plus_signed_recipe_3600_x9",
        "full_replacement_occurrence_count": len(full_values),
    }


def bind_b0_controlled_rows(
    rows: Sequence[InputRow],
    b0_records: Sequence[Mapping[str, Any]],
    b0_token: torch.Tensor,
    parent_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Any],
    tokenizer: Any,
    identity_ids: Sequence[int],
    deadline: _Deadline,
) -> tuple[list[InputRow], dict[str, Any]]:
    """Recover and verify B0 replacement offsets/IDs from published metadata."""
    from scripts.trr0005_prepare_public_corpus import _candidate

    by_slot = {int(item.get("slot", -1)): dict(item) for item in parent_rows}
    controlled_slots = [index for index, row in enumerate(b0_records) if bool(row.get("synthetic", False))]
    if len(controlled_slots) != 120 or set(controlled_slots) != set(by_slot):
        raise PreparationErrorLocal("B0 controlled slots do not match published parent metadata")
    specials = _special_token_ids(tokenizer)
    updated = list(rows)
    cycle_values: list[int] = []
    cycle_offsets: list[int] = []
    for item in sorted(parent_rows, key=lambda value: (-int(value["target_post_bos_token_count"]), int(value["slot"]))):
        slot = int(item["slot"])
        domain = str(item["domain"])
        dataset_key = str(item["dataset_key"])
        source_index = int(item["row_index"])
        if dataset_key not in datasets:
            raise PreparationErrorLocal(f"B0 controlled parent dataset unavailable: {dataset_key}")
        candidate = _candidate(
            dataset_key, datasets[dataset_key], source_index, tokenizer, deadline=deadline,
            excluded_ids=set(), excluded_row_keys=set(), excluded_hashes=set(), scan_stats=None,
        )
        if candidate is None or candidate.record_id != str(item["source_record_id"]):
            raise PreparationErrorLocal(f"B0 controlled parent identity changed at slot {slot}")
        target = int(item["target_post_bos_token_count"])
        parent = clip_ids(candidate.token_ids, target)
        synthetic_id = str(item["synthetic_record_id"])
        offsets = tuple(_planned_replacement_positions(parent, target_post_bos_token_count=target, record_key=synthetic_id, seed=STABLE_SEED, structural_token_ids=specials))
        actual = tuple(int(b0_token[slot, int(offset) + 1]) for offset in offsets)
        if len(actual) != REPLACEMENTS_PER_ROW or any(value in specials for value in actual):
            raise PreparationErrorLocal(f"B0 controlled replacement IDs are malformed at slot {slot}")
        reconstructed = tuple(apply_replacements(parent, offsets, actual, target_post_bos_token_count=target, structural_token_ids=specials))
        active = int(b0_records[slot].get("active_token_count", b0_records[slot].get("target_full_token_count", 0)))
        observed = tuple(int(value) for value in b0_token[slot, :active].tolist())
        if reconstructed != observed:
            raise PreparationErrorLocal(f"B0 controlled replacement reconstruction differs at slot {slot}")
        cycle_offsets.extend(offsets)
        cycle_values.extend(actual)
        current = updated[slot]
        updated[slot] = replace(current, replacement_positions=offsets, replacement_token_ids=actual, parent_h128_sha256=h128_digest(parent), sequence_h128_sha256=h128_digest(observed))
        deadline.check("B0 controlled sidecar binding")
    # Published B0 is immutable evidence.  Its observed replacement sequence
    # is bound separately from the current signed cycle used for B1 additions.
    audit = replacement_cycle_audit(cycle_values, identity_ids)
    audit.update({
        "b0_controlled_rows": len(controlled_slots),
        "b0_replacement_offset_digest": digest_bytes(canonical_bytes(cycle_offsets)),
        "b0_sidecar_positions_verified": True,
        "b0_sidecar_token_ids_verified": True,
        "parent_metadata_sha256": "7de2ce7fca1d9489f652e0229b7cbfeacd625ca6ebd37c56d4405962da79f547",
        "verified_against_public_b0_tensor": True,
    })
    return updated, audit

def b0_stratum(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset_key", ""))
    synthetic = bool(row.get("synthetic", False))
    if synthetic:
        return f"{dataset}_controlled"
    return f"{dataset}_natural"


def bin_index(length: int) -> int:
    for index, (lower, upper) in enumerate(BIN_RANGES):
        if lower <= length <= upper:
            return index
    raise PreparationErrorLocal(f"post-BOS length {length} is outside signed bins")


def validate_b0_quota(records: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for row in records:
        counts[(b0_stratum(row), int(row.get("target_post_bos_token_count", row.get("post_bos_token_count", -1))))] += 1
    by_stratum = Counter(stratum for stratum, _ in counts.elements())
    # counts.elements() is not useful for a Counter with tuple keys; explicit.
    by_stratum = Counter()
    for (stratum, _), count in counts.items():
        by_stratum[stratum] += count
    expected_b0 = {name: sum(target) // 10 for name, _, _, target, _ in STRATA}
    if dict(by_stratum) != expected_b0:
        raise PreparationErrorLocal(f"B0 stratum counts changed: {dict(by_stratum)}")
    for name, _, _, target, _ in STRATA:
        for index, (lower, upper) in enumerate(BIN_RANGES):
            expected = int(target[index] // 10)
            observed = sum(count for (stratum, length), count in counts.items() if stratum == name and lower <= length <= upper)
            if observed != expected:
                raise PreparationErrorLocal(f"B0 {name} bin {lower}-{upper}: {observed} != {expected}")
    return counts


def _candidate_source_h128(candidate: _Candidate) -> str | None:
    return h128_digest(candidate.token_ids)


def _blocked(candidate: _Candidate, exclusions: Mapping[str, Any], used_ids: set[str], used_rendered: set[str], used_h128: set[str], used_public: set[str] | None = None) -> str | None:
    if candidate.record_id in exclusions["record_ids"] or candidate.record_id in used_ids:
        return "record_id"
    # The shared renderer computes stable_public_text_digest, which is the
    # canonical public-record digest used by the approved selection ledgers.
    # Keep the metadata namespaces separate even though the candidate digest
    # is byte-identical to the public-record field by construction.
    if candidate.rendered_sha256 in exclusions["rendered_sha256"] or candidate.rendered_sha256 in used_rendered:
        return "rendered_sha256"
    public_hash = getattr(candidate, "public_record_sha256", None)
    if public_hash and (public_hash in exclusions["public_record_sha256"] or (used_public is not None and public_hash in used_public)):
        return "public_record_sha256"
    if int(candidate.row_index) in exclusions["source_indices"].get(candidate.dataset_key, set()):
        return "source_index"
    h = _candidate_source_h128(candidate)
    if h is not None and (h in exclusions["h128_sha256"] or h in used_h128):
        return "final_sequence_sha256"
    return None


def exact_addition_quotas(b0_records: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, int]]:
    """Return nine additions for every exact B0 post-BOS length."""
    counts: Counter[tuple[str, int]] = Counter()
    for row in b0_records:
        length = int(row.get("target_post_bos_token_count", row.get("post_bos_token_count", -1)))
        counts[(b0_stratum(row), length)] += 1
    result: dict[str, dict[int, int]] = {name: {} for name in STRATUM_ORDER}
    for (stratum, length), count in counts.items():
        if length < 40 or length > 191:
            raise PreparationErrorLocal(f"B0 exact length {length} is outside the signed range")
        result[stratum][length] = int(count) * 9
    if sum(sum(values.values()) for values in result.values()) != ADDITION_ROWS:
        raise PreparationErrorLocal("exact-length addition quotas do not sum to 10,800")
    return result


def select_candidates(candidates: Sequence[_Candidate], *, stratum: str, exact_quota: Mapping[int, int], exclusions: Mapping[str, Any], used_ids: set[str], used_rendered: set[str], used_h128: set[str], used_public: set[str] | None = None) -> list[tuple[_Candidate, int]]:
    """Select exact lengths target-slot-first, then stable source order."""
    remaining = {int(length): int(count) for length, count in exact_quota.items()}
    ordered = sorted(candidates, key=lambda c: (stable_source_key(c.dataset_id, c.split, c.revision, c.row_index), c.row_index))
    selected: list[tuple[_Candidate, int]] = []
    # The signed rule allocates descending target length and then the first
    # eligible row in the frozen content-blind source order.  Candidate-first
    # greedy assignment is not equivalent when a row can satisfy several bins.
    for target in sorted(remaining, reverse=True):
        for _ in range(remaining[target]):
            chosen: _Candidate | None = None
            for candidate in ordered:
                if _blocked(candidate, exclusions, used_ids, used_rendered, used_h128, used_public) is not None:
                    continue
                if stratum.endswith("_controlled") and candidate.full_token_count - 1 < 128:
                    continue
                if candidate.full_token_count - 1 < target:
                    continue
                chosen = candidate
                break
            if chosen is None:
                raise PreparationErrorLocal(f"{stratum} source pool cannot satisfy target length {target}")
            selected.append((chosen, int(target)))
            used_ids.add(chosen.record_id)
            used_rendered.add(chosen.rendered_sha256)
            if used_public is not None:
                public_hash = getattr(chosen, "public_record_sha256", None)
                if public_hash:
                    used_public.add(public_hash)
            h = _candidate_source_h128(chosen)
            if h is not None:
                used_h128.add(h)
    return selected


def clip_ids(ids: Sequence[int], target_post_bos: int) -> tuple[int, ...]:
    values = tuple(int(x) for x in ids[: target_post_bos + 1])
    if len(values) != target_post_bos + 1 or values[0] != BOS_TOKEN_ID:
        raise PreparationErrorLocal("source row cannot satisfy target BOS/length")
    return values


def make_natural_rows(selected: Sequence[tuple[_Candidate, int]], stratum: str) -> list[InputRow]:
    return [
        InputRow(
            record_id=c.record_id,
            source_record_id=c.record_id,
            dataset_key=c.dataset_key,
            stratum=stratum,
            source_row_index=int(c.row_index),
            rendered_sha256=c.rendered_sha256,
            public_record_sha256=getattr(c, "public_record_sha256", None),
            source_full_token_count=c.full_token_count,
            target_post_bos_token_count=target,
            token_ids=clip_ids(c.token_ids, target),
            sequence_h128_sha256=h128_digest(c.token_ids),
        )
        for c, target in selected
    ]


def make_controlled_rows(selected: Sequence[tuple[_Candidate, int]], stratum: str, identity_ids: Sequence[int], cursor: int, structural_token_ids: Iterable[int] = (BOS_TOKEN_ID, PAD_TOKEN_ID)) -> tuple[list[InputRow], int]:
    special = {int(value) for value in structural_token_ids}
    rows: list[InputRow] = []
    for ordinal, (candidate, target) in enumerate(sorted(selected, key=lambda pair: (-pair[1], pair[0].row_index))):
        parent = clip_ids(candidate.token_ids, target)
        record_id = f"TRR-P09/controlled-v1/{candidate.record_id}::row-{ordinal:03d}"
        offsets = tuple(_planned_replacement_positions(parent, target_post_bos_token_count=target, record_key=record_id, seed=STABLE_SEED, structural_token_ids=special))
        if len(offsets) != REPLACEMENTS_PER_ROW:
            raise PreparationErrorLocal("controlled replacement helper returned the wrong count")
        replacements = tuple(int(identity_ids[(cursor + i) % len(identity_ids)]) for i in range(len(offsets)))
        built = tuple(apply_replacements(parent, offsets, replacements, target_post_bos_token_count=target, structural_token_ids=special))
        rows.append(InputRow(
            record_id=record_id,
            source_record_id=candidate.record_id,
            dataset_key=candidate.dataset_key,
            stratum=stratum,
            source_row_index=int(candidate.row_index),
            rendered_sha256=candidate.rendered_sha256,
            public_record_sha256=getattr(candidate, "public_record_sha256", None),
            source_full_token_count=candidate.full_token_count,
            target_post_bos_token_count=target,
            token_ids=built,
            synthetic=True,
            parent_h128_sha256=h128_digest(candidate.token_ids),
            sequence_h128_sha256=h128_digest(built),
            replacement_positions=offsets,
            replacement_token_ids=replacements,
        ))
        cursor += len(offsets)
    return rows, cursor


def diagnostic_indices(rows: Sequence[InputRow]) -> dict[str, Any]:
    chosen: list[int] = []
    by_stratum: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = digest_bytes(f"TRR-0010|fixed-diagnostic|{DIAGNOSTIC_SEED}|{row.record_id}".encode())
        by_stratum[row.stratum].append((key, index))
    for stratum in STRATUM_ORDER:
        quota = DIAGNOSTIC_QUOTAS[stratum]
        values = sorted(by_stratum[stratum])[:quota]
        if len(values) != quota:
            raise PreparationErrorLocal(f"diagnostic stratum {stratum} is short")
        chosen.extend(index for _, index in values)
    chosen = sorted(chosen)
    return {"seed": DIAGNOSTIC_SEED, "stratum_quotas": DIAGNOSTIC_QUOTAS, "indices": chosen, "indices_sha256": digest_bytes(canonical_bytes(chosen))}


def per_bank_diagnostic_indices(rows: Sequence[InputRow]) -> dict[str, Any]:
    """Freeze independent 64-row diagnostics for the current and expanded banks."""
    if len(rows) < B0_ROWS:
        raise PreparationErrorLocal("expanded bank is shorter than the immutable B0 prefix")
    current = diagnostic_indices(rows[:B0_ROWS])
    expanded = diagnostic_indices(rows)
    return {
        "seed": DIAGNOSTIC_SEED,
        "per_bank_record_count": sum(DIAGNOSTIC_QUOTAS.values()),
        "current_bank": current,
        "expanded_bank": expanded,
        "shared_across": "fixed_and_directional_arms_within_each_bank_only",
    }


def exposure_summary(rows: Sequence[InputRow]) -> dict[str, Any]:
    positions = int(sum(row.target_post_bos_token_count for row in rows))
    exact_n = max(12000, 1000 * math.ceil((5 * positions / 512) / 1000))
    counts = Counter(int(token) for row in rows for token in row.token_ids[1:])
    return {
        "actual_B1_valid_positions": positions,
        "distinct_post_bos_token_ids": len(counts),
        "post_bos_token_occurrences": int(sum(counts.values())),
        "token_occurrence_digest": digest_bytes(canonical_bytes(sorted((int(k), int(v)) for k, v in counts.items()))),
        "formula": "N = max(12000, 1000 * ceil((5 * actual_B1_valid_positions / 512) / 1000))",
        "exact_common_N": int(exact_n),
        "grid": [0, 1000, 2000, 4000, 8000, 12000, int(exact_n)],
    }


def _write_json_create(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PreparationErrorLocal(f"create-only artifact exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def publish_output(output_root: Path, rows: Sequence[InputRow], *, plan: Mapping[str, Any], countersign: Mapping[str, Any], source_files: Mapping[str, Any], exclusions: Mapping[str, Any], b0_meta: Mapping[str, Any], b0_token: torch.Tensor, b0_mask: torch.Tensor, b0_pos: torch.Tensor, b0_prefix_digest: str, diagnostic: Mapping[str, Any], exposure: Mapping[str, Any], b0_control_audit: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise PreparationErrorLocal(f"output root is create-only and already exists: {output_root}")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(parent)))
    try:
        token = torch.full((len(rows), SEQUENCE_WIDTH), PAD_TOKEN_ID, dtype=torch.int32)
        mask = torch.zeros((len(rows), SEQUENCE_WIDTH), dtype=torch.uint8)
        pos = torch.zeros((len(rows), SEQUENCE_WIDTH), dtype=torch.int64)
        # Copy all B0 tensors, including padding values and position IDs.  The
        # first 1,200 rows are a byte-equivalent immutable prefix.
        token[:B0_ROWS] = b0_token.to(dtype=torch.int32)
        mask[:B0_ROWS] = b0_mask.to(dtype=torch.uint8)
        pos[:B0_ROWS] = b0_pos.to(dtype=torch.int64)
        for index, row in enumerate(rows[B0_ROWS:], start=B0_ROWS):
            if len(row.token_ids) > SEQUENCE_WIDTH or not row.token_ids or row.token_ids[0] != BOS_TOKEN_ID:
                raise PreparationErrorLocal(f"row {index} violates padded input geometry")
            active = len(row.token_ids)
            token[index, :active] = torch.tensor(row.token_ids, dtype=torch.int32)
            mask[index, :active] = 1
            # Capture accepts the inherited active-arange/zero-padding form.
            pos[index, :active] = torch.arange(active, dtype=torch.int64)
        if not torch.equal(token[:B0_ROWS], b0_token.to(dtype=torch.int32)) or not torch.equal(mask[:B0_ROWS], b0_mask.to(dtype=torch.uint8)) or not torch.equal(pos[:B0_ROWS], b0_pos.to(dtype=torch.int64)):
            raise PreparationErrorLocal("B0 prefix changed while staging input tensors")
        input_path = temp / "inputs.safetensors"
        save_file({"token_ids": token, "attention_mask": mask, "position_ids": pos}, str(input_path))
        records = [row.sidecar(index) for index, row in enumerate(rows)]
        records_path = temp / "records.json"
        _write_json_create(records_path, records)
        summary = {
            "schema": "token-reconstruction.trr-p09-stage1-public-inputs.v1",
            "task_id": TASK_ID,
            "status": "CPU_INPUTS_COMPILED_NO_ACTIVATIONS",
            "input_source_schema": "token-reconstruction.trr-p09-stage1-public-inputs.v1",
            "created_utc": utc_now(),
            "plan": {"path": str(countersign["plan"]["path"]), "bytes": PLAN_BYTES, "sha256": PLAN_SHA256, "commit": COUNTERSIGN_COMMIT},
            "countersignature": {"path": "experiments/TRR-P09/setup/stage1-plan-countersignature-r1.json", "attested": True},
            "geometry": {"records": len(rows), "sequence_tokens": SEQUENCE_WIDTH, "b0_prefix_records": B0_ROWS, "new_records": ADDITION_ROWS, "logical_capture_shard_records": 64, "logical_capture_shards": 169, "input_dtype": "int32", "mask_dtype": "uint8", "position_dtype": "int64"},
            "capture_contract": {"expanded_row_origin": B0_ROWS, "new_row_slice": [B0_ROWS, TARGET_ROWS], "capture_consumes_only_new_rows": True, "input_payload_is_one_hash_bound_cpu_file": True},
            "artifacts": {
                "inputs": {"path": "inputs.safetensors", "bytes": input_path.stat().st_size, "sha256": digest_file(input_path)},
                "records": {"path": "records.json", "bytes": records_path.stat().st_size, "sha256": digest_file(records_path)},
            },
            "b0_prefix": {"semantic_input_digest": b0_prefix_digest, "published": dict(b0_meta), "byte_equivalent_prefix_required": True, "full_tensor_copy_verified": True},
            "controlled_replacement_audit": dict(b0_control_audit),
            "strata": {name: {"records": sum(1 for row in rows if row.stratum == name), "post_bos_positions": sum(row.target_post_bos_token_count for row in rows if row.stratum == name)} for name in STRATUM_ORDER},
            "diagnostic": dict(diagnostic),
            "exposure": dict(exposure),
            "exclusions": {"summary": dict(exclusions["summary"]), "source_files": list(exclusions["source_files"])},
            "source_files": dict(source_files),
            "truth_boundary": {
                "public_fitting_labels_loaded": True,
                "source_text_persisted": False,
                "evaluation_truth_opened": False,
                "model_loaded": False,
                "activations_created": False,
            },
        }
        manifest_path = temp / "preparation_manifest.json"
        _write_json_create(manifest_path, summary)
        (temp / "COMPLETE").write_text("CPU_INPUTS_COMPILED_NO_ACTIVATIONS\n", encoding="utf-8", newline="\n")
        if output_root.exists() or output_root.is_symlink():
            raise PreparationErrorLocal("output root appeared during create-only publication")
        os.replace(temp, output_root)
        summary["artifacts"]["manifest"] = {"path": "preparation_manifest.json", "bytes": (output_root / "preparation_manifest.json").stat().st_size, "sha256": digest_file(output_root / "preparation_manifest.json")}
        return summary
    except Exception:
        # Preserve the create-only contract: a failed staging directory is
        # removed only before publication; no existing output is touched.
        import shutil
        shutil.rmtree(temp, ignore_errors=True)
        raise


def compile_inputs(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve()
    plan = load_and_verify_plan(plan_path)
    countersign = verify_countersignature(Path(args.plan_countersignature))
    published_root = Path(args.published_trr0007_root).expanduser().resolve()
    b0_records, b0_token, b0_mask, b0_pos, b0_meta = load_b0(published_root)
    validate_b0_quota(b0_records)
    exclusions = collect_identity_exclusions(plan, Path(args.repository_root).expanduser().resolve(), published_root)
    identity_ids = load_identity_ids(published_root)
    exact_quotas = exact_addition_quotas(b0_records)
    # B0's published rows are copied verbatim into the new input tensor.
    rows: list[InputRow] = []
    for index, meta in enumerate(b0_records):
        active = int(b0_mask[index].sum().item())
        ids = tuple(int(x) for x in b0_token[index, :active].tolist())
        rows.append(InputRow(
            record_id=str(meta["record_id"]), source_record_id=str(meta.get("source_record_id", meta["record_id"])), dataset_key=str(meta["dataset_key"]), stratum=b0_stratum(meta), source_row_index=int(meta.get("source_record_id", "row-0").rsplit("row-", 1)[-1]) if "row-" in str(meta.get("source_record_id", "")) else None, rendered_sha256=str(meta.get("rendered_sha256", "")), source_full_token_count=int(meta.get("source_full_token_count", meta.get("full_token_count", active))), target_post_bos_token_count=active - 1, token_ids=ids, synthetic=bool(meta.get("synthetic", False)), sequence_h128_sha256=h128_digest(ids)))
    used_ids = {row.source_record_id for row in rows}
    used_rendered = {row.rendered_sha256 for row in rows if row.rendered_sha256}
    used_public = {row.public_record_sha256 for row in rows if row.public_record_sha256}
    used_h128 = {row.sequence_h128_sha256 for row in rows if row.sequence_h128_sha256}
    tokenizer = _load_tokenizer(Path(args.tokenizer).expanduser().resolve())
    deadline = _Deadline(__import__("time").monotonic(), float(args.max_seconds))
    source_paths = {
        "alpaca": [Path(args.alpaca_arrow).expanduser().resolve()],
        "pile": [Path(args.pile_arrow).expanduser().resolve()],
        "finance": [Path(x).expanduser().resolve() for x in args.finance_arrow],
    }
    bound_sources = plan["already_bound_inputs"]["source_datasets"]
    source_files = {}
    for key, paths in source_paths.items():
        descriptor = bound_sources[key]
        expected = descriptor.get("arrow_shard_sha256", descriptor.get("arrow_sha256"))
        if isinstance(expected, list):
            if len(expected) != len(paths):
                raise PreparationErrorLocal(f"{key} Arrow shard count changed")
            hashes = [str(x) for x in expected]
        else:
            hashes = [str(expected)] * len(paths)
        source_files[key] = []
        for p, expected_hash in zip(paths, hashes, strict=True):
            actual_hash = digest_file(p)
            if actual_hash != expected_hash:
                raise PreparationErrorLocal(f"{key} Arrow source hash changed: {p}")
            source_files[key].append({"path": str(p), "bytes": p.stat().st_size, "sha256": actual_hash, "hash_status": "VERIFIED_ON_READ"})
    datasets = {key: _load_public_dataset(paths, label=key) for key, paths in source_paths.items()}
    candidate_pools: dict[str, list[_Candidate]] = {}
    scan_stats: dict[str, Any] = {}
    for key in ("alpaca", "pile", "finance"):
        if key == "alpaca":
            index_range = range(len(datasets[key]))
        else:
            spec = SOURCE_PARTITIONS[key]
            index_range = range(int(spec["fit_frequency_start"]), int(spec["fit_frequency_stop"]))
        # Existing TRR-0005 scanner is the shared renderer/exclusion boundary.
        report: dict[str, Any] = {}
        # Avoid reading rows whose dataset-scoped source index is already
        # excluded.  The later candidate check remains as a defense in depth,
        # while this binding keeps source-index and hash namespaces distinct.
        excluded_row_keys = {
            (key, int(index))
            for index in exclusions["source_indices"].get(key, set())
        }
        base_candidates = _scan_fit_candidates(
            key,
            datasets[key],
            index_range,
            tokenizer,
            deadline=deadline,
            excluded_ids=set(exclusions["record_ids"]),
            excluded_row_keys=excluded_row_keys,
            excluded_hashes=set(exclusions["rendered_sha256"]),
            scan_stats=report,
        )
        candidate_pools[key] = []
        for base in base_candidates:
            public_hash = _public_record_sha256(key, datasets[key][int(base.row_index)], base.rendered_sha256)
            if public_hash in exclusions["public_record_sha256"]:
                report["rows_excluded_by_public_record_hash"] = int(report.get("rows_excluded_by_public_record_hash", 0)) + 1
                report["rows_eligible"] = max(0, int(report.get("rows_eligible", 0)) - 1)
                continue
            candidate_pools[key].append(_Stage1Candidate(
                dataset_key=base.dataset_key, dataset_id=base.dataset_id, split=base.split,
                revision=base.revision, row_index=base.row_index, record_id=base.record_id,
                domain=base.domain, rendered_sha256=base.rendered_sha256,
                public_record_sha256=public_hash, token_ids=base.token_ids,
            ))
        scan_stats[key] = report
    b0_parent_rows, b0_parent_meta = load_b0_control_parents(published_root)
    rows, b0_control_audit = bind_b0_controlled_rows(rows, b0_records, b0_token, b0_parent_rows, datasets, tokenizer, identity_ids, deadline)
    additions_by_stratum: dict[str, list[InputRow]] = {}
    cursor = 3600  # B0 controlled rows consume the first published cycle.
    for name, domain, controlled, _target, addition_quota in STRATA:
        chosen = select_candidates(candidate_pools[domain], stratum=name, exact_quota=exact_quotas[name], exclusions=exclusions, used_ids=used_ids, used_rendered=used_rendered, used_h128=used_h128, used_public=used_public)
        if controlled:
            additions_by_stratum[name], cursor = make_controlled_rows(chosen, name, identity_ids, cursor, _special_token_ids(tokenizer))
        else:
            additions_by_stratum[name] = make_natural_rows(chosen, name)
        deadline.check(f"{name} selection")
    for name in STRATUM_ORDER:
        rows.extend(additions_by_stratum[name])
    if len(rows) != TARGET_ROWS:
        raise PreparationErrorLocal(f"compiled {len(rows)} rows, expected {TARGET_ROWS}")
    if cursor != 36000:
        raise PreparationErrorLocal(f"controlled identity exposure cursor {cursor} != 36000")
    diagnostic = per_bank_diagnostic_indices(rows)
    exposure = exposure_summary(rows)
    # Semantic B0 input prefix digest binds the actual copied token/mask/position bytes.
    prefix_payload = canonical_bytes({"token_ids": b0_token.tolist(), "attention_mask": b0_mask.tolist(), "position_ids": b0_pos.tolist()})
    b0_prefix_digest = digest_bytes(prefix_payload)
    b0_control_audit["parent_metadata"] = b0_parent_meta
    # Bind the historical B0 control-plan reference separately from the
    # current signed identity recipe.  The plan is metadata-only and is used
    # to explain the published prefix provenance; it is never substituted for
    # the observed B0 token IDs.
    legacy_plan_path = published_root / "experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json"
    _regular_hash(
        legacy_plan_path,
        "35d74424df60a1d62f7962850eaa3132fa57dd7001024f915cc4fc470a6d0e76",
        label="B0 historical corpus plan",
    )
    legacy_plan = json.loads(legacy_plan_path.read_text(encoding="utf-8"))
    legacy_control = legacy_plan.get("controlled_token_selection", {})
    b0_control_audit["legacy_b0_control_plan"] = {
        "path": str(legacy_plan_path),
        "bytes": legacy_plan_path.stat().st_size,
        "sha256": digest_file(legacy_plan_path),
        "selected_token_id_count": int(legacy_control.get("selected_token_id_count", -1)),
        "selected_token_ids_observed": isinstance(legacy_control.get("selected_token_ids"), list),
        "role": "published_B0_provenance_reference_only",
    }
    b0_control_audit["signed_recipe_reference"] = {
        "selected_token_id_count": len(identity_ids),
        "selected_token_ids_sha256": "0e84f625e24fb4a2038972cf417923dc20a8cc2fd85e55ac6e9b7a846f5ca576",
        "role": "B1_additions_and_signed_future_recipe",
    }
    b0_control_audit["total_replacement_occurrences"] = 36000
    summary = publish_output(Path(args.output_root), rows, plan=plan, countersign=countersign, source_files=source_files, exclusions=exclusions, b0_meta=b0_meta, b0_token=b0_token, b0_mask=b0_mask, b0_pos=b0_pos, b0_prefix_digest=b0_prefix_digest, diagnostic=diagnostic, exposure=exposure, b0_control_audit=b0_control_audit)
    summary["scan_stats"] = scan_stats
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_REPO)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-countersignature", type=Path, required=True)
    parser.add_argument("--published-trr0007-root", type=Path, default=DEFAULT_TRR0007_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--alpaca-arrow", type=Path, default=Path("/home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow"))
    parser.add_argument("--pile-arrow", type=Path, default=Path("/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"))
    parser.add_argument("--finance-arrow", type=Path, nargs="+", default=[Path("/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow"), Path("/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow")])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        compile_inputs(build_parser().parse_args(argv))
    except (PreparationError, PreparationErrorLocal, OSError, ValueError, RuntimeError) as exc:
        print(f"prepare_stage1_inputs: ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
