#!/usr/bin/env python3
"""Prepare the TRR-0006 public-label sidecar after the prediction freeze.

The selected natural records are public source material. This utility
materializes those records only after the complete eight-entry prediction and
timing freeze has been revalidated, then writes the 128-token labels outside
the repository. One label tensor is stored per domain and the scorer joins it
to both paired target conditions; the source panel and observation artifacts
never receive the labels.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

try:  # Repository import form.
    from scripts import trr0005_produce_confirmation as trusted
    from scripts import trr0005_truth_alias_adapter as alias_adapter
    from scripts import trr0006_capture_public as capture
    from scripts import trr0006_freeze_pair as freeze_pair
    from scripts import trr0006_prediction_contract as contract
except ModuleNotFoundError:  # Direct execution with PYTHONPATH=src:scripts.
    import trr0005_produce_confirmation as trusted
    import trr0005_truth_alias_adapter as alias_adapter
    import trr0006_capture_public as capture
    import trr0006_freeze_pair as freeze_pair
    import trr0006_prediction_contract as contract


TASK_ID = contract.TASK_ID
TRUTH_SCHEMA = "token-reconstruction.trr0006-private-label-binding.v1"
TRUTH_FILE_SCHEMA = "token-reconstruction.trr0006-private-label-sidecar.v1"
TRUTH_STATUS = "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT"
TRUTH_TENSOR_KEYS = ("finance__token_ids", "pile__token_ids")
SEQUENCE_TOKENS = contract.STORED_SEQUENCE_TOKENS
POST_BOS_TOKENS = contract.SCORED_POST_BOS_TOKENS
CAPTURE_TOKENS = contract.CAPTURE_SEQUENCE_TOKENS
BOS_TOKEN_ID = contract.BOS_TOKEN_ID
VOCAB_SIZE = contract.VOCAB_SIZE
CELL_ORDER = tuple(contract.CELL_ORDER)
STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
PANEL_SCHEMA = "token-reconstruction.trr0006-public-source-panel.v1"
PANEL_STATUS = "FROZEN_SOURCE_PANEL_NO_TRUTH"
OBSERVATION_SCHEMA = contract.OBSERVATION_SCHEMA


class TruthPreparationError(RuntimeError):
    """Raised when private label preparation cannot satisfy the frozen gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthPreparationError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthPreparationError(f"asset is not a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthPreparationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TruthPreparationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TruthPreparationError(f"{description} must be a JSON object")
    return dict(value)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_inside_task(path: Path, *, root: Path, description: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise TruthPreparationError(f"{description} must not be a symbolic link")
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise TruthPreparationError(f"{description} is unavailable: {resolved}")
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        resolved.relative_to(task_root)
    except ValueError as exc:
        raise TruthPreparationError(f"{description} escaped the task root: {resolved}") from exc
    return resolved


def _outside_destination(path: Path, *, root: Path, description: str) -> Path:
    raw = path.expanduser()
    if raw.exists() or raw.is_symlink():
        raise TruthPreparationError(f"{description} is create-only and already exists: {raw}")
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    raise TruthPreparationError(f"{description} must be outside the repository: {resolved}")


def _selection_and_rows(
    *,
    root: Path,
    decision_plan_path: Path,
    source_selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    try:
        frozen, selection, rows = capture._validate_frozen_selection(
            decision_plan_path.expanduser().resolve(),
            source_selection_path.expanduser().resolve(),
        )
    except (capture.CaptureError, capture.selector.SelectionError, OSError, ValueError) as exc:
        raise TruthPreparationError(f"frozen source selection is invalid: {exc}") from exc
    if frozen["records_per_domain"] != selection["records_per_domain"]:
        raise TruthPreparationError("selection count differs from the frozen decision plan")
    return frozen, selection, rows


def _panel_bindings(
    *,
    root: Path,
    panel_path: Path,
    observation_manifest_path: Path,
    source_selection_path: Path,
    selection: Mapping[str, Any],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    records_per_domain: int,
) -> dict[str, Any]:
    panel_file = _resolve_inside_task(panel_path, root=root, description="TRR-0006 source panel")
    observation_file = _resolve_inside_task(
        observation_manifest_path,
        root=root,
        description="TRR-0006 observation manifest",
    )
    selection_file = _resolve_inside_task(
        source_selection_path,
        root=root,
        description="TRR-0006 source selection",
    )
    panel = _load_json(panel_file, description="TRR-0006 source panel")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise TruthPreparationError("source panel schema or task ID changed")
    if panel.get("status") != PANEL_STATUS:
        raise TruthPreparationError("source panel is not frozen without truth")
    if panel.get("records_per_domain") != records_per_domain:
        raise TruthPreparationError("source panel record count changed")
    if panel.get("cell_order") != list(CELL_ORDER):
        raise TruthPreparationError("source panel cell order changed")
    if panel.get("target_conditions") != list(CONDITION_ORDER):
        raise TruthPreparationError("source panel target conditions changed")
    expected_ranges = {"pile": [7000, 10000], "finance": [12000, 20000]}
    if panel.get("source_ranges_half_open") != expected_ranges:
        raise TruthPreparationError("source panel ranges changed")
    if panel.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS:
        raise TruthPreparationError("source panel clip length changed")
    if panel.get("scored_post_bos_tokens") != POST_BOS_TOKENS:
        raise TruthPreparationError("source panel post-BOS length changed")
    if panel.get("capture_batch_records") != contract.CAPTURE_BATCH_RECORDS:
        raise TruthPreparationError("source panel capture batch changed")
    if panel.get("capture_sequence_tokens") != CAPTURE_TOKENS:
        raise TruthPreparationError("source panel capture sequence changed")
    if panel.get("truth_opened") is not False or panel.get("public_material_only") is not True:
        raise TruthPreparationError("source panel truth/public-material flags changed")

    selection_record = _file_record(selection_file)
    declared_selection = panel.get("selection_plan")
    if not isinstance(declared_selection, Mapping) or dict(declared_selection) != selection_record:
        raise TruthPreparationError("source panel selection binding changed")
    observation_record = _file_record(observation_file)
    declared_observation = panel.get("observation_manifest")
    if not isinstance(declared_observation, Mapping) or dict(declared_observation) != observation_record:
        raise TruthPreparationError("source panel observation binding changed")

    expected_digests = {
        style: _json_digest([row["record_id"] for row in rows[style]]) for style in STYLE_ORDER
    }
    if panel.get("record_ids_sha256") != expected_digests:
        raise TruthPreparationError("source panel record-ID digest differs from frozen selection")
    if panel.get("method_freeze_sha256") != selection.get("method_freeze_sha256"):
        raise TruthPreparationError("source panel method freeze differs from selection")

    observation_manifest = _load_json(observation_file, description="TRR-0006 observation manifest")
    if observation_manifest.get("schema") != OBSERVATION_SCHEMA or observation_manifest.get("task_id") != TASK_ID:
        raise TruthPreparationError("observation manifest schema or task ID changed")
    if observation_manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise TruthPreparationError("observation manifest is not frozen without truth")
    if observation_manifest.get("records_per_domain") != records_per_domain:
        raise TruthPreparationError("observation manifest record count changed")
    if observation_manifest.get("cell_order") != list(CELL_ORDER):
        raise TruthPreparationError("observation manifest cell order changed")
    if observation_manifest.get("source_text_written") is not False or observation_manifest.get("token_ids_written") is not False:
        raise TruthPreparationError("observation manifest private flags changed")
    if observation_manifest.get("truth_opened") is not False:
        raise TruthPreparationError("observation manifest truth flag changed")
    observation_hashes: dict[str, str] = {}
    cells = observation_manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != len(CELL_ORDER):
        raise TruthPreparationError("observation manifest cells are incomplete")
    for row in cells:
        if not isinstance(row, Mapping) or row.get("cell_id") not in CELL_ORDER:
            raise TruthPreparationError("observation manifest cell is malformed")
        cell_id = str(row["cell_id"])
        observation = row.get("observation")
        if not isinstance(observation, Mapping) or not isinstance(observation.get("sha256"), str):
            raise TruthPreparationError(f"observation binding is absent: {cell_id}")
        observation_hashes[cell_id] = str(observation["sha256"])
    if set(observation_hashes) != set(CELL_ORDER):
        raise TruthPreparationError("observation manifest cell set changed")
    return {
        "panel": panel,
        "panel_record": _file_record(panel_file),
        "observation_manifest": observation_manifest,
        "observation_record": observation_record,
        "observation_hashes": observation_hashes,
        "selection_record": selection_record,
        "record_ids_sha256": expected_digests,
    }


def _source_paths(selection: Mapping[str, Any], style: str) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise TruthPreparationError(f"selection has no frozen {style} Arrow descriptor")
    result: list[Path] = []
    for value in files:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise TruthPreparationError(f"selection {style} Arrow descriptor is malformed")
        result.append(Path(str(value["path"])).expanduser().resolve())
    return tuple(result)


def _tokenizer_path(selection: Mapping[str, Any]) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise TruthPreparationError("selection has no frozen tokenizer descriptor")
    return Path(str(descriptor["path"])).expanduser().resolve()


def _build_truth_tensors(
    records: Mapping[str, Sequence[Any]],
    *,
    records_per_domain: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Build one cloned label tensor per source domain.

    The two target conditions are paired on the same natural records.  The
    binding therefore stores one domain tensor and the scorer aliases that
    tensor to both target cells after the public gate.  ``clone`` at this
    boundary makes the serialized storage independent from the materialized
    source objects and from any future alias map.
    """

    tensors: dict[str, torch.Tensor] = {}
    record_digests: dict[str, str] = {}
    for style in STYLE_ORDER:
        rows = list(records[style])
        if len(rows) != records_per_domain:
            raise TruthPreparationError(f"truth source count changed: {style}")
        ids = [str(row.record_id) for row in rows]
        if len(set(ids)) != records_per_domain:
            raise TruthPreparationError(f"truth source IDs are not unique: {style}")
        labels = torch.tensor(
            [list(row.token_ids[:SEQUENCE_TOKENS]) for row in rows],
            dtype=torch.int64,
        )
        expected_shape = (records_per_domain, SEQUENCE_TOKENS)
        if tuple(labels.shape) != expected_shape:
            raise TruthPreparationError(f"truth label geometry changed: {style}")
        if not labels[:, 0].eq(BOS_TOKEN_ID).all().item():
            raise TruthPreparationError(f"truth BOS changed: {style}")
        if labels.lt(0).any().item() or labels.ge(VOCAB_SIZE).any().item():
            raise TruthPreparationError(f"truth label vocabulary changed: {style}")
        # Keep the domain/condition aliasing explicit in the caller's model:
        # one clone is serialized and reused for both paired target cells.
        tensors[f"{style}__token_ids"] = labels.detach().contiguous().clone()
        record_digests[style] = _json_digest(ids)
    if set(tensors) != set(TRUTH_TENSOR_KEYS):
        raise TruthPreparationError("truth tensor key matrix is incomplete")
    return tensors, record_digests


def _source_id_digests(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {style: _json_digest([str(row["record_id"]) for row in rows[style]]) for style in STYLE_ORDER}


def _code_bindings(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "truth_preparation": Path(__file__).resolve(),
        "truth_alias_adapter": Path(alias_adapter.__file__).resolve(),
        "trusted_trr0005_producer": Path(trusted.__file__).resolve(),
        "public_capture_materializer": Path(capture.__file__).resolve(),
        "public_freeze_gate": root / "scripts/trr0006_freeze_pair.py",
        "prediction_contract": root / "scripts/trr0006_prediction_contract.py",
    }
    return {name: _file_record(path) for name, path in paths.items()}


def prepare_truth(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise TruthPreparationError("truth preparation requires explicit --execute after the prediction freeze")
    root = args.repository_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TruthPreparationError(f"repository root is unavailable: {root}")
    truth_path = _outside_destination(args.truth_output, root=root, description="truth sidecar")
    manifest_arg = args.truth_manifest
    manifest_path = (
        manifest_arg.expanduser()
        if manifest_arg is not None
        else truth_path.with_name(truth_path.stem + ".manifest.json")
    )
    manifest_path = _outside_destination(manifest_path, root=root, description="truth binding manifest")
    started_utc = _utc_now()
    started_clock = time.perf_counter()

    # This gate is deliberately the first operation that can authorize label
    # preparation. It rehashes the complete public prediction/timing matrix
    # and does not inspect the truth path or truth bytes.
    try:
        gate = freeze_pair.validate_before_truth(
            repository_root=root,
            registration_path=args.registration,
            receipt_path=args.prediction_freeze,
            observation_manifest_path=args.observation_manifest,
            plan_path=args.decision_plan,
            truth_path=truth_path,
        )
    except Exception as exc:
        raise TruthPreparationError(f"complete public prediction freeze is required before truth: {exc}") from exc
    if gate.get("verified_before_truth") is not True or gate.get("truth_opened") is not False:
        raise TruthPreparationError("public prediction gate did not return a verified closed receipt")

    frozen, selection, rows = _selection_and_rows(
        root=root,
        decision_plan_path=args.decision_plan,
        source_selection_path=args.source_selection,
    )
    records_per_domain = int(frozen["records_per_domain"])
    panel = _panel_bindings(
        root=root,
        panel_path=args.panel,
        observation_manifest_path=args.observation_manifest,
        source_selection_path=args.source_selection,
        selection=selection,
        rows=rows,
        records_per_domain=records_per_domain,
    )
    expected_ids = _source_id_digests(rows)
    if expected_ids != panel["record_ids_sha256"]:
        raise TruthPreparationError("selection and panel source-ID digests differ")
    decision_plan_file = _resolve_inside_task(
        args.decision_plan,
        root=root,
        description="TRR-0006 decision plan",
    )
    decision_plan_record = _file_record(decision_plan_file)

    tokenizer_path = (
        args.tokenizer.expanduser().resolve() if args.tokenizer is not None else _tokenizer_path(selection)
    )
    pile_paths = tuple(path.expanduser().resolve() for path in (args.pile_arrow or _source_paths(selection, "pile")))
    finance_paths = tuple(path.expanduser().resolve() for path in (args.finance_arrow or _source_paths(selection, "finance")))
    try:
        capture._validate_source_descriptors(
            selection,
            pile_paths=pile_paths,
            finance_paths=finance_paths,
            tokenizer_path=tokenizer_path,
        )
        tokenizer = trusted._load_tokenizer(tokenizer_path)
        datasets = {
            "pile": trusted._load_arrow_dataset(pile_paths),
            "finance": trusted._load_arrow_dataset(finance_paths),
        }
        records = capture._materialize_selected(rows, datasets=datasets, tokenizer=tokenizer)
    except Exception as exc:
        raise TruthPreparationError(f"selected public materialization changed: {exc}") from exc
    actual_ids = {
        style: [str(record.record_id) for record in records[style]] for style in STYLE_ORDER
    }
    declared_ids = {style: [str(row["record_id"]) for row in rows[style]] for style in STYLE_ORDER}
    if actual_ids != declared_ids:
        raise TruthPreparationError("materialized truth row order differs from frozen selection")

    tensors, record_digests = _build_truth_tensors(records, records_per_domain=records_per_domain)
    observation_metadata = json.dumps(
        panel["observation_hashes"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    record_metadata = json.dumps(
        record_digests, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    metadata = {
        "schema": TRUTH_FILE_SCHEMA,
        "task_id": TASK_ID,
        "decision_plan_sha256": str(decision_plan_record["sha256"]),
        "source_selection_sha256": str(panel["selection_record"]["sha256"]),
        "observation_sha256": observation_metadata,
        "record_ids_sha256": record_metadata,
        "record_ids_pile": json.dumps(actual_ids["pile"], separators=(",", ":"), ensure_ascii=False),
        "record_ids_finance": json.dumps(actual_ids["finance"], separators=(",", ":"), ensure_ascii=False),
        "method_freeze_sha256": str(selection["method_freeze_sha256"]),
        "truth_source": "selected public rows; evaluator-side labels only",
        "labels_shared_across_target_conditions": "true",
        "truth_opened": "false",
    }
    # The adapter clones each tensor at the serialization boundary.  The
    # scorer subsequently aliases each domain tensor to its two paired public
    # target conditions; no target labels or model state are loaded here.
    alias_adapter.save_file_alias_safe(tensors, str(truth_path), metadata=metadata)
    truth_record = _file_record(truth_path)
    truth_tensor_digests = {
        key: contract.tensor_digest(tensors[key]) for key in TRUTH_TENSOR_KEYS
    }
    code_bindings = _code_bindings(root)
    truth_binding = {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "truth_file": truth_record,
        "decision_plan": decision_plan_record,
        "source_selection": panel["selection_record"],
        "panel": panel["panel_record"],
        "observation_manifest": panel["observation_record"],
        "decision_plan_sha256": str(decision_plan_record["sha256"]),
        "source_selection_sha256": str(panel["selection_record"]["sha256"]),
        "observation_sha256": panel["observation_hashes"],
        "record_ids_sha256": record_digests,
        "truth_tensor_keys": list(TRUTH_TENSOR_KEYS),
        "truth_tensor_sha256": truth_tensor_digests,
        "records_per_domain": records_per_domain,
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": POST_BOS_TOKENS,
        "cell_order": list(CELL_ORDER),
        "target_conditions": list(CONDITION_ORDER),
        "labels_shared_across_target_conditions": True,
        "truth_source": "selected public rows; evaluator-side labels only",
        "source_text_loaded_for_label_materialization": True,
        "target_model_or_target_labels_loaded": False,
        "reconstruction_root_contains_truth": False,
        "truth_opened": False,
        "pretruth_gate": {
            "status": gate["status"],
            "verified_before_truth": True,
            "truth_opened": False,
            "receipt_path": gate["receipt_path"],
            "entry_count": gate["entry_count"],
            "code_commit": gate["code_commit"],
        },
        "code_bindings": code_bindings,
        "execution": {
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "command": list(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_loaded": False,
            "target_model_loaded": False,
            "truth_created": True,
            "truth_opened": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(truth_binding, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    binding_record = _file_record(manifest_path)
    return {
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "truth_file": truth_record,
        "truth_binding": binding_record,
        "truth_manifest": binding_record,
        "records_per_domain": records_per_domain,
        "truth_tensor_keys": list(TRUTH_TENSOR_KEYS),
        "truth_opened": False,
    }


def load_truth_tensor_map(manifest_path: Path) -> dict[str, torch.Tensor]:
    """Load the two domain labels after the caller has passed the public gate.

    The sidecar stores exactly ``pile__token_ids`` and
    ``finance__token_ids``.  A scorer joins each domain tensor to both paired
    target conditions; this loader deliberately returns no masks or
    positions, which remain public observation geometry.
    """

    manifest = _load_json(manifest_path, description="TRR-0006 truth binding")
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise TruthPreparationError("truth binding schema or task ID changed")
    if manifest.get("status") != TRUTH_STATUS or manifest.get("truth_opened") is not False:
        raise TruthPreparationError("truth binding is not a prepared closed binding")
    if manifest.get("reconstruction_root_contains_truth") is not False:
        raise TruthPreparationError("truth sidecar is inside the reconstruction root")
    truth_file = manifest.get("truth_file")
    if not isinstance(truth_file, Mapping) or not isinstance(truth_file.get("path"), str):
        raise TruthPreparationError("truth file binding is absent")
    truth_path = Path(str(truth_file["path"])).expanduser().resolve()
    if truth_path.is_symlink() or not truth_path.is_file():
        raise TruthPreparationError(f"truth sidecar is unavailable: {truth_path}")
    actual = _file_record(truth_path)
    if dict(truth_file) != actual:
        raise TruthPreparationError("truth sidecar bytes or hash differ from binding")
    count = manifest.get("records_per_domain")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise TruthPreparationError("truth record count is malformed")
    declared_keys = manifest.get("truth_tensor_keys")
    if not isinstance(declared_keys, list) or set(declared_keys) != set(TRUTH_TENSOR_KEYS) or len(declared_keys) != len(TRUTH_TENSOR_KEYS):
        raise TruthPreparationError("truth tensor key declaration changed")
    result: dict[str, torch.Tensor] = {}
    with safe_open(str(truth_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(TRUTH_TENSOR_KEYS):
            raise TruthPreparationError("truth tensor key matrix changed")
        metadata = dict(handle.metadata() or {})
        if metadata.get("schema") != TRUTH_FILE_SCHEMA or metadata.get("task_id") != TASK_ID:
            raise TruthPreparationError("truth tensor schema/task binding changed")
        if metadata.get("truth_opened") not in (None, "false", "False"):
            raise TruthPreparationError("truth tensor is marked open")
        if metadata.get("decision_plan_sha256") != manifest.get("decision_plan_sha256"):
            raise TruthPreparationError("truth tensor decision-plan binding changed")
        if metadata.get("source_selection_sha256") != manifest.get("source_selection_sha256"):
            raise TruthPreparationError("truth tensor source-selection binding changed")
        if metadata.get("observation_sha256") != json.dumps(
            manifest.get("observation_sha256"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ):
            raise TruthPreparationError("truth tensor observation binding changed")
        for key in TRUTH_TENSOR_KEYS:
            labels = handle.get_tensor(key).to(torch.int64).contiguous()
            expected_shape = (count, SEQUENCE_TOKENS)
            if tuple(labels.shape) != expected_shape:
                raise TruthPreparationError(f"truth tensor geometry changed: {key}")
            if not labels[:, 0].eq(BOS_TOKEN_ID).all().item() or labels.lt(0).any().item() or labels.ge(VOCAB_SIZE).any().item():
                raise TruthPreparationError(f"truth labels changed: {key}")
            result[key] = labels
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    prepare = parser.add_subparsers(dest="command", required=True).add_parser(
        "prepare", help="prepare labels after a complete frozen public prediction matrix"
    )
    prepare.add_argument("--execute", action="store_true", help="required acknowledgment for truth preparation")
    prepare.add_argument("--repository-root", type=Path, default=Path("."))
    prepare.add_argument("--decision-plan", type=Path, required=True)
    prepare.add_argument("--source-selection", type=Path, required=True)
    prepare.add_argument("--registration", type=Path, required=True)
    prepare.add_argument("--prediction-freeze", type=Path, required=True)
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--observation-manifest", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path)
    prepare.add_argument("--pile-arrow", type=Path, nargs="*")
    prepare.add_argument("--finance-arrow", type=Path, nargs="*")
    prepare.add_argument("--truth-output", type=Path, required=True)
    prepare.add_argument("--truth-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "prepare":  # pragma: no cover
        raise TruthPreparationError(f"unknown command: {args.command}")
    result = prepare_truth(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TruthPreparationError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-0006 truth preparation error: {exc}")
