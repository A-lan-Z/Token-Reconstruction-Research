"""TRR-0009 truth preparation and scoring boundary.

``prepare`` revalidates the public freeze, then materializes labels from the
already-frozen metadata-only selection using the trusted public producer
helpers.  It writes the private sidecar outside the repository and a
metadata-only binding header under the task root.  ``score`` validates that
header and the public freeze before opening the sidecar once.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors.torch import save_file, safe_open

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0009_eval_capture as capture
from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_gate as gate
from scripts import trr0009_score as scorer

TRUTH_BINDING_SCHEMA = "token-reconstruction.trr0009-truth-binding.v1"
TRUTH_SIDECAR_SCHEMA = "token-reconstruction.trr0009-truth-sidecar.v1"
TRUTH_STATUS = "TRR0009_TRUTH_PREPARED_AFTER_PUBLIC_GATE"
TRUTH_SIDECAR_KEYS = tuple(f"{cell}__token_ids" for cell in contract.CELL_ORDER)


class TruthGateError(contract.ContractError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TruthGateError(f"repository root unavailable: {root}")
    return root


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        return contract.validate_file_record({"path": str(path), "bytes": path.stat().st_size, "sha256": contract.sha256_file(path)}, repository_root=root, description=description, verify=True)
    except (OSError, contract.ContractError) as exc:
        raise TruthGateError(str(exc)) from exc


def _write_json(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        record = contract.write_create_only(path, payload)
    except contract.ContractError as exc:
        raise TruthGateError(str(exc)) from exc
    return record


def _resolve(value: Path | str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _task_header(value: Path, *, root: Path) -> Path:
    path = _resolve(value, root=root)
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise TruthGateError(f"truth binding header must be under {task_root}") from exc
    if path.exists() or path.is_symlink():
        raise TruthGateError(f"truth binding header is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _outside_sidecar(value: Path, *, root: Path, output_root: Path) -> Path:
    path = Path(value).expanduser().resolve()
    for directory, label in ((root, "repository"), (output_root, "prediction output")):
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            continue
        raise TruthGateError(f"truth sidecar must be outside {label}: {path}")
    if path.exists() or path.is_symlink():
        raise TruthGateError(f"truth sidecar is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise TruthGateError(f"{description} {key} binding changed")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise TruthGateError(str(exc)) from exc


def _load_public_context(*, freeze_path: Path, repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _root(repository_root)
    freeze = gate.validate_before_truth(freeze_path=freeze_path, repository_root=root, require_current_head=True)
    freeze_record = _record(freeze_path, root=root, description="TRR-0009 public freeze receipt")
    registration = _load_json(Path(str(freeze["registration"]["path"])), description="TRR-0009 registration")
    observation = _load_json(Path(str(registration["observation_manifest"]["path"])), description="TRR-0009 observations")
    if registration.get("frequency_reference") is None or registration.get("source_selection") is None or registration.get("capture_receipt") is None:
        raise TruthGateError("registration lacks sealed public context bindings")
    return freeze, registration, observation, root, freeze_record


def _source_paths(selection: Mapping[str, Any], *, root: Path, style: str) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise TruthGateError(f"selection has no {style} Arrow descriptors")
    return tuple(_resolve(str(item["path"]), root=root) for item in files if isinstance(item, Mapping) and isinstance(item.get("path"), str))


def _tokenizer_path(selection: Mapping[str, Any], *, root: Path) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise TruthGateError("selection has no tokenizer descriptor")
    return _resolve(str(descriptor["path"]), root=root)


def _source_evidence(selection: Mapping[str, Any], *, root: Path, tokenizer_path: Path, pile_paths: Sequence[Path], finance_paths: Sequence[Path]) -> dict[str, Any]:
    """Bind the exact public input files used by truth materialization.

    This records file hashes rather than only paths so a later score cannot
    silently reuse a sidecar with a changed tokenizer or Arrow input.
    """
    try:
        expected = selection["public_sources_frozen"]
        if not isinstance(expected, Mapping):
            raise KeyError("public_sources_frozen")
        expected_tokenizer = str(expected["tokenizer"]["path"])
        expected_pile = [str(item["path"]) for item in expected["pile"]["arrow_files"]]
        expected_finance = [str(item["path"]) for item in expected["finance"]["arrow_files"]]
    except (KeyError, TypeError, IndexError) as exc:
        raise TruthGateError("selection public source descriptors are incomplete") from exc
    actual_paths = {
        "tokenizer": [str(Path(tokenizer_path).resolve())],
        "pile_arrow": [str(Path(path).resolve()) for path in pile_paths],
        "finance_arrow": [str(Path(path).resolve()) for path in finance_paths],
    }
    if actual_paths["tokenizer"] != [str(_resolve(expected_tokenizer, root=root))] or actual_paths["pile_arrow"] != [str(_resolve(path, root=root)) for path in expected_pile] or actual_paths["finance_arrow"] != [str(_resolve(path, root=root)) for path in expected_finance]:
        raise TruthGateError("truth input paths differ from frozen public selection")
    return {
        "tokenizer": _record(Path(actual_paths["tokenizer"][0]), root=root, description="truth tokenizer input"),
        "pile_arrow": [_record(Path(path), root=root, description="truth Pile Arrow input") for path in actual_paths["pile_arrow"]],
        "finance_arrow": [_record(Path(path), root=root, description="truth Finance Arrow input") for path in actual_paths["finance_arrow"]],
    }


def _validate_sidecar_header(*, freeze: Mapping[str, Any], registration: Mapping[str, Any], freeze_record: Mapping[str, Any], header_path: Path, repository_root: Path, open_tensor_data: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _root(repository_root)
    header_record = _record(header_path, root=root, description="TRR-0009 truth binding header")
    header = _load_json(header_path, description="TRR-0009 truth binding header")
    if header.get("schema") != TRUTH_BINDING_SCHEMA or header.get("task_id") != contract.TASK_ID or header.get("status") != TRUTH_STATUS:
        raise TruthGateError("truth binding identity/status changed")
    if header.get("truth_opened") is not False or header.get("prepared_after_public_gate") is not True:
        raise TruthGateError("truth binding is not closed before truth")
    for key, expected in (("registration", freeze["registration"]), ("receipt", freeze_record), ("source_selection", registration["source_selection"]), ("observation_manifest", registration["observation_manifest"]), ("capture_receipt", registration["capture_receipt"])):
        actual = header.get(key)
        if not isinstance(actual, Mapping):
            raise TruthGateError(f"truth binding {key} is absent")
        _same_record(actual, expected, description=f"truth binding {key}")
    if header.get("records_by_domain") != registration.get("records_by_domain") or header.get("cell_order") != list(contract.CELL_ORDER) or header.get("target_conditions") != list(contract.TARGET_ORDER):
        raise TruthGateError("truth binding panel geometry changed")
    cells = header.get("cells")
    if not isinstance(cells, list) or {str(row.get("cell_id")) for row in cells if isinstance(row, Mapping)} != set(contract.CELL_ORDER):
        raise TruthGateError("truth binding cell matrix changed")
    sidecar_declared = header.get("sidecar")
    if not isinstance(sidecar_declared, Mapping):
        raise TruthGateError("truth sidecar binding is absent")
    sidecar_path = Path(str(sidecar_declared.get("path"))).expanduser().resolve()
    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    for directory, label in ((root, "repository"), (output_root.resolve(), "prediction output")):
        try:
            sidecar_path.relative_to(directory)
        except ValueError:
            continue
        raise TruthGateError(f"truth sidecar must be outside {label}: {sidecar_path}")
    sidecar_record = _record(sidecar_path, root=root, description="TRR-0009 truth sidecar")
    _same_record(sidecar_record, sidecar_declared, description="truth sidecar")
    try:
        with safe_open(str(sidecar_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            keys = set(handle.keys())
            if open_tensor_data:
                tensors = {key: handle.get_tensor(key) for key in sorted(keys)}
            else:
                tensors = None
    except Exception as exc:
        raise TruthGateError("truth sidecar is not a readable safetensors file") from exc
    expected_registration_sha = str(freeze["registration"]["sha256"])
    if metadata.get("schema") != TRUTH_SIDECAR_SCHEMA or metadata.get("task_id") != contract.TASK_ID or metadata.get("truth_opened") != "false":
        raise TruthGateError("truth sidecar metadata identity changed")
    if metadata.get("registration_sha256") != expected_registration_sha or metadata.get("source_selection_sha256") != str(registration["source_selection"]["sha256"]) or metadata.get("capture_receipt_sha256") != str(registration["capture_receipt"]["sha256"]) or metadata.get("observation_manifest_sha256") != str(registration["observation_manifest"]["sha256"]):
        raise TruthGateError("truth sidecar public binding changed")
    if metadata.get("target_model_or_target_labels_loaded") != "false" or metadata.get("sequence_tokens") != str(contract.STORED_SEQUENCE_TOKENS) or metadata.get("scored_post_bos_tokens") != str(contract.SCORED_POST_BOS_TOKENS):
        raise TruthGateError("truth sidecar geometry/target metadata changed")
    try:
        observed_counts = json.loads(str(metadata.get("records_by_domain")))
        observed_digests = json.loads(str(metadata.get("observation_record_ids_sha256")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise TruthGateError("truth sidecar count/digest metadata is malformed") from exc
    observation_manifest = _load_json(Path(str(registration["observation_manifest"]["path"])), description="TRR-0009 observations")
    expected_digests = {
        style: str(contract._as_cells(observation_manifest)[f"{style}__public_base"].get("record_ids_sha256"))
        for style in contract.DOMAIN_ORDER
    }
    if observed_counts != registration.get("records_by_domain") or observed_digests != expected_digests:
        raise TruthGateError("truth sidecar record counts or observation pairing changed")
    if keys != set(TRUTH_SIDECAR_KEYS):
        raise TruthGateError("truth sidecar tensor key matrix changed")
    selection = _load_json(Path(str(registration["source_selection"]["path"])), description="TRR-0009 source selection")
    expected_source_evidence = _source_evidence(
        selection,
        root=root,
        tokenizer_path=_tokenizer_path(selection, root=root),
        pile_paths=_source_paths(selection, root=root, style="pile"),
        finance_paths=_source_paths(selection, root=root, style="finance"),
    )
    source_evidence = header.get("source_evidence")
    if not isinstance(source_evidence, Mapping) or source_evidence != expected_source_evidence:
        raise TruthGateError("truth binding source identity evidence changed")
    return header, {"header": header_record, "sidecar": sidecar_record, "metadata": metadata, "tensors": tensors}


def _load_predictions(freeze: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    rows = freeze.get("predictions")
    if not isinstance(rows, Mapping):
        raise TruthGateError("freeze prediction matrix is absent")
    result: dict[str, dict[str, torch.Tensor]] = {method: {} for method in contract.METHOD_ORDER}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            binding = rows.get(key)
            if not isinstance(binding, Mapping):
                raise TruthGateError(f"frozen prediction is absent: {key}")
            try:
                values, _metadata = contract.load_prediction_file(Path(str(binding["path"])), records=int(binding["records"]), expected_metadata={"schema": contract.PREDICTION_SCHEMA, "task_id": contract.TASK_ID, "cell_id": cell_id, "method_id": method_id, "truth_opened": "false", "candidate_arrays_persisted": "false"})
            except contract.ContractError as exc:
                raise TruthGateError(str(exc)) from exc
            if contract.tensor_digest(values) != str(binding.get("prediction_sha256")):
                raise TruthGateError(f"frozen prediction tensor changed: {key}")
            result[method_id][cell_id] = values
    return result


def _frequency_counts(path: Path, *, root: Path, registration: Mapping[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    actual = _record(path, root=root, description="public fitting-frequency reference")
    _same_record(actual, registration["frequency_reference"], description="frequency reference")
    payload = _load_json(path, description="public fitting-frequency reference")
    if payload.get("schema") != "token-reconstruction.trr0005-frequency-references.v1" or payload.get("task_id") != "TRR-0005" or payload.get("status") != "PUBLIC_FITTING_FREQUENCY_REFERENCES":
        raise TruthGateError("frequency reference is not the public fitting-only map")
    metadata = payload.get("metadata")
    if not contract.public_frequency_metadata_is_truth_free(metadata):
        raise TruthGateError("frequency reference records forbidden access")
    try:
        counts = scorer.normalize_frequency_reference(payload)
    except scorer.ScoreError as exc:
        raise TruthGateError(str(exc)) from exc
    return counts, actual


def _cost_evidence(freeze: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    registration = _load_json(Path(str(freeze["registration"]["path"])), description="TRR-0009 registration")
    run = _load_json(Path(str(freeze["run_manifest"]["path"])), description="TRR-0009 run manifest")
    methods = registration.get("methods")
    startup = run.get("model_startup")
    if not isinstance(methods, list) or not isinstance(startup, Mapping):
        raise TruthGateError("fit/deployment/preparation cost evidence is absent")
    costs = {}
    for row in methods:
        if not isinstance(row, Mapping) or row.get("id") not in contract.METHOD_ORDER:
            raise TruthGateError("method cost row is malformed")
        costs[str(row["id"])] = {"fit_cost": row.get("fit_cost"), "deployment_cost": row.get("deployment_cost")}
    return {"method_fit_and_deployment": costs, "model_preparation": dict(startup), "run_manifest": dict(freeze["run_manifest"])}


def _materialize_truth(*, selection: Mapping[str, Any], rows: Mapping[str, Sequence[Mapping[str, Any]]], root: Path, tokenizer_path: Path | None, pile_paths: Sequence[Path] | None, finance_paths: Sequence[Path] | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    actual_tokenizer = tokenizer_path or _tokenizer_path(selection, root=root)
    actual_pile = tuple(pile_paths or _source_paths(selection, root=root, style="pile"))
    actual_finance = tuple(finance_paths or _source_paths(selection, root=root, style="finance"))
    trr6_capture._validate_source_descriptors(selection, pile_paths=actual_pile, finance_paths=actual_finance, tokenizer_path=actual_tokenizer)
    tokenizer = trusted._load_tokenizer(actual_tokenizer)
    datasets = {"pile": trusted._load_arrow_dataset(actual_pile), "finance": trusted._load_arrow_dataset(actual_finance)}
    records = trr6_capture._materialize_selected(rows, datasets=datasets, tokenizer=tokenizer)
    labels: dict[str, torch.Tensor] = {}
    for style in contract.DOMAIN_ORDER:
        selected_ids = [str(row["record_id"]) for row in rows[style]]
        actual_ids = [str(record.record_id) for record in records[style]]
        if selected_ids != actual_ids:
            raise TruthGateError(f"materialized truth row order differs from frozen selection: {style}")
        values = torch.tensor([list(record.token_ids[:contract.STORED_SEQUENCE_TOKENS]) for record in records[style]], dtype=torch.long).contiguous()
        expected = int(selection["records_by_domain"][style])
        if tuple(values.shape) != (expected, contract.STORED_SEQUENCE_TOKENS) or not values[:, 0].eq(contract.BOS_TOKEN_ID).all().item() or values.lt(0).any().item() or values.ge(contract.VOCABULARY_SIZE).any().item():
            raise TruthGateError(f"truth tensor geometry or vocabulary changed: {style}")
        labels[style] = values
    tensors = {f"{cell_id}__token_ids": labels[cell_id.split("__", 1)[0]] for cell_id in contract.CELL_ORDER}
    return tensors, {
        **_source_evidence(selection, root=root, tokenizer_path=actual_tokenizer, pile_paths=actual_pile, finance_paths=actual_finance),
        "records_by_domain": dict(selection["records_by_domain"]),
    }


def prepare_truth(*, freeze_path: Path, registration_path: Path | None, selection_path: Path, truth_output: Path, truth_binding_path: Path, repository_root: Path, tokenizer_path: Path | None = None, pile_paths: Sequence[Path] | None = None, finance_paths: Sequence[Path] | None = None, execute: bool = False) -> dict[str, Any]:
    if not execute:
        raise TruthGateError("truth preparation requires explicit execute authorization")
    root = _root(repository_root)
    freeze, registration, observation, _, freeze_record = _load_public_context(freeze_path=freeze_path, repository_root=root)
    if registration_path is not None and _resolve(registration_path, root=root) != Path(str(freeze["registration"]["path"])).resolve():
        raise TruthGateError("supplied registration differs from the frozen receipt")
    selected_path = _resolve(selection_path, root=root)
    if selected_path != Path(str(registration["source_selection"]["path"])).resolve():
        raise TruthGateError("supplied selection differs from frozen registration")
    selection, selection_record, rows, counts = capture.load_selection(selected_path, repository_root=root)
    if counts != registration.get("records_by_domain"):
        raise TruthGateError("selection counts differ from registration")
    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    sidecar_path = _outside_sidecar(truth_output, root=root, output_root=output_root.resolve())
    tensors, source_evidence = _materialize_truth(selection=selection, rows=rows, root=root, tokenizer_path=tokenizer_path, pile_paths=pile_paths, finance_paths=finance_paths)
    observation_cells = contract._as_cells(observation)
    record_digests = {style: observation_cells[f"{style}__public_base"].get("record_ids_sha256") for style in contract.DOMAIN_ORDER}
    if any(observation_cells[cell].get("record_ids_sha256") != record_digests[cell.split("__", 1)[0]] for cell in contract.CELL_ORDER):
        raise TruthGateError("observation target pairing changed")
    save_file(tensors, str(sidecar_path), metadata={"schema": TRUTH_SIDECAR_SCHEMA, "task_id": contract.TASK_ID, "truth_opened": "false", "registration_sha256": str(freeze["registration"]["sha256"]), "source_selection_sha256": str(registration["source_selection"]["sha256"]), "capture_receipt_sha256": str(registration["capture_receipt"]["sha256"]), "observation_manifest_sha256": str(registration["observation_manifest"]["sha256"]), "observation_record_ids_sha256": json.dumps(record_digests, sort_keys=True, separators=(",", ":")), "records_by_domain": json.dumps(dict(counts), sort_keys=True, separators=(",", ":")), "sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS), "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS), "target_model_or_target_labels_loaded": "false", "source_text_loaded_for_label_materialization": "true"})
    sidecar_record = _record(sidecar_path, root=root, description="TRR-0009 truth sidecar")
    header = {"schema": TRUTH_BINDING_SCHEMA, "task_id": contract.TASK_ID, "status": TRUTH_STATUS, "truth_opened": False, "prepared_after_public_gate": True, "registration": dict(freeze["registration"]), "receipt": dict(freeze_record), "source_selection": dict(registration["source_selection"]), "capture_receipt": dict(registration["capture_receipt"]), "observation_manifest": dict(registration["observation_manifest"]), "sidecar": sidecar_record, "records_by_domain": dict(counts), "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS, "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS, "cell_order": list(contract.CELL_ORDER), "target_conditions": list(contract.TARGET_ORDER), "cells": [{"cell_id": cell, "records": int(counts[cell.split("__", 1)[0]]), "record_ids_sha256": record_digests[cell.split("__", 1)[0]]} for cell in contract.CELL_ORDER], "source_evidence": source_evidence, "execution": {"started_utc": _utc_now(), "ended_utc": _utc_now(), "truth_created": True, "truth_opened": False}}
    header_path = _task_header(truth_binding_path, root=root)
    header_record = _write_json(header_path, header, root=root, description="TRR-0009 truth binding header")
    _validate_sidecar_header(freeze=freeze, registration=registration, freeze_record=freeze_record, header_path=header_path, repository_root=root, open_tensor_data=False)
    return {"status": TRUTH_STATUS, "truth_binding": header_record, "truth_sidecar": sidecar_record, "truth_opened": False, "public_gate_verified": True, "records_by_domain": dict(counts)}


def score_truth_sidecar(*, freeze_path: Path, truth_binding_path: Path, truth_sidecar_path: Path | None, frequency_reference_path: Path | None, output_path: Path, repository_root: Path, bootstrap_seed: int = 9009, bootstrap_draws: int = 10000, exact_alpha: float = 0.025, token_one_sided_alpha: float = 0.025) -> dict[str, Any]:
    root = _root(repository_root)
    freeze, registration, _observation, _, freeze_record = _load_public_context(freeze_path=freeze_path, repository_root=root)
    header, sidecar = _validate_sidecar_header(freeze=freeze, registration=registration, freeze_record=freeze_record, header_path=_resolve(truth_binding_path, root=root), repository_root=root, open_tensor_data=False)
    declared_sidecar = Path(str(header["sidecar"]["path"])).resolve()
    if truth_sidecar_path is not None and _resolve(truth_sidecar_path, root=root) != declared_sidecar:
        raise TruthGateError("supplied truth sidecar differs from binding header")
    frequency_path = Path(str(registration["frequency_reference"]["path"])).resolve()
    if frequency_reference_path is not None and _resolve(frequency_reference_path, root=root) != frequency_path:
        raise TruthGateError("supplied frequency reference differs from frozen registration")
    frequency_counts, frequency_record = _frequency_counts(frequency_path, root=root, registration=registration)
    predictions = _load_predictions(freeze)
    # Sole private truth tensor opening, after all public/header checks.
    try:
        with safe_open(str(declared_sidecar), framework="pt", device="cpu") as handle:
            truth = {cell: handle.get_tensor(f"{cell}__token_ids") for cell in contract.CELL_ORDER}
    except Exception as exc:
        raise TruthGateError("truth sidecar tensor read failed closed") from exc
    score = scorer.score_predictions(predictions, truth, frequency_counts=frequency_counts, bootstrap_seed=bootstrap_seed, bootstrap_draws=bootstrap_draws, exact_alpha=exact_alpha, token_one_sided_alpha=token_one_sided_alpha)
    payload = {**score, "status": "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE", "freeze_receipt": dict(freeze_record), "truth_binding": _record(_resolve(truth_binding_path, root=root), root=root, description="truth binding header"), "truth_sidecar": sidecar["sidecar"], "frequency_reference": frequency_record, "cost_evidence": _cost_evidence(freeze, root=root), "truth_opened": True, "source_text_or_target_labels": False, "candidate_arrays_persisted": False, "scored_utc": _utc_now()}
    artifact = _write_json(output_path, payload, root=root, description="TRR-0009 score result")
    return payload | {"score_artifact": artifact}


def score_after_gate(*, freeze_path: Path, repository_root: Path, truth_loader: Callable[[], Mapping[str, torch.Tensor]], frequency_reference_path: Path, output_path: Path, truth_binding: Mapping[str, Any], bootstrap_seed: int = 9009, bootstrap_draws: int = 10000, exact_alpha: float = 0.025, token_one_sided_alpha: float = 0.025) -> dict[str, Any]:
    """Synthetic/unit adapter; production scoring uses ``score_truth_sidecar``."""
    root = _root(repository_root)
    freeze, registration, _observation, _, _freeze_record = _load_public_context(freeze_path=freeze_path, repository_root=root)
    if not isinstance(truth_binding, Mapping) or truth_binding.get("schema") != TRUTH_BINDING_SCHEMA or truth_binding.get("status") != TRUTH_STATUS or truth_binding.get("truth_opened") is not False:
        raise TruthGateError("synthetic truth binding is mandatory and malformed")
    predictions = _load_predictions(freeze)
    frequency_counts, frequency_record = _frequency_counts(frequency_reference_path, root=root, registration=registration)
    truth = truth_loader()
    if not isinstance(truth, Mapping) or set(truth) != set(contract.CELL_ORDER):
        raise TruthGateError("truth loader cell matrix changed")
    score = scorer.score_predictions(predictions, truth, frequency_counts=frequency_counts, bootstrap_seed=bootstrap_seed, bootstrap_draws=bootstrap_draws, exact_alpha=exact_alpha, token_one_sided_alpha=token_one_sided_alpha)
    payload = {**score, "status": "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE", "freeze_receipt": _record(freeze_path, root=root, description="public freeze receipt"), "truth_binding": dict(truth_binding), "frequency_reference": frequency_record, "cost_evidence": _cost_evidence(freeze, root=root), "truth_opened": True, "source_text_or_target_labels": False, "candidate_arrays_persisted": False, "scored_utc": _utc_now()}
    artifact = _write_json(output_path, payload, root=root, description="TRR-0009 score result")
    return payload | {"score_artifact": artifact}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--execute", action="store_true")
    prep.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    prep.add_argument("--freeze", type=Path, required=True)
    prep.add_argument("--registration", type=Path)
    prep.add_argument("--selection", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path)
    prep.add_argument("--pile-arrow", type=Path, nargs="*")
    prep.add_argument("--finance-arrow", type=Path, nargs="*")
    prep.add_argument("--truth-output", type=Path, required=True)
    prep.add_argument("--truth-binding", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--truth-binding", type=Path, required=True)
    score.add_argument("--truth-sidecar", type=Path)
    score.add_argument("--frequency-reference", type=Path)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_truth(freeze_path=args.freeze, registration_path=args.registration, selection_path=args.selection, truth_output=args.truth_output, truth_binding_path=args.truth_binding, repository_root=args.repository_root, tokenizer_path=args.tokenizer, pile_paths=args.pile_arrow, finance_paths=args.finance_arrow, execute=args.execute)
        else:
            result = score_truth_sidecar(freeze_path=args.freeze, truth_binding_path=args.truth_binding, truth_sidecar_path=args.truth_sidecar, frequency_reference_path=args.frequency_reference, output_path=args.output, repository_root=args.repository_root)
    except Exception as exc:
        print(f"TRR-0009 truth boundary failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "truth_opened": result.get("truth_opened"), "artifact": result.get("score_artifact", result.get("truth_binding"))}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
