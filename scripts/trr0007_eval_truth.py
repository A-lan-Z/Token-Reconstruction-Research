#!/usr/bin/env python3
"""Prepare the sealed TRR-0007 truth sidecar after the public matrix gate.

This is the sole task-local truth preparation entry point.  It revalidates the
complete public matrix, materializes the already-selected public rows only
after that gate, writes the two domain label tensors outside the repository,
and creates a metadata-only binding header under the TRR-0007 task root.  The
header is compatible with trr0007_eval_gate.py and trr0007_score.py; the
sidecar is not opened by the gate and is opened once by the scorer.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from safetensors.torch import save_file
import torch

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0007_eval_capture as trr7_capture
from scripts import trr0007_eval_contract as contract
from scripts import trr0007_eval_gate as gate
from token_reconstruction.trr0005_contract import STYLE_ORDER


TRUTH_BINDING_SCHEMA = "token-reconstruction.trr0007-truth-binding.v1"
TRUTH_SIDECAR_SCHEMA = "token-reconstruction.trr0007-truth-sidecar.v1"
TRUTH_STATUS = "TRR0007_TRUTH_PREPARED_AFTER_PUBLIC_GATE"
TRUTH_KEYS = ("pile__token_ids", "finance__token_ids")


class TruthError(contract.ContractError):
    """Raised when sealed truth preparation cannot complete safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthError(f"{description} is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": contract.sha256_file(path)}


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TruthError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path, description=description)


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TruthError(f"repository root is unavailable: {root}")
    return root


def _task_header(value: Path, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise TruthError(f"truth binding header must be under {task_root}: {path}") from exc
    if path.exists() or path.is_symlink():
        raise TruthError(f"truth binding header is create-only and already exists: {path}")
    return path


def _outside_sidecar(value: Path, *, root: Path, output_root: Path) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve(strict=False)
    for directory, label in ((root, "repository"), (output_root, "prediction output")):
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            continue
        raise TruthError(f"truth sidecar must be outside {label}: {path}")
    if path.exists() or path.is_symlink():
        raise TruthError(f"truth sidecar is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(value: Path | None, *, root: Path) -> Path | None:
    if value is None:
        return None
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _load_inputs(
    *,
    root: Path,
    selection_path: Path,
    tokenizer_path: Path | None,
    pile_paths: Sequence[Path] | None,
    finance_paths: Sequence[Path] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[Any]]]:
    selection, selection_record, rows = trr7_capture._load_selection(
        selection_path, repository_root=root
    )
    actual_tokenizer = tokenizer_path or trr7_capture._tokenizer_path(selection, root=root)
    actual_pile = tuple(pile_paths or trr7_capture._source_paths(selection, "pile", root=root))
    actual_finance = tuple(finance_paths or trr7_capture._source_paths(selection, "finance", root=root))
    trr6_capture._validate_source_descriptors(
        selection,
        pile_paths=actual_pile,
        finance_paths=actual_finance,
        tokenizer_path=actual_tokenizer,
    )
    tokenizer = trusted._load_tokenizer(actual_tokenizer)
    datasets = {
        "pile": trusted._load_arrow_dataset(actual_pile),
        "finance": trusted._load_arrow_dataset(actual_finance),
    }
    records = trr6_capture._materialize_selected(rows, datasets=datasets, tokenizer=tokenizer)
    declared_ids = {
        style: [str(row["record_id"]) for row in rows[style]] for style in STYLE_ORDER
    }
    actual_ids = {
        style: [str(record.record_id) for record in records[style]] for style in STYLE_ORDER
    }
    if actual_ids != declared_ids:
        raise TruthError("materialized truth row order differs from frozen selection")
    return selection, selection_record, rows, records


def _label_tensors(records: Mapping[str, Sequence[Any]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for style in STYLE_ORDER:
        values = list(records[style])
        if len(values) != contract.RECORDS_PER_DOMAIN:
            raise TruthError(f"truth record count changed: {style}")
        labels = torch.tensor(
            [list(record.token_ids[:contract.STORED_SEQUENCE_TOKENS]) for record in values],
            dtype=torch.int64,
        ).contiguous()
        expected_shape = (contract.RECORDS_PER_DOMAIN, contract.STORED_SEQUENCE_TOKENS)
        if tuple(labels.shape) != expected_shape:
            raise TruthError(f"truth geometry changed: {style}")
        if not labels[:, 0].eq(contract.BOS_TOKEN_ID).all().item():
            raise TruthError(f"truth BOS changed: {style}")
        if labels.lt(0).any().item() or labels.ge(contract.VOCAB_SIZE).any().item():
            raise TruthError(f"truth vocabulary range changed: {style}")
        result[f"{style}__token_ids"] = labels
    if set(result) != set(TRUTH_KEYS):
        raise TruthError("truth tensor key matrix changed")
    return result


def prepare_truth(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise TruthError("truth preparation requires explicit --execute")
    root = _root(args.repository_root)
    selection_path = Path(args.selection).expanduser().resolve()
    header_path = _task_header(args.truth_binding, root=root)
    registration_path = Path(args.registration).expanduser().resolve()
    receipt_path = Path(args.receipt).expanduser().resolve()
    started_utc = _utc_now()
    started_clock = time.perf_counter()

    # Resolve registration and output root without touching any private truth.
    registration = contract.load_registration(registration_path)
    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    truth_path = _outside_sidecar(args.truth_output, root=root, output_root=output_root)

    # This public-only call is the authorization boundary for opening source
    # rows to construct labels. It validates all 22 artifacts and writes no
    # truth. The receipt must already exist and is rechecked below.
    public_gate = gate.validate_public_matrix(
        registration=registration,
        repository_root=root,
        output_root=output_root,
    )
    declared_receipt = _file_record(receipt_path, description="public freeze receipt")
    receipt_doc = contract.load_json(receipt_path, description="public freeze receipt")
    if (
        receipt_doc.get("schema") != contract.FREEZE_SCHEMA
        or receipt_doc.get("task_id") != contract.TASK_ID
        or receipt_doc.get("status") != "FROZEN_COMPLETE_PUBLIC_MATRIX_NO_TRUTH"
        or receipt_doc.get("truth_opened") is not False
        or receipt_doc.get("registration_sha256") != registration["registration_sha256"]
    ):
        raise TruthError("provided public freeze receipt is not bound to the closed registration")

    selection, selection_record, rows, records = _load_inputs(
        root=root,
        selection_path=selection_path,
        tokenizer_path=_resolve_path(args.tokenizer, root=root),
        pile_paths=tuple(_resolve_path(path, root=root) for path in (args.pile_arrow or ())),
        finance_paths=tuple(_resolve_path(path, root=root) for path in (args.finance_arrow or ())),
    )
    observations = contract.load_observation_manifest(
        registration, repository_root=root, verify_assets=True
    )[1]
    record_ids_sha256 = {
        style: str(selection["selection_rule"]["record_ids_sha256"][style])
        for style in STYLE_ORDER
    }
    if record_ids_sha256 != observations["record_ids_sha256"]:
        raise TruthError("selection and public observation record order differ")

    tensors = _label_tensors(records)
    metadata = {
        "schema": TRUTH_SIDECAR_SCHEMA,
        "task_id": contract.TASK_ID,
        "truth_opened": "false",
        "registration_sha256": registration["registration_sha256"],
        "source_selection_sha256": selection_record["sha256"],
        "observation_record_ids_sha256": json.dumps(record_ids_sha256, sort_keys=True, separators=(",", ":")),
        "sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
        "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS),
        "labels_shared_across_target_conditions": "true",
        "source_text_loaded_for_label_materialization": "true",
        "target_model_or_target_labels_loaded": "false",
    }
    save_file(tensors, str(truth_path), metadata=metadata)
    sidecar_record = _file_record(truth_path, description="truth sidecar")
    header = {
        "schema": TRUTH_BINDING_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": TRUTH_STATUS,
        "truth_opened": False,
        "prepared_after_public_gate": True,
        "registration": {
            "path": registration["_path"],
            "sha256": registration["registration_sha256"],
        },
        "receipt": declared_receipt,
        "source_selection": selection_record,
        "observation_manifest": registration["observation_manifest"],
        "sidecar": sidecar_record,
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "cell_order": list(contract.CELL_ORDER),
        "target_conditions": list(contract.TARGET_ORDER),
        "labels_shared_across_target_conditions": True,
        "reconstruction_root_contains_truth": False,
        "private_truth_payload_persisted_in_repository": False,
        "cells": [
            {
                "cell_id": cell_id,
                "records": contract.RECORDS_PER_DOMAIN,
                "record_ids_sha256": observations["cells"][cell_id]["record_ids_sha256"],
            }
            for cell_id in contract.CELL_ORDER
        ],
        "execution": {
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "truth_created": True,
            "truth_opened": False,
        },
    }
    header_record = _write_create_only(header_path, header, description="truth binding header")

    # Re-run the metadata-only pre-truth gate against the now-bound sidecar;
    # this still does not stat, hash, or open the sidecar. The scorer owns the
    # first sidecar read after this check.
    pretruth = gate.validate_before_truth(
        receipt_path=receipt_path,
        registration_path=registration_path,
        repository_root=root,
        truth_binding_path=header_path,
    )
    if pretruth.get("status") != "PUBLIC_MATRIX_VERIFIED_NO_TRUTH_OPENED" or pretruth.get("truth_opened") is not False:
        raise TruthError("truth binding failed the metadata-only pre-truth gate")
    return {
        "task_id": contract.TASK_ID,
        "status": TRUTH_STATUS,
        "truth_binding": header_record,
        "truth_sidecar": sidecar_record,
        "truth_opened": False,
        "public_gate_verified": True,
        "public_gate_expected_artifact_count": public_gate["expected_artifact_count"],
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", nargs="?", default="prepare")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--pile-arrow", type=Path, nargs="*")
    parser.add_argument("--finance-arrow", type=Path, nargs="*")
    parser.add_argument("--truth-output", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, default=Path("experiments/TRR-0007/truth_binding.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare != "prepare":
        print("TRR-0007 truth preparation requires the prepare command", file=sys.stderr)
        return 2
    try:
        result = prepare_truth(args)
    except (TruthError, contract.ContractError, gate.GateError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0007 truth preparation failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
