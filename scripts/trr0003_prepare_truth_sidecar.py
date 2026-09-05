#!/usr/bin/env python3
"""Prepare the TRR-0003 evaluator sidecar in a separate preparation role.

This process may read the public source ledgers' token labels solely to create
one private paired sidecar.  It never prints token values.  The committed JSON
outputs contain hashes, row identities, and geometry only; prediction methods
must receive the sidecar path only after the public freeze gate passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from token_reconstruction.footing import (
    CONDITION_ORDER,
    PANEL_SCHEMA,
    TASK_ID,
    FootingError,
    build_truth_binding,
    external_file_record,
    file_record,
    load_all_cells,
    load_panel,
    sha256_file,
    truth_sidecar_metadata,
)


SOURCE_SCHEMA = "token-reconstruction.trr0003-truth-preparation.v1"
DEFAULT_PANEL = Path("experiments/TRR-0003/footing/panel.json")
DEFAULT_PILE_RECORDS = Path("outputs/TRR-0002/public-calibration/records.json")
DEFAULT_PILE_TRUTH = Path("outputs/TRR-0002/public-calibration/truth.safetensors")
DEFAULT_FINANCE_RECORDS = Path(
    "outputs/TRR-0002/configuration-search/public-finance/records.json"
)
DEFAULT_FINANCE_TRUTH = Path(
    "outputs/TRR-0002/configuration-search/public-finance/truth.safetensors"
)
DEFAULT_PREPARATION = Path("experiments/TRR-0003/footing/truth_preparation.json")
DEFAULT_BINDING = Path("experiments/TRR-0003/footing/truth_binding.json")
DEFAULT_SIDECAR = Path("outputs/TRR-0003/evaluator_private/panel_truth.safetensors")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _current_commit(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FootingError("unable to resolve preparation execution commit") from exc
    if len(value) != 40:
        raise FootingError("preparation execution commit is not a full commit")
    return value


def _write_json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FootingError(f"refusing to overwrite preparation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _regular(path: Path, *, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"{description} is unavailable: {path}")
    return path.resolve()


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(_regular(path, description=description).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootingError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FootingError(f"{description} root is not an object: {path}")
    return value


def _source_rows(records_path: Path, *, section: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _load_json(records_path, description="source record ledger")
    rows = raw.get(section)
    if not isinstance(rows, list) or not rows:
        raise FootingError(f"source record section is absent: {records_path}#{section}")
    if any(not isinstance(row, dict) for row in rows):
        raise FootingError(f"source record rows are malformed: {records_path}#{section}")
    return rows, raw


def _load_tokens(path: Path, *, expected_key: str = "token_ids") -> torch.Tensor:
    path = _regular(path, description="source truth asset")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if expected_key not in keys:
                raise FootingError(f"source truth key is absent: {path}")
            value = handle.get_tensor(expected_key).to(torch.long).contiguous()
    except FootingError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FootingError(f"source truth asset is unreadable: {path}") from exc
    if value.ndim != 2:
        raise FootingError(f"source truth tensor geometry is invalid: {path}")
    if value.lt(0).any().item() or value.ge(128256).any().item():
        raise FootingError(f"source truth tensor has an invalid token range: {path}")
    if value[:, 0].ne(128000).any().item():
        raise FootingError(f"source truth tensor has an invalid BOS row: {path}")
    return value


def _token_rows_by_id(rows: list[dict[str, Any]], *, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise FootingError(f"{description} has an invalid record ID")
        if record_id in result:
            raise FootingError(f"{description} has duplicate record ID: {record_id}")
        result[record_id] = row
    return result


def _record_ids_from_panel(panel: Mapping[str, Any], *, cell_id: str) -> tuple[str, ...]:
    for row in panel.get("cells", []):
        if isinstance(row, Mapping) and row.get("id") == cell_id:
            records = row.get("records")
            if not isinstance(records, list):
                break
            ids = tuple(item.get("record_id") for item in records if isinstance(item, Mapping))
            if len(ids) != len(records) or any(not isinstance(value, str) for value in ids):
                break
            return ids
    raise FootingError(f"panel record IDs are unavailable: {cell_id}")


def _check_source_alignment(
    *,
    panel: Mapping[str, Any],
    cell: Any,
    token_ids: torch.Tensor,
    source_rows: Mapping[str, Mapping[str, Any]],
    finance_attention: torch.Tensor | None = None,
    finance_positions: torch.Tensor | None = None,
) -> None:
    ids = _record_ids_from_panel(panel, cell_id=cell.cell_id)
    if token_ids.shape != cell.attention_mask.shape:
        raise FootingError(f"source truth geometry differs from panel: {cell.cell_id}")
    for index, record_id in enumerate(ids):
        row = source_rows.get(record_id)
        if row is None:
            raise FootingError(f"source ledger is missing panel record: {record_id}")
        if cell.style == "pile":
            raw_tokens = row.get("token_ids")
            if not isinstance(raw_tokens, list) or len(raw_tokens) != cell.sequence_tokens:
                raise FootingError(f"Pile source token row geometry changed: {record_id}")
            expected = torch.tensor(raw_tokens, dtype=torch.long)
            if not torch.equal(token_ids[index], expected):
                raise FootingError(f"Pile source truth row does not match ledger: {record_id}")
        else:
            expected_sha = row.get("token_ids_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise FootingError(f"Finance token hash is absent: {record_id}")
            # The Finance ledger publishes the token-row hash.  The source
            # truth tensor is checked against that hash by the main caller.
        if int(token_ids[index, 0]) != 128000:
            raise FootingError(f"source truth BOS changed: {record_id}")
    if finance_attention is not None and not torch.equal(
        finance_attention.to(torch.long), cell.attention_mask.to(torch.long)
    ):
        raise FootingError(f"Finance source mask differs from panel: {cell.cell_id}")
    if finance_positions is not None and not torch.equal(
        finance_positions.to(torch.long), cell.position_ids.to(torch.long)
    ):
        raise FootingError(f"Finance source positions differ from panel: {cell.cell_id}")


def _row_sha256(value: torch.Tensor) -> str:
    import hashlib

    contiguous = value.to(torch.int32).cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _source_files(root: Path, path: Path) -> dict[str, Any]:
    try:
        return file_record(path, repository_root=root)
    except FootingError:
        # Public source ledgers are repository files in the normal checkout.
        # Keep a uniform explicit record if a caller stages them elsewhere.
        return external_file_record(path)


def prepare(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    panel_path = _regular(args.panel, description="panel")
    panel = load_panel(panel_path, repository_root=root)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise FootingError("panel identity changed")
    cells = load_all_cells(panel, repository_root=root)
    cell_by_id = {cell.cell_id: cell for cell in cells}

    pile_rows, pile_ledger = _source_rows(args.pile_records, section="development")
    finance_rows, finance_ledger = _source_rows(args.finance_records, section="records")
    pile_by_id = _token_rows_by_id(pile_rows, description="Pile development ledger")
    finance_by_id = _token_rows_by_id(finance_rows, description="Finance public ledger")
    pile_tokens = _load_tokens(args.pile_truth)
    finance_truth_path = _regular(args.finance_truth, description="Finance source truth")
    try:
        with safe_open(finance_truth_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"attention_mask", "position_ids", "token_ids"}:
                raise FootingError("Finance source truth tensor set changed")
            finance_tokens = handle.get_tensor("token_ids").to(torch.long).contiguous()
            finance_attention = handle.get_tensor("attention_mask").to(torch.long).contiguous()
            finance_positions = handle.get_tensor("position_ids").to(torch.long).contiguous()
    except FootingError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FootingError("Finance source truth is unreadable") from exc
    if pile_tokens.shape[0] != len(pile_rows):
        raise FootingError("Pile source truth/ledger row count changed")
    if finance_tokens.shape[0] != len(finance_rows):
        raise FootingError("Finance source truth/ledger row count changed")

    # Map each panel row to its source ledger position before reading any
    # source token values into the sidecar.  The target condition is paired and
    # therefore reuses the same source token tensor for both conditions.
    pile_index = {str(row["record_id"]): index for index, row in enumerate(pile_rows)}
    finance_index = {str(row["record_id"]): index for index, row in enumerate(finance_rows)}
    truth: dict[str, torch.Tensor] = {}
    tensors: dict[str, torch.Tensor] = {}
    source_rows_by_style = {"pile": pile_by_id, "finance": finance_by_id}
    source_index_by_style = {"pile": pile_index, "finance": finance_index}
    row_manifest: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        ids = _record_ids_from_panel(panel, cell_id=cell.cell_id)
        style = cell.style
        source_rows = source_rows_by_style[style]
        source_index = source_index_by_style[style]
        missing = [record_id for record_id in ids if record_id not in source_index]
        if missing:
            raise FootingError(f"source ledger is missing panel IDs: {missing}")
        indices = [source_index[record_id] for record_id in ids]
        if style == "pile":
            token_ids = pile_tokens[indices].clone()
            mask = cell.attention_mask.to(torch.int32)
            positions = cell.position_ids.to(torch.int32)
            _check_source_alignment(
                panel=panel,
                cell=cell,
                token_ids=token_ids,
                source_rows=source_rows,
            )
        else:
            token_ids = finance_tokens[indices].clone()
            mask = finance_attention[indices].to(torch.int32)
            positions = finance_positions[indices].to(torch.int32)
            _check_source_alignment(
                panel=panel,
                cell=cell,
                token_ids=token_ids,
                source_rows=source_rows,
                finance_attention=mask,
                finance_positions=positions,
            )
            # The public Finance row ledger includes the token-row SHA.  Check
            # it without printing or persisting token values.
            for local_index, record_id in enumerate(ids):
                expected_sha = str(source_rows[record_id]["token_ids_sha256"])
                valid_tokens = int(source_rows[record_id]["valid_tokens"])
                if valid_tokens <= 0 or valid_tokens > token_ids.shape[1]:
                    raise FootingError(f"Finance valid-token count changed: {record_id}")
                if _row_sha256(token_ids[local_index, :valid_tokens]) != expected_sha:
                    raise FootingError(f"Finance source token hash differs: {record_id}")
        if token_ids[:, 0].ne(128000).any().item():
            raise FootingError(f"source truth BOS changed: {cell.cell_id}")
        if token_ids.lt(0).any().item() or token_ids.ge(128256).any().item():
            raise FootingError(f"source truth token range changed: {cell.cell_id}")
        truth[cell.cell_id] = token_ids.to(torch.int32)
        tensors[f"{cell.cell_id}__token_ids"] = token_ids.to(torch.int32)
        tensors[f"{cell.cell_id}__attention_mask"] = mask
        tensors[f"{cell.cell_id}__position_ids"] = positions
        row_manifest[cell.cell_id] = []
        for local_index, (record_id, source_index) in enumerate(zip(ids, indices)):
            row_tokens = truth[cell.cell_id][local_index]
            if style == "finance":
                row_tokens = row_tokens[: int(source_rows[record_id]["valid_tokens"])]
            row_manifest[cell.cell_id].append(
                {
                    "record_id": record_id,
                    "source_index": int(source_index),
                    "source_row_sha256": _row_sha256(row_tokens),
                }
            )

    preparation = {
        "schema": SOURCE_SCHEMA,
        "task_id": TASK_ID,
        "role": "private truth sidecar preparation",
        "status": "PREPARED_WITHOUT_PRINTING_SOURCE_TOKENS",
        "created_utc": _utc_now(),
        "preparation_script": file_record(Path(__file__).resolve(), repository_root=root),
        "execution_commit": _current_commit(root),
        "panel": file_record(panel_path, repository_root=root),
        "panel_sha256": sha256_file(panel_path),
        "source_ledgers": {
            "pile_records": _source_files(root, args.pile_records),
            "pile_truth": _source_files(root, args.pile_truth),
            "finance_records": _source_files(root, args.finance_records),
            "finance_truth": _source_files(root, args.finance_truth),
        },
        "source_splits": {
            "pile": "public-calibration development rows selected by panel record_id",
            "finance": "public-finance rows selected by panel record_id",
        },
        "conditions": list(CONDITION_ORDER),
        "paired_conditions": True,
        "cells": row_manifest,
        "source_labels_read_in_this_process": True,
        "prediction_methods_run": False,
        "evaluation_truth_opened": False,
        "source_tokens_printed": False,
    }
    _write_json_create(args.preparation, preparation)
    prep_record = external_file_record(args.preparation)
    placeholder = {
        "path": str(args.sidecar.resolve()),
        "bytes": 0,
        "sha256": "0" * 64,
    }
    binding = build_truth_binding(
        panel_sha256=sha256_file(panel_path),
        cells=cells,
        truth=truth,
        preparation=prep_record,
        sidecar=placeholder,
    )
    if args.sidecar.exists() or args.sidecar.is_symlink():
        raise FootingError(f"refusing to overwrite truth sidecar: {args.sidecar}")
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.sidecar), metadata=truth_sidecar_metadata(binding))
    binding["sidecar"] = external_file_record(args.sidecar)
    _write_json_create(args.binding, binding)
    print(json.dumps({
        "status": "PRIVATE_SIDECAR_PREPARED",
        "panel_sha256": sha256_file(panel_path),
        "preparation": str(args.preparation.resolve()),
        "preparation_sha256": prep_record["sha256"],
        "sidecar": str(args.sidecar.resolve()),
        "sidecar_sha256": binding["sidecar"]["sha256"],
        "cells": len(cells),
        "records_per_condition": sum(cell.records for cell in cells[::2]),
        "source_tokens_printed": False,
    }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, default=Path("."))
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--pile-records", type=Path, default=DEFAULT_PILE_RECORDS)
    p.add_argument("--pile-truth", type=Path, default=DEFAULT_PILE_TRUTH)
    p.add_argument("--finance-records", type=Path, default=DEFAULT_FINANCE_RECORDS)
    p.add_argument("--finance-truth", type=Path, default=DEFAULT_FINANCE_TRUTH)
    p.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    p.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    p.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.repository_root.resolve()
    for name in (
        "panel", "pile_records", "pile_truth", "finance_records", "finance_truth",
        "preparation", "binding", "sidecar",
    ):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else root / path)
    try:
        return prepare(args)
    except (FootingError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0003 truth preparation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
