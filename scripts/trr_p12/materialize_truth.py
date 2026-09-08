#!/usr/bin/env python3
"""Materialize P12 evaluator truth only after the complete prediction freeze.

The command validates the registered P12 evaluation freeze without opening
truth, checks that its B1 and A1+A2 source orders are the frozen panel H128
order (A1+A2 is the first 32 rows), and only then can write private truth
safetensors.  Public observations and target artifacts are never rewritten.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import torch
from safetensors.torch import save_file

from scripts.trr_p12 import capture as public_capture
from scripts.trr_p12 import evaluation
from scripts.trr_p12 import analysis


TASK_ID = "TRR-P12"
TRUTH_SCHEMA = "token-reconstruction.trr-p12-evaluation-truth.v1"
TRUTH_STATUS = "TRUTH_READY_AFTER_FREEZE"
DOMAINS = ("pile", "finance")
B1_STAGES = (0, 64, 128, 256)
A1_STAGES = (0, 256)
RECORDS = 128
A1_RECORDS = 32
STORED_TOKENS = 128
BOS = 128000
PANEL_SHA256 = "f48a126fbae6a5e88e1292caf0675d8973f67ab415b706b7fd35e966659ad08e"
SOURCE_INPUTS_SHA256 = "1fa6aea4485ad602d396e6a57dc53257977a0f4581375892ef22ab176d52b409"
DEFAULT_PANEL = ROOT / "outputs/TRR-P12/sources-r2/panel.json"
DEFAULT_FREEZE = ROOT / "experiments/TRR-P12/evaluation-freeze.json"
DEFAULT_OUTPUT = ROOT / "outputs/TRR-P12/private-evaluation/truth-r1"
DEFAULT_SOURCE_INPUTS = ROOT / "experiments/TRR-P11/selector/public_source_inputs_r1.json"


class TruthError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def file_binding(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    base: Path = ROOT,
) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = base / raw
    path = raw.resolve()
    if raw.is_symlink() or not path.is_file():
        raise TruthError(f"{label} must be a regular file: {path}")
    digest = sha256_file(path)
    size = int(path.stat().st_size)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise TruthError(f"{label} SHA-256 changed")
    if expected_bytes is not None and size != int(expected_bytes):
        raise TruthError(f"{label} byte count changed")
    return {"path": str(path), "bytes": size, "sha256": digest, "readonly": True}


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TruthError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TruthError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise TruthError(f"{label} must be an object")
    return dict(value)


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise TruthError(f"create-only truth descriptor already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_binding(path, label="truth descriptor")


def panel_orders(panel_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    panel, panel_binding, pile_rows = public_capture.load_panel(panel_path, "pile")
    _panel_again, _binding_again, finance_rows = public_capture.load_panel(panel_path, "finance")
    if panel_binding["panel"]["sha256"] != PANEL_SHA256:
        raise TruthError("P12 panel SHA-256 is not the frozen release")
    source_binding = panel_binding["source_inputs"]
    if source_binding["sha256"] != SOURCE_INPUTS_SHA256:
        raise TruthError("P12 source-input SHA-256 changed")
    rows = {"pile": pile_rows, "finance": finance_rows}
    orders: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        record_ids = [str(row["record_id"]) for row in rows[domain]]
        h128 = [str(row["h128_sequence_sha256"]).lower() for row in rows[domain]]
        if len(record_ids) != RECORDS or len(set(record_ids)) != RECORDS:
            raise TruthError(f"P12 {domain} record order is not unique")
        if len(h128) != RECORDS or len(set(h128)) != RECORDS:
            raise TruthError(f"P12 {domain} H128 order is not unique")
        orders[domain] = {
            "record_ids": record_ids,
            "h128": h128,
            "record_ids_sha256": public_capture.record_ids_sha256(record_ids),
            "h128_order_sha256": hashlib.sha256(
                json.dumps(h128, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "source_order_sha256": analysis.source_order_digest(h128),
        }
    return panel, panel_binding, rows, orders


def load_complete_freeze(path: Path, *, panel_orders_by_domain: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding(path, label="P12 evaluation freeze receipt")
    try:
        freeze_path, freeze = evaluation._load_freeze_receipt(
            path,
            strict_record_counts=True,
        )
    except Exception as exc:
        raise TruthError("P12 evaluation freeze is not complete and pretruth") from exc
    if freeze_path.resolve() != path.resolve():
        raise TruthError("freeze path changed during validation")
    if freeze.get("schema") != evaluation.FREEZE_RECEIPT_SCHEMA:
        raise TruthError("P12 freeze schema changed")
    if freeze.get("status") != "PREDICTIONS_FROZEN_BEFORE_TRUTH":
        raise TruthError("P12 freeze status is not pretruth")
    for flag in ("truth_opened", "source_text_loaded", "target_weights_loaded"):
        if freeze.get(flag) is not False:
            raise TruthError(f"P12 freeze boundary flag {flag} is open")
    methods = freeze.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {"B1", "A1+A2"}:
        raise TruthError("P12 freeze must contain B1 and A1+A2")
    for method, expected_stages, expected_count in (
        ("B1", B1_STAGES, RECORDS),
        ("A1+A2", A1_STAGES, A1_RECORDS),
    ):
        domains = methods[method].get("domains")
        if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
            raise TruthError(f"P12 freeze {method} domains changed")
        for domain in DOMAINS:
            source_order = domains[domain]["source_order"]
            hashes = list(source_order["hashes"])
            expected = list(panel_orders_by_domain[domain]["h128"])
            if expected_count == A1_RECORDS:
                expected = expected[:A1_RECORDS]
            if hashes != expected:
                raise TruthError(f"P12 freeze {method}/{domain} order differs from panel H128 order")
            if source_order["sha256"] != analysis.source_order_digest(expected):
                raise TruthError(f"P12 freeze {method}/{domain} order digest changed")
            stages = domains[domain].get("stages")
            if not isinstance(stages, Mapping) or {int(str(key)) for key in stages} != set(expected_stages):
                raise TruthError(f"P12 freeze {method}/{domain} stage set changed")
    return freeze, binding


def materialize_source_truth(
    *,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    orders: Mapping[str, Mapping[str, Any]],
    source_inputs: Path,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for domain in DOMAINS:
        batch, order = public_capture.rerender_batches(
            rows=rows[domain],
            domain=domain,
            source_inputs=source_inputs,
        )
        actual_h128 = list(order["ordered_h128_sequence_sha256"])
        expected_h128 = list(orders[domain]["h128"])
        if actual_h128 != expected_h128:
            raise TruthError(f"canonical rerender H128 order changed for {domain}")
        values = batch.token_ids[:, :STORED_TOKENS].to(dtype=torch.int64, device="cpu").contiguous()
        if tuple(values.shape) != (RECORDS, STORED_TOKENS):
            raise TruthError(f"truth tensor geometry changed for {domain}")
        if not bool(values[:, 0].eq(BOS).all().item()):
            raise TruthError(f"truth tensor BOS changed for {domain}")
        if bool(values.lt(0).any().item()) or bool(values.ge(public_capture.VOCAB_SIZE).any().item()):
            raise TruthError(f"truth tensor vocabulary range changed for {domain}")
        tensors[domain] = values
        del batch
    return tensors


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_path = resolve_path(args.panel)
    panel, panel_binding, rows, orders = panel_orders(panel_path)
    source_inputs = resolve_path(args.source_inputs)
    if source_inputs != Path(panel_binding["source_inputs"]["path"]):
        raise TruthError("source-inputs argument differs from the panel-bound descriptor")
    freeze, freeze_binding = load_complete_freeze(
        resolve_path(args.freeze_receipt),
        panel_orders_by_domain=orders,
    )
    if not args.execute:
        return {
            "status": "READY_AFTER_COMPLETE_FREEZE_NO_TRUTH_OPEN",
            "panel": panel_binding,
            "freeze": freeze_binding,
            "source_order_sha256": {
                domain: orders[domain]["source_order_sha256"] for domain in DOMAINS
            },
            "cells": 12,
            "truth_opened": False,
            "output_root": str(resolve_path(args.output_root)),
        }
    output_root = resolve_path(args.output_root)
    allowed_root = (ROOT / "outputs" / "TRR-P12" / "private-evaluation").resolve()
    try:
        output_root.relative_to(allowed_root)
    except ValueError as exc:
        raise TruthError("truth output must be below outputs/TRR-P12/private-evaluation") from exc
    if output_root.exists() or output_root.is_symlink():
        raise TruthError(f"truth output is create-only: {output_root}")
    started = time.monotonic()
    tensors = materialize_source_truth(
        rows=rows,
        orders=orders,
        source_inputs=source_inputs,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    truth_files: dict[tuple[str, str], dict[str, Any]] = {}
    for method, count in (("B1", RECORDS), ("A1+A2", A1_RECORDS)):
        for domain in DOMAINS:
            path = output_root / f"truth__{method.replace('+', '_')}__{domain}.safetensors"
            values = tensors[domain][:count].contiguous()
            save_file(
                {"truth": values},
                str(path),
                metadata={
                    "schema": TRUTH_SCHEMA,
                    "task_id": TASK_ID,
                    "method": method,
                    "domain": domain,
                    "sequence_tokens": str(STORED_TOKENS),
                    "records": str(count),
                    "bos_token_id": str(BOS),
                    "truth_opened": "true",
                    "source_text_written": "false",
                    "source_text_materialized_transiently": "true",
                    "target_labels_loaded": "false",
                    "p03_holdout_accessed": "false",
                },
            )
            truth_files[(method, domain)] = {
                **file_binding(path, label=f"private {method}/{domain} truth"),
                "key": "truth",
                "shape": [count, STORED_TOKENS],
            }
    cells: list[dict[str, Any]] = []
    methods = freeze["methods"]
    for method, expected_stages in (("B1", B1_STAGES), ("A1+A2", A1_STAGES)):
        for domain in DOMAINS:
            source_digest = methods[method]["domains"][domain]["source_order"]["sha256"]
            for stage in expected_stages:
                cells.append(
                    {
                        "method": method,
                        "domain": domain,
                        "stage": stage,
                        "source_order_sha256": source_digest,
                        "truth": truth_files[(method, domain)],
                    }
                )
    descriptor = {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "created_utc": utc_now(),
        "freeze_receipt_sha256": freeze_binding["sha256"],
        "panel_sha256": panel_binding["panel"]["sha256"],
        "cells": cells,
        "truth_opened": True,
        "source_text_loaded": True,
        "source_text_written": False,
        "token_ids_written": True,
        "target_labels_loaded": False,
        "p03_holdout_accessed": False,
        "execution": {
            "command": list(sys.argv),
            "code_commit": git_commit(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "network_used": False,
            "source_text_materialized_transiently": True,
            "truth_opened_only_after_complete_freeze": True,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    descriptor_binding = write_json_create_only(output_root / "truth.json", descriptor)
    return {
        "status": TRUTH_STATUS,
        "truth_descriptor": descriptor_binding,
        "cells": len(cells),
        "freeze": freeze_binding,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--source-inputs", type=Path, default=DEFAULT_SOURCE_INPUTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("use one of --execute or --dry-run")
    if not args.execute and not args.dry_run:
        raise SystemExit("pass --dry-run or --execute")
    try:
        print(json.dumps(run(args), sort_keys=True))
    except TruthError as exc:
        print(f"P12 truth refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
