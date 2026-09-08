#!/usr/bin/env python3
"""Task-local TRR-0010 curator adapter.

The frozen public gate still receives the original public freeze.  This adapter
only repairs the metadata boundary between the bound source-selection wrapper
and its already-bound native selection ledger.  It validates both records,
passes a deep copy whose source_selection input is the verified ledger to the
frozen materializer, and leaves the original freeze unchanged.  No source
selection, model, prediction, or scoring rule is changed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_prepare_truth as frozen_truth


TASK_ID = "TRR-0010"
ADAPTER_SCHEMA = "token-reconstruction.trr0010-curator-adapter.v1"
ADAPTER_STATUS = "CURATOR_ADAPTER_METADATA_CHAIN_VALIDATED"


class CuratorAdapterError(RuntimeError):
    """Raised when the frozen wrapper-to-ledger chain fails closed."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CuratorAdapterError(f"repository root is unavailable: {root}")
    return root


def _checked_record(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return gate._record(value, root=root, description=description)
    except gate.GateError as exc:
        raise CuratorAdapterError(str(exc)) from exc


def _checked_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = gate._load_json(path, description=description)
    except gate.GateError as exc:
        raise CuratorAdapterError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise CuratorAdapterError(f"{description} must be an object")
    return payload


def validate_selection_binding_chain(
    freeze: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Validate wrapper and nested selection-ledger bytes/hashes only.

    This function reads metadata JSON and never opens Arrow rows, tokenizers,
    labels, observations, model state, or the truth sidecar.
    """
    root = _root(repository_root)
    inputs = freeze.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise CuratorAdapterError("public freeze input_bindings are absent")
    wrapper_value = inputs.get("source_selection")
    wrapper_record = _checked_record(
        wrapper_value, root=root, description="frozen source-selection wrapper"
    )
    wrapper = _checked_json(
        Path(wrapper_record["path"]), description="frozen source-selection wrapper"
    )
    if wrapper.get("task_id") != TASK_ID:
        raise CuratorAdapterError("source-selection wrapper task identity changed")
    if wrapper.get("schema") != "token-reconstruction.trr0010-source-selection-binding.v1":
        raise CuratorAdapterError("source-selection wrapper schema changed")
    if wrapper.get("status") != "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH":
        raise CuratorAdapterError("source-selection wrapper is not frozen and truth-free")
    if wrapper.get("source_text_loaded") is not False or wrapper.get("target_labels_loaded") is not False:
        raise CuratorAdapterError("source-selection wrapper has loaded labels or source text")
    if wrapper.get("truth_opened") is not False:
        raise CuratorAdapterError("source-selection wrapper is already truth-opened")

    ledger_value = wrapper.get("selection_ledger")
    ledger_record = _checked_record(
        ledger_value, root=root, description="nested frozen selection ledger"
    )
    ledger = _checked_json(
        Path(ledger_record["path"]), description="nested frozen selection ledger"
    )
    if ledger.get("task_id") != TASK_ID:
        raise CuratorAdapterError("selection ledger task identity changed")
    if ledger.get("schema") != "token-reconstruction.trr0010-source-selection.v1":
        raise CuratorAdapterError("selection ledger schema changed")
    if ledger.get("status") != "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH":
        raise CuratorAdapterError("selection ledger is not frozen")
    if ledger.get("truth_opened") is not False or ledger.get("truth_created") is not False:
        raise CuratorAdapterError("selection ledger is not truth-free")
    if wrapper.get("selection_ledger_schema") != ledger.get("schema"):
        raise CuratorAdapterError("wrapper selection-ledger schema metadata changed")
    if wrapper.get("selection_ledger_status") != ledger.get("status"):
        raise CuratorAdapterError("wrapper selection-ledger status metadata changed")
    if ledger.get("records_by_domain") != gate.RECORDS_BY_DOMAIN:
        raise CuratorAdapterError("selection ledger domain counts changed")
    if list(ledger.get("target_conditions", ())) != list(gate.TARGET_ORDER):
        raise CuratorAdapterError("selection ledger target conditions changed")
    return {
        "wrapper_record": wrapper_record,
        "wrapper": wrapper,
        "ledger_record": ledger_record,
        "ledger": ledger,
    }


def build_delegate_freeze(
    freeze: Mapping[str, Any], chain: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy the gate result and substitute only the verified ledger binding."""
    ledger_record = chain.get("ledger_record")
    if not isinstance(ledger_record, Mapping):
        raise CuratorAdapterError("validated ledger record is absent")
    delegated = deepcopy(dict(freeze))
    inputs = delegated.get("input_bindings")
    if not isinstance(inputs, dict):
        raise CuratorAdapterError("public freeze input_bindings are absent")
    inputs["source_selection"] = dict(ledger_record)
    return delegated


def _write_receipt(path: Path, payload: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise CuratorAdapterError("adapter receipt must remain under the task root") from exc
    if path.exists() or path.is_symlink():
        raise CuratorAdapterError(f"adapter receipt is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return gate.file_record(path, root=root)


def prepare_truth_with_bound_ledger(
    *,
    freeze_path: Path,
    truth_sidecar_path: Path,
    descriptor_path: Path,
    adapter_receipt_path: Path,
    repository_root: Path,
    approval_receipt_path: Path | None = None,
    require_current_head: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the frozen gate/materializer order with the verified ledger copy."""
    root = _root(repository_root)
    chain_holder: dict[str, Any] = {}

    def delegate(freeze: Mapping[str, Any], sidecar_path: Path) -> Any:
        chain = validate_selection_binding_chain(freeze, repository_root=root)
        chain_holder.update(chain)
        delegated = build_delegate_freeze(freeze, chain)
        return frozen_truth.materialize_public_truth(delegated, sidecar_path)

    result = frozen_truth.prepare_truth_after_freeze(
        freeze_path=Path(freeze_path),
        truth_sidecar_path=Path(truth_sidecar_path),
        descriptor_path=Path(descriptor_path),
        repository_root=root,
        materializer=delegate,
        require_current_head=require_current_head,
    )
    if not chain_holder:
        raise CuratorAdapterError("adapter did not validate the selection binding chain")
    adapter_record = gate.file_record(Path(__file__).resolve(), root=root)
    frozen_truth_record = gate.file_record(Path(frozen_truth.__file__).resolve(), root=root)
    approval_record = None
    if approval_receipt_path is not None:
        approval_record = gate.file_record(Path(approval_receipt_path), root=root)
    receipt_payload = {
        "schema": ADAPTER_SCHEMA,
        "task_id": TASK_ID,
        "status": "CURATOR_LABELS_MATERIALIZED_DESCRIPTOR_VALIDATED",
        "adapter_status": ADAPTER_STATUS,
        "user_authorization": {
            "receipt": approval_record,
            "exact_reply": "Approved.",
            "scope": "creating the curator adapter and opening the frozen panel’s labels to score the existing predictions",
        },
        "original_freeze_gate": {
            "path": str(Path(freeze_path).expanduser().resolve()),
            "revalidated_before_materialization": True,
            "require_current_head": bool(require_current_head),
        },
        "selection_binding_chain": {
            "wrapper": dict(chain_holder["wrapper_record"]),
            "nested_selection_ledger": dict(chain_holder["ledger_record"]),
            "wrapper_schema": chain_holder["wrapper"].get("schema"),
            "ledger_schema": chain_holder["ledger"].get("schema"),
            "wrapper_to_ledger_substitution_only": True,
            "original_freeze_unchanged": True,
        },
        "delegation": {
            "materializer": "scripts.trr0010_prepare_truth:materialize_public_truth",
            "frozen_materializer_source": frozen_truth_record,
            "delegate_freeze_source_selection": "verified nested selection ledger record",
            "source_selection_or_model_or_rule_changed": False,
        },
        "outputs": {
            "truth_sidecar": result.get("truth_payload"),
            "truth_descriptor": result.get("descriptor"),
            "labels_materialized": True,
            "descriptor_validated": True,
            "truth_payload_score_opened": False,
        },
        "command": [str(value) for value in (command or [])],
        "truth_opened": False,
        "source_text_loaded_for_public_label_materialization": True,
    }
    receipt_record = _write_receipt(adapter_receipt_path, receipt_payload, root=root)
    result = dict(result)
    result["adapter_receipt"] = receipt_record
    result["selection_binding_chain"] = {
        "wrapper": dict(chain_holder["wrapper_record"]),
        "nested_selection_ledger": dict(chain_holder["ledger_record"]),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth-sidecar", type=Path)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--approval-receipt", type=Path)
    parser.add_argument("--require-current-head", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = _root(args.repository_root)
        freeze = gate.validate_before_truth(
            freeze_path=args.freeze,
            repository_root=root,
            require_current_head=args.require_current_head,
        )
        chain = validate_selection_binding_chain(freeze, repository_root=root)
        if args.command == "validate":
            print(json.dumps({"status": "PASS_METADATA_ONLY", "chain": {"wrapper": chain["wrapper_record"], "nested_selection_ledger": chain["ledger_record"]}}, indent=2, sort_keys=True))
            return 0
        required = (args.truth_sidecar, args.descriptor, args.adapter_receipt)
        if any(value is None for value in required):
            raise CuratorAdapterError("prepare requires --truth-sidecar, --descriptor, and --adapter-receipt")
        result = prepare_truth_with_bound_ledger(
            freeze_path=args.freeze,
            truth_sidecar_path=args.truth_sidecar,
            descriptor_path=args.descriptor,
            adapter_receipt_path=args.adapter_receipt,
            repository_root=root,
            approval_receipt_path=args.approval_receipt,
            require_current_head=args.require_current_head,
            command=list(sys.argv),
        )
    except (CuratorAdapterError, gate.GateError, frozen_truth.TruthPreparationError, OSError, ValueError) as exc:
        print(f"TRR-0010 curator adapter failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
