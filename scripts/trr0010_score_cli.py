"""Fail-closed TRR-0010 truth-descriptor scoring entrypoint.

The curator supplies a truth descriptor only after the public prediction freeze.
This adapter validates that descriptor against the public freeze without opening
its payload, then delegates public-gate validation and the single truth-opening
step to :func:`scripts.trr0010_score.score_after_gate`.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register
from scripts import trr0010_score as score


class ScoreCliError(ValueError):
    """Raised when the truth descriptor or sealed payload fails closed."""


BOUNDARY_RECEIPT_SCHEMA = "token-reconstruction.trr0010-score-boundary-receipt.v1"
BOUNDARY_RECEIPT_STATUS = "SCORE_BOUNDARY_PROVENANCE_RECORDED"


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ScoreCliError(f"repository root is unavailable: {root}")
    return root


def _payload_path(binding: Mapping[str, Any], *, repository_root: Path) -> Path:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ScoreCliError("truth payload path is absent")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    if path.is_symlink():
        raise ScoreCliError("truth payload must not be a symlink")
    path = path.resolve()
    if not path.is_file():
        raise ScoreCliError(f"truth payload is unavailable: {path}")
    expected_bytes = binding.get("bytes")
    expected_sha256 = binding.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ScoreCliError("truth payload byte binding is malformed")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ScoreCliError("truth payload hash binding is malformed")
    if path.stat().st_size != expected_bytes:
        raise ScoreCliError("truth payload byte binding changed")
    actual_sha256 = gate.sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ScoreCliError("truth payload hash binding changed")
    return path


def _load_sealed_truth(binding: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """Open a validated sealed tensor payload after the public gate passes."""
    path = _payload_path(binding, repository_root=repository_root)
    expected_keys = [f"{cell}__token_ids" for cell in gate.CELL_ORDER]
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if set(keys) != set(expected_keys):
                raise ScoreCliError("truth tensor key set changed")
            return {
                cell: handle.get_tensor(f"{cell}__token_ids").detach().cpu().contiguous()
                for cell in gate.CELL_ORDER
            }
    except ScoreCliError:
        raise
    except Exception as exc:
        raise ScoreCliError(f"truth payload is not a readable safetensors file: {path}") from exc


def _default_boundary_receipt_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.boundary.json")


def _write_boundary_receipt(
    *,
    result: Mapping[str, Any],
    checked: Mapping[str, Any],
    repository_root: Path,
    output_path: Path,
    command: Sequence[str] | None = None,
    boundary_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Create a provenance sidecar after the immutable score artifact exists.

    ``score_after_gate`` deliberately writes its score before this wrapper adds
    curator-header fields to the returned in-memory result.  The sidecar keeps
    that immutable score unchanged while linking its exact bytes to the
    validated descriptor and sealed-payload bindings.
    """
    root = _root(repository_root)
    score_value = result.get("score_artifact")
    if not isinstance(score_value, Mapping):
        raise ScoreCliError("score artifact binding is absent; cannot write boundary receipt")
    score_path = Path(str(score_value.get("path", "")))
    if not score_path.is_absolute():
        score_path = root / score_path
    score_record = gate.file_record(score_path, root=root)
    for key in ("bytes", "sha256"):
        if str(score_record.get(key)) != str(score_value.get(key)):
            raise ScoreCliError(f"score artifact {key} changed before boundary receipt")
    descriptor_record = checked.get("truth_descriptor")
    payload_record = checked.get("truth_payload")
    if not isinstance(descriptor_record, Mapping) or not isinstance(payload_record, Mapping):
        raise ScoreCliError("validated truth descriptor or payload binding is absent")
    receipt_path = boundary_receipt_path or _default_boundary_receipt_path(output_path)
    receipt_path = Path(receipt_path).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = receipt_path.resolve()
    task_root = (root / "experiments" / gate.TASK_ID).resolve()
    try:
        receipt_path.relative_to(task_root)
    except ValueError as exc:
        raise ScoreCliError("boundary receipt must remain under the task root") from exc
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ScoreCliError(f"boundary receipt is create-only: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": BOUNDARY_RECEIPT_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": BOUNDARY_RECEIPT_STATUS,
        "created_after_immutable_score": True,
        "score_artifact": dict(score_record),
        "truth_descriptor": dict(descriptor_record),
        "truth_payload": dict(payload_record),
        "cli_code": gate.file_record(Path(__file__).resolve(), root=root, readonly=True),
        "command": [str(value) for value in (command or ["scripts/trr0010_score_cli.py"])],
        "truth_opened": True,
        "score_artifact_rewritten": False,
    }
    try:
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ScoreCliError(f"boundary receipt is create-only: {receipt_path}") from exc
    return gate.file_record(receipt_path, root=root)


def validate_descriptor_before_truth(
    *,
    freeze_path: Path,
    truth_descriptor_path: Path,
    repository_root: Path,
    require_current_head: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate public outputs and the curator header without opening truth."""
    root = _root(repository_root)
    freeze = gate.validate_before_truth(
        freeze_path=freeze_path,
        repository_root=root,
        require_current_head=require_current_head,
    )
    checked = register.validate_truth_descriptor(
        truth_descriptor_path,
        repository_root=root,
        freeze=freeze,
    )
    return freeze, checked


def score_from_truth_descriptor(
    *,
    freeze_path: Path,
    truth_descriptor_path: Path,
    repository_root: Path,
    frequency_reference_paths: Mapping[str, Path],
    output_path: Path,
    cost_evidence_path: Path | None = None,
    bootstrap_seed: int = 10010,
    bootstrap_draws: int = 10000,
    require_current_head: bool = False,
    boundary_receipt_path: Path | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the descriptor, then score using its sealed payload exactly once."""
    root = _root(repository_root)
    if set(frequency_reference_paths) != {"B0", "B1"}:
        raise ScoreCliError("frequency_reference_paths must bind exactly B0 and B1")
    _freeze, checked = validate_descriptor_before_truth(
        freeze_path=freeze_path,
        truth_descriptor_path=truth_descriptor_path,
        repository_root=root,
        require_current_head=require_current_head,
    )
    payload_binding = checked["truth_payload"]

    def truth_loader() -> dict[str, Any]:
        return _load_sealed_truth(payload_binding, repository_root=root)

    result = score.score_after_gate(
        freeze_path=freeze_path,
        repository_root=root,
        truth_loader=truth_loader,
        frequency_reference_paths={
            "B0": Path(frequency_reference_paths["B0"]),
            "B1": Path(frequency_reference_paths["B1"]),
        },
        output_path=output_path,
        cost_evidence_path=cost_evidence_path,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_draws=int(bootstrap_draws),
        require_current_head=require_current_head,
    )
    result["truth_descriptor"] = checked["truth_descriptor"]
    result["truth_payload_binding"] = dict(payload_binding)
    if "score_artifact" not in result:
        raise ScoreCliError("score artifact was not persisted; boundary receipt cannot be created")
    result["boundary_receipt"] = _write_boundary_receipt(
        result=result,
        checked=checked,
        repository_root=root,
        output_path=output_path,
        command=command,
        boundary_receipt_path=boundary_receipt_path,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth-descriptor", type=Path, required=True)
    parser.add_argument("--frequency-reference-b0", type=Path, required=True)
    parser.add_argument("--frequency-reference-b1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cost-evidence",
        type=Path,
        default=None,
        help="hash-bound public timing cost artifact from trr0010_cost_evidence.py",
    )
    parser.add_argument("--boundary-receipt", type=Path, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=10010)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--require-current-head", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize the post-gate sealed truth open and score; omit for a fail-closed dry invocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print("TRR-0010 score refused: --execute is required for sealed truth access", file=sys.stderr)
        return 2
    try:
        result = score_from_truth_descriptor(
            freeze_path=args.freeze,
            truth_descriptor_path=args.truth_descriptor,
            repository_root=args.repository_root,
            frequency_reference_paths={"B0": args.frequency_reference_b0, "B1": args.frequency_reference_b1},
            output_path=args.output,
            cost_evidence_path=args.cost_evidence,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_draws=args.bootstrap_draws,
            require_current_head=args.require_current_head,
            boundary_receipt_path=args.boundary_receipt,
            command=list(sys.argv),
        )
    except (gate.GateError, register.RegisterError, score.ScoreError, ScoreCliError, OSError, ValueError) as exc:
        print(f"TRR-0010 scoring boundary failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "truth_opened": result.get("truth_opened"),
                "truth_loader_invocations": result.get("truth_loader_invocations"),
                "score_artifact": result.get("score_artifact"),
                "boundary_receipt": result.get("boundary_receipt"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
