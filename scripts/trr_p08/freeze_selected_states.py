#!/usr/bin/env python3
"""Create a metadata-only freeze receipt for the eight selected P08 states.

This is the first post-fit freeze boundary.  It verifies the already-created
main-fit and qualification receipts through the shared P08 freeze validator,
then records only selected-state file/tensor hashes, selection steps, and
schedule bindings.  It never loads safetensors, observations, predictions, or
truth.  The output is create-only and must be generated after the main fit has
stopped.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trr_p08 import freeze_matrix  # noqa: E402


TASK_ID = "TRR-P08"
SCHEMA = "token-reconstruction.trr-p08-selected-state-freeze.v1"
STATUS = "SELECTED_STATES_FROZEN_NO_TRUTH"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SelectedStateFreezeError(RuntimeError):
    """Raised when the selected-state freeze contract fails closed."""


def _write_create_only(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    path = path.expanduser().resolve()
    try:
        path.relative_to(root / "experiments" / "TRR-P08")
    except ValueError as exc:
        raise SelectedStateFreezeError("selected-state freeze output must be task-owned") from exc
    if path.exists() or path.is_symlink():
        raise SelectedStateFreezeError(f"selected-state freeze output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _selected_record(state: Mapping[str, Any], *, root: Path, key: str) -> dict[str, Any]:
    raw = state.get("selected")
    if not isinstance(raw, Mapping):
        raise SelectedStateFreezeError(f"selected state record is missing: {key}")
    record = freeze_matrix._record(Path(str(raw.get("path"))).expanduser().resolve(), root=root)
    if record["bytes"] != raw.get("bytes") or record["sha256"] != raw.get("sha256"):
        raise SelectedStateFreezeError(f"selected state file changed during freeze: {key}")
    tensor_sha = state.get("selected_tensor_sha256")
    if not isinstance(tensor_sha, str) or _SHA256.fullmatch(tensor_sha) is None:
        raise SelectedStateFreezeError(f"selected state tensor hash is missing: {key}")
    return {"file": record, "state_tensor_sha256": tensor_sha}


def assemble_selected_state_freeze(
    *,
    repository_root: Path,
    plan_path: Path,
    fit_receipt_path: Path,
    output_path: Path,
    expected_plan_sha256: str = freeze_matrix.APPROVED_PLAN_SHA256,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate and write one create-only selected-state freeze receipt."""

    root = Path(repository_root).expanduser().resolve()
    if not isinstance(expected_plan_sha256, str) or _SHA256.fullmatch(expected_plan_sha256) is None:
        raise SelectedStateFreezeError("expected plan SHA-256 is malformed")
    if expected_source_commit is not None and _COMMIT.fullmatch(expected_source_commit) is None:
        raise SelectedStateFreezeError("expected source commit is malformed")

    plan_record = freeze_matrix._validate_plan(Path(plan_path).expanduser().resolve(), root=root, expected_plan_sha256=expected_plan_sha256)
    fit = freeze_matrix._validate_fit_receipt(Path(fit_receipt_path).expanduser().resolve(), root=root)
    source_commit = fit["source_commit"]
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise SelectedStateFreezeError(
            f"main-fit source commit differs: expected {expected_source_commit}, got {source_commit}"
        )

    selected_states: dict[str, dict[str, Any]] = {}
    expected_keys = [
        f"{seed}::{method}"
        for seed in freeze_matrix.SEEDS
        for method in freeze_matrix.METHOD_ORDER
    ]
    for key in expected_keys:
        raw_seed, method = key.split("::", 1)
        state = fit["states"].get((int(raw_seed), method))
        if not isinstance(state, Mapping):
            raise SelectedStateFreezeError(f"selected state matrix is incomplete: {key}")
        schedule_sha = state.get("schedule_sha256")
        if not isinstance(schedule_sha, str) or _SHA256.fullmatch(schedule_sha) is None:
            raise SelectedStateFreezeError(f"schedule hash is missing: {key}")
        selected = _selected_record(state, root=root, key=key)
        selected_states[key] = {
            "seed": int(state["seed"]),
            "arm_id": str(state["arm_id"]),
            "selection_step": int(state["selected_step"]),
            "schedule_sha256": schedule_sha,
            **selected,
        }

    qualification_record = fit["qualification"]
    receipt = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": STATUS,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "create_only": True,
        "plan": plan_record,
        "plan_sha256": plan_record["sha256"],
        "main_fit_receipt": fit["receipt_record"],
        "main_fit_receipt_sha256": fit["receipt_record"]["sha256"],
        "qualification_receipt": qualification_record,
        "qualification_receipt_sha256": qualification_record["sha256"],
        "source_commit": source_commit,
        "selected_state_order": expected_keys,
        "selected_state_count": len(selected_states),
        "selected_states": selected_states,
        "state_payloads_loaded": False,
        "freeze_boundary": "all eight selected states are hash-bound after main fit completion and before fresh-panel selection or prediction",
    }
    _write_create_only(Path(output_path), receipt, root=root)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", default=freeze_matrix.APPROVED_PLAN_SHA256)
    parser.add_argument("--expected-source-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assemble_selected_state_freeze(
            repository_root=args.repository_root,
            plan_path=args.plan,
            fit_receipt_path=args.fit_receipt,
            output_path=args.output,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_source_commit=args.expected_source_commit,
        )
    except (SelectedStateFreezeError, freeze_matrix.P08FreezeError, OSError, ValueError, TypeError) as exc:
        print(f"TRR-P08 selected-state freeze error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
