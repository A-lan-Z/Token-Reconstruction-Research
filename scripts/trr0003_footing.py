#!/usr/bin/env python3
"""Prepare and preflight the shared TRR-0003 development panel.

This script only reads public ledgers and public activation assets.  It writes
no evaluator truth and has no scoring path.  The resulting panel is the common
input contract for the Track A, Track B, and historical comparator runners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open
import torch

from token_reconstruction.footing import (
    CONDITION_ORDER,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PANEL_SCHEMA,
    STYLE_ORDER,
    FootingError,
    file_record,
    load_panel,
    sha256_file,
)


PLAN_SCHEMA = "token-reconstruction.trr0003-footing-plan.v1"
TASK_ID = "TRR-0003"
PANEL_RECORDS_PER_STYLE = 8


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"public ledger is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootingError(f"public ledger is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FootingError(f"public ledger root is not an object: {path}")
    return value


def _write_create_or_same(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise FootingError(f"refusing to overwrite changed footing artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _pile_rows(
    *,
    source_root: Path,
    source_plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger_path = source_root / "records.json"
    ledger = _json(ledger_path)
    rows = ledger.get("development")
    if not isinstance(rows, list) or len(rows) < PANEL_RECORDS_PER_STYLE:
        raise FootingError("Pile public development ledger is too short")
    selected = rows[:PANEL_RECORDS_PER_STYLE]
    selected_ids = [row.get("record_id") for row in selected]
    if any(not isinstance(record_id, str) or not record_id for record_id in selected_ids):
        raise FootingError("Pile public development IDs are malformed")
    declared = source_plan["data"]["selection"]["splits"]
    split_ids = {
        split: {row["record_id"] for row in declared[split]["records"]}
        for split in ("inverse_train", "target_update_train", "blind_evaluation")
    }
    for split, ids in split_ids.items():
        overlap = set(selected_ids) & ids
        if overlap:
            raise FootingError(f"panel Pile records overlap {split}: {sorted(overlap)}")
    sanitized: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise FootingError("Pile public record row is malformed")
        sanitized.append(
            {
                "record_id": str(row["record_id"]),
                "public_record_sha256": str(row["text_sha256"]),
            }
        )
    return sanitized, {
        "dataset": "NeelNanda/pile-10k",
        "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
        "ledger": file_record(ledger_path, repository_root=source_root.parent.parent.parent),
        "selected_split": "development",
        "selection": "first 8 rows in the existing TRR-0002 public-development ledger",
        "disjoint_against": ["inverse_train", "target_update_train", "blind_evaluation"],
        "disjoint_checked": True,
    }


def _finance_rows(*, source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger_path = source_root / "records.json"
    ledger = _json(ledger_path)
    rows = ledger.get("records")
    if not isinstance(rows, list) or len(rows) < PANEL_RECORDS_PER_STYLE:
        raise FootingError("Finance public ledger is too short")
    selected = rows[:PANEL_RECORDS_PER_STYLE]
    sanitized: list[dict[str, Any]] = []
    masks: list[list[int]] | None = None
    positions: list[list[int]] | None = None
    for row in selected:
        if not isinstance(row, Mapping):
            raise FootingError("Finance public record row is malformed")
        row_mask = row.get("attention_mask")
        row_positions = row.get("position_ids")
        if not isinstance(row_mask, list) or not isinstance(row_positions, list):
            raise FootingError("Finance public masks/positions are missing")
        if len(row_mask) != 128 or len(row_positions) != 128:
            raise FootingError("Finance public sequence geometry changed")
        if masks is None:
            masks = []
            positions = []
        masks.append([int(value) for value in row_mask])
        positions.append([int(value) for value in row_positions])
        sanitized.append(
            {
                "record_id": str(row["record_id"]),
                "public_record_sha256": str(row["content_sha256"]),
                "raw_index": int(row["raw_index"]),
                "valid_tokens": int(row["valid_tokens"]),
                "tokenized_record_sha256": str(row["token_ids_sha256"]),
            }
        )
    assert masks is not None and positions is not None
    return sanitized, {
        "dataset": str(ledger["dataset"]),
        "revision_or_fingerprint": str(ledger["fingerprint"]),
        "ledger": file_record(ledger_path, repository_root=source_root.parent.parent.parent),
        "selected_split": "public_cursor",
        "selection": "first 8 rows in the existing TRR-0002 public-finance ledger",
        "public_cursor": list(ledger["public_cursor"]),
        "historical_cursor": list(ledger["historical_cursor"]),
        "overlap": dict(ledger["overlap"]),
        "disjoint_checked": dict(ledger["overlap"]) == {
            "normalized_content_sha256": 0,
            "raw_index": 0,
        },
        "attention_mask": masks,
        "position_ids": positions,
    }


def _asset(path: Path, *, repository_root: Path, observation: bool = False) -> dict[str, Any]:
    result = file_record(path, repository_root=repository_root)
    if observation:
        result["tensor_key"] = "activations"
        result["row_indices"] = list(range(PANEL_RECORDS_PER_STYLE))
    return result


def _lora_shift_evidence(
    *,
    root: Path,
    pile_root: Path,
    finance_root: Path,
    masks: Mapping[str, list[list[int]]],
) -> dict[str, Any]:
    """Record public LoRA construction and activation drift, excluding BOS."""

    generation_path = pile_root / "generation.json"
    generation = _json(generation_path)
    condition = next(
        (row for row in generation.get("conditions", []) if row.get("id") == "public_lora_2601"),
        None,
    )
    if not isinstance(condition, Mapping) or not isinstance(condition.get("training"), Mapping):
        raise FootingError("public_lora_2601 training metadata is absent")
    config = condition["training"].get("config")
    update = condition.get("update")
    if not isinstance(config, Mapping) or not isinstance(update, Mapping):
        raise FootingError("public_lora_2601 construction metadata is incomplete")
    update_path = root / str(update["path"])
    update_asset = _asset(update_path, repository_root=root)
    if update_asset["sha256"] != update.get("sha256") or update_asset["bytes"] != update.get("bytes"):
        raise FootingError("public_lora_2601 update state hash changed")

    base_paths = {
        "pile": pile_root / "observations" / "public_base_cut4.safetensors",
        "finance": finance_root / "observations" / "public_base_cut4.safetensors",
    }
    shift_paths = {
        "pile": pile_root / "observations" / "public_lora_2601_cut4.safetensors",
        "finance": finance_root / "observations" / "public_lora_2601_cut4.safetensors",
    }
    drift: dict[str, Any] = {}
    for style in STYLE_ORDER:
        with safe_open(base_paths[style], framework="pt", device="cpu") as handle:
            base = handle.get_tensor("activations")[:PANEL_RECORDS_PER_STYLE].float()
        with safe_open(shift_paths[style], framework="pt", device="cpu") as handle:
            shifted = handle.get_tensor("activations")[:PANEL_RECORDS_PER_STYLE].float()
        mask = torch.tensor(masks[style], dtype=torch.bool)
        mask[:, 0] = False
        left = base[mask]
        right = shifted[mask]
        delta = right - left
        token_relative = delta.norm(dim=1) / (left.norm(dim=1) + 1e-12)
        token_cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
        record_relative = [
            float((shifted[index][mask[index]] - base[index][mask[index]]).norm() / (base[index][mask[index]].norm() + 1e-12))
            for index in range(PANEL_RECORDS_PER_STYLE)
        ]
        drift[style] = {
            "scored_tokens": int(mask.sum().item()),
            "relative_l2_all_scored": float(delta.norm() / (left.norm() + 1e-12)),
            "relative_l2_token_mean": float(token_relative.mean()),
            "relative_l2_token_median": float(token_relative.median()),
            "relative_l2_token_min": float(token_relative.min()),
            "relative_l2_token_max": float(token_relative.max()),
            "cosine_token_mean": float(token_cosine.mean()),
            "cosine_token_min": float(token_cosine.min()),
            "relative_l2_record_values": record_relative,
        }
    return {
        "construction": {
            "rank": int(config["rank"]),
            "alpha": float(config["alpha"]),
            "layers": list(config["layers"]),
            "modules": list(config["modules"]),
            "learning_rate": float(config["learning_rate"]),
            "steps": int(config["steps"]),
            "seed": int(config["seed"]),
            "batch_records": int(config["batch_records"]),
            "weight_decay": float(config["weight_decay"]),
            "gradient_clip_norm": float(config["gradient_clip_norm"]),
        },
        "update_state": update_asset,
        "generation_metadata": file_record(generation_path, repository_root=root),
        "paired_activation_drift_excluding_bos": drift,
    }


def _cell(
    *,
    style: str,
    condition: str,
    records: list[dict[str, Any]],
    mask: list[list[int]],
    positions: list[list[int]],
    observation: dict[str, Any],
    sequence_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"{style}__{condition}",
        "style": style,
        "condition": condition,
        "shift_role": (
            "matched_public_control"
            if condition == "public_base"
            else "single_public_shift_diagnostic"
        ),
        "records": records,
        "attention_mask": mask,
        "position_ids": positions,
        "observation": observation,
        "geometry": {
            "records": PANEL_RECORDS_PER_STYLE,
            "sequence_tokens": sequence_tokens,
            "hidden_size": HIDDEN_SIZE,
            "cut_depth": CUT_DEPTH,
        },
    }


def prepare(args: argparse.Namespace) -> None:
    root = args.repository_root.resolve()
    output = args.output_root.resolve()
    source_plan_path = root / "experiments/TRR-0001/plan.json"
    source_plan = _json(source_plan_path)
    pile_root = (root / args.pile_source_root).resolve()
    finance_root = (root / args.finance_source_root).resolve()
    pile_records, pile_source = _pile_rows(source_root=pile_root, source_plan=source_plan)
    finance_records, finance_source = _finance_rows(source_root=finance_root)
    pile_ledger = _json(pile_root / "records.json")
    development_rows = pile_ledger.get("development")
    if not isinstance(development_rows, list) or len(development_rows) < 32:
        raise FootingError("Pile development ledger lacks the fixed decoder-fit rows")
    validation_rows = development_rows[8:32]
    validation_ids = [str(row["record_id"]) for row in validation_rows]
    panel_ids = {row["record_id"] for row in pile_records}
    if panel_ids & set(validation_ids) or len(set(validation_ids)) != 24:
        raise FootingError("Track-B validation rows overlap the shared panel")

    pile_mask = [[1] * 40 for _ in range(PANEL_RECORDS_PER_STYLE)]
    pile_positions = [list(range(40)) for _ in range(PANEL_RECORDS_PER_STYLE)]
    finance_mask = finance_source.pop("attention_mask")
    finance_positions = finance_source.pop("position_ids")
    lora_shift_evidence = _lora_shift_evidence(
        root=root,
        pile_root=pile_root,
        finance_root=finance_root,
        masks={"pile": pile_mask, "finance": finance_mask},
    )
    observation_paths = {
        "pile": {
            condition: pile_root / "observations" / f"{condition}_cut4.safetensors"
            for condition in CONDITION_ORDER
        },
        "finance": {
            condition: finance_root / "observations" / f"{condition}_cut4.safetensors"
            for condition in CONDITION_ORDER
        },
    }
    observations = {
        style: {
            condition: _asset(path, repository_root=root, observation=True)
            for condition, path in per_condition.items()
        }
        for style, per_condition in observation_paths.items()
    }
    cells = []
    for style, records, mask, positions, sequence_tokens in (
        ("pile", pile_records, pile_mask, pile_positions, 40),
        ("finance", finance_records, finance_mask, finance_positions, 128),
    ):
        for condition in CONDITION_ORDER:
            cells.append(
                _cell(
                    style=style,
                    condition=condition,
                    records=records,
                    mask=mask,
                    positions=positions,
                    observation=observations[style][condition],
                    sequence_tokens=sequence_tokens,
                )
            )
    panel = {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": "RETROSPECTIVE_DEVELOPMENT_PANEL",
        "panel_id": "trr0003-footing-dev16-v1",
        "source_material_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "cut_depth": CUT_DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "styles": [
            {
                "id": "pile",
                "records": PANEL_RECORDS_PER_STYLE,
                "sequence_tokens": 40,
                "hidden_size": HIDDEN_SIZE,
                "input_style": "plain Pile text",
                "source": pile_source,
            },
            {
                "id": "finance",
                "records": PANEL_RECORDS_PER_STYLE,
                "sequence_tokens": 128,
                "hidden_size": HIDDEN_SIZE,
                "input_style": "Finance chat-template rows",
                "source": finance_source,
            },
        ],
        "conditions": [
            {
                "id": "public_base",
                "role": "matched_public_control",
                "weights_available_to_reconstructor": True,
                "online_prefix_calls_allowed": True,
            },
            {
                "id": "public_lora_2601",
                "role": "one synthetic public target shift diagnostic",
                "weights_available_to_reconstructor": False,
                "online_prefix_calls_allowed": True,
            },
        ],
        "cells": cells,
        "method_output_contract": {
            "method_ids": [
                "historical_alpaca_a1",
                "frozen_a1_a2_k256",
                "direct_inverse",
            ],
            "artifact_template": "<output>/<style>/<condition>/<method_id>.safetensors",
            "required_tensors": ["predictions"],
            "optional_diagnostics": [
                "candidates",
                "candidate_scores",
                "selection_scores",
            ],
            "all_cells_required_before_evaluation": True,
        },
        "canonical_status": {
            "new_track_a_methods": "NOT_RUN",
            "new_track_b_methods": "NOT_RUN",
            "dual_benchmark_comparison": "INCOMPLETE",
        },
    }
    panel_path = output / "panel.json"
    _write_create_or_same(panel_path, panel)
    loaded = load_panel(panel_path, repository_root=root)
    panel_sha = sha256_file(panel_path)
    plan = {
        "schema": PLAN_SCHEMA,
        "task_id": TASK_ID,
        "status": "FOOTING_READY_FOR_REVIEW",
        "panel": {
            "path": str(panel_path.relative_to(root).as_posix()),
            "bytes": panel_path.stat().st_size,
            "sha256": panel_sha,
            "cells": 4,
            "distinct_records_per_condition": 16,
            "observed_sequences_total": 32,
        },
        "resources": {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "device": "RTX 5080 16GB shared development panel",
            "network": "offline local caches only",
            "paid_compute": False,
        },
        "public_assets": {
            "panel_observations": observations,
            "historical_lens": _asset(
                root / "outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt",
                repository_root=root,
            ),
            "existing_direct_inverse": _asset(
                root / "outputs/TRR-0001/reconstructor_public/inverses/cut4.safetensors",
                repository_root=root,
            ),
            "public_prefix_snapshot": {
                "path": "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/" + MODEL_REVISION,
                "revision": MODEL_REVISION,
                "local_files_only": True,
            },
        },
        "fitting_separation": {
            "track_b": {
                "fit_split": "inverse_train",
                "fit_record_count": 128,
                "validation_split": "development_rows_8_31",
                "validation_row_slice": [8, 32],
                "source_ledger": str((pile_root / "records.json").relative_to(root).as_posix()),
                "source_plan": str(source_plan_path.relative_to(root).as_posix()),
                "validation_record_ids": [str(row["record_id"]) for row in development_rows[8:32]],
                "panel_rows_forbidden": True,
                "panel_overlap_checked": True,
                "finance_fitting": "must be separately declared by Track B; shared panel rows are forbidden",
            },
            "track_a": {
                "fitted_state": False,
                "panel_used_for_development_only": True,
            },
            "existing_comparator_state": {
                "direct_inverse_training_split": "inverse_train",
                "historical_lens_training": "prior TRR-0002 public A1 fit; frozen comparator only",
            },
        },
        "condition_notes": {
            "public_base": "matched public checkpoint diagnostic control",
            "public_lora_2601": {
                "role": "single synthetic LoRA condition; update weights unavailable to reconstruction; not full-SFT evidence",
                **lora_shift_evidence,
            },
            "paired_rows": True,
            "same_observation_execution": True,
            "numerical_difference_reporting": "record activation dtype, prefix implementation, and timing differences",
        },
        "comparison_status": {
            "historical_comparators": "pending shared-panel run",
            "track_a_canonical": "NOT RUN",
            "track_b_canonical": "NOT RUN",
            "dual_benchmark_matrix": "comparison-incomplete by design for this pilot",
        },
        "validated_panel": {
            "styles": list(STYLE_ORDER),
            "conditions": list(CONDITION_ORDER),
            "cells": [cell.cell_id for cell in load_panel_cells(loaded, root)],
            "largest_geometry": [8, 128, HIDDEN_SIZE],
        },
    }
    _write_create_or_same(
        output / "lora_2601_drift.json",
        {
            "schema": "token-reconstruction.trr0003-public-shift-drift.v1",
            "task_id": TASK_ID,
            "condition": "public_lora_2601",
            **lora_shift_evidence,
        },
    )
    _write_create_or_same(output / "plan.json", plan)
    print(json.dumps({"panel": str(panel_path), "plan": str(output / "plan.json"), "panel_sha256": panel_sha}, indent=2))


def load_panel_cells(panel: Mapping[str, Any], root: Path) -> tuple[Any, ...]:
    from token_reconstruction.footing import load_all_cells

    return load_all_cells(panel, repository_root=root)


def preflight(args: argparse.Namespace) -> None:
    root = args.repository_root.resolve()
    output = args.output_root.resolve()
    panel_path = output / "panel.json"
    plan_path = output / "plan.json"
    panel = load_panel(panel_path, repository_root=root)
    cells = load_panel_cells(panel, root)
    largest = max((cell.shape for cell in cells), key=lambda shape: shape[1])
    # Activation bytes are a lower bound for loading one cell; candidate and
    # model state are accounted for by each method's own preflight.
    activation_bytes = largest[0] * largest[1] * largest[2] * 2
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "panel": str(panel_path),
                "panel_sha256": sha256_file(panel_path),
                "plan_sha256": sha256_file(plan_path),
                "cells": len(cells),
                "largest_cell_shape": list(largest),
                "largest_activation_bytes_bfloat16": activation_bytes,
                "qualification_order": "finance first; 128 positions is largest",
                "gpu_run": "not launched by footing preflight",
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/TRR-0003/footing"))
    parser.add_argument(
        "--pile-source-root",
        type=Path,
        default=Path("outputs/TRR-0002/public-calibration"),
    )
    parser.add_argument(
        "--finance-source-root",
        type=Path,
        default=Path("outputs/TRR-0002/configuration-search/public-finance"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        preflight(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FootingError as exc:
        raise SystemExit(f"TRR-0003 footing error: {exc}")
