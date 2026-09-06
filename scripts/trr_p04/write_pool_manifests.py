#!/usr/bin/env python3
"""Extract P04 public pool record manifests from a frozen selection.

This command writes only public record identities and geometry metadata.  It
never materializes source text or token IDs, loads a model, opens evaluator
truth, or hashes large activation/embedding assets.  The implementation owner
can use the correction/validation manifests with its public activation capture
command; the immutable PR7 replay artifact is recorded as a parent-manifest
pointer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence


TASK_ID = "TRR-P04"
SELECTION_SCHEMA = "token-reconstruction.trr-p04-public-selection.v1"
SCHEMA = "token-reconstruction.trr-p04-public-pool-manifest.v1"
PANEL_LENGTHS = (16, 32, 64, 128)
STYLES = ("pile_plain", "finance_chat", "alpaca_instruction")
SENSITIVE_FIELDS = {
    "token_ids",
    "input_ids",
    "labels",
    "source_text",
    "truth",
    "oracle",
    "target_weights",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_selection(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selection is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"selection is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != SELECTION_SCHEMA or value.get("task_id") != TASK_ID:
        raise ValueError("selection schema or task changed")
    if value.get("execution", {}).get("evaluation_truth_accessed") is not False:
        raise ValueError("selection truth-access marker is not false")
    return value, _sha256_file(path)


def _records(value: Mapping[str, Any], pool: str) -> list[dict[str, Any]]:
    pools = value.get("pools")
    if not isinstance(pools, Mapping) or not isinstance(pools.get(pool), Mapping):
        raise ValueError(f"selection has no {pool} pool")
    rows = pools[pool].get("records")
    if not isinstance(rows, list):
        raise ValueError(f"selection {pool} records are malformed")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"selection {pool} row {index} is malformed")
        if SENSITIVE_FIELDS.intersection(row):
            raise ValueError(f"selection {pool} row {index} contains private fields")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"selection {pool} row {index} has a missing or duplicate record ID")
        seen.add(record_id)
        result.append(dict(row))
    return result


def _panel_checks(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 72:
        raise ValueError(f"fresh panel must contain 72 records, got {len(rows)}")
    if {row.get("style") for row in rows} != set(STYLES):
        raise ValueError("fresh panel styles changed")
    for style in STYLES:
        for length in PANEL_LENGTHS:
            cell = [row for row in rows if row.get("style") == style and row.get("length_stratum") == length]
            if len(cell) != 6:
                raise ValueError(f"fresh panel quota changed for {style}:{length}")
    anchors = [row for row in rows if bool(row.get("anchor", False))]
    if len(anchors) != 12 or any(row.get("length_stratum") != 32 for row in anchors):
        raise ValueError("fresh panel anchor quota changed")


def _record_manifest(
    *,
    pool: str,
    role: str,
    rows: Sequence[Mapping[str, Any]],
    selection_path: Path,
    selection_sha256: str,
    source_code: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "token-reconstruction.trr-p04-public-records.v1",
        "task_id": TASK_ID,
        "status": "PUBLIC_RECORD_METADATA_READY_NO_MODEL_NO_EVALUATION_TRUTH",
        "pool": pool,
        "role": role,
        "record_count": len(rows),
        "records": [dict(row) for row in rows],
        "token_ids_included": False,
        "source_text_included": False,
        "evaluation_truth_included": False,
        "selection": {
            "path": str(selection_path),
            "sha256": selection_sha256,
            "selection_adapts_to_scores": False,
        },
        "source_code": dict(source_code),
    }


def _file_pointer(path: Path, *, sha256: str | None = None, hash_source: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"replay asset is unavailable: {path}")
    result: dict[str, Any] = {
        "path": str(path.expanduser()),
        "resolved_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256,
        "hash_source": hash_source or "computed_by_manifest_writer",
    }
    if sha256 is None:
        result["sha256"] = _sha256_file(resolved)
    return result


def build_manifests(
    *,
    selection_path: Path,
    output_dir: Path,
    replay_observations: Path,
    replay_observations_sha256: str,
    replay_records: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.monotonic()
    started_utc = _utc_now()
    selection, selection_sha256 = _load_selection(selection_path)
    correction = _records(selection, "correction")
    validation = _records(selection, "validation")
    panel = _records(selection, "fresh_evaluation")
    if len(correction) != 256 or len(validation) != 192:
        raise ValueError("public correction/validation quotas changed")
    _panel_checks(panel)
    all_ids = [row["record_id"] for row in correction + validation + panel]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("public correction, validation, and panel records overlap")
    script = Path(__file__).resolve()
    source_code = {"path": str(script), "sha256": _sha256_file(script)}
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory is create-only: {output_dir}")
    output_dir.mkdir(parents=True)
    selection_resolved = selection_path.expanduser().resolve()
    manifests = {
        "correction": _record_manifest(
            pool="public_correction",
            role="public correction-training pool; labels may be materialized by implementation",
            rows=correction,
            selection_path=selection_resolved,
            selection_sha256=selection_sha256,
            source_code=source_code,
        ),
        "validation": _record_manifest(
            pool="public_validation",
            role="public validation pool; labels may be materialized by implementation",
            rows=validation,
            selection_path=selection_resolved,
            selection_sha256=selection_sha256,
            source_code=source_code,
        ),
        "fresh_panel": {
            "schema": "token-reconstruction.trr-p04-fresh-panel-index.v1",
            "task_id": TASK_ID,
            "status": "PUBLIC_PANEL_INDEX_READY_NO_MODEL_NO_EVALUATION_TRUTH",
            "role": "fresh evaluator-side panel index",
            "record_count": len(panel),
            "independent_source_records": len(panel),
            "paired_target_conditions": ["public_base", "p04_evaluator_target_update_v1"],
            "records": [dict(row) for row in panel],
            "anchor_record_count": sum(bool(row.get("anchor", False)) for row in panel),
            "anchor_scored_positions_per_target": 12 * 32,
            "token_ids_included": False,
            "source_text_included": False,
            "evaluation_truth_included": False,
            "selection": {"path": str(selection_resolved), "sha256": selection_sha256},
            "source_code": source_code,
        },
    }
    output_files: dict[str, dict[str, Any]] = {}
    names = {"correction": "correction_records.json", "validation": "validation_records.json", "fresh_panel": "fresh_panel_index.json"}
    for key, payload in manifests.items():
        path = output_dir / names[key]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_files[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    replay_pointer = _file_pointer(
        replay_observations,
        sha256=replay_observations_sha256,
        hash_source="immutable PR7 public-fit manifest; large asset not rehashed",
    )
    replay_record_pointer = _file_pointer(replay_records)
    execution = {
        "argv": list(argv),
        "python": sys.executable,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "max_rss_bytes": _max_rss_bytes(),
        "model_loaded": False,
        "evaluation_truth_accessed": False,
        "source_text_or_token_ids_materialized": False,
    }
    manifest = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_POOLS_READY_RECORD_METADATA_ONLY",
        "selection": {"path": str(selection_resolved), "bytes": selection_path.stat().st_size, "sha256": selection_sha256},
        "pool_files": output_files,
        "replay": {
            "record_manifest": replay_record_pointer,
            "observations": replay_pointer,
            "rows": 1200,
            "positions": 192,
            "hidden_size": 2048,
            "tensor_keys": ["activations", "attention_mask", "position_ids", "post_bos_selector_large", "post_bos_selector_small", "token_ids"],
            "role": "immutable PR7 public fit/replay; labels are public training data",
        },
        "correction": {
            "record_manifest": output_files["correction"],
            "rows": 256,
            "observations": None,
            "capture_owner": "implementation",
            "capture_status": "PENDING_PUBLIC_PREFIX_CAPTURE",
            "must_precede": ["capacity_probe", "teacher_qualification", "student_training"],
        },
        "validation": {
            "record_manifest": output_files["validation"],
            "rows": 192,
            "observations": None,
            "capture_owner": "implementation",
            "capture_status": "PENDING_PUBLIC_PREFIX_CAPTURE",
        },
        "fresh_panel": {
            "index": output_files["fresh_panel"],
            "rows": 72,
            "observations": None,
            "truth": None,
            "target_update": None,
            "access_status": "SEALED_UNTIL_JOINT_PREDICTION_FREEZE",
        },
        "access_contract": {
            "public_labels_only_for_replay_correction_validation": True,
            "evaluator_truth_opened": False,
            "target_update_accessed": False,
            "model_loaded": False,
            "source_text_or_token_ids_in_metadata": False,
        },
        "execution": execution,
        "source_code": source_code,
    }
    manifest_path = output_dir / "pool_manifest.json"
    # Keep the manifest self-hash outside the JSON to avoid a recursive
    # pointer. The file is create-only and callers can hash these final bytes.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-observations", type=Path, required=True)
    parser.add_argument("--replay-observations-sha256", required=True)
    parser.add_argument("--replay-records", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    values = list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv])
    try:
        result = build_manifests(
            selection_path=args.selection,
            output_dir=args.output_dir,
            replay_observations=args.replay_observations,
            replay_observations_sha256=str(args.replay_observations_sha256),
            replay_records=args.replay_records,
            argv=values,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.expanduser().resolve()), "correction_records": 256, "validation_records": 192, "fresh_panel_records": 72, "model_loaded": False, "evaluation_truth_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
