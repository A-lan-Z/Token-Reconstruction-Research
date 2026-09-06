#!/usr/bin/env python3
"""Compare public token support in the current and improved TRR-0007 banks.

This is a read-only, model-free comparison of two already materialized public
fit token banks.  It reports distinct token identities over all post-BOS
positions, the first one-based positions 1..127, late positions 128..191,
and IDs that occur only late.  It also verifies the public corpus-plan
replacement ledgers without writing token lists or source metadata.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch
from safetensors import safe_open


TASK_ID = "TRR-0007"
SCHEMA = "token-reconstruction.trr0007-public-bank-coverage-comparison.v1"
STATUS = "DESCRIPTIVE_PUBLIC_BANK_COVERAGE_ONLY"
BOS_TOKEN_ID = 128000
EXPECTED_ROWS = 1200
EXPECTED_WIDTH = 192
EXPECTED_POST_BOS_POSITIONS = 124371
EXPECTED_CONTROLLED_ROWS = 120
EXPECTED_REPLACEMENT_OCCURRENCES = 3600
EXPECTED_CURRENT_REPLACEMENT_IDS = 2000
EXPECTED_IMPROVED_REPLACEMENT_IDS = 3600


class CoverageError(ValueError):
    """Raised when a public bank or plan violates the comparison contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CoverageError(f"{label} is unavailable or is a symlink: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoverageError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CoverageError(f"{label} must be a JSON object")
    return value, record


def _set_digest(values: set[int]) -> str:
    payload = "\n".join(str(value) for value in sorted(values)) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


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
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _load_bank(token_path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _file_record(token_path, label=f"{label} token bank")
    try:
        with safe_open(str(token_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"token_ids", "attention_mask"}
            if not required.issubset(keys):
                raise CoverageError(f"{label} bank lacks token_ids/attention_mask")
            token_ids = handle.get_tensor("token_ids").detach().cpu().contiguous()
            mask = handle.get_tensor("attention_mask").detach().cpu().contiguous()
    except CoverageError:
        raise
    except Exception as exc:
        raise CoverageError(f"cannot read {label} public token bank") from exc
    if tuple(token_ids.shape) != (EXPECTED_ROWS, EXPECTED_WIDTH):
        raise CoverageError(f"{label} token geometry changed: {tuple(token_ids.shape)}")
    if tuple(mask.shape) != tuple(token_ids.shape):
        raise CoverageError(f"{label} token/mask geometry differs")
    if token_ids.dtype != torch.int32:
        raise CoverageError(f"{label} token dtype changed: {token_ids.dtype}")
    active = mask.to(dtype=torch.bool)
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise CoverageError(f"{label} mask is not binary")
    active_counts = active.sum(dim=1)
    if int(active_counts.sum()) != EXPECTED_POST_BOS_POSITIONS + EXPECTED_ROWS:
        raise CoverageError(f"{label} active geometry changed")
    if not bool(torch.all(active[:, 0])):
        raise CoverageError(f"{label} lost BOS-active rows")
    for row_index, count in enumerate(active_counts.tolist()):
        count = int(count)
        if not bool(torch.equal(active[row_index], torch.tensor([1] * count + [0] * (EXPECTED_WIDTH - count), dtype=torch.bool))):
            raise CoverageError(f"{label} mask is not a contiguous prefix at row {row_index}")
    if not bool(torch.all(token_ids[:, 0] == BOS_TOKEN_ID)):
        raise CoverageError(f"{label} BOS token changed")
    post = token_ids[:, 1:][active[:, 1:]]
    early_mask = active[:, 1:128]
    early = token_ids[:, 1:128][early_mask]
    late_mask = active[:, 128:]
    late = token_ids[:, 128:][late_mask]
    post_ids = {int(value) for value in post.tolist()}
    early_ids = {int(value) for value in early.tolist()}
    late_ids = {int(value) for value in late.tolist()}
    late_only = late_ids - early_ids
    summary = {
        "geometry": {
            "rows": EXPECTED_ROWS,
            "stored_width": EXPECTED_WIDTH,
            "active_positions_including_bos": int(active.sum()),
            "post_bos_positions": int(post.numel()),
            "first_1_127_positions": int(early.numel()),
            "late_128_191_positions": int(late.numel()),
        },
        "distinct": {
            "post_bos": {"count": len(post_ids), "ids_sha256": _set_digest(post_ids)},
            "first_1_127": {"count": len(early_ids), "ids_sha256": _set_digest(early_ids)},
            "late_128_191": {"count": len(late_ids), "ids_sha256": _set_digest(late_ids)},
            "late_only": {"count": len(late_only), "ids_sha256": _set_digest(late_only)},
        },
    }
    return summary, {"descriptor": descriptor, "keys": sorted(keys)}


def _plan_replacements(
    plan_path: Path,
    *,
    bank_path: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, plan_record = _json(plan_path, label=f"{label} corpus plan")
    try:
        rows = plan["arms"]["coverage_mix_v1"]["records"]
    except (KeyError, TypeError) as exc:
        raise CoverageError(f"{label} corpus plan has no coverage_mix_v1 records") from exc
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise CoverageError(f"{label} corpus plan record count changed")
    synthetic = [row for row in rows if isinstance(row, Mapping) and row.get("synthetic") is True]
    if len(synthetic) != EXPECTED_CONTROLLED_ROWS:
        raise CoverageError(f"{label} controlled row count changed")
    with safe_open(str(bank_path), framework="pt", device="cpu") as handle:
        token_ids = handle.get_tensor("token_ids").detach().cpu().contiguous()
        mask = handle.get_tensor("attention_mask").detach().cpu().contiguous().bool()
    replacement_ids: list[int] = []
    mismatch_count = 0
    replacement_count_values: set[int] = set()
    position_values: list[int] = []
    for row in synthetic:
        slot = row.get("slot")
        positions = row.get("replacement_positions")
        values = row.get("replacement_token_ids")
        if isinstance(slot, bool) or not isinstance(slot, int) or not isinstance(positions, list) or not isinstance(values, list):
            raise CoverageError(f"{label} replacement row is malformed")
        if len(positions) != len(values) or len(positions) != int(row.get("replacement_count", -1)):
            raise CoverageError(f"{label} replacement row count metadata changed")
        replacement_count_values.add(len(positions))
        for offset, value in zip(positions, values):
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise CoverageError(f"{label} replacement offset is invalid")
            token_value = int(value)
            replacement_ids.append(token_value)
            position_values.append(offset + 1)
            if int(token_ids[slot, offset + 1]) != token_value:
                mismatch_count += 1
        if int(mask[slot].sum()) != 1 + int(row.get("target_post_bos_token_count", -1)):
            raise CoverageError(f"{label} controlled row length changed at slot {slot}")
        one_based = row.get("replacement_positions_one_based")
        if one_based is not None and [int(value) + 1 for value in positions] != [int(value) for value in one_based]:
            raise CoverageError(f"{label} one-based replacement ledger disagrees")
    if mismatch_count:
        raise CoverageError(f"{label} has {mismatch_count} token/plan replacement mismatches")
    if replacement_count_values != {EXPECTED_REPLACEMENT_OCCURRENCES // EXPECTED_CONTROLLED_ROWS}:
        raise CoverageError(f"{label} per-row replacement count changed")
    replacement_set = set(replacement_ids)
    controlled = plan.get("controlled_token_selection")
    if not isinstance(controlled, Mapping) or not isinstance(controlled.get("selected_token_ids"), list):
        raise CoverageError(f"{label} controlled identity selection is absent")
    legacy_ids = {int(value) for value in controlled["selected_token_ids"]}
    if legacy_ids != replacement_set and label == "current_enriched":
        raise CoverageError(f"{label} legacy controlled IDs do not match replacements")
    support = plan.get("trr0007_support")
    support_ids = None
    baseline_ids = None
    additional_ids = None
    if label == "improved_public_bank":
        if not isinstance(support, Mapping):
            raise CoverageError("improved bank support ledger is absent")
        support_ids = {int(value) for value in support.get("selected_token_ids", [])}
        baseline_ids = {int(value) for value in support.get("baseline_token_ids", [])}
        additional_ids = {int(value) for value in support.get("additional_token_ids", [])}
        if support_ids != replacement_set:
            raise CoverageError("improved support selected IDs do not match replacements")
        if baseline_ids | additional_ids != support_ids or baseline_ids & additional_ids:
            raise CoverageError("improved baseline/additional identity sets are malformed")
    return {
        "plan": plan_record,
        "controlled_rows": len(synthetic),
        "replacement_occurrences": len(replacement_ids),
        "replacement_distinct_ids": len(replacement_set),
        "replacement_ids_sha256": _set_digest(replacement_set),
        "replacement_position_min": min(position_values),
        "replacement_position_max": max(position_values),
        "replacement_positions_1_127": sum(1 <= value <= 127 for value in position_values),
        "legacy_selected_id_count": len(legacy_ids),
        "legacy_selected_ids_sha256": _set_digest(legacy_ids),
        "replacement_ids_match_legacy_selection": legacy_ids == replacement_set,
        "support_selected_id_count": len(support_ids) if support_ids is not None else None,
        "support_selected_ids_sha256": _set_digest(support_ids) if support_ids is not None else None,
        "support_baseline_id_count": len(baseline_ids) if baseline_ids is not None else None,
        "support_baseline_ids_sha256": _set_digest(baseline_ids) if baseline_ids is not None else None,
        "support_additional_id_count": len(additional_ids) if additional_ids is not None else None,
        "support_additional_ids_sha256": _set_digest(additional_ids) if additional_ids is not None else None,
        "support_selected_ids_match_replacements": support_ids == replacement_set if support_ids is not None else None,
    }, plan


def _scope_comparison(current: set[int], improved: set[int]) -> dict[str, Any]:
    common = current & improved
    current_only = current - improved
    improved_only = improved - current
    return {
        "current_count": len(current),
        "improved_count": len(improved),
        "common_count": len(common),
        "current_only_count": len(current_only),
        "improved_only_count": len(improved_only),
        "new_in_improved_count": len(improved_only),
        "current_ids_sha256": _set_digest(current),
        "improved_ids_sha256": _set_digest(improved),
        "common_ids_sha256": _set_digest(common),
        "current_only_ids_sha256": _set_digest(current_only),
        "improved_only_ids_sha256": _set_digest(improved_only),
    }


def build_comparison(
    *,
    repository_root: Path,
    current_tokens: Path,
    current_plan: Path,
    improved_tokens: Path,
    improved_plan: Path,
) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    current_coverage, current_source = _load_bank(current_tokens, label="current_enriched")
    improved_coverage, improved_source = _load_bank(improved_tokens, label="improved_public_bank")
    current_replacement, current_plan_value = _plan_replacements(
        current_plan, bank_path=current_tokens, label="current_enriched"
    )
    improved_replacement, improved_plan_value = _plan_replacements(
        improved_plan, bank_path=improved_tokens, label="improved_public_bank"
    )
    # Reconstruct only compact set identities from the public tensors.  No ID
    # lists are written; these are used to make the comparison fields exact.
    sets: dict[str, dict[str, set[int]]] = {}
    for label, path in (("current_enriched", current_tokens), ("improved_public_bank", improved_tokens)):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            ids = handle.get_tensor("token_ids").detach().cpu().contiguous()
            mask = handle.get_tensor("attention_mask").detach().cpu().contiguous().bool()
        post = ids[:, 1:][mask[:, 1:]]
        early = ids[:, 1:128][mask[:, 1:128]]
        late = ids[:, 128:][mask[:, 128:]]
        post_set = {int(value) for value in post.tolist()}
        early_set = {int(value) for value in early.tolist()}
        late_set = {int(value) for value in late.tolist()}
        sets[label] = {
            "post_bos": post_set,
            "first_1_127": early_set,
            "late_128_191": late_set,
            "late_only": late_set - early_set,
        }
    # The improved support ledger is the frozen 2,000 + 1,600 identity split.
    current_ids = set(int(value) for value in current_plan_value["controlled_token_selection"]["selected_token_ids"])
    improved_support = improved_plan_value["trr0007_support"]
    improved_baseline = {int(value) for value in improved_support["baseline_token_ids"]}
    improved_additional = {int(value) for value in improved_support["additional_token_ids"]}
    preserved = current_ids & improved_baseline
    current_only = current_ids - improved_baseline
    additional_not_current = improved_additional - current_ids
    current_additional_overlap = current_ids & improved_additional
    comparison = {
        scope: _scope_comparison(sets["current_enriched"][scope], sets["improved_public_bank"][scope])
        for scope in ("post_bos", "first_1_127", "late_128_191", "late_only")
    }
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "purpose": "Compact descriptive coverage comparison of the existing public current and improved fitting banks.",
        "method": {
            "post_bos_scope": "all active columns after the declared BOS column 0",
            "first_1_127_scope": "one-based post-BOS positions 1 through 127, inclusive",
            "late_scope": "one-based post-BOS positions 128 through 191, inclusive",
            "late_only_scope": "late-scope identities absent from first_1_127 identities within the same bank",
            "set_digest": "SHA-256 of newline-terminated sorted decimal token IDs; full ID lists are not emitted",
            "bos_token_id": BOS_TOKEN_ID,
            "source_text_or_truth_accessed": False,
        },
        "banks": {
            "current_enriched": {"tokens": current_source["descriptor"], "coverage": current_coverage},
            "improved_public_bank": {"tokens": improved_source["descriptor"], "coverage": improved_coverage},
        },
        "comparison": comparison,
        "controlled_replacement_support": {
            "current_enriched": current_replacement,
            "improved_public_bank": improved_replacement,
            "identity_split": {
                "current_replacement_id_count": len(current_ids),
                "improved_baseline_preserved_count": len(preserved),
                "improved_additional_current_unseen_count": len(improved_additional),
                "improved_additional_not_in_current_count": len(additional_not_current),
                "current_ids_not_in_improved_baseline_count": len(current_only),
                "current_ids_overlapping_improved_additional_count": len(current_additional_overlap),
                "current_ids_preserved_plus_new_total": len(preserved) + len(additional_not_current),
                "preserved_ids_sha256": _set_digest(preserved),
                "new_current_unseen_ids_sha256": _set_digest(additional_not_current),
                "preserved_plus_new_ids_sha256": _set_digest(preserved | additional_not_current),
                "preserved_set_equals_current_replacement_set": preserved == current_ids,
                "new_set_disjoint_from_current_replacement_set": not current_additional_overlap,
            },
        },
        "matched_geometry": {
            "rows": EXPECTED_ROWS,
            "post_bos_positions_per_bank": EXPECTED_POST_BOS_POSITIONS,
            "controlled_rows_per_bank": EXPECTED_CONTROLLED_ROWS,
            "replacement_occurrences_per_bank": EXPECTED_REPLACEMENT_OCCURRENCES,
        },
        "execution": {
            "script": str(Path(__file__).resolve()),
            "code_commit": _git_commit(root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "network_used": False,
            "gpu_used": False,
            "fresh_selection_opened": False,
            "evaluation_truth_opened": False,
            "model_loaded": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--current-tokens",
        type=Path,
        default=Path("../TRR-0005/experiments/TRR-0005/corpus/coverage_mix_v1/constructed_public_tokens.safetensors"),
    )
    parser.add_argument(
        "--current-plan",
        type=Path,
        default=Path("../TRR-0005/experiments/TRR-0005/corpus/corpus_plan.json"),
    )
    parser.add_argument(
        "--improved-tokens",
        type=Path,
        default=Path("experiments/TRR-0007/support/broader_bank_v5/constructed_public_tokens.safetensors"),
    )
    parser.add_argument(
        "--improved-plan",
        type=Path,
        default=Path("experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-0007/support/coverage_comparison.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_comparison(
            repository_root=args.repository_root,
            current_tokens=args.current_tokens,
            current_plan=args.current_plan,
            improved_tokens=args.improved_tokens,
            improved_plan=args.improved_plan,
        )
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise CoverageError(f"output is create-only and already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        result["output"] = {
            "path": str(output),
            "bytes": int(output.stat().st_size),
            "sha256": _sha256_file(output),
        }
    except (OSError, CoverageError, RuntimeError, ValueError) as exc:
        print(f"TRR-0007 public coverage comparison failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
