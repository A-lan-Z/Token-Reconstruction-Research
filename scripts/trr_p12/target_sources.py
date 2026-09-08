"""Prepare the evaluator-only TRR-P12 Alpaca target-source bundle.

The source worker reads the pinned public Alpaca cache and tokenizer on CPU,
but emits no public text or selection payload to reconstruction.  The retained
target trainer consumes the two safetensors files produced here; they are
explicitly evaluator-only and are never passed to the frozen decoder.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from safetensors.torch import save_file

from scripts.trr_p10 import build_exclusion_audit as p10
from scripts.trr_p11 import source_selector as p11
from scripts.trr_p12 import select_panel
from token_reconstruction.alpaca_split import (
    ALPACA_CACHE_REVISION,
    ALPACA_DATASET_ID,
    ALPACA_SPLIT,
    historical_rendered_text,
    public_record_id,
)
from token_reconstruction.trr0005_public_corpus import deterministic_row_order


TASK_ID = "TRR-P12"
SOURCE_SCHEMA = "token-reconstruction.trr-p12-target-source-bundle.v1"
RELEASE_SCHEMA = "token-reconstruction.trr-p12-target-source-release.v1"
RELEASE_STATUS = "P12_TARGET_SOURCE_RELEASED"
PANEL_SCHEMA = "token-reconstruction.trr-p12-source-panel.v1"
PANEL_STATUS = "FROZEN_P12_SOURCE_PANEL_NO_TRUTH"
UNION_SHA256 = select_panel.UNION_SHA256
UNION_BYTES = select_panel.UNION_BYTES
STUDY_MANIFEST_SHA256 = select_panel.STUDY_MANIFEST_SHA256
SELECTION_SEED = 6012
TRAIN_ROWS = 256
VALIDATION_ROWS = 64
SEQUENCE_LENGTH = 128
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
CANDIDATE_RANGE = (20000, 30000)
DEFAULT_ARROW = Path(
    "/home/alanz/.cache/huggingface/datasets/tatsu-lab/alpaca/default/0.0.0/"
    f"{ALPACA_CACHE_REVISION}/alpaca-train.arrow"
)
DEFAULT_TOKENIZER = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)
ALLOWED_FIELDS = select_panel.ALLOWED_UNION_FIELDS
HEX = select_panel.HEX


class TargetSourceError(RuntimeError):
    """Raised when target-source preparation cannot satisfy its bindings."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(value: str | Path, *, root: Path, label: str) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if path.is_symlink() or not path.is_file():
        raise TargetSourceError(f"{label} is unavailable or symlinked: {path}")
    return path


def file_binding(value: str | Path, *, root: Path, label: str,
                 expected_sha256: str | None = None, expected_bytes: int | None = None) -> tuple[Path, dict[str, Any]]:
    path = _resolve(value, root=root, label=label)
    digest = sha256_file(path)
    size = int(path.stat().st_size)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise TargetSourceError(f"{label} SHA-256 changed: {digest} != {expected_sha256}")
    if expected_bytes is not None and size != int(expected_bytes):
        raise TargetSourceError(f"{label} byte count changed: {size} != {expected_bytes}")
    return path, {"path": str(path), "bytes": size, "sha256": digest, "readonly": True}


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetSourceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise TargetSourceError(f"{label} must be a JSON object")
    return dict(value)


def _binding(value: Any, *, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    if isinstance(value, Mapping):
        if not isinstance(value.get("path"), str):
            raise TargetSourceError(f"{label} path is absent")
        return file_binding(value["path"], root=root, label=label,
                            expected_sha256=value.get("sha256"), expected_bytes=value.get("bytes"))
    if isinstance(value, (str, Path)):
        return file_binding(value, root=root, label=label)
    raise TargetSourceError(f"{label} binding is malformed")


def validate_release(path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    release_file, binding = _binding(path, root=root, label="P12 target-source release")
    payload = load_json(release_file, label="P12 target-source release")
    is_study_manifest = (
        payload.get("schema") == "token-reconstruction.trr-p12-plan.v1"
        and payload.get("task_id") == TASK_ID
        and binding["sha256"] == STUDY_MANIFEST_SHA256
    )
    if not is_study_manifest and (payload.get("schema") != RELEASE_SCHEMA or payload.get("task_id") != TASK_ID):
        raise TargetSourceError("P12 target-source release schema/task identity changed")
    if not is_study_manifest and (payload.get("status") != RELEASE_STATUS or payload.get("selection_release") is not True):
        raise TargetSourceError("P12 target-source release is not active")
    target = payload.get("target") if is_study_manifest else payload
    if not isinstance(target, Mapping):
        raise TargetSourceError("target release constants are absent")
    if target.get("selection_seed", target.get("seed")) != SELECTION_SEED:
        raise TargetSourceError("target selection seed changed")
    if target.get("train_records") != TRAIN_ROWS or target.get("validation_records") != VALIDATION_ROWS:
        raise TargetSourceError("target source counts changed")
    if tuple(target.get("candidate_range_half_open", ())) != CANDIDATE_RANGE:
        raise TargetSourceError("target candidate range changed")
    if target.get("sequence_length", SEQUENCE_LENGTH) != SEQUENCE_LENGTH or target.get("bos_token_id", BOS_TOKEN_ID) != BOS_TOKEN_ID:
        raise TargetSourceError("target sequence geometry changed")
    if target.get("dataset_id", ALPACA_DATASET_ID) != ALPACA_DATASET_ID or target.get("split", ALPACA_SPLIT) != ALPACA_SPLIT or target.get("revision", ALPACA_CACHE_REVISION) != ALPACA_CACHE_REVISION:
        raise TargetSourceError("target Alpaca dataset binding changed")
    study = ({"path": str(release_file), "bytes": binding["bytes"], "sha256": binding["sha256"], "readonly": True}
             if is_study_manifest else payload.get("study_manifest"))
    if not isinstance(study, Mapping) or study.get("sha256") != STUDY_MANIFEST_SHA256:
        raise TargetSourceError("target release is not bound to the immutable study manifest")
    panel = None if is_study_manifest else (payload.get("panel") or payload.get("evaluation_panel"))
    if panel is not None and (not isinstance(panel, Mapping) or not isinstance(panel.get("sha256"), str)):
        raise TargetSourceError("target release panel binding is malformed")
    if payload.get("truth_opened", False) is not False or payload.get("p03_holdout_accessed", False) is not False:
        raise TargetSourceError("target release boundary is open")
    return payload, binding

def _namespace(text: str) -> p10.Namespace:
    parts = text.split("|")
    if len(parts) != 4:
        raise TargetSourceError(f"malformed namespace: {text!r}")
    return p10.Namespace(*parts)


def _panel_bundle(panel_path: Path, *, root: Path) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    resolved, binding = _binding(panel_path, root=root, label="P12 source panel")
    payload = load_json(resolved, label="P12 source panel")
    if payload.get("schema") != PANEL_SCHEMA or payload.get("status") != PANEL_STATUS:
        raise TargetSourceError("P12 source panel is not frozen")
    if payload.get("truth_opened") is not False or payload.get("p03_holdout_accessed") is not False:
        raise TargetSourceError("P12 source panel boundary is open")
    rule = payload.get("selection_rule")
    rows_by_domain = rule.get("records") if isinstance(rule, Mapping) else None
    if not isinstance(rows_by_domain, Mapping):
        raise TargetSourceError("P12 source panel rows are absent")
    bundle = p10.IdentityBundle("P12 reconstruction panel", "opened_evaluation", resolved,
                                binding["sha256"], binding["bytes"], schema=PANEL_SCHEMA, status=PANEL_STATUS)
    for domain in select_panel.DOMAIN_ORDER:
        rows = rows_by_domain.get(domain)
        if not isinstance(rows, list) or len(rows) != select_panel.RECORDS_PER_DOMAIN:
            raise TargetSourceError(f"P12 panel {domain} count changed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise TargetSourceError("P12 panel row is malformed")
            try:
                namespace = p10.Namespace(str(row["dataset_key"]), str(row["dataset_id"]),
                                          str(row["split"]), str(row["revision"]))
                bundle.add("record_id", str(row["record_id"]), namespace)
                bundle.add("source_index", int(row["source_index"]), namespace)
                bundle.add("rendered_sha256", str(row["public_record_sha256"]).lower(), namespace)
                bundle.add("h128_sequence_sha256", str(row["h128_sequence_sha256"]).lower(), namespace)
                if row.get("h129_sequence_sha256") is not None:
                    bundle.add("h129_sequence_sha256", str(row["h129_sequence_sha256"]).lower(), namespace)
            except (KeyError, TypeError, ValueError) as exc:
                raise TargetSourceError("P12 panel identity row is malformed") from exc
    return bundle, binding


def _token_ids(tokenizer: Any, rendered: str) -> list[int]:
    try:
        encoded = tokenizer(rendered, add_special_tokens=False)
        values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    except Exception as exc:
        raise TargetSourceError("Alpaca tokenizer failed") from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TargetSourceError("Alpaca tokenizer returned malformed IDs")
    result = [int(value) for value in values]
    if any(value < 0 or value >= VOCAB_SIZE for value in result):
        raise TargetSourceError("Alpaca tokenizer returned an out-of-vocabulary ID")
    if not result or result[0] != BOS_TOKEN_ID:
        raise TargetSourceError("Alpaca rendered row does not start with the fixed BOS")
    return result


def _candidate(row: Mapping[str, Any], index: int, tokenizer: Any) -> tuple[dict[str, Any], list[int]]:
    if not isinstance(row, Mapping):
        raise TargetSourceError(f"Alpaca row {index} is malformed")
    rendered = historical_rendered_text(row, tokenizer)
    values = _token_ids(tokenizer, rendered)
    if len(values) < SEQUENCE_LENGTH:
        raise TargetSourceError(f"Alpaca row {index} is shorter than {SEQUENCE_LENGTH} tokens")
    fingerprints = p10.candidate_sequence_fingerprints(values)
    metadata: dict[str, Any] = {
        "record_id": public_record_id(index, dataset_revision=ALPACA_CACHE_REVISION),
        "public_record_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "dataset_key": "alpaca",
        "dataset_id": ALPACA_DATASET_ID,
        "split": ALPACA_SPLIT,
        "revision": ALPACA_CACHE_REVISION,
        "row_index": index,
        "source_index": index,
        "full_token_count": len(values),
        "post_bos_token_count": len(values) - 1,
        "valid_tokens": SEQUENCE_LENGTH,
        "final_sequence_sha256": fingerprints["h128_sequence_sha256"],
        "h40_sequence_sha256": p10._raw_int32_digest(values[:40]),
        "h128_sequence_sha256": fingerprints["h128_sequence_sha256"],
        "h129_sequence_sha256": fingerprints.get("h129_sequence_sha256"),
        "trr0002_active_token_ids_sha256": fingerprints["trr0002_active_token_ids_sha256"],
        "trr0002_h40_token_ids_sha256": fingerprints.get("trr0002_h40_token_ids_sha256"),
    }
    return metadata, values


def _pad(values: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    clipped = list(values[:SEQUENCE_LENGTH])
    if not clipped or clipped[0] != BOS_TOKEN_ID:
        raise TargetSourceError("target tensor row lost BOS")
    ids = clipped + [PAD_TOKEN_ID] * (SEQUENCE_LENGTH - len(clipped))
    mask = [True] * len(clipped) + [False] * (SEQUENCE_LENGTH - len(clipped))
    return torch.tensor(ids, dtype=torch.int64), torch.tensor(mask, dtype=torch.bool)


def _git_commit(root: Path) -> str | None:
    try:
        value = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _create_safetensor(path: Path, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TargetSourceError(f"refusing to overwrite tensor artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(path), metadata={"purpose": "TRR-P12 evaluator-only target source"})
    return {"path": path.name, "absolute_path": str(path), "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path), "readonly": True}


def _create_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TargetSourceError(f"refusing to overwrite source bundle manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path), "readonly": True}


def prepare_target_bundle(*, release_path: Path, union_path: Path, panel_path: Path, arrow_path: Path,
                          tokenizer_path: Path, output_manifest: Path, root: Path = ROOT,
                          opaque_reservation_paths: Sequence[Path] = ()) -> dict[str, Any]:
    release, release_binding = validate_release(release_path, root=root)
    union, union_binding = select_panel.load_union(union_path, root=root)
    panel, panel_binding = _panel_bundle(panel_path, root=root)
    bundles = [union, panel]
    opaque_bindings: list[dict[str, Any]] = []
    for opaque_path in opaque_reservation_paths:
        opaque, binding = select_panel.load_opaque_reservation(opaque_path, root=root)
        bundles.append(opaque)
        opaque_bindings.append(binding)
    exclusions = select_panel.merge_identity_bundles(*bundles)
    arrow_file, arrow_binding = file_binding(arrow_path, root=root, label="Alpaca Arrow cache")
    tokenizer_dir = tokenizer_path.expanduser().resolve()
    if tokenizer_dir.is_symlink() or not tokenizer_dir.is_dir():
        raise TargetSourceError(f"tokenizer snapshot is unavailable: {tokenizer_dir}")
    from scripts.trr0005_produce_confirmation import _load_tokenizer
    tokenizer = _load_tokenizer(tokenizer_dir)
    try:
        from datasets import Dataset
        dataset = Dataset.from_file(str(arrow_file))
    except Exception as exc:
        raise TargetSourceError("Alpaca Arrow cache could not be loaded offline") from exc
    if len(dataset) < CANDIDATE_RANGE[1]:
        raise TargetSourceError(f"Alpaca cache has {len(dataset)} rows; need {CANDIDATE_RANGE[1]}")

    selected: list[tuple[dict[str, Any], list[int]]] = []
    diagnostics = {"excluded_identity": 0, "duplicate_record": 0, "duplicate_rendered": 0,
                   "duplicate_h128": 0, "invalid_short": 0}
    seen_ids: set[str] = set()
    seen_rendered: set[str] = set()
    seen_h128: set[str] = set()
    order = deterministic_row_order(range(*CANDIDATE_RANGE), dataset_key="trr-p12-alpaca", seed=SELECTION_SEED)
    for index in order:
        try:
            metadata, values = _candidate(dataset[index], index, tokenizer)
        except TargetSourceError as exc:
            if "shorter than" in str(exc):
                diagnostics["invalid_short"] += 1
                continue
            raise
        if p11._candidate_exclusion_reasons(metadata, exclusions):
            diagnostics["excluded_identity"] += 1
            continue
        record_id = str(metadata["record_id"])
        rendered = str(metadata["public_record_sha256"])
        h128 = str(metadata["h128_sequence_sha256"])
        if record_id in seen_ids:
            diagnostics["duplicate_record"] += 1
            continue
        if rendered in seen_rendered:
            diagnostics["duplicate_rendered"] += 1
            continue
        if h128 in seen_h128:
            diagnostics["duplicate_h128"] += 1
            continue
        seen_ids.add(record_id)
        seen_rendered.add(rendered)
        seen_h128.add(h128)
        selected.append((metadata, values))
        if len(selected) == TRAIN_ROWS + VALIDATION_ROWS:
            break
    if len(selected) != TRAIN_ROWS + VALIDATION_ROWS:
        raise TargetSourceError(f"eligible Alpaca pool yielded {len(selected)}; need {TRAIN_ROWS + VALIDATION_ROWS}")

    train_pairs = selected[:TRAIN_ROWS]
    validation_pairs = selected[TRAIN_ROWS:]
    train_ids, train_masks = zip(*(_pad(values) for _, values in train_pairs))
    validation_ids, validation_masks = zip(*(_pad(values) for _, values in validation_pairs))
    train_file = _create_safetensor(
        output_manifest.parent / "train.safetensors",
        {"train_input_ids": torch.stack(train_ids), "train_attention_mask": torch.stack(train_masks)},
    )
    validation_file = _create_safetensor(
        output_manifest.parent / "validation.safetensors",
        {"validation_input_ids": torch.stack(validation_ids), "validation_attention_mask": torch.stack(validation_masks)},
    )
    train_rows = [metadata for metadata, _ in train_pairs]
    validation_rows = [metadata for metadata, _ in validation_pairs]
    payload: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_P12_TARGET_SOURCE_BUNDLE_EVALUATOR_ONLY",
        "created_utc": utc_now(),
        "release": release_binding,
        "release_sha256": release_binding["sha256"],
        "study_manifest": release.get("study_manifest") or {
            "path": str((root / "experiments/TRR-P12/manifest.json").resolve()),
            "bytes": int((root / "experiments/TRR-P12/manifest.json").stat().st_size),
            "sha256": STUDY_MANIFEST_SHA256,
            "readonly": True,
        },
        "union": union_binding,
        "panel": panel_binding,
        "opaque_reservations": opaque_bindings,
        "dataset": {
            "dataset_id": ALPACA_DATASET_ID,
            "split": ALPACA_SPLIT,
            "revision": ALPACA_CACHE_REVISION,
            "arrow": arrow_binding,
            "candidate_range_half_open": list(CANDIDATE_RANGE),
            "selection_seed": SELECTION_SEED,
            "order_sha256": json_digest([row["row_index"] for row in selected]),
        },
        "tokenizer": {"path": str(tokenizer_dir), "bos_token_id": BOS_TOKEN_ID, "pad_token_id": PAD_TOKEN_ID,
                      "tokenizer_json_sha256": sha256_file(tokenizer_dir / "tokenizer.json")},
        "train_records": TRAIN_ROWS,
        "validation_records": VALIDATION_ROWS,
        "sequence_length": SEQUENCE_LENGTH,
        "bos_token_id": BOS_TOKEN_ID,
        "selection": {
            "algorithm": "deterministic_row_order(dataset_key=trr-p12-alpaca, seed=6012); reject reviewed union, panel identities, opaque reservations, and within-bundle record/rendered/H128 duplicates; retain first 256 train then 64 validation eligible rows",
            "train": train_rows,
            "validation": validation_rows,
            "train_row_indices_sha256": json_digest([row["row_index"] for row in train_rows]),
            "validation_row_indices_sha256": json_digest([row["row_index"] for row in validation_rows]),
            "source_text_written": False,
            "reconstruction_source_text_written": False,
            "reconstruction_target_tokens_written": False,
            "target_training_token_ids_evaluator_only": True,
        },
        "train_tensor": {"path": train_file["path"], "bytes": train_file["bytes"], "sha256": train_file["sha256"]},
        "validation_tensor": {"path": validation_file["path"], "bytes": validation_file["bytes"], "sha256": validation_file["sha256"]},
        "selection_diagnostics": {**diagnostics, "eligible": len(selected), "candidate_pool": CANDIDATE_RANGE[1] - CANDIDATE_RANGE[0]},
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "network_used": False,
            "gpu_used": False,
            "source_text_read": True,
            "source_text_written": False,
            "target_training_token_ids_emitted": True,
            "target_training_token_ids_evaluator_only": True,
            "reconstruction_payload_emitted": False,
            "truth_opened": False,
            "selection_performed": True,
        },
        "truth_opened": False,
        "p03_holdout_accessed": False,
    }
    manifest_file = _create_json(output_manifest, payload)
    return {"manifest": manifest_file, "status": payload["status"], "train_records": TRAIN_ROWS,
            "validation_records": VALIDATION_ROWS, "selection_diagnostics": diagnostics}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("experiments/TRR-P12/manifest.json"))
    parser.add_argument("--union", type=Path, default=Path("experiments/TRR-P12/exclusions/identity_union_extension_r2.json"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--opaque-reservation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True, help="create-only source bundle manifest")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="validate release/union/panel bindings without reading Alpaca")
    parser.add_argument("--execute", action="store_true", help="perform CPU-only public source preparation")
    args = parser.parse_args(argv)
    if not args.execute and not args.dry_run:
        parser.error("pass --dry-run for binding checks or --execute to prepare the evaluator-only bundle")
    root = args.root.expanduser().resolve()
    release, _ = validate_release(args.release, root=root)
    union_value = release.get("union") or release.get("exclusions")
    union_path = Path(union_value["path"]) if isinstance(union_value, Mapping) and isinstance(union_value.get("path"), str) else args.union
    if args.dry_run:
        select_panel.load_union(union_path, root=root)
        _panel_bundle(args.panel, root=root)
        print(json.dumps({"status": "PASS_CPU_BINDING_DRY_RUN", "release": release["status"]}, sort_keys=True))
        return 0
    result = prepare_target_bundle(release_path=args.release, union_path=union_path, panel_path=args.panel,
                                   arrow_path=args.arrow, tokenizer_path=args.tokenizer,
                                   output_manifest=args.output, root=root,
                                   opaque_reservation_paths=[Path(value) for value in args.opaque_reservation])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
