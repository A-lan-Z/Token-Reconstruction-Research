"""Prepare the hash-only TRR-P12 reconstruction source panel.

This module is deliberately a small, CPU-only phase boundary.  It reuses the
validated P11 renderer and row-order helper, but requires a separately signed
P12 release receipt before it reads public Arrow rows.  The output contains
only approved row metadata and identity hashes; source text and token IDs are
transient and are never written.
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

from scripts.trr_p10 import build_exclusion_audit as p10
from scripts.trr_p11 import source_selector as p11


TASK_ID = "TRR-P12"
PANEL_SCHEMA = "token-reconstruction.trr-p12-source-panel.v1"
RELEASE_SCHEMA = "token-reconstruction.trr-p12-panel-release.v1"
RELEASE_STATUS = "P12_PANEL_SELECTION_RELEASED"
PANEL_STATUS = "FROZEN_P12_SOURCE_PANEL_NO_TRUTH"
UNION_SCHEMA = "token-reconstruction.trr-p12-identity-union.v1"
UNION_STATUS = "IDENTITY_UNION_COMPLETE_NO_PAYLOAD"
UNION_SHA256 = "bd2e641f5f10d249595b89f48aeeba2d97a2b71688190177f4ebe4f4ac022a0b"
UNION_BYTES = 6338870
STUDY_MANIFEST_SHA256 = "302cef73349927615e39b116e985f796a9c70b6f9c97932a119c0acb1d3f0a8d"
AGENT1_RESERVATION_SHA256 = "dee8b7e1d70342c154a4217c982d47d3917d4e11c9cad33b6ebdc87c79a6cf33"
AGENT1_RESERVATION_PATH = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0013/outputs/TRR-0013/correction_selection_r1/opaque_reservations.json"
)
AGENT1_EVAL_RESERVATION_PATH = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0013/outputs/TRR-0013/evaluation_selection_r1/opaque_reservations.json"
)
AGENT1_EVAL_RESERVATION_SHA256 = "a6a47eb11731fca900b5a5f91e5512b1193ffcbe812a6e42ec678a10cec8c6a1"
SOURCE_INPUTS_DEFAULT = Path("experiments/TRR-P11/selector/public_source_inputs_r1.json")
DOMAIN_ORDER = ("pile", "finance")
SOURCE_RANGES = {"pile": (8000, 9000), "finance": (50000, 52000)}
SELECTION_SEED = 5012
RECORDS_PER_DOMAIN = 128
DESIGN_RECORDS = 64
VALIDATION_RECORDS = 64
STORED_SEQUENCE_TOKENS = 128
PAIRED_STAGES = (0, 64, 128, 256)
ALLOWED_UNION_FIELDS = frozenset(
    {
        "record_id",
        "source_index",
        "rendered_sha256",
        "tokenized_record_sha256",
        "h40_sequence_sha256",
        "h128_sequence_sha256",
        "h129_sequence_sha256",
        "trr0002_active_token_ids_sha256",
        "trr0002_h40_token_ids_sha256",
    }
)
HASH_FIELDS = ALLOWED_UNION_FIELDS - {"record_id", "source_index"}
HEX = frozenset("0123456789abcdef")


class PanelError(RuntimeError):
    """Raised when panel preparation cannot satisfy a bound contract."""


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


def file_binding(value: str | Path, *, root: Path, label: str, expected_sha256: str | None = None,
                 expected_bytes: int | None = None) -> tuple[Path, dict[str, Any]]:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if path.is_symlink() or not path.is_file():
        raise PanelError(f"{label} is unavailable or symlinked: {path}")
    digest = sha256_file(path)
    size = int(path.stat().st_size)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise PanelError(f"{label} SHA-256 changed: {digest} != {expected_sha256}")
    if expected_bytes is not None and size != int(expected_bytes):
        raise PanelError(f"{label} byte count changed: {size} != {expected_bytes}")
    return path, {"path": str(path), "bytes": size, "sha256": digest, "readonly": True}


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanelError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise PanelError(f"{label} must be a JSON object")
    return dict(value)


def _namespace(text: Any) -> p10.Namespace:
    if not isinstance(text, str) or len(text.split("|")) != 4:
        raise PanelError(f"malformed identity namespace: {text!r}")
    return p10.Namespace(*text.split("|"))


def _validate_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in HEX for ch in value.lower()):
        raise PanelError(f"malformed {field} identity")
    return value.lower()


def _load_identity_fields(payload: Mapping[str, Any], *, label: str, path: Path, schema: str,
                          status: str | None = None) -> p10.IdentityBundle:
    if payload.get("schema") != schema:
        raise PanelError(f"{label} schema changed")
    if status is not None and payload.get("status") != status:
        raise PanelError(f"{label} status changed")
    boundary = payload.get("access_boundary")
    if isinstance(boundary, Mapping):
        forbidden = (
            "source_text_read", "source_text_serialized", "source_tokens_serialized",
            "token_values_emitted", "truth_or_scores_read", "new_selection_started",
            "p03_holdout_accessed", "payload_emitted",
        )
        if any(boundary.get(key) is True for key in forbidden):
            raise PanelError(f"{label} records a forbidden boundary")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise PanelError(f"{label} fields are absent")
    bundle = p10.IdentityBundle(
        label, "opaque_exclusion", path, sha256_file(path), int(path.stat().st_size),
        schema=str(payload.get("schema")), status=str(payload.get("status")) if payload.get("status") else None,
    )
    for raw_field, by_namespace in fields.items():
        field = str(raw_field)
        if field not in ALLOWED_UNION_FIELDS or not isinstance(by_namespace, Mapping):
            raise PanelError(f"{label} contains an unapproved identity field: {field}")
        for raw_ns, values in by_namespace.items():
            namespace = _namespace(raw_ns)
            if not isinstance(values, list):
                raise PanelError(f"{label} {field}/{raw_ns} values are malformed")
            for value in values:
                if field == "source_index":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise PanelError(f"{label} source index is malformed")
                    bundle.add(field, value, namespace)
                elif field == "record_id":
                    if not isinstance(value, str) or not value:
                        raise PanelError(f"{label} record ID is malformed")
                    bundle.add(field, value, namespace)
                else:
                    bundle.add(field, _validate_hash(value, field=field), namespace)
    declared = payload.get("identity_counts", payload.get("counts"))
    if isinstance(declared, Mapping) and dict(declared) != bundle.counts():
        raise PanelError(f"{label} identity counts changed")
    return bundle


def load_union(path: Path, *, root: Path = ROOT) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    resolved, binding = file_binding(path, root=root, label="P12 identity union", expected_sha256=UNION_SHA256,
                                     expected_bytes=UNION_BYTES)
    payload = load_json(resolved, label="P12 identity union")
    if payload.get("task_id") != TASK_ID:
        raise PanelError("P12 identity union task identity changed")
    bundle = _load_identity_fields(payload, label="P12 identity union", path=resolved,
                                   schema=UNION_SCHEMA, status=UNION_STATUS)
    if dict(payload.get("identity_counts", {})) != bundle.counts():
        raise PanelError("P12 identity union counts changed")
    return bundle, binding


def load_opaque_reservation(path: Path, *, root: Path = ROOT,
                            expected_sha256: str | None = None) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    resolved, binding = file_binding(path, root=root, label="opaque reservation", expected_sha256=expected_sha256)
    payload = load_json(resolved, label="opaque reservation")
    if payload.get("task_id") != "TRR-0013" or payload.get("source_text_or_tokens") is not False:
        raise PanelError("opaque reservation does not certify metadata-only content")
    fields = payload.get("fields")
    counts = payload.get("counts")
    if not isinstance(fields, Mapping) or not isinstance(counts, Mapping):
        raise PanelError("opaque reservation fields/counts are absent")
    expected_fields = {"source_index", "record_id", "rendered_sha256", "h128_sequence_sha256",
                       "h129_sequence_sha256", "h40_sequence_sha256"}
    if set(fields) != expected_fields or set(counts) != {"pile", "finance"}:
        raise PanelError("opaque reservation field/domain inventory changed")
    bundle = p10.IdentityBundle("opaque reservation", "opaque_exclusion", resolved,
                                binding["sha256"], binding["bytes"], status="metadata_only")
    for raw_field, by_namespace in fields.items():
        if not isinstance(by_namespace, Mapping):
            raise PanelError("opaque reservation field namespace map is malformed")
        for raw_ns, values in by_namespace.items():
            namespace = _namespace(raw_ns)
            if not isinstance(values, list):
                raise PanelError("opaque reservation values are malformed")
            for value in values:
                if raw_field == "source_index":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise PanelError("opaque reservation source index is malformed")
                    bundle.add(raw_field, value, namespace)
                elif raw_field == "record_id":
                    if not isinstance(value, str) or not value:
                        raise PanelError("opaque reservation record ID is malformed")
                    bundle.add(raw_field, value, namespace)
                else:
                    bundle.add(raw_field, _validate_hash(value, field=raw_field), namespace)
    for domain in ("pile", "finance"):
        if not isinstance(counts[domain], Mapping):
            raise PanelError("opaque reservation domain count is malformed")
        declared = counts[domain]
        observed = {
            field: sum(len(values) for namespace, values in bundle.values.get(field, {}).items()
                       if namespace.style == domain)
            for field in expected_fields
        }
        for field, value in declared.items():
            if field in observed and int(value) != observed[field]:
                raise PanelError(f"opaque reservation {domain}/{field} count changed")
    return bundle, binding

def merge_identity_bundles(*bundles: p10.IdentityBundle) -> p10.IdentityBundle:
    merged = p10.IdentityBundle("P12 all known opaque exclusions", "union", Path("."), "", 0,
                                status="metadata_only_union")
    for bundle in bundles:
        for field, by_namespace in bundle.values.items():
            for namespace, values in by_namespace.items():
                merged.values[field][namespace].update(values)
    return merged


def _binding_from_value(value: Any, *, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    if isinstance(value, Mapping):
        if not isinstance(value.get("path"), str):
            raise PanelError(f"{label} path is absent")
        return file_binding(value["path"], root=root, label=label,
                            expected_sha256=value.get("sha256"), expected_bytes=value.get("bytes"))
    if isinstance(value, (str, Path)):
        return file_binding(value, root=root, label=label)
    raise PanelError(f"{label} binding is malformed")


def validate_release(release_path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    release_file, release_binding = _binding_from_value(release_path, root=root, label="P12 panel release")
    payload = load_json(release_file, label="P12 panel release")
    is_study_manifest = (
        payload.get("schema") == "token-reconstruction.trr-p12-plan.v1"
        and payload.get("task_id") == TASK_ID
        and release_binding["sha256"] == STUDY_MANIFEST_SHA256
    )
    if not is_study_manifest and (payload.get("schema") != RELEASE_SCHEMA or payload.get("task_id") != TASK_ID):
        raise PanelError("P12 panel release schema/task identity changed")
    if not is_study_manifest and (payload.get("status") != RELEASE_STATUS or payload.get("selection_release") is not True):
        raise PanelError("P12 panel release is not active")
    expected = {
        "selection_seed": SELECTION_SEED,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "stored_sequence_tokens_including_bos": STORED_SEQUENCE_TOKENS,
        "design_records": DESIGN_RECORDS,
        "validation_records": VALIDATION_RECORDS,
    }
    evaluation = payload.get("evaluation") if is_study_manifest else payload
    if not isinstance(evaluation, Mapping):
        raise PanelError("P12 panel evaluation binding is absent")
    actual = {
        "selection_seed": evaluation.get("source_seed", evaluation.get("selection_seed")),
        "records_per_domain": evaluation.get("records_per_domain"),
        "stored_sequence_tokens_including_bos": evaluation.get("stored_sequence_tokens_including_bos", STORED_SEQUENCE_TOKENS),
        "design_records": evaluation.get("design_records", DESIGN_RECORDS),
        "validation_records": evaluation.get("validation_records", VALIDATION_RECORDS),
    }
    if any(actual[key] != value for key, value in expected.items()):
        raise PanelError("P12 panel release constants changed")
    if tuple(evaluation.get("paired_stages", payload.get("paired_stages", PAIRED_STAGES))) != PAIRED_STAGES:
        raise PanelError("P12 panel paired stages changed")
    ranges = evaluation.get("source_ranges_half_open")
    if not isinstance(ranges, Mapping) or {domain: tuple(ranges.get(domain, ())) for domain in DOMAIN_ORDER} != SOURCE_RANGES:
        raise PanelError("P12 panel source ranges changed")
    union_value = payload.get("union") or payload.get("exclusions")
    if not isinstance(union_value, Mapping) or union_value.get("sha256") != UNION_SHA256 or int(union_value.get("bytes", -1)) != UNION_BYTES:
        raise PanelError("P12 panel release is not bound to the reviewed P12 union")
    study = ({"path": str(release_file), "bytes": release_binding["bytes"], "sha256": release_binding["sha256"], "readonly": True}
             if is_study_manifest else payload.get("study_manifest"))
    if not isinstance(study, Mapping) or study.get("sha256") != STUDY_MANIFEST_SHA256:
        raise PanelError("P12 panel release is not bound to the immutable study manifest")
    if payload.get("truth_opened", False) is not False or payload.get("p03_holdout_accessed", False) is not False:
        raise PanelError("P12 panel release boundary is open")
    return payload, release_binding

def split_panel_rows(rows: Sequence[Mapping[str, Any]], *, records_per_domain: int = RECORDS_PER_DOMAIN) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != records_per_domain:
        raise PanelError(f"panel row count {len(rows)} != {records_per_domain}")
    copied = [dict(row) for row in rows]
    return copied[:DESIGN_RECORDS], copied[DESIGN_RECORDS:]


def _git_commit(root: Path) -> str | None:
    try:
        value = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _create_only_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PanelError(f"refusing to overwrite create-only panel: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path), "readonly": True}


def select_panel(*, release_path: Path, union_path: Path, source_inputs_path: Path,
                 output_path: Path, root: Path = ROOT, opaque_reservation_path: Path | None = None,
                 opaque_reservation_sha256: str | None = AGENT1_RESERVATION_SHA256,
                 opaque_reservation_paths: Sequence[Path] = ()) -> dict[str, Any]:
    release, release_binding = validate_release(release_path, root=root)
    union, union_binding = load_union(union_path, root=root)
    combined = union
    reservation_paths = list(opaque_reservation_paths)
    if opaque_reservation_path is not None:
        reservation_paths.insert(0, opaque_reservation_path)
    reservation_bindings: list[dict[str, Any]] = []
    for reservation_path in reservation_paths:
        reservation, reservation_binding = load_opaque_reservation(
            reservation_path, root=root,
            expected_sha256=opaque_reservation_sha256 if reservation_path == opaque_reservation_path else None,
        )
        combined = merge_identity_bundles(combined, reservation)
        reservation_bindings.append(reservation_binding)
    inputs_path, inputs_binding = file_binding(source_inputs_path, root=root, label="P11 source inputs")
    inputs = p11._normalize_source_inputs(inputs_path, root=root, require_tokenizer_dir=True)
    try:
        from scripts import trr0005_produce_confirmation as trusted
        from token_reconstruction.trr0005_public_corpus import deterministic_row_order
        tokenizer = trusted._load_tokenizer(Path(inputs["tokenizer"]["path"]))
        datasets = {
            domain: trusted._load_arrow_dataset(tuple(Path(item["path"]) for item in inputs[domain]["arrow_files"]))
            for domain in DOMAIN_ORDER
        }
    except Exception as exc:
        raise PanelError("trusted public tokenizer or Arrow sources could not be loaded") from exc

    selected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    diagnostics: dict[str, dict[str, int]] = {}
    seen_ids: set[str] = set()
    seen_rendered: set[str] = set()
    seen_h128: set[str] = set()
    for domain in DOMAIN_ORDER:
        start, stop = SOURCE_RANGES[domain]
        if len(datasets[domain]) < stop:
            raise PanelError(f"{domain} source has {len(datasets[domain])} rows; need {stop}")
        counts = {"excluded_identity": 0, "duplicate_record": 0, "duplicate_rendered": 0,
                  "duplicate_h128": 0, "invalid_short_or_render": 0}
        order = deterministic_row_order(range(start, stop), dataset_key=f"trr-p12-{domain}", seed=SELECTION_SEED)
        for index in order:
            try:
                candidate = trusted._render_row(domain, datasets[domain][index], index, tokenizer)
                metadata = p11._candidate_identity(candidate)
            except Exception as exc:
                message = str(exc).lower()
                if any(text in message for text in ("shorter than", "no user/assistant", "malformed")):
                    counts["invalid_short_or_render"] += 1
                    continue
                raise PanelError(f"trusted renderer failed at {domain}/{index}") from exc
            if p11._candidate_exclusion_reasons(metadata, combined):
                counts["excluded_identity"] += 1
                continue
            record_id = str(metadata["record_id"])
            rendered = str(metadata["public_record_sha256"])
            h128 = str(metadata["h128_sequence_sha256"])
            if record_id in seen_ids:
                counts["duplicate_record"] += 1
                continue
            if rendered in seen_rendered:
                counts["duplicate_rendered"] += 1
                continue
            if h128 in seen_h128:
                counts["duplicate_h128"] += 1
                continue
            seen_ids.add(record_id)
            seen_rendered.add(rendered)
            seen_h128.add(h128)
            selected[domain].append(p11._selection_row(candidate))
            if len(selected[domain]) == RECORDS_PER_DOMAIN:
                break
        if len(selected[domain]) != RECORDS_PER_DOMAIN:
            raise PanelError(f"{domain} eligible pool yielded {len(selected[domain])}; need {RECORDS_PER_DOMAIN}")
        diagnostics[domain] = {**counts, "selected": len(selected[domain]), "pool_size": stop - start}

    ids = {domain: [row["record_id"] for row in selected[domain]] for domain in DOMAIN_ORDER}
    h128 = {domain: [row["h128_sequence_sha256"] for row in selected[domain]] for domain in DOMAIN_ORDER}
    payload: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": PANEL_STATUS,
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
        "opaque_reservations": reservation_bindings,
        "source_inputs": inputs_binding,
        "selection_seed": SELECTION_SEED,
        "source_ranges_half_open": {domain: list(SOURCE_RANGES[domain]) for domain in DOMAIN_ORDER},
        "records_per_domain": RECORDS_PER_DOMAIN,
        "stored_sequence_tokens_including_bos": STORED_SEQUENCE_TOKENS,
        "paired_stages": list(PAIRED_STAGES),
        "same_records_all_snapshots": True,
        "selection_rule": {
            "algorithm": "deterministic_row_order(dataset_key=trr-p12-{domain}, seed=5012); trusted P11 canonical rerender/tokenization; reject reviewed union, opaque reservation identities, and within-panel record/rendered/H128 duplicates; retain first 128 eligible rows",
            "records": selected,
            "design_records": {domain: selected[domain][:DESIGN_RECORDS] for domain in DOMAIN_ORDER},
            "validation_records": {domain: selected[domain][DESIGN_RECORDS:] for domain in DOMAIN_ORDER},
            "record_ids_sha256": {domain: json_digest(ids[domain]) for domain in DOMAIN_ORDER},
            "h128_sequence_sha256": {domain: json_digest(h128[domain]) for domain in DOMAIN_ORDER},
            "source_text_or_token_ids_written": False,
        },
        "selection_diagnostics": diagnostics,
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "network_used": False,
            "model_loaded": False,
            "source_text_read": True,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_opened": False,
            "selection_performed": True,
        },
        "truth_opened": False,
        "target_labels_loaded": False,
        "p03_holdout_accessed": False,
    }
    return {"panel": _create_only_json(output_path, payload), "status": PANEL_STATUS,
            "records_per_domain": {domain: len(selected[domain]) for domain in DOMAIN_ORDER}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("experiments/TRR-P12/manifest.json"))
    parser.add_argument("--union", type=Path, default=Path("experiments/TRR-P12/exclusions/identity_union_extension_r2.json"))
    parser.add_argument("--source-inputs", type=Path, default=SOURCE_INPUTS_DEFAULT)
    parser.add_argument("--opaque-reservation", type=Path, action="append",
                        default=[AGENT1_RESERVATION_PATH, AGENT1_EVAL_RESERVATION_PATH])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="validate release/bindings without reading source rows")
    parser.add_argument("--execute", action="store_true", help="perform CPU-only public source selection")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    release, _ = validate_release(args.release, root=root)
    union_binding = release.get("union") or release.get("exclusions")
    union_path = args.union
    if isinstance(union_binding, Mapping) and isinstance(union_binding.get("path"), str):
        union_path = Path(union_binding["path"])
    if args.dry_run:
        _, binding = load_union(union_path, root=root)
        print(json.dumps({"status": "PASS_CPU_BINDING_DRY_RUN", "release": release.get("status"), "union": binding}, sort_keys=True))
        return 0
    if not args.execute:
        parser.error("pass --dry-run for binding checks or --execute to prepare the panel")
    result = select_panel(release_path=args.release, union_path=union_path, source_inputs_path=args.source_inputs,
                          output_path=args.output, root=root, opaque_reservation_paths=args.opaque_reservation)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
