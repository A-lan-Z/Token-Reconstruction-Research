#!/usr/bin/env python3
"""Build the TRR-P10 metadata-only identity exclusion audit.

The audit is intentionally a bounded, explicit union of previously opened
identity ledgers.  It never scans a new public source, selects a record, loads
an activation/model, or opens evaluation truth.  The two TRR-0002 public
record ledgers are an authorized historical-public curator input: token IDs
are used only transiently to reproduce their declared digest conventions and
are never retained in a bundle or output artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


TASK_ID = "TRR-P10"
SCHEMA = "token-reconstruction.trr-p10-identity-exclusion-audit.v1"
SOURCE_SCHEMA = "token-reconstruction.trr-p10-identity-source.v1"

# These are the only payload keys that the generic metadata walker treats as
# unsafe.  Panel sources use the narrow ``panel_identity`` mode, which skips
# these keys without retaining their values.
PAYLOAD_KEYS = {
    "attention_mask",
    "input_ids",
    "labels",
    "plaintext",
    "position_ids",
    "source_text",
    "target_tokens",
    "token_ids",
}

STYLE_KEYS = {"dataset", "dataset_key", "domain", "input_style", "source_style", "style"}
DATASET_KEYS = {"dataset_id", "source_dataset_id"}
SPLIT_KEYS = {"dataset_split", "split", "source_split"}
REVISION_KEYS = {"dataset_revision", "revision", "source_revision"}
STYLE_NAMES = ("alpaca", "finance", "pile")

# Identity namespaces.  ``record_id`` and verified sequence/content digests
# are global; source indices remain scoped to a compatible dataset namespace.
GLOBAL_FIELDS = {
    "record_id",
    "rendered_sha256",
    "tokenized_record_sha256",
    "h128_sequence_sha256",
    "h129_sequence_sha256",
    "trr0002_active_token_ids_sha256",
    "trr0002_h40_token_ids_sha256",
}

DEFAULT_ALIASES = {
    "record_id": "record_id",
    "source_record_id": "record_id",
    "public_record_id": "record_id",
    "public_record_sha256": "rendered_sha256",
    "rendered_sha256": "rendered_sha256",
    "content_sha256": "rendered_sha256",
    "text_sha256": "rendered_sha256",
    "tokenized_record_sha256": "tokenized_record_sha256",
    "dataset_index": "source_index",
    "index": "source_index",
    "raw_index": "source_index",
    "row_index": "source_index",
    "source_index": "source_index",
    "source_row_index": "source_index",
}

_SHA256_HEX = set("0123456789abcdefABCDEF")
_UNKNOWN_HASH_MARKERS = ("sha256", "hash", "digest", "fingerprint")
_UNKNOWN_IDENTITY_HINTS = (
    "active",
    "content",
    "final",
    "index",
    "public",
    "record",
    "rendered",
    "sequence",
    "source",
    "text",
    "token",
    "truncated",
)

class AuditError(RuntimeError):
    """Raised when the identity-only audit cannot fail closed."""


@dataclass(frozen=True)
class Namespace:
    style: str = "*"
    dataset_id: str = "*"
    split: str = "*"
    revision: str = "*"

    def as_string(self) -> str:
        return "|".join((self.style, self.dataset_id, self.split, self.revision))

    def compatible(self, other: "Namespace") -> bool:
        return all(
            left == right or left == "*" or right == "*"
            for left, right in zip(
                (self.style, self.dataset_id, self.split, self.revision),
                (other.style, other.dataset_id, other.split, other.revision),
            )
        )


@dataclass
class IdentityBundle:
    label: str
    role: str
    path: Path
    sha256: str
    bytes: int
    schema: str | None = None
    status: str | None = None
    values: dict[str, dict[Namespace, set[str | int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    unsafe_keys: list[str] = field(default_factory=list)
    unknown_identity_keys: Counter[str] = field(default_factory=Counter)
    metadata_notes: list[str] = field(default_factory=list)

    def add(self, field_name: str, value: str | int, namespace: Namespace) -> None:
        self.values[field_name][namespace].add(value)

    def counts(self) -> dict[str, int]:
        return {
            field: sum(len(values) for values in namespaces.values())
            for field, namespaces in sorted(self.values.items())
        }

    def namespace_counts(self) -> dict[str, dict[str, int]]:
        return {
            field: {
                namespace.as_string(): len(values)
                for namespace, values in sorted(by_namespace.items(), key=lambda item: item[0].as_string())
            }
            for field, by_namespace in sorted(self.values.items())
        }


@dataclass(frozen=True)
class SourceSpec:
    label: str
    role: str
    path: str
    required: bool = True
    # Raw producer field -> verified canonical identity namespace.
    hash_aliases: tuple[tuple[str, str], ...] = ()
    metadata_mode: str | None = None
    expected_sha256: str | None = None

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self.hash_aliases)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _SHA256_HEX for char in value)


def _style(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.casefold()
    for name in STYLE_NAMES:
        if lowered == name or name in lowered:
            return name
    return None


def _namespace_from_mapping(context: Namespace, mapping: Mapping[str, Any], hint: str = "") -> Namespace:
    style, dataset_id, split, revision = context.style, context.dataset_id, context.split, context.revision
    for raw_key, value in mapping.items():
        key = str(raw_key).casefold().replace("-", "_")
        if key in STYLE_KEYS:
            style = _style(value) or style
        elif key in DATASET_KEYS and isinstance(value, str) and value:
            dataset_id = value
        elif key in SPLIT_KEYS and isinstance(value, str) and value:
            split = value
        elif key in REVISION_KEYS and isinstance(value, str) and value:
            revision = value
    return Namespace(style=style or _style(hint) or "*", dataset_id=dataset_id, split=split, revision=revision)


def _namespace_for_child(context: Namespace, key: str, value: Any, hint: str) -> Namespace:
    return Namespace(
        style=_style(key) or _style(value) or context.style or _style(hint) or "*",
        dataset_id=context.dataset_id,
        split=context.split,
        revision=context.revision,
    )


def _canonical_key(key: str, aliases: Mapping[str, str]) -> str | None:
    lowered = key.casefold().replace("-", "_")
    if lowered in aliases:
        return aliases[lowered]
    return DEFAULT_ALIASES.get(lowered)


def _looks_like_unknown_identity_key(lowered: str) -> bool:
    return (
        any(marker in lowered for marker in _UNKNOWN_HASH_MARKERS)
        and any(hint in lowered for hint in _UNKNOWN_IDENTITY_HINTS)
    )


def _emit(bundle: IdentityBundle, canonical: str, value: Any, namespace: Namespace) -> None:
    if canonical == "record_id":
        if isinstance(value, str) and value:
            bundle.add(canonical, value, namespace)
        return
    if canonical == "source_index":
        # Namespace-scoped indices are not comparable without a known style.
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0 and namespace.style != "*":
            bundle.add(canonical, value, namespace)
        return
    if canonical in GLOBAL_FIELDS and _is_sha256(value):
        bundle.add(canonical, value.casefold(), namespace)


def _walk(
    value: Any,
    *,
    bundle: IdentityBundle,
    namespace: Namespace,
    hint: str,
    location: str,
    aliases: Mapping[str, str],
    panel_mode: bool = False,
    active_field: str | None = None,
) -> None:
    """Walk metadata, retaining only declared identity fields.

    Generic sources fail closed on payload-bearing keys.  A trusted historical
    panel may use ``panel_mode``: the walker skips payload values and proceeds
    to the record metadata in surrounding objects.  It never stores skipped
    values or includes them in a receipt.
    """
    if isinstance(value, Mapping):
        local = _namespace_from_mapping(namespace, value, hint)
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.casefold().replace("-", "_")
            if lowered in PAYLOAD_KEYS and isinstance(child, (list, Mapping, tuple, str)):
                if not panel_mode:
                    bundle.unsafe_keys.append(f"{location}.{key}")
                continue
            canonical = _canonical_key(key, aliases)
            if canonical is not None:
                _emit(bundle, canonical, child, local)
                child_active = canonical if canonical in GLOBAL_FIELDS else None
            else:
                if _looks_like_unknown_identity_key(lowered):
                    bundle.unknown_identity_keys[lowered] += 1
                child_active = active_field
            _walk(
                child,
                bundle=bundle,
                namespace=_namespace_for_child(local, key, child, hint),
                hint=f"{hint} {key}",
                location=f"{location}.{key}",
                aliases=aliases,
                panel_mode=panel_mode,
                active_field=child_active,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if active_field is not None and not isinstance(child, (Mapping, list)):
                _emit(bundle, active_field, child, namespace)
            else:
                _walk(
                    child,
                    bundle=bundle,
                    namespace=namespace,
                    hint=hint,
                    location=f"{location}[{index}]",
                    aliases=aliases,
                    panel_mode=panel_mode,
                    active_field=active_field,
                )


def resolve_path(path: str, root: Path, pr20_root: Path | None = None) -> Path:
    if path.startswith("@pr20/"):
        if pr20_root is None:
            raise AuditError("a @pr20 source requires --pr20-root")
        return (pr20_root / path.removeprefix("@pr20/")).resolve()
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _new_bundle(spec: SourceSpec, path: Path, value: Mapping[str, Any]) -> IdentityBundle:
    return IdentityBundle(
        label=spec.label,
        role=spec.role,
        path=path,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        schema=value.get("schema") if isinstance(value.get("schema"), str) else None,
        status=value.get("status") if isinstance(value.get("status"), str) else None,
    )


def _trr0002_active_int32_digest(token_ids: Sequence[int]) -> str:
    values = [int(token) for token in token_ids]
    if any(value < -(2**31) or value >= 2**31 for value in values):
        raise AuditError("TRR2 active token ID is outside signed int32")
    return hashlib.sha256(struct.pack("<" + "i" * len(values), *values)).hexdigest()


def _raw_int32_digest(token_ids: Sequence[int]) -> str:
    values = [int(token) for token in token_ids]
    if any(value < -(2**31) or value >= 2**31 for value in values):
        raise AuditError("sequence token ID is outside signed int32")
    return hashlib.sha256(struct.pack("<" + "i" * len(values), *values)).hexdigest()


def _trr0002_h40_digest(token_ids: Sequence[int]) -> str:
    values = [int(token) for token in token_ids]
    header = json.dumps(
        {"dtype": "torch.int64", "shape": [len(values)]}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = struct.pack("<" + "q" * len(values), *values)
    return hashlib.sha256(header + payload).hexdigest()


def candidate_sequence_fingerprints(token_ids: Sequence[int]) -> dict[str, str]:
    """Return full-active, H40, H128 and H129 digests when widths permit.

    The prefix hashes are always computed from the first N IDs including BOS;
    they remain available for a candidate row longer than N.  H128/H129 are
    separate namespaces and use the producer raw signed-int32 convention.
    """
    if not token_ids or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids):
        raise AuditError("candidate token IDs must be a non-empty integer sequence")
    values = [int(token) for token in token_ids]
    result = {"trr0002_active_token_ids_sha256": _raw_int32_digest(values)}
    if len(values) >= 40:
        result["trr0002_h40_token_ids_sha256"] = _trr0002_h40_digest(values[:40])
    if len(values) >= 128:
        result["h128_sequence_sha256"] = _raw_int32_digest(values[:128])
    if len(values) >= 129:
        result["h129_sequence_sha256"] = _raw_int32_digest(values[:129])
    return result


def _load_trr0002(spec: SourceSpec, path: Path, value: Mapping[str, Any]) -> IdentityBundle:
    bundle = _new_bundle(spec, path, value)
    if spec.metadata_mode == "trr0002_finance":
        rows = value.get("records")
        dataset_id = value.get("dataset")
        if not isinstance(rows, list) or not isinstance(dataset_id, str) or not dataset_id:
            raise AuditError("TRR2 Finance identity ledger lacks records/dataset metadata")
        namespace = Namespace("finance", dataset_id, "train", "*")
        for row_number, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise AuditError(f"TRR2 Finance row {row_number} is malformed")
            ids = row.get("input_ids")
            valid = row.get("valid_tokens")
            supplied = row.get("token_ids_sha256")
            if (
                not isinstance(row.get("record_id"), str)
                or not isinstance(row.get("raw_index"), int)
                or not isinstance(ids, list)
                or not isinstance(valid, int)
                or valid <= 0
                or valid > len(ids)
                or not isinstance(supplied, str)
            ):
                raise AuditError(f"TRR2 Finance row {row_number} violates the identity contract")
            digest = _trr0002_active_int32_digest(ids[:valid])
            if supplied.casefold() != digest:
                raise AuditError(f"TRR2 Finance row {row_number} producer digest mismatch")
            bundle.add("record_id", row["record_id"], namespace)
            bundle.add("source_index", row["raw_index"], namespace)
            if _is_sha256(row.get("content_sha256")):
                bundle.add("rendered_sha256", row["content_sha256"].casefold(), namespace)
            bundle.add("trr0002_active_token_ids_sha256", digest, namespace)
        bundle.metadata_notes.append("TRR2 Finance token arrays hashed transiently as producer-declared active signed-int32 IDs")
        return bundle
    if spec.metadata_mode == "trr0002_pile":
        namespace = Namespace("pile", "NeelNanda/pile-10k", "train", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
        row_count = 0
        for group in ("development", "update_train"):
            rows = value.get(group)
            if not isinstance(rows, list):
                raise AuditError(f"TRR2 Pile missing {group} rows")
            for row_number, row in enumerate(rows):
                tokens = row.get("token_ids") if isinstance(row, Mapping) else None
                if (
                    not isinstance(row, Mapping)
                    or not isinstance(row.get("record_id"), str)
                    or not isinstance(row.get("dataset_index"), int)
                    or not _is_sha256(row.get("text_sha256"))
                    or not isinstance(tokens, list)
                    or len(tokens) != 40
                    or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens)
                ):
                    raise AuditError(f"TRR2 Pile {group} row {row_number} violates the H40 identity contract")
                bundle.add("record_id", row["record_id"], namespace)
                bundle.add("source_index", row["dataset_index"], namespace)
                bundle.add("rendered_sha256", row["text_sha256"].casefold(), namespace)
                bundle.add("trr0002_h40_token_ids_sha256", _trr0002_h40_digest(tokens), namespace)
                row_count += 1
        bundle.metadata_notes.append(f"TRR2 Pile {row_count} rows hashed transiently under the H40 int64 tensor convention")
        return bundle
    raise AuditError(f"unknown TRR2 metadata mode: {spec.metadata_mode}")


def load_bundle(spec: SourceSpec, *, root: Path, pr20_root: Path | None = None) -> IdentityBundle | None:
    path = resolve_path(spec.path, root, pr20_root)
    if not path.is_file() or path.is_symlink():
        if spec.required:
            raise AuditError(f"required identity source is unavailable: {path}")
        return None
    if spec.expected_sha256 and sha256_file(path) != spec.expected_sha256:
        raise AuditError(f"identity source hash changed: {spec.label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"identity source is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise AuditError(f"identity source root is not an object: {path}")
    if spec.metadata_mode in {"trr0002_finance", "trr0002_pile"}:
        return _load_trr0002(spec, path, value)
    bundle = _new_bundle(spec, path, value)
    _walk(
        value,
        bundle=bundle,
        namespace=Namespace(),
        hint=f"{spec.label} {path.name}",
        location="$",
        aliases=spec.aliases,
        panel_mode=spec.metadata_mode == "panel_identity",
    )
    if bundle.unsafe_keys:
        raise AuditError(f"payload-bearing identity source rejected: {spec.label}: {bundle.unsafe_keys[:4]}")
    if spec.metadata_mode == "panel_identity":
        bundle.metadata_notes.append("Panel payload keys skipped; only record/hash/index metadata retained")
    return bundle


def _spec(label: str, role: str, path: str, *, aliases: Mapping[str, str] | None = None, mode: str | None = None, sha: str | None = None) -> SourceSpec:
    return SourceSpec(label, role, path, hash_aliases=tuple(sorted((aliases or {}).items())), metadata_mode=mode, expected_sha256=sha)


def source_specs() -> tuple[SourceSpec, ...]:
    """Return the explicit 45-ledger inventory plus three opened panels.

    P03 is intentionally absent.  No wildcard path or repository-wide scan is
    permitted to substitute for a missing source specification.
    """
    h128 = {"final_sequence_sha256": "h128_sequence_sha256", "h128_sequence_sha256": "h128_sequence_sha256", "sequence_h128_sha256": "h128_sequence_sha256"}
    h129 = {"truncated_sequence_sha256": "h129_sequence_sha256", "sequence_h129_sha256": "h129_sequence_sha256"}
    opaque = {"source_hashes": "rendered_sha256", "public_record_sha256": "rendered_sha256", "sequence_hashes_h128": "h128_sequence_sha256", "final_sequence_sha256": "h128_sequence_sha256", "h128_sequence_sha256": "h128_sequence_sha256", "sequence_h128_sha256": "h128_sequence_sha256"}
    h128_h129 = {**h128, **h129}
    specs = [
        _spec("trr0001_manifest", "opened_development", "experiments/TRR-0001/manifest.json"),
        _spec("trr0001_plan", "opened_development", "experiments/TRR-0001/plan.json"),
        _spec("trr0001_r1_selection_reveal", "opened_development", "experiments/TRR-0001/revision-r1/selection_reveal.json"),
        _spec("trr0002_calibration", "checkpoint_calibration", "experiments/TRR-0002/calibration/frozen_calibration.json"),
        _spec("trr0002_configuration_winner", "checkpoint_selection", "experiments/TRR-0002/configuration-search/causal-selection/winner.json"),
        _spec("trr0002_fresh_observation_index", "opened_development", "experiments/TRR-0002/configuration-search/fresh-blind/observation-index.json"),
        _spec("trr0002_public_finance_records", "opened_development", "experiments/TRR-0002/configuration-search/public-finance/records.json", aliases={"content_sha256": "rendered_sha256", "token_ids_sha256": "trr0002_active_token_ids_sha256"}, mode="trr0002_finance"),
        _spec("trr0002_public_pile_records", "opened_development", "experiments/TRR-0002/configuration-search/public-pile/records.json", aliases={"text_sha256": "rendered_sha256", "token_ids": "trr0002_h40_token_ids_sha256"}, mode="trr0002_pile"),
        _spec("trr0003_fit_records", "fitting_bank", "experiments/TRR-0003/evidence/control/track_b_fit_records.json"),
        _spec("trr0004_affine_fit", "fitting_bank", "experiments/TRR-0004/fit/affine_fit_records.json"),
        _spec("trr0004_affine_validation", "checkpoint_calibration", "experiments/TRR-0004/fit/affine_validation_records.json"),
        _spec("trr0004_adapter_v2_fit", "fitting_bank", "experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json"),
        _spec("trr0004_adapter_v2_validation", "checkpoint_calibration", "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json"),
        _spec("trr0004_selection_plan", "opened_evaluation", "experiments/TRR-0004/fresh_confirmation_v1/selection_plan.json", aliases=h129),
        _spec("trr0005_original_fit", "fitting_bank", "experiments/TRR-0005/public_activation_v1/original_fit_records.json"),
        _spec("trr0005_enriched_fit", "fitting_bank", "experiments/TRR-0005/public_activation_v1/enriched_fit_records.json"),
        _spec("trr0005_public_validation_selection", "checkpoint_calibration", "experiments/TRR-0005/public_validation_selection.json"),
        _spec("trr0005_selection_plan", "opened_evaluation", "experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json"),
        _spec("trr0006_selection", "opened_evaluation", "experiments/TRR-0006/source_selection.json", aliases=h128_h129),
        _spec("trr0006_panel", "opened_evaluation", "experiments/TRR-0006/panel_capture_v1/panel.json"),
        _spec("trr0006_public_observation_panel", "opened_evaluation", "experiments/TRR-0006/public_observations_v1/panel.json"),
        _spec("trr0006_p04_opaque", "opaque_reservation", "experiments/TRR-0006/coordination/p04_reservation_hashes.json", aliases=h129),
        _spec("trr0007_selection", "opened_evaluation", "experiments/TRR-0007/selection/source_selection.json", aliases=h128_h129),
        _spec("trr0007_selection_exclusions", "opened_development", "experiments/TRR-0007/selection/source_exclusions.json", aliases=h129),
        _spec("trr0007_original_fit", "fitting_bank", "experiments/TRR-0007/support/broader_capture_v2/original_fit_records.json"),
        _spec("trr0007_enriched_fit", "fitting_bank", "experiments/TRR-0007/support/broader_capture_v2/enriched_fit_records.json"),
        _spec("trr0007_prefix_exclusions", "fitting_bank", "experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json", aliases=h128),
        _spec("trr0007_eval_panel", "opened_evaluation", "experiments/TRR-0007/evaluation/public_observations/panel.json", aliases=h128),
        _spec("trr0007_p06_opaque", "opaque_reservation", "experiments/TRR-0007/coordination/p06_opaque_source_sequence_reservation.json", aliases=opaque),
        _spec("trr0008_selection", "opened_evaluation", "experiments/TRR-0008/selection/source_selection.json", aliases=h128_h129),
        _spec("trr0008_selection_exclusions", "opened_development", "experiments/TRR-0008/selection/source_exclusions.json", aliases=h129),
        _spec("trr0008_eval_panel", "opened_evaluation", "experiments/TRR-0008/evaluation/public_observations_v1/panel.json", aliases=h128),
        _spec("trr0008_p06_opaque", "opaque_reservation", "experiments/TRR-0008/planning/approved_opaque/p06_opaque_source_sequence_reservation.json", aliases=opaque),
        _spec("trr0008_selection_opaque", "opaque_reservation", "experiments/TRR-0008/selection/opaque_source_sequence_reservation.json", aliases=opaque),
        _spec("trr0009_original_selection", "opened_development", "experiments/TRR-0009/selection/source_selection.json", aliases=h128_h129),
        _spec("trr0009_selection_v2", "checkpoint_selection", "experiments/TRR-0009/selection_v2/source_selection.json", aliases=h128_h129),
        _spec("trr0009_selection_v2_exclusions", "opened_development", "experiments/TRR-0009/selection_v2/source_exclusions.json", aliases=h129),
        _spec("trr0009_eval_panel", "opened_evaluation", "experiments/TRR-0009/evaluation/public_observations_v2/panel.json", aliases=h128),
        _spec("trr0009_opaque_reservation", "opaque_reservation", "experiments/TRR-0009/selection_v2/opaque_source_sequence_reservation.json", aliases=opaque),
        _spec("trr0009_p08_opaque", "opaque_reservation", "experiments/TRR-0009/coordination/approved_opaque/p08_opaque_hash_exchange_sanitized_r2.json", aliases=opaque),
        _spec("trr_p09_nested_b1", "fitting_bank", "experiments/TRR-P09/setup/nested-b1-exclusion-ledger-r1.json", aliases=h128),
        _spec("trr_p09_public_validation_audit", "checkpoint_calibration", "experiments/TRR-P09/setup/public-validation-r1-audit.json", aliases=h128),
        _spec("pr20_source_selection_binding", "opened_evaluation", "@pr20/experiments/TRR-0010/evaluation/source_selection_binding_r4.json"),
        _spec("pr20_source_selection", "opened_evaluation", "@pr20/experiments/TRR-0010/evaluation/source_selection.json", aliases=h128),
        _spec("pr20_source_panel", "opened_evaluation", "@pr20/experiments/TRR-0010/evaluation/public_capture_watchdog_r5/observations_v1/panel.json", aliases=h128),
        # The three opened panel files were previously omitted because they
        # contain payload arrays.  Their identity fields are now parsed by a
        # narrow skip-payload path and bound to immutable file hashes.
        _spec("trr0003_opened_panel_metadata", "opened_evaluation", "experiments/TRR-0003/evidence/control/panel.json", mode="panel_identity", sha="d1810330f53ebb7e149be45b8c07414e09d30b31b68c7c09d24df3854fec7333"),
        _spec("trr0004_opened_panel_metadata", "opened_evaluation", "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/panel.json", mode="panel_identity", sha="da65242f395c2c96a25ed8e30d62415db9108c9ded0c9525d4f9358691cb44da"),
        _spec("trr0005_opened_panel_metadata", "opened_evaluation", "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json", mode="panel_identity", sha="d72bc8338de90028ae87a00fef3193c2c494a65d529ffad70978b8f706fd690d"),
    ]
    return tuple(specs)


def merge_bundles(bundles: Sequence[IdentityBundle]) -> IdentityBundle:
    merged = IdentityBundle("all_accessible_identity_sources", "union", Path("."), "", 0, status="metadata_only_union")
    for bundle in bundles:
        for field_name, by_namespace in bundle.values.items():
            for namespace, values in by_namespace.items():
                merged.values[field_name][namespace].update(values)
    return merged


def _value_overlap(left: IdentityBundle, right: IdentityBundle, field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for left_ns, left_values in left.values.get(field_name, {}).items():
        for right_ns, right_values in right.values.get(field_name, {}).items():
            if field_name == "source_index" and not left_ns.compatible(right_ns):
                continue
            common = left_values & right_values
            if common:
                style = left_ns.style if left_ns.style != "*" else right_ns.style
                counts[style] += len(common)
    return dict(sorted(counts.items()))


def pair_intersection(left: IdentityBundle, right: IdentityBundle) -> dict[str, Any]:
    fields = sorted(set(left.values) | set(right.values))
    return {"left": left.label, "right": right.label, "counts": {field: _value_overlap(left, right, field) for field in fields}}


def check_candidate(candidate: Mapping[str, Any], exclusions: IdentityBundle) -> list[dict[str, str]]:
    """Return every matching identity reason for a candidate.

    Record IDs and verified content/sequence hashes are global.  Source
    indices are compared only inside compatible dataset/split/revision/style
    namespaces.  H128 and H129 remain distinct canonical fields.
    """
    namespace = _namespace_from_mapping(Namespace(), candidate, "candidate")
    reasons: list[dict[str, str]] = []
    for raw_key, value in candidate.items():
        key = str(raw_key).casefold().replace("-", "_")
        canonical = {
            **DEFAULT_ALIASES,
            "final_sequence_sha256": "h128_sequence_sha256",
            "h128_sequence_sha256": "h128_sequence_sha256",
            "sequence_h128_sha256": "h128_sequence_sha256",
            "truncated_sequence_sha256": "h129_sequence_sha256",
            "h129_sequence_sha256": "h129_sequence_sha256",
            "sequence_h129_sha256": "h129_sequence_sha256",
            "trr0002_active_token_ids_sha256": "trr0002_active_token_ids_sha256",
            "trr0002_h40_token_ids_sha256": "trr0002_h40_token_ids_sha256",
        }.get(key)
        if canonical is None:
            continue
        if canonical in GLOBAL_FIELDS and (
            (canonical == "record_id" and isinstance(value, str))
            or (canonical != "record_id" and _is_sha256(value))
        ):
            comparable = value if canonical == "record_id" else value.casefold()
            for source_ns, values in exclusions.values.get(canonical, {}).items():
                if comparable in values:
                    reasons.append({"field": canonical, "namespace": source_ns.as_string()})
        elif canonical == "source_index" and isinstance(value, int) and namespace.style != "*":
            for source_ns, values in exclusions.values.get(canonical, {}).items():
                if value in values and source_ns.compatible(namespace):
                    reasons.append({"field": canonical, "namespace": source_ns.as_string()})
    return sorted({(item["field"], item["namespace"]): item for item in reasons}.values(), key=lambda item: (item["field"], item["namespace"]))


def _bundle_receipt(bundle: IdentityBundle) -> dict[str, Any]:
    return {
        "label": bundle.label,
        "role": bundle.role,
        "path": str(bundle.path),
        "bytes": bundle.bytes,
        "sha256": bundle.sha256,
        "schema": bundle.schema,
        "status": bundle.status,
        "identity_counts": bundle.counts(),
        "namespace_counts": bundle.namespace_counts(),
        "unrecognized_identity_key_counts": dict(sorted(bundle.unknown_identity_keys.items())),
        "metadata_notes": bundle.metadata_notes,
    }


def _compare_ordered_selection_records(
    trr9_records: Mapping[str, Any],
    pr20_records: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the registered PR20 rows to the TRR9 selection rows.

    This helper is deliberately pure so the negative regression can prove that
    a missing or malformed row cannot be counted as an equal None field.
    """

    expected_counts = {"finance": (256, 128), "pile": (128, 128)}
    domains: dict[str, Any] = {}
    for domain, (expected_trr9, expected_pr20) in expected_counts.items():
        left = trr9_records.get(domain)
        right = pr20_records.get(domain)
        if not isinstance(left, list) or not isinstance(right, list):
            return {"status": "FAIL_DOMAIN_RECORDS_NOT_LIST", "domain": domain}
        if len(left) != expected_trr9 or len(right) != expected_pr20:
            return {
                "status": "FAIL_DOMAIN_COUNTS",
                "domain": domain,
                "trr0009_records": len(left),
                "pr20_records": len(right),
                "expected_trr0009_records": expected_trr9,
                "expected_pr20_records": expected_pr20,
            }
        required_fields = ("record_id", "public_record_sha256", "final_sequence_sha256")
        for row_number, row in enumerate(left + right):
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("record_id"), str)
                or not row["record_id"]
                or any(not _is_sha256(row.get(field_name)) for field_name in required_fields[1:])
            ):
                return {
                    "status": "FAIL_MALFORMED_IDENTITY",
                    "domain": domain,
                    "row_number": row_number,
                }
        expected = left[:128] if domain == "finance" else left
        matches = {
            field_name: sum(
                a.get(field_name) == b.get(field_name)
                for a, b in zip(expected, right)
            )
            for field_name in required_fields
        }
        domains[domain] = {
            "trr0009_records": len(left),
            "pr20_records": len(right),
            "ordered_expected_subset": "finance_first_128" if domain == "finance" else "pile_all_128",
            "ordered_match_counts": matches,
            "ordered_exact": all(value == 128 for value in matches.values()),
        }
    exact = all(
        domains[domain]["ordered_exact"]
        for domain in expected_counts
    )
    return {
        "status": "PASS_ORDERED_FINANCE_FIRST128_PILE_ALL128"
        if exact
        else "FAIL_ORDERED_IDENTITY_MISMATCH",
        "domains": domains,
    }


def _ordered_pr20_binding(*, root: Path, pr20_root: Path | None) -> dict[str, Any]:
    if pr20_root is None:
        raise AuditError("PR20 root is required for the ordered binding check")
    trr9_path = resolve_path("experiments/TRR-0009/selection_v2/source_selection.json", root, pr20_root)
    pr20_path = resolve_path("@pr20/experiments/TRR-0010/evaluation/source_selection.json", root, pr20_root)
    try:
        trr9 = json.loads(trr9_path.read_text(encoding="utf-8"))
        pr20 = json.loads(pr20_path.read_text(encoding="utf-8"))
        left = trr9["selection_rule"]["records"]
        right = pr20["selection_rule"]["records"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AuditError("TRR9/PR20 ordered selection metadata is unavailable") from exc
    result = _compare_ordered_selection_records(left, right)
    if result["status"] != "PASS_ORDERED_FINANCE_FIRST128_PILE_ALL128":
        raise AuditError(f"TRR9/PR20 ordered binding failed: {result['status']}")
    return {
        **result,
        "trr0009_sha256": sha256_file(trr9_path),
        "pr20_sha256": sha256_file(pr20_path),
    }

def build_audit(*, root: Path, pr20_root: Path | None = None) -> dict[str, Any]:
    specs = source_specs()
    bundles = [load_bundle(spec, root=root, pr20_root=pr20_root) for spec in specs]
    bundles = [bundle for bundle in bundles if bundle is not None]
    by_label = {bundle.label: bundle for bundle in bundles}
    if len(by_label) != len(specs):
        missing = sorted({spec.label for spec in specs} - set(by_label))
        raise AuditError(f"required identity sources missing: {missing}")
    union = merge_bundles(bundles)
    trr9 = by_label["trr0009_selection_v2"]
    pr20 = by_label["pr20_source_selection"]
    overlap = pair_intersection(trr9, pr20)
    by_domain: dict[str, dict[str, int]] = {}
    for field_name, styles in overlap["counts"].items():
        for style, count in styles.items():
            by_domain.setdefault(style, {})[field_name] = count
    # A synthetic fixture proves the rejection rule without exposing an
    # actual source identity.  The actual TRR9 fixture is tested separately.
    fixture_ns = Namespace("pile", "fixture", "train", "fixture")
    fixture = IdentityBundle("fixture", "regression", Path("."), "", 0)
    fixture.add("record_id", "reused-validation-record", fixture_ns)
    fixture.add("rendered_sha256", "a" * 64, fixture_ns)
    fixture.add("h128_sequence_sha256", "b" * 64, fixture_ns)
    fixture.add("source_index", 17, fixture_ns)
    probe = {"record_id": "reused-validation-record", "public_record_sha256": "a" * 64, "final_sequence_sha256": "b" * 64, "source_index": 17, "style": "pile", "dataset_id": "fixture", "split": "train", "revision": "fixture"}
    probe_reasons = check_candidate(probe, fixture)
    source_count = len(bundles)
    identity_gaps = [{"label": b.label, "role": b.role, "reason": "loaded metadata contains no individual comparable identity fields"} for b in bundles if not b.values]
    descriptor_only_labels = {
        "trr0002_calibration",
        "trr0005_public_validation_selection",
        "trr0007_selection_exclusions",
        "trr0008_selection_exclusions",
        "trr0009_selection_v2_exclusions",
        "trr_p09_public_validation_audit",
        "pr20_source_selection_binding",
    }
    unresolved_identity_gaps = [
        gap for gap in identity_gaps if gap["label"] not in descriptor_only_labels
    ]
    unknown_identity_key_gaps = [
        {
            "label": bundle.label,
            "role": bundle.role,
            "keys": dict(sorted(bundle.unknown_identity_keys.items())),
            "reason": "unrecognized identity/hash keys are reported but not admitted without a verified producer mapping",
        }
        for bundle in bundles
        if bundle.unknown_identity_keys
    ]
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PARTIAL_METADATA_ONLY_EXCLUSION_AUDIT",
        "coverage_complete": False,
        "source_inventory": [_bundle_receipt(bundle) for bundle in bundles],
        "source_count": source_count,
        "union_identity_counts": union.counts(),
        "union_namespace_counts": union.namespace_counts(),
        "required_sources": [spec.label for spec in specs],
        "coverage": {
            "coverage_complete": False,
            "descriptor_only_sources": sorted(descriptor_only_labels),
            "unresolved_identity_gaps": unresolved_identity_gaps,
            "accessible_explicit_sources_loaded": source_count == len(specs),
            "accessible_explicit_source_count": source_count,
            "explicit_source_spec_count": len(specs),
            "trr0009_selection_manifest_included": "trr0009_selection_v2" in by_label,
            "pr20_selection_included": "pr20_source_selection" in by_label,
            "opened_panel_identity_extractor_used": [label for label in by_label if label.endswith("_opened_panel_metadata")],
            "sources_with_no_individual_identity_fields": [
                gap["label"] for gap in identity_gaps
            ],
            "unrecognized_identity_key_gaps": unknown_identity_key_gaps,
        },
        "coverage_gaps": [
            "The following panel artifacts expose only aggregate record identity digests; their individual rows are covered only if the bound selection ledger is independently verified: trr0006_panel, trr0006_public_observation_panel, trr0007_eval_panel, trr0008_eval_panel, trr0009_eval_panel, pr20_source_panel.",
            "P03 sealed holdout is intentionally unopened and absent from the inventory.",
            "No new candidate rows were scanned; this receipt binds prior identities only and does not certify remaining-range capacity.",
            "Opaque reservations without domain labels remain wildcard namespaces; they are not silently assigned to a dataset.",
            "TRR-0006 P04 fields opaque_truncated_sequence_fingerprints and truncated_sequence_fingerprint have no verified producer convention and are excluded from the H129 union.",
            "Unrecognized identity/hash keys are reported by source and key name but are not admitted until their producer convention is verified.",
        ],
        "identity_coverage_gaps": identity_gaps,
        "access_boundary": {
            "source_text_read": False,
            "historical_panel_json_parsed": True,
            "historical_panel_identity_fields_inspected": True,
            "historical_panel_payload_fields_skipped": True,
            "historical_panel_payload_values_emitted": False,
            "previously_opened_public_tokens_hashed": True,
            "token_payload_values_emitted": False,
            "activation_payload_read": False,
            "truth_or_scores_read": False,
            "new_panel_selected": False,
            "p03_holdout_accessed": False,
        },
        "hash_conventions": {
            "record_id": "global exact string identity",
            "rendered_sha256": "global only where producer field is explicitly bound to rendered/public record bytes",
            "h128_sequence_sha256": "global SHA-256 of exactly 128 little-endian signed-int32 IDs including BOS",
            "h129_sequence_sha256": "separate truncated/H129 namespace; never compared with H128",
            "source_index": "dataset/style/split/revision namespace scoped",
            "trr0002_active_token_ids_sha256": "TRR2 Finance active IDs, little-endian signed int32 bytes",
            "trr0002_h40_token_ids_sha256": "TRR2 Pile H40 tensor header plus little-endian int64 bytes",
        },
        "ordered_pr20_binding": _ordered_pr20_binding(root=root, pr20_root=pr20_root),
        "focused_intersections": {
            "trr0009_selection_v2_vs_pr20": {
                "counts_by_style": by_domain,
                "expected_record_id_counts": {"finance": 128, "pile": 128},
                "status": "PASS_128_PER_DOMAIN" if by_domain.get("finance", {}).get("record_id") == 128 and by_domain.get("pile", {}).get("record_id") == 128 else "FAIL_OR_UNAVAILABLE",
            }
        },
        "regression": {
            "synthetic_reused_validation_rejected": len(probe_reasons) == 4,
            "synthetic_reason_fields": sorted(reason["field"] for reason in probe_reasons),
        },
        "notes": [
            "This is a prior-identity exclusion union, not a source-selection or capacity result.",
            "Overall status remains partial even when all explicit accessible files load.",
            "Descriptor-only pointer/count receipts are not interpreted as zero overlap; unresolved panel aggregates remain explicit gaps.",
            "Earlier recovery_identity_audit.json, recovery_identity_audit_r2.json, and recovery_identity_audit_r3.json receipts are superseded and excluded by this fresh receipt.",
        ],
        "superseded_artifacts": [
            {
                "path": "experiments/TRR-P10/exclusions/recovery_identity_audit.json",
                "reason": "superseded by this receipt after prefix-hash and ordered-binding corrections",
            },
            {
                "path": "experiments/TRR-P10/exclusions/recovery_identity_audit_r2.json",
                "reason": "superseded by this receipt after explicit partial-coverage classification",
            },
            {
                "path": "experiments/TRR-P10/exclusions/recovery_identity_audit_r3.json",
                "reason": "superseded by this receipt after explicit unknown identity/hash-key reporting",
            },
        ],
    }


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pr20-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_audit(root=args.root.resolve(), pr20_root=args.pr20_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as handle:
            handle.write(canonical_json(result))
    except FileExistsError as exc:
        raise AuditError(f"refusing to overwrite existing audit receipt: {args.output}") from exc
    print(json.dumps({"output": str(args.output), "status": result["status"], "source_count": result["source_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
