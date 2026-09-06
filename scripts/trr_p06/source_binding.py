"""Metadata-only source and exclusion bindings for TRR-P06.

The P06 source selector is deliberately separate from the published P04/P05
selectors.  This module binds the public dataset revisions and audits prior
identity metadata without retaining source rows, token arrays, labels, or
answers in the resulting setup artifact.  P01--P03 and TRR-0007 inputs are
opaque optional reservations: a missing input is recorded as incomplete
coverage and never silently treated as an empty exclusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


class SourceBindingError(RuntimeError):
    """Raised when a public source or exclusion binding is unsafe."""


TASK_ID = "TRR-P06"
# Approved by the parent task as a hash-only reservation export.  The digest
# check applies only when this exact basename is supplied explicitly; P06 never
# scans TRR-0007 directories or follows its mutable worktree.
APPROVED_OPAQUE_EXPECTED_SHA256 = {
    "p06_opaque_source_sequence_reservation.json": (
        "09e845fec244a38873c5bf127f6d984af91af503fb42d3a8411451ce41cdedf4"
    ),
}
SOURCE_HASH_KEYS = frozenset(
    {
        "public_record_sha256",
        "public_record_hash",
        "rendered_sha256",
        "rendered_hash",
        "final_sequence_sha256",
        "sequence_sha256",
        "sequence_hash",
        "truncated_sequence_sha256",
        "source_text_sha256",
        "text_sha256",
        "text_hash",
        "prompt_id",
        "content_sha256",
        "content_hash",
        "tokenized_record_sha256",
        "record_sha256",
        "record_hash",
        "artifact_sha256",
    }
)
SOURCE_ID_KEYS = frozenset({
    "record_id",
    "source_record_id",
    "public_record_id",
    "context_id",
    "endpoint_id",
    "case_id",
})
SOURCE_INDEX_KEYS = frozenset({
    "dataset_index",
    "source_index",
    "row_index",
    "index",
})
PRIVATE_BRANCH_KEYS = frozenset(
    {
        "token_ids",
        "input_ids",
        "labels",
        "target_labels",
        "truth_ids",
        "truth_tokens",
        "truth_labels",
        "truth",
        "correctness",
        "source_text",
        "plaintext",
        "answer",
        "answers",
        "private_truth",
        "evaluator_truth",
    }
)


@dataclass(frozen=True)
class ExclusionSourceSpec:
    """One immutable metadata input, with no source payload embedded."""

    label: str
    path: str
    expected_sha256: str | None = None
    required: bool = False
    role: str = "published_identity_metadata"


# These files are published identity/selection metadata rather than model or
# truth payloads.  Keep the list explicit: a broad recursive walk could ingest
# a private truth sidecar or a large prediction artifact by accident.
PUBLISHED_METADATA_RELATIVE = (
    # TRR-0001/P01 historical selection metadata.
    "experiments/TRR-0001/manifest.json",
    "experiments/TRR-0001/revision-r1/manifest.json",
    "experiments/TRR-0001/revision-r1/selection_commitment.json",
    "experiments/TRR-0001/revision-r1/selection_reveal.json",
    "experiments/TRR-0001/revision-r1/selection_verification.json",
    "experiments/TRR-0001/revision-r2/manifest.json",
    "experiments/TRR-0001/revision-r2/dual_benchmark_matrix.json",
    # TRR-0002/P02 public selection metadata.  The diagnostic exclusion is
    # supplied through the approved external copy below; this old checkout
    # relative path is intentionally not treated as a required input.
    "experiments/TRR-0002/manifest.json",
    "experiments/TRR-0002/blind/access_manifest.json",
    "experiments/TRR-0002/blind/selection_commitment.json",
    "experiments/TRR-0002/configuration-search/fresh-blind/access_manifest.json",
    "experiments/TRR-0002/configuration-search/fresh-blind/selection_commitment.json",
    "experiments/TRR-0002/configuration-search/fresh-blind/observation-index.json",
    "experiments/TRR-0002/configuration-search/public-pile/records.json",
    "experiments/TRR-0002/configuration-search/public-finance/records.json",
    "experiments/TRR-0002/strict-surrogate-heavy/heavy-selection-commitment.json",
    "experiments/TRR-0002/strict-surrogate-heavy/heavy-selection-commitment-c1.json",
    "experiments/TRR-0002/strict-surrogate-heavy/selection-reveal.json",
    # TRR-0003/P03 published public panel/resource identities.
    "experiments/TRR-0003/manifest.json",
    "experiments/TRR-0003/evidence/control/panel.json",
    "experiments/TRR-0003/evidence/control/public_resource_manifest.json",
    "experiments/TRR-0003/evidence/control/track_b_panel_bindings.json",
    "experiments/TRR-0003/footing/panel.json",
    "experiments/TRR-0003/footing/inventory.json",
    "experiments/TRR-0003/track_a/public_resource_manifest.json",
    "experiments/TRR-0003/track_b/inventory_v1.json",
    "experiments/TRR-0003/track_b/checkpoint_selection_amendment.json",
    # TRR-0004/P04 public fit/validation/evaluation identity metadata.
    "experiments/TRR-0004/manifest.json",
    "experiments/TRR-0004/fit/public_fit_manifest.json",
    "experiments/TRR-0004/fit/affine_fit_records.json",
    "experiments/TRR-0004/fit/affine_validation_records.json",
    "experiments/TRR-0004/fit/adapter_v2/public_fit_manifest.json",
    "experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json",
    "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
    "experiments/TRR-0004/fresh_confirmation_v1/selection_plan.json",
    "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/panel.json",
    "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/registration_v2.json",
    # TRR-0005 public fit/validation and natural fresh-panel metadata.
    "experiments/TRR-0005/manifest.json",
    "experiments/TRR-0005/public_activation_v1/enriched_manifest.json",
    "experiments/TRR-0005/public_activation_v1/original_manifest.json",
    "experiments/TRR-0005/public_activation_v1/enriched_fit_records.json",
    "experiments/TRR-0005/public_activation_v1/original_fit_records.json",
    "experiments/TRR-0005/public_activation_v1/capture_manifest_receipt.json",
    "experiments/TRR-0005/public_validation_selection.json",
    "experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json",
    "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json",
    "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations.json",
    # TRR-0006/P04 natural panel and its public observation metadata.
    "experiments/TRR-0006/manifest.json",
    "experiments/TRR-0006/source_selection.json",
    "experiments/TRR-0006/panel_capture_v1/panel.json",
    "experiments/TRR-0006/panel_capture_v1/observations.json",
    "experiments/TRR-0006/panel_capture_v1/capture.json",
    "experiments/TRR-0006/public_observations_v1/panel.json",
    "experiments/TRR-0006/public_observations_v1/capture.json",
    "experiments/TRR-0006/duplicate_capture_exclusion.json",
    "experiments/TRR-0006/eligibility_inventory_1536_projection.json",
    "experiments/TRR-0006/eligibility_inventory_1536_projection_v2.json",
    "experiments/TRR-0006/eligibility_inventory_1536_projection_v3.json",
    "experiments/TRR-0006/predictions_v1/run_manifest.json",
    "experiments/TRR-0006/scored_v1/manifest.json",
    "experiments/TRR-0006/coordination/p04_reservation_hashes.json",
)


# These are explicitly published P01--P03 reservation/audit artifacts found in
# the approved read-only task workspaces.  They are optional at setup time so a
# clean checkout records missing coverage instead of pretending it is absent.
EXTERNAL_APPROVED_METADATA = (
    ExclusionSourceSpec(
        "TRR-P01 public panel metadata",
        "/tmp/trr-p03/experiments/TRR-P01/runtime/panel-20260905/panel_manifest.json",
        "cf4b03d06109635a7aa69e7fbfca386abfb5d1b03ad4347ee43dfe307a096ef9",
        required=True,
        role="approved_p01_public_panel_identity",
    ),
    ExclusionSourceSpec(
        "TRR-P01 selection evidence",
        "/tmp/trr-p03/experiments/TRR-P01/runtime/panel-20260905/selection_evidence.json",
        "78246bcf0e2c392e0c6abce51ab96062629e2a22be72ca9712f8ed5557248a84",
        required=True,
        role="approved_p01_selection_identity",
    ),
    ExclusionSourceSpec(
        "TRR-P01 completed reservation",
        "/tmp/trr-p03/experiments/TRR-P01/runtime/reconstruct-final-r2-arm-000-reservation.json",
        "e39229e616d2b552e53f1f87bd3cb2b485dbedd35bfef8a1764c8a087f269b73",
        required=True,
        role="approved_p01_reservation_metadata",
    ),
    ExclusionSourceSpec(
        "TRR-P01 paired reservation",
        "/tmp/trr-p03/experiments/TRR-P01/runtime/reconstruct-final-r2-arm-001-reservation.json",
        "9242eab4d0061f3b7bc079b385dbb20a117f09fe35fe604c393c7374a7a0d0bd",
        required=True,
        role="approved_p01_reservation_metadata",
    ),
    ExclusionSourceSpec(
        "TRR-P02 public diagnostic exclusion",
        "/tmp/trr-p02/experiments/TRR-P02/setup/public-diagnostic-exclusion.final.json",
        "3b671dea06371834dfaf8863fd2b667fb2894f82d171d29d314236ec7abaa6dc",
        required=True,
        role="approved_p02_public_exclusion",
    ),
    ExclusionSourceSpec(
        "TRR-P03 prior exclusion audit",
        "/tmp/trr-p03/experiments/TRR-P03/setup/panel-20260906-frozen/prior-exclusion-audit.json",
        "1e4aa3179aa2411b4fed301374d31d9f65f5a64a3959aff9a29ca3148a276428",
        required=True,
        role="approved_p03_opaque_exclusion_audit",
    ),
    ExclusionSourceSpec(
        "TRR-P03 selected reservation identities",
        "experiments/TRR-P06/setup/approved-p03/p03_selected_reservation.json",
        "47c22cae9ee6103e249ca789e5d9f421d9577d5acef320f052dcdf21ed6cba4b",
        required=True,
        role="approved_p03_selected_reservation",
    ),
)


@dataclass
class _IdentityAccumulator:
    ids: set[str] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)
    sequence_hashes: set[str] = field(default_factory=set)
    text_hashes: set[str] = field(default_factory=set)
    indices: set[int] = field(default_factory=set)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SourceBindingError(f"metadata input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_key(value: Any) -> str:
    return str(value).casefold().replace("-", "_")


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def _scan_identity_metadata(
    value: Any,
    *,
    accumulator: _IdentityAccumulator,
    private_branch: bool = False,
    field_name: str | None = None,
) -> None:
    """Collect opaque identity strings without traversing payload arrays.

    Some published ledgers store hashes/IDs as arrays under a field name.
    ``field_name`` carries that declared identity type through the array so
    those reservations are bound without reading token or source-value
    branches.
    """

    def record(field: str | None, child: Any) -> None:
        if field in SOURCE_ID_KEYS and isinstance(child, str) and child:
            accumulator.ids.add(child)
        if field in SOURCE_INDEX_KEYS and isinstance(child, int) and not isinstance(child, bool):
            accumulator.indices.add(int(child))
        if field in SOURCE_HASH_KEYS and _is_hash(child):
            lowered = child.lower()
            accumulator.hashes.add(lowered)
            if field is not None and ("sequence" in field or "truncated" in field):
                accumulator.sequence_hashes.add(lowered)
            if field is not None and ("text" in field or "rendered" in field or "content" in field):
                accumulator.text_hashes.add(lowered)

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normal_key(key)
            if (
                normalized in PRIVATE_BRANCH_KEYS
                or normalized in {"private_truth", "evaluator_truth", "plaintext", "position_ids", "attention_mask", "mask"}
                or normalized.endswith("_token_ids")
            ):
                # Never traverse token arrays, labels, masks, source text, or
                # private truth.  Their enclosing metadata may still contain
                # approved identity fields elsewhere in the same object.
                continue
            record(normalized, child)
            child_field = (
                normalized
                if normalized in SOURCE_ID_KEYS or normalized in SOURCE_HASH_KEYS
                else field_name if field_name is not None and normalized in {"values", "items"} else None
            )
            _scan_identity_metadata(child, accumulator=accumulator, private_branch=private_branch, field_name=child_field)
    elif isinstance(value, list):
        for child in value:
            record(field_name, child)
            _scan_identity_metadata(child, accumulator=accumulator, private_branch=private_branch, field_name=field_name)
    else:
        record(field_name, value)


def _canonical_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(set(values))) + "\n").encode("utf-8"))


def _spec_descriptor(spec: ExclusionSourceSpec, *, root: Path) -> dict[str, Any]:
    original = Path(spec.path).expanduser()
    path = original if original.is_absolute() else root / original
    path = path.resolve()
    descriptor: dict[str, Any] = {
        "label": spec.label,
        "path": str(path),
        "role": spec.role,
        "required": bool(spec.required),
        "expected_sha256": spec.expected_sha256,
        "available": False,
        "identity_only": True,
    }
    if path.is_symlink() or not path.is_file():
        descriptor["status"] = "MISSING_COVERAGE"
        return descriptor
    actual = sha256_file(path)
    descriptor.update({"available": True, "bytes": int(path.stat().st_size), "sha256": actual})
    if spec.expected_sha256 is not None and actual != spec.expected_sha256:
        raise SourceBindingError(
            f"approved metadata hash changed for {spec.label}: expected {spec.expected_sha256}, got {actual}"
        )
    descriptor["status"] = "BOUND"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBindingError(f"approved identity metadata is invalid JSON: {path}") from exc
    accumulator = _IdentityAccumulator()
    _scan_identity_metadata(payload, accumulator=accumulator)
    descriptor.update(
        {
            "identity_id_count": len(accumulator.ids),
            "identity_hash_count": len(accumulator.hashes),
            "identity_sequence_hash_count": len(accumulator.sequence_hashes),
            "identity_text_hash_count": len(accumulator.text_hashes),
            "identity_index_count": len(accumulator.indices),
            "ids_digest": _canonical_digest(tuple(accumulator.ids)),
            "hashes_digest": _canonical_digest(tuple(accumulator.hashes)),
        }
    )
    return descriptor


def catalog_specs(root: Path) -> tuple[ExclusionSourceSpec, ...]:
    root = Path(root).expanduser().resolve()
    return tuple(
        ExclusionSourceSpec(
            label=f"published:{relative}",
            path=relative,
            required=False,
            role="published_prior_identity_metadata",
        )
        for relative in PUBLISHED_METADATA_RELATIVE
    ) + EXTERNAL_APPROVED_METADATA


@dataclass(frozen=True)
class ExclusionIndex:
    """In-memory identity index plus a payload-free provenance summary."""

    ids: frozenset[str]
    hashes: frozenset[str]
    sequence_hashes: frozenset[str]
    text_hashes: frozenset[str]
    indices: frozenset[int]
    descriptors: tuple[dict[str, Any], ...]
    coverage_complete: bool
    missing_labels: tuple[str, ...]
    catalog_sha256: str
    required_coverage_ready: bool = False
    required_missing_labels: tuple[str, ...] = ()
    optional_missing_labels: tuple[str, ...] = ()
    universal_coverage_complete: bool = False

    def block_reason(
        self,
        *,
        record_id: str,
        public_record_sha256: str,
        final_sequence_sha256: str,
        sequence_sha256_129: str | None = None,
        row_index: int | None = None,
    ) -> str | None:
        if record_id in self.ids:
            return "prior_record_id"
        if public_record_sha256 in self.hashes or public_record_sha256 in self.text_hashes:
            return "prior_rendered_or_text_hash"
        if final_sequence_sha256 in self.hashes or final_sequence_sha256 in self.sequence_hashes:
            return "prior_sequence_hash"
        if sequence_sha256_129 and sequence_sha256_129 in self.sequence_hashes:
            return "prior_129_sequence_hash"
        # Index-only commitments are retained for audit, but never applied
        # without a dataset binding because row numbers are not globally unique.
        _ = row_index
        return None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "catalog_sha256": self.catalog_sha256,
            "coverage_complete": self.coverage_complete,
            "descriptor_coverage_complete": self.coverage_complete,
            "missing_labels": list(self.missing_labels),
            "required_approved_identities_bound": bool(self.required_coverage_ready),
            "required_missing_labels": list(self.required_missing_labels),
            "optional_missing_labels": list(self.optional_missing_labels),
            "selection_ready": bool(self.required_coverage_ready),
            "universal_coverage_complete": bool(self.universal_coverage_complete),
            "source_id_count": len(self.ids),
            "source_hash_count": len(self.hashes),
            "sequence_hash_count": len(self.sequence_hashes),
            "text_hash_count": len(self.text_hashes),
            "source_index_count": len(self.indices),
            "source_id_digest": _canonical_digest(tuple(self.ids)),
            "source_hash_digest": _canonical_digest(tuple(self.hashes)),
            "sequence_hash_digest": _canonical_digest(tuple(self.sequence_hashes)),
            "descriptors": [dict(value) for value in self.descriptors],
            "global_disjoint_claim_allowed": False,
        }


def _catalog_digest(descriptors: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(value) for value in descriptors],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def collect_exclusions(
    root: Path,
    *,
    metadata_paths: Sequence[Path | str] | None = None,
    approved_opaque_paths: Sequence[Path | str] = (),
    include_default_catalog: bool = True,
) -> ExclusionIndex:
    """Bind prior metadata while retaining only opaque identity sets in memory.

    ``metadata_paths`` is used by synthetic tests and can extend the explicit
    catalog.  Approved opaque paths are caller-supplied because TRR-0007's
    file is not available at P06 setup time; no directory scan is performed.
    """

    root = Path(root).expanduser().resolve()
    specs: list[ExclusionSourceSpec] = []
    if include_default_catalog:
        specs.extend(catalog_specs(root))
    if metadata_paths:
        specs.extend(
            ExclusionSourceSpec(label=f"explicit:{Path(path).name}", path=str(path), required=True)
            for path in metadata_paths
        )
    specs.extend(
        ExclusionSourceSpec(
            label=f"approved-opaque:{Path(path).name}",
            path=str(path),
            expected_sha256=APPROVED_OPAQUE_EXPECTED_SHA256.get(Path(path).name),
            required=True,
            role="approved_opaque_reservation",
        )
        for path in approved_opaque_paths
    )
    descriptors: list[dict[str, Any]] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    sequence_hashes: set[str] = set()
    text_hashes: set[str] = set()
    indices: set[int] = set()
    missing: list[str] = []
    required_missing: list[str] = []
    optional_missing: list[str] = []
    for spec in specs:
        descriptor = _spec_descriptor(spec, root=root)
        descriptors.append(descriptor)
        if descriptor.get("available") is not True:
            if spec.required:
                required_missing.append(spec.label)
                raise SourceBindingError(f"required exclusion metadata is missing: {spec.path}")
            missing.append(spec.label)
            optional_missing.append(spec.label)
            continue
        path = Path(str(descriptor["path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        accumulator = _IdentityAccumulator()
        _scan_identity_metadata(payload, accumulator=accumulator)
        ids.update(accumulator.ids)
        hashes.update(accumulator.hashes)
        sequence_hashes.update(accumulator.sequence_hashes)
        text_hashes.update(accumulator.text_hashes)
        indices.update(accumulator.indices)
    # ``coverage_complete`` describes only whether every declared descriptor
    # file was present.  It is not a universal non-overlap claim: identity
    # extraction is intentionally metadata-only and hidden commitments remain
    # outside the available ledger.  Required approved inputs are tracked
    # separately from optional historical aliases.
    coverage_complete = not missing
    required_ready = not required_missing
    return ExclusionIndex(
        ids=frozenset(ids),
        hashes=frozenset(hashes),
        sequence_hashes=frozenset(sequence_hashes),
        text_hashes=frozenset(text_hashes),
        indices=frozenset(indices),
        descriptors=tuple(descriptors),
        coverage_complete=coverage_complete,
        missing_labels=tuple(missing),
        catalog_sha256=_catalog_digest(descriptors),
        required_coverage_ready=required_ready,
        required_missing_labels=tuple(required_missing),
        optional_missing_labels=tuple(optional_missing),
        universal_coverage_complete=False,
    )
