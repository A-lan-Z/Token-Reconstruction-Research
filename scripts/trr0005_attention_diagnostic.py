#!/usr/bin/env python3
"""Post-fit attention-weight diagnostic for the two TRR-0005 causal states.

This helper is intentionally observation-only.  It reads the reused public
validation H activations and validity mask, loads the already-selected causal
states, and recomputes the attention weights used by
``JointAffineAttentionDecoder._added_path``.  It never reads token labels,
truth, or the public embedding table, and it never trains or mutates a state.

The command is to be run only after the TRR-0005 fits have completed and the
post-fit resource window has been granted.  The default mode avoids hashing
large activation files; ``--hash-inputs`` enables reproducibility hashes for
the later evidence receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_contract import POSITION_BINS
from token_reconstruction.trr0005_joint_decoder import (
    ATTENTION_SCORE_MODE_COSINE_SCALE4,
    ATTENTION_SCORE_MODE_DOT_PRODUCT,
    ATTENTION_SCORE_MODES,
    CAUSAL_ATTENTION_METHOD,
    COSINE_ATTENTION_SCORE_SCALE,
    DATA_SCHEMA,
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    SCHEMA as DECODER_SCHEMA,
    VOCAB_SIZE,
    load_decoder_state,
)


TASK_ID = "TRR-0005"
DIAGNOSTIC_SCHEMA = "token-reconstruction.trr0005-attention-diagnostic.v1"
ACCEPTED_DATA_SCHEMAS = {
    DATA_SCHEMA,
    "token-reconstruction.trr0005-joint-fit-data.v1",
    "token-reconstruction.trr0004-public-fit-data.v1",
}
DISTRIBUTION_ORDER = ("original", "enriched")


class AttentionDiagnosticError(RuntimeError):
    """Raised when the public-H attention diagnostic contract is invalid."""


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise AttentionDiagnosticError(f"{label} must be a regular file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AttentionDiagnosticError(f"cannot resolve {label}: {candidate}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise AttentionDiagnosticError(f"{label} must be a regular file: {candidate}")
    return resolved


def _json_load(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AttentionDiagnosticError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise AttentionDiagnosticError(f"{label} must be a JSON object: {path}")
    return payload


def _resource_path(manifest_path: Path, resource: Mapping[str, Any], *, label: str) -> Path:
    raw = resource.get("path")
    if not isinstance(raw, str) or not raw:
        raise AttentionDiagnosticError(f"{label} resource has no path")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return _regular_file(path, label=label)


def _resource_tensor(
    manifest_path: Path,
    resources: Mapping[str, Any],
    names: Sequence[str],
    *,
    default_key: str,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected_name = next((name for name in names if name in resources), None)
    if selected_name is None:
        raise AttentionDiagnosticError(
            f"manifest is missing {label} resource; expected one of {tuple(names)!r}"
        )
    resource = resources[selected_name]
    if not isinstance(resource, Mapping):
        raise AttentionDiagnosticError(f"{selected_name} resource is malformed")
    path = _resource_path(manifest_path, resource, label=label)
    key = resource.get("tensor_key", default_key)
    if not isinstance(key, str) or not key:
        raise AttentionDiagnosticError(f"{selected_name} tensor key is malformed")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            if key not in available:
                raise AttentionDiagnosticError(
                    f"{label} is missing tensor {key!r}: {path}"
                )
            value = handle.get_tensor(key).contiguous()
    except AttentionDiagnosticError:
        raise
    except Exception as exc:
        raise AttentionDiagnosticError(f"cannot load {label}: {path}") from exc
    declared_shape = resource.get("shape")
    if declared_shape is not None:
        if not isinstance(declared_shape, list) or not all(
            isinstance(item, int) and item >= 0 for item in declared_shape
        ):
            raise AttentionDiagnosticError(f"{label} declared shape is malformed")
        if tuple(declared_shape) != tuple(value.shape):
            raise AttentionDiagnosticError(
                f"{label} declared shape {declared_shape!r} differs from tensor {list(value.shape)!r}"
            )
    return value, {
        "logical_resource": selected_name,
        "path": str(path),
        "tensor_key": key,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _validate_observations(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise AttentionDiagnosticError("validation observations must be [records, positions, hidden]")
    if tuple(value.shape[1:]) != (DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE):
        raise AttentionDiagnosticError(
            "validation observations must have registered geometry "
            f"[records,{DEFAULT_SEQUENCE_LENGTH},{DEFAULT_HIDDEN_SIZE}]"
        )
    if not value.dtype.is_floating_point:
        raise AttentionDiagnosticError("validation observations must be floating point")
    if not torch.isfinite(value).all().item():
        raise AttentionDiagnosticError("validation observations contain non-finite values")
    return value.contiguous()


def _validate_mask(value: torch.Tensor, *, records: int, positions: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (records, positions):
        raise AttentionDiagnosticError(
            f"validation validity mask must have shape [{records},{positions}]"
        )
    if value.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise AttentionDiagnosticError("validation validity mask must be boolean or integer")
    result = value.to(device="cpu", dtype=torch.bool).contiguous()
    if not result[:, 0].all().item():
        raise AttentionDiagnosticError("validation validity mask must contain BOS at every row")
    for row in result:
        false_seen = False
        for item in row.tolist():
            if not item:
                false_seen = True
            elif false_seen:
                raise AttentionDiagnosticError("validation validity mask must be right-padded")
    return result


def _load_public_validation_h(manifest_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    manifest_path = _regular_file(manifest_path, label="validation manifest")
    payload = _json_load(manifest_path, label="validation manifest")
    if payload.get("schema") not in ACCEPTED_DATA_SCHEMAS:
        raise AttentionDiagnosticError(
            f"unsupported validation manifest schema: {payload.get('schema')!r}"
        )
    if payload.get("task_id") not in (None, TASK_ID):
        raise AttentionDiagnosticError("validation manifest task ID changed")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise AttentionDiagnosticError("validation manifest has no resources object")

    observations, observation_descriptor = _resource_tensor(
        manifest_path,
        resources,
        ("validation_observations", "validation_artifact"),
        default_key="activations",
        label="validation observations",
    )
    observations = _validate_observations(observations)

    mask_name = next(
        (name for name in ("validation_valid_mask", "validation_artifact") if name in resources),
        None,
    )
    if mask_name is None:
        valid_mask = torch.ones(observations.shape[:2], dtype=torch.bool)
        mask_descriptor = {
            "logical_resource": None,
            "path": None,
            "tensor_key": None,
            "shape": list(valid_mask.shape),
            "dtype": str(valid_mask.dtype),
            "implicit_all_valid": True,
        }
    else:
        valid_mask, mask_descriptor = _resource_tensor(
            manifest_path,
            resources,
            (mask_name,),
            default_key="attention_mask",
            label="validation validity mask",
        )
        valid_mask = _validate_mask(
            valid_mask,
            records=int(observations.shape[0]),
            positions=int(observations.shape[1]),
        )
        mask_descriptor["implicit_all_valid"] = False

    return observations, valid_mask, {
        "manifest": str(manifest_path),
        "manifest_schema": payload.get("schema"),
        "observation": observation_descriptor,
        "validity_mask": mask_descriptor,
        "geometry": list(observations.shape),
        "truth_loaded": False,
        "embedding_table_loaded": False,
    }


def _state_metadata(path: Path, *, distribution: str) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise AttentionDiagnosticError(f"cannot read decoder-state metadata: {path}") from exc
    score_mode = metadata.get(
        "attention_score_mode", ATTENTION_SCORE_MODE_DOT_PRODUCT
    )
    if score_mode not in ATTENTION_SCORE_MODES:
        raise AttentionDiagnosticError(
            f"{path} has unsupported attention score mode: {score_mode!r}"
        )
    metadata.setdefault("attention_score_mode", score_mode)
    expected = {
        "schema": DECODER_SCHEMA,
        "task_id": TASK_ID,
        "method_id": CAUSAL_ATTENTION_METHOD,
        "attention_mode": "causal",
        "context_width": str(DEFAULT_CONTEXT_WIDTH),
        "distribution": distribution,
        "canonical_method_id": f"{distribution}__{CAUSAL_ATTENTION_METHOD}",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AttentionDiagnosticError(
                f"{path} metadata {key!r} must be {value!r}, observed {metadata.get(key)!r}"
            )
    return metadata


def _causal_attention_weights(
    model: Any,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Recompute the exact H-only causal attention weights used by the model."""

    # Keep this sequence aligned with JointAffineAttentionDecoder._added_path:
    # layer norm (eps=1e-5), Q/K/V projections, scaled scores, right-padded
    # causal mask, safe softmax, and zeroed invalid queries.
    mask = model._check_inputs(activation, valid_mask)
    if model.attention_mode != "causal":
        raise AttentionDiagnosticError("diagnostic state is not causal")
    if model.query is None or model.key is None or model.value is None:
        raise AttentionDiagnosticError("causal state has no Q/K/V projections")
    value = F.layer_norm(
        activation.float(),
        (model.hidden_size,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    query = model.query(value)
    key = model.key(value)
    positions = torch.arange(int(activation.shape[1]), device=activation.device)
    allowed = positions[None, :] <= positions[:, None]
    allowed = allowed.unsqueeze(0) & mask[:, None, :]
    if model.attention_score_mode == ATTENTION_SCORE_MODE_COSINE_SCALE4:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        scores = (query @ key.transpose(-1, -2)) * COSINE_ATTENTION_SCORE_SCALE
    elif model.attention_score_mode == ATTENTION_SCORE_MODE_DOT_PRODUCT:
        scores = query @ key.transpose(-1, -2) / math.sqrt(model.context_width)
    else:
        raise AttentionDiagnosticError(
            f"unsupported attention score mode: {model.attention_score_mode!r}"
        )
    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    valid_query = mask.unsqueeze(-1)
    has_key = allowed.any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_key, masked_scores, torch.zeros_like(masked_scores))
    weights = torch.softmax(safe_scores, dim=-1)
    return torch.where(valid_query & has_key, weights, torch.zeros_like(weights))


def _summary_for_queries(
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    query_selector: torch.Tensor,
) -> dict[str, Any]:
    rows, positions = torch.nonzero(query_selector, as_tuple=True)
    if int(rows.numel()) == 0:
        return {
            "query_count": 0,
            "average_current_position_mass": None,
            "average_earlier_position_mass": None,
            "average_bos_mass": None,
            "average_entropy_nats": None,
            "self_mass_gt_0_99_fraction": None,
            "maximum_partition_error": None,
        }
    query_weights = weights[rows, positions].to(dtype=torch.float64)
    query_positions = positions.to(dtype=torch.long)
    current_mass = query_weights.gather(1, query_positions[:, None]).squeeze(1)
    bos_mass = query_weights[:, 0]
    key_positions = torch.arange(query_weights.shape[1], device=query_weights.device)
    earlier = (key_positions[None, :] > 0) & (key_positions[None, :] < query_positions[:, None])
    earlier_mass = (query_weights * earlier.to(dtype=query_weights.dtype)).sum(dim=1)
    safe_weights = torch.where(
        query_weights > 0,
        query_weights,
        torch.ones_like(query_weights),
    )
    entropy = -(query_weights * safe_weights.log()).sum(dim=1)
    partition_error = (current_mass + earlier_mass + bos_mass - 1.0).abs()
    return {
        "query_count": int(rows.numel()),
        "average_current_position_mass": float(current_mass.mean().item()),
        "average_earlier_position_mass": float(earlier_mass.mean().item()),
        "average_bos_mass": float(bos_mass.mean().item()),
        "average_entropy_nats": float(entropy.mean().item()),
        "self_mass_gt_0_99_fraction": float((current_mass > 0.99).to(dtype=torch.float64).mean().item()),
        "maximum_partition_error": float(partition_error.max().item()),
    }


def summarize_attention(weights: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    """Return overall and fixed post-BOS position-bin attention summaries."""

    if weights.ndim != 3:
        raise AttentionDiagnosticError("attention weights must be [records, queries, keys]")
    if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(weights.shape[:2]):
        raise AttentionDiagnosticError("attention weights and validity mask geometry differ")
    post_bos = valid_mask.to(dtype=torch.bool).clone()
    post_bos[:, 0] = False
    overall = _summary_for_queries(weights, valid_mask, post_bos)
    by_position: dict[str, Any] = {}
    for name, lower, upper in POSITION_BINS:
        positions = torch.arange(weights.shape[1], device=weights.device)
        selector = post_bos & positions[None, :].ge(lower)
        if upper is not None:
            selector &= positions[None, :].le(upper)
        by_position[name] = _summary_for_queries(weights, valid_mask, selector)
    return {
        "overall": overall,
        "position_bins": by_position,
        "position_bin_definition": "zero-based tensor positions; BOS is index 0 and bins cover post-BOS indices",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, hash_bytes: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": None,
    }
    if hash_bytes:
        result["sha256"] = _sha256_file(path)
    return result


def run_attention_diagnostic(
    validation_manifest: Path,
    state_paths: Mapping[str, Path],
    *,
    hash_inputs: bool = False,
) -> dict[str, Any]:
    """Run the bounded public-validation H diagnostic for original/enriched."""

    if tuple(state_paths) != DISTRIBUTION_ORDER:
        raise AttentionDiagnosticError(
            f"state paths must be supplied in exact order {DISTRIBUTION_ORDER!r}"
        )
    observations, valid_mask, input_metadata = _load_public_validation_h(validation_manifest)
    input_metadata["manifest_file"] = _file_record(
        _regular_file(validation_manifest, label="validation manifest"),
        hash_bytes=hash_inputs,
    )
    for descriptor_key in ("observation", "validity_mask"):
        descriptor = input_metadata[descriptor_key]
        path_value = descriptor.get("path")
        if isinstance(path_value, str):
            descriptor["file"] = _file_record(
                _regular_file(Path(path_value), label=f"validation {descriptor_key}"),
                hash_bytes=hash_inputs,
            )

    state_results: dict[str, Any] = {}
    for distribution in DISTRIBUTION_ORDER:
        state_path = _regular_file(state_paths[distribution], label=f"{distribution} causal state")
        metadata = _state_metadata(state_path, distribution=distribution)
        try:
            model = load_decoder_state(
                state_path,
                method_id=CAUSAL_ATTENTION_METHOD,
                hidden_size=int(observations.shape[-1]),
                vocabulary_size=VOCAB_SIZE,
                context_width=DEFAULT_CONTEXT_WIDTH,
            )
        except Exception as exc:
            if isinstance(exc, AttentionDiagnosticError):
                raise
            raise AttentionDiagnosticError(
                f"cannot load {distribution} causal decoder state: {state_path}"
            ) from exc
        model.eval()
        with torch.inference_mode():
            weights = _causal_attention_weights(model, observations, valid_mask)
            summary = summarize_attention(weights, valid_mask)
        state_results[distribution] = {
            "state_file": _file_record(state_path, hash_bytes=hash_inputs),
            "metadata": metadata,
            "method_id": CAUSAL_ATTENTION_METHOD,
            "attention_mode": "causal",
            "attention_score_mode": model.attention_score_mode,
            "summary": summary,
        }
        del weights, model

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_VALIDATION_H_ONLY",
        "truth_accessed": False,
        "embedding_table_loaded": False,
        "state_mutated": False,
        "new_fitting": False,
        "input": input_metadata,
        "states": state_results,
    }


def _parse_state(value: str) -> tuple[str, Path]:
    distribution, separator, raw_path = value.partition("=")
    if not separator or distribution not in DISTRIBUTION_ORDER or not raw_path:
        raise argparse.ArgumentTypeError(
            "state must use DISTRIBUTION=PATH with distribution original or enriched"
        )
    return distribution, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize causal attention weights on public validation H only."
    )
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--state",
        action="append",
        type=_parse_state,
        required=True,
        metavar="DISTRIBUTION=PATH",
        help="repeat exactly for original and enriched selected causal states",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="hash manifest, H/mask files, and state files for the evidence receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state_paths: dict[str, Path] = {}
    for distribution, path in args.state:
        if distribution in state_paths:
            raise AttentionDiagnosticError(f"duplicate causal state: {distribution}")
        state_paths[distribution] = path
    if set(state_paths) != set(DISTRIBUTION_ORDER):
        raise AttentionDiagnosticError(
            f"exactly one causal state is required for each distribution: {DISTRIBUTION_ORDER!r}"
        )
    ordered_paths = {distribution: state_paths[distribution] for distribution in DISTRIBUTION_ORDER}
    result = run_attention_diagnostic(
        args.validation_manifest,
        ordered_paths,
        hash_inputs=bool(args.hash_inputs),
    )
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise AttentionDiagnosticError(f"diagnostic output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AttentionDiagnosticError as exc:
        raise SystemExit(f"trr0005 attention diagnostic failed: {exc}") from exc
