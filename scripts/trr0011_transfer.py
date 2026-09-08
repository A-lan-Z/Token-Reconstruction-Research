"""Controlled-target transfer diagnostics and truth-free runner bridge.

This module deliberately does not select records, open source text, load a
target model, or read evaluation truth.  A target-side producer supplies
paired cut activations and the frozen decoder supplies projected features and
top-two public-readout information.  The functions here summarize transfer
geometry only.  ``evaluator_only_transfer_inventory`` is a separate,
explicitly evaluator-only helper for use after the public gate and truth
freeze.

The selected TRR-0010 current-fixed state is loaded by
``scripts.trr0010_p09_fixed_loader.load_p09_fixed_state`` as the 15-tensor
``ResidualMLPPositionwiseDecoder``.  Its serialized state has a scalar logit
scale and no vocabulary gain or bias tensors; after the residual decoder's
feature projection it scores ``z_j = exp(s) * <u, E_j>``.  The diagnostic
therefore reports the clean predicted-class versus clean runner-up boundary
only.  It does not claim that the runner-up is the nearest boundary among all
vocabulary classes, and it is not a deployment certificate or a fitted
threshold.

The small runner bridge validates hashed transfer observations before calling
the existing TRR-0010 registration/decoder loader.  It is truth-free and
does not replace the frozen TRR-0010 public-output gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TASK_ID = "TRR-0011"
SCHEMA = "token-reconstruction.trr0011-controlled-target-transfer.v1"
CUT_DEPTH = 4
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127

VARIANT_ROLES = (
    "clean_public_target",
    "early_prefix_perturbation",
    "near_cut_prefix_perturbation",
    "after_cut_suffix_null_control",
    "historical_trained_benchmark_target_condition",
)
PERTURBATION_AMPLITUDES = (1e-3, 1e-2)
P09_METHOD_ID = "continued_fixed_readout"
P09_LOADER_MODULE = "scripts.trr0010_p09_fixed_loader"
P09_TENSOR_KEY_COUNT = 15
TRANSFER_RUN_SCHEMA = "token-reconstruction.trr0011-transfer-public-inference.v1"
TRANSFER_RUN_STATUS = "PUBLIC_TRANSFER_INPUTS_VALIDATED_BEFORE_TRUTH"
TRANSFER_MATRIX_SCHEMA = "token-reconstruction.trr0011-transfer-public-prediction-matrix.v1"
TRANSFER_MATRIX_STATUS = "PUBLIC_TRANSFER_PREDICTIONS_COMPLETE_BEFORE_TRUTH"
TRANSFER_METHOD_ORDER = ("expanded_fixed", "current_fixed")
TRANSFER_DOMAIN_ORDER = ("finance", "pile")
TRANSFER_RECORDS_PER_DOMAIN = 32
TRANSFER_EXPECTED_VARIANT_COUNT = 7
FIXED_AUC_SCORE_NAMES = (
    "margin_normalized_feature_shift",
    "raw_activation_l2",
    "relative_raw_activation_l2",
    "projected_feature_l2",
    "parameter_relative_l2",
)
TRANSFER_VARIANT_ORDER = (
    "clean_public_base",
    "early_prefix_eps1e3",
    "early_prefix_eps1e2",
    "near_cut_prefix_eps1e3",
    "near_cut_prefix_eps1e2",
    "after_cut_suffix_eps1e2_null",
    "historical_trained_benchmark_public_lora_2601",
)
TRANSFER_VARIANT_ALIASES = {
    "clean_public": "clean_public_base",
    "layer1_block_rel1e-3": "early_prefix_eps1e3",
    "layer1_block_rel1e-2": "early_prefix_eps1e2",
    "layer3_block_rel1e-3": "near_cut_prefix_eps1e3",
    "layer3_block_rel1e-2": "near_cut_prefix_eps1e2",
    "layer4_null_block_rel1e-2": "after_cut_suffix_eps1e2_null",
    "historical_public_lora_2601": "historical_trained_benchmark_public_lora_2601",
}
CAPTURE_MANIFEST_SCHEMA = "token-reconstruction.trr0011-controlled-target-capture-manifest.v1"
TRUTH_FLAGS = (
    "truth_opened",
    "source_text_loaded",
    "target_labels_loaded",
    "candidate_arrays_persisted",
    "private_or_truth_payload_read",
    "fresh_evaluation_started",
)


class TransferDiagnosticError(ValueError):
    """Raised when a transfer diagnostic input is incomplete or unsafe."""


def _as_float_tensor(value: Any, *, label: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value).detach().cpu().float().contiguous()
    except Exception as exc:  # pragma: no cover - torch error wording varies
        raise TransferDiagnosticError(f"{label} is not tensor-like") from exc
    if tensor.numel() == 0:
        raise TransferDiagnosticError(f"{label} is empty")
    if not torch.isfinite(tensor).all().item():
        raise TransferDiagnosticError(f"{label} contains non-finite values")
    return tensor


def _flatten_features(value: Any, *, label: str) -> torch.Tensor:
    tensor = _as_float_tensor(value, label=label)
    if tensor.ndim == 2:
        if tensor.shape[1] <= 0:
            raise TransferDiagnosticError(f"{label} has an invalid hidden dimension")
        return tensor
    if tensor.ndim == 3:
        if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise TransferDiagnosticError(f"{label} has invalid record geometry")
        return tensor.reshape(-1, tensor.shape[-1]).contiguous()
    raise TransferDiagnosticError(f"{label} must be [rows, hidden] or [records, positions, hidden]")


def _validate_paired_feature_geometry(left: Any, right: Any, *, left_label: str, right_label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve record/position pairing before flattening diagnostic rows."""

    left_tensor = _as_float_tensor(left, label=left_label)
    right_tensor = _as_float_tensor(right, label=right_label)
    if left_tensor.ndim != right_tensor.ndim or tuple(left_tensor.shape) != tuple(right_tensor.shape):
        raise TransferDiagnosticError(
            f"{left_label} and {right_label} must have identical record/position geometry"
        )
    if left_tensor.ndim not in (2, 3):
        raise TransferDiagnosticError(
            f"{left_label} and {right_label} must be [rows, hidden] or [records, positions, hidden]"
        )
    if left_tensor.shape[-1] <= 0 or (left_tensor.ndim == 3 and left_tensor.shape[1] <= 0):
        raise TransferDiagnosticError(f"{left_label} has invalid paired geometry")
    return left_tensor, right_tensor


def _flatten_vector(value: Any, *, label: str) -> torch.Tensor:
    tensor = _as_float_tensor(value, label=label).reshape(-1).contiguous()
    return tensor


def _flatten_ids(value: Any, *, label: str) -> torch.Tensor:
    try:
        raw = torch.as_tensor(value).detach().cpu().contiguous()
    except Exception as exc:  # pragma: no cover
        raise TransferDiagnosticError(f"{label} is not tensor-like") from exc
    if raw.numel() == 0 or raw.ndim not in (1, 2):
        raise TransferDiagnosticError(f"{label} must be a non-empty vector or matrix")
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TransferDiagnosticError(f"{label} must use an integer dtype")
    result = raw.reshape(-1).to(dtype=torch.long)
    if result.lt(0).any().item() or result.ge(VOCABULARY_SIZE).any().item():
        raise TransferDiagnosticError(f"{label} contains an invalid vocabulary ID")
    return result


def _require_same_rows(named: Mapping[str, torch.Tensor]) -> int:
    counts = {name: int(value.shape[0]) for name, value in named.items()}
    if len(set(counts.values())) != 1:
        raise TransferDiagnosticError(f"paired diagnostic row counts differ: {counts}")
    return next(iter(counts.values()))


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise TransferDiagnosticError("cannot summarize an empty metric")
    index = max(0, min(len(ordered) - 1, int(math.ceil(probability * len(ordered)) - 1)) )
    return float(ordered[index])


def _summary(values: torch.Tensor, *, label: str) -> dict[str, Any]:
    finite = values.detach().cpu().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"status": "UNKNOWN", "label": label, "valid_rows": 0}
    values_list = [float(value) for value in finite.tolist()]
    return {
        "status": "COMPUTED",
        "label": label,
        "valid_rows": len(values_list),
        "mean": float(sum(values_list) / len(values_list)),
        "median": _nearest_rank(values_list, 0.5),
        "p90": _nearest_rank(values_list, 0.9),
        "maximum": float(max(values_list)),
        "percentile_convention": "nearest-rank index ceil(p*n)-1",
    }


def tensor_digest(value: torch.Tensor) -> str:
    """Hash dtype, shape, and contiguous CPU bytes for a receipt."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_variant_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a truth-free controlled-target variant declaration.

    The declaration describes target-side production work; it contains no
    target weights.  A real run must bind the producer's resource and code
    hashes before observations are admitted to the decoder.
    """

    if not isinstance(plan, Mapping):
        raise TransferDiagnosticError("variant plan must be a mapping")
    if plan.get("schema") != SCHEMA or plan.get("task_id") != TASK_ID:
        raise TransferDiagnosticError("variant plan schema or task identity changed")
    if plan.get("cut_depth") != CUT_DEPTH:
        raise TransferDiagnosticError("variant plan cut depth must remain four")
    geometry = plan.get("geometry")
    expected_geometry = {
        "hidden_size": HIDDEN_SIZE,
        "vocabulary_size": VOCABULARY_SIZE,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "bos_token_id": BOS_TOKEN_ID,
    }
    if not isinstance(geometry, Mapping) or any(geometry.get(key) != value for key, value in expected_geometry.items()):
        raise TransferDiagnosticError("variant plan geometry is not the frozen decoder geometry")
    for flag in TRUTH_FLAGS:
        value = plan.get(flag)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise TransferDiagnosticError(f"variant plan records forbidden access: {flag}")

    variants = plan.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes, bytearray)):
        raise TransferDiagnosticError("variant plan variants are absent")
    rows: dict[str, dict[str, Any]] = {}
    for item in variants:
        if not isinstance(item, Mapping):
            raise TransferDiagnosticError("variant row is malformed")
        variant_id = item.get("variant_id")
        role = item.get("role")
        if not isinstance(variant_id, str) or not isinstance(role, str) or role not in VARIANT_ROLES:
            raise TransferDiagnosticError("variant row has an unknown role")
        if variant_id in rows:
            raise TransferDiagnosticError(f"duplicate variant: {variant_id}")
        rows[variant_id] = dict(item)
        location = item.get("update_location")
        if role == "clean_public_target" and location not in (None, "none"):
            raise TransferDiagnosticError("clean target cannot declare a weight update")
        if role == "early_prefix_perturbation":
            if not isinstance(location, Mapping) or location.get("layer_index") not in (0, 1, 2):
                raise TransferDiagnosticError("early perturbation must target a layer before the cut")
        if role == "near_cut_prefix_perturbation":
            if not isinstance(location, Mapping) or location.get("layer_index") != CUT_DEPTH - 1:
                raise TransferDiagnosticError("near-cut perturbation must target layer three")
        if role in {"early_prefix_perturbation", "near_cut_prefix_perturbation"}:
            location = item.get("update_location")
            if not isinstance(location, Mapping):
                raise TransferDiagnosticError(f"{role} must declare an update location")
            amplitude = location.get("relative_parameter_l2")
            if not isinstance(amplitude, (int, float)) or isinstance(amplitude, bool):
                raise TransferDiagnosticError(f"{role} must declare a numeric perturbation amplitude")
            if not math.isclose(float(amplitude), float(PERTURBATION_AMPLITUDES[0]), rel_tol=0.0, abs_tol=1e-12) and not math.isclose(float(amplitude), float(PERTURBATION_AMPLITUDES[1]), rel_tol=0.0, abs_tol=1e-12):
                raise TransferDiagnosticError(f"{role} amplitude is not one of the two preregistered levels")
        if role == "after_cut_suffix_null_control":
            if not isinstance(location, Mapping) or int(location.get("layer_index", -1)) < CUT_DEPTH:
                raise TransferDiagnosticError("after-cut null must target a suffix layer")
            if item.get("expected_activation_equivalence") is not True:
                raise TransferDiagnosticError("after-cut null must require exact H equivalence")
            amplitude = location.get("relative_parameter_l2")
            if not isinstance(amplitude, (int, float)) or not math.isclose(float(amplitude), float(PERTURBATION_AMPLITUDES[1]), rel_tol=0.0, abs_tol=1e-12):
                raise TransferDiagnosticError("after-cut null must use the fixed 1e-2 null perturbation")
        if role == "historical_trained_benchmark_target_condition":
            if item.get("artificial") is not False:
                raise TransferDiagnosticError("historical trained benchmark target must be labelled non-artificial")
            if item.get("independent_target") is not False:
                raise TransferDiagnosticError("historical trained benchmark target cannot be labelled independent")
            if item.get("available_for_run") is not True:
                raise TransferDiagnosticError("historical trained benchmark target must bind its published read-only asset")
            if not isinstance(item.get("availability_reason"), str) or not item["availability_reason"]:
                raise TransferDiagnosticError("historical trained benchmark target needs an availability reason")
            asset_binding = item.get("update_binding")
            if not isinstance(asset_binding, Mapping) or asset_binding.get("readonly") is not True:
                raise TransferDiagnosticError("historical trained benchmark target needs a readonly asset binding")
    roles = {row["role"] for row in rows.values()}
    missing = [role for role in VARIANT_ROLES if role not in roles]
    if missing:
        raise TransferDiagnosticError(f"variant plan is missing roles: {missing}")
    for role in ("early_prefix_perturbation", "near_cut_prefix_perturbation"):
        amplitudes = sorted(float(row["update_location"]["relative_parameter_l2"]) for row in rows.values() if row["role"] == role)
        expected = sorted(float(value) for value in PERTURBATION_AMPLITUDES)
        if amplitudes != expected:
            raise TransferDiagnosticError(f"{role} must include each preregistered amplitude exactly once")
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "cut_depth": CUT_DEPTH,
        "variant_ids": list(rows),
        "roles": sorted(roles),
        "preregistered_amplitudes": list(PERTURBATION_AMPLITUDES),
        "historical_trained_benchmark_target_available": True,
        "truth_free": True,
    }


def validate_panel_descriptor(panel: Mapping[str, Any], *, strict_reservation: bool = False) -> dict[str, Any]:
    """Validate a reserved, source-paired panel descriptor without selecting it.

    A literal pending marker is never accepted as a reservation.  The
    ``strict_reservation`` switch is used by production matrix callers once
    the owner supplies a complete reservation status/descriptor; the compact
    synthetic fixture intentionally omits those optional fields.
    """

    if not isinstance(panel, Mapping):
        raise TransferDiagnosticError("panel descriptor must be a mapping")
    count = panel.get("records_per_domain")
    if not isinstance(count, int) or isinstance(count, bool) or not 32 <= count <= 64:
        raise TransferDiagnosticError("panel must declare 32 to 64 records per domain")
    if panel.get("same_record_order_across_targets") is not True:
        raise TransferDiagnosticError("panel is not paired across targets")
    reservation = panel.get("opaque_reservation")
    if not isinstance(reservation, Mapping):
        # Capture manifests carry the checked receipt as a flattened field.
        flattened = panel.get("opaque_reservation_sha256")
        if isinstance(flattened, str):
            reservation = {"receipt_sha256": flattened, "status": panel.get("status")}
    if not isinstance(reservation, Mapping):
        raise TransferDiagnosticError("panel lacks an opaque reservation receipt")
    receipt = reservation.get("receipt_sha256")
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or receipt.lower() != receipt
        or any(char not in "0123456789abcdef" for char in receipt)
        or receipt.startswith("pending")
    ):
        raise TransferDiagnosticError("panel opaque reservation receipt is pending or malformed")
    reservation_status = reservation.get("status", panel.get("status"))
    if isinstance(reservation_status, str) and (
        reservation_status.upper().startswith("PENDING")
        or "PENDING" in reservation_status.upper()
    ):
        raise TransferDiagnosticError("panel opaque reservation is still pending")
    if strict_reservation and (
        not isinstance(reservation_status, str)
        or reservation_status not in {
            "RESERVED_OPAQUE_PANEL_BEFORE_CAPTURE",
            "RESERVED",
            "OPAQUE_RESERVATION_COMPLETE",
        }
    ):
        raise TransferDiagnosticError("panel lacks a completed opaque reservation status")
    for key in ("descriptor_sha256", "panel_sha256", "selection_descriptor_sha256"):
        value = reservation.get(key, panel.get(key))
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or value.lower() != value
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise TransferDiagnosticError(f"panel {key} binding is malformed")
    if panel.get("selection_performed") is True:
        raise TransferDiagnosticError("diagnostic module cannot perform source selection")
    for flag in TRUTH_FLAGS:
        value = panel.get(flag)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise TransferDiagnosticError(f"panel descriptor records forbidden access: {flag}")
    return {
        "records_per_domain": count,
        "paired_targets": True,
        "selection_performed": False,
        "opaque_reservation_sha256": receipt,
        "reservation_status": reservation_status,
    }


def summarize_transfer(
    clean_activation: Any,
    changed_activation: Any,
    clean_feature: Any,
    changed_feature: Any,
    clean_top_logits: Any,
    clean_runner_logits: Any,
    readout_pairs: Any,
    *,
    logit_scale: float,
    clean_top_ids: Any | None = None,
    changed_top_ids: Any | None = None,
    parameter_delta_l2: float | None = None,
    parameter_base_l2: float | None = None,
    validate_readout_equation: bool = False,
) -> dict[str, Any]:
    """Summarize truth-free transfer geometry for one target variant.

    ``readout_pairs`` contains the effective public readout rows for the clean
    top class and clean runner-up, shaped ``[rows, 2, hidden]``.  Keeping only
    these rows avoids materializing a second full-vocabulary logit matrix.
    """

    clean_h_raw, changed_h_raw = _validate_paired_feature_geometry(
        clean_activation, changed_activation,
        left_label="clean activation", right_label="changed activation",
    )
    clean_u_raw, changed_u_raw = _validate_paired_feature_geometry(
        clean_feature, changed_feature,
        left_label="clean projected feature", right_label="changed projected feature",
    )
    if tuple(clean_h_raw.shape) != tuple(clean_u_raw.shape):
        raise TransferDiagnosticError(
            "activation and projected feature record/position geometry differ"
        )
    clean_h = clean_h_raw.reshape(-1, clean_h_raw.shape[-1]).contiguous()
    changed_h = changed_h_raw.reshape(-1, changed_h_raw.shape[-1]).contiguous()
    clean_u = clean_u_raw.reshape(-1, clean_u_raw.shape[-1]).contiguous()
    changed_u = changed_u_raw.reshape(-1, changed_u_raw.shape[-1]).contiguous()
    pairs = _as_float_tensor(readout_pairs, label="clean readout rows")
    if pairs.ndim != 3 or pairs.shape[1] != 2:
        raise TransferDiagnosticError("clean readout rows must be [rows, 2, hidden]")
    pairs = pairs.contiguous()
    rows = _require_same_rows({
        "clean activation": clean_h,
        "changed activation": changed_h,
        "clean projected feature": clean_u,
        "changed projected feature": changed_u,
        "clean readout rows": pairs.reshape(pairs.shape[0], -1),
    })
    if int(clean_h.shape[1]) != int(clean_u.shape[1]) or int(clean_h.shape[1]) != int(pairs.shape[2]):
        raise TransferDiagnosticError("activation, feature, and readout hidden sizes differ")
    top_logits = _flatten_vector(clean_top_logits, label="clean top logits")
    runner_logits = _flatten_vector(clean_runner_logits, label="clean runner-up logits")
    _require_same_rows({"top logits": top_logits[:, None], "runner logits": runner_logits[:, None]})
    if int(top_logits.shape[0]) != rows or int(runner_logits.shape[0]) != rows:
        raise TransferDiagnosticError("clean margin rows differ from transfer rows")
    scale = float(logit_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise TransferDiagnosticError("logit scale must be finite and positive")
    if (top_logits < runner_logits).any().item():
        raise TransferDiagnosticError("clean top logits are not sorted above runner-up logits")
    clean_feature_norm = torch.linalg.vector_norm(clean_u, dim=-1)
    changed_feature_norm = torch.linalg.vector_norm(changed_u, dim=-1)
    unit_tolerance = 2e-3
    if not torch.allclose(clean_feature_norm, torch.ones_like(clean_feature_norm), rtol=unit_tolerance, atol=unit_tolerance):
        raise TransferDiagnosticError("clean projected features are not normalized decoder outputs")
    if not torch.allclose(changed_feature_norm, torch.ones_like(changed_feature_norm), rtol=unit_tolerance, atol=unit_tolerance):
        raise TransferDiagnosticError("changed projected features are not normalized decoder outputs")
    if validate_readout_equation:
        expected_top = scale * (clean_u * pairs[:, 0]).sum(dim=-1)
        expected_runner = scale * (clean_u * pairs[:, 1]).sum(dim=-1)
        equation_tolerance = 2e-4
        if not torch.allclose(top_logits, expected_top, rtol=equation_tolerance, atol=equation_tolerance):
            raise TransferDiagnosticError("clean top logits do not match the bound decoder/readout equation")
        if not torch.allclose(runner_logits, expected_runner, rtol=equation_tolerance, atol=equation_tolerance):
            raise TransferDiagnosticError("clean runner logits do not match the bound decoder/readout equation")

    raw_shift = torch.linalg.vector_norm(changed_h - clean_h, dim=-1)
    clean_h_norm = torch.linalg.vector_norm(clean_h, dim=-1).clamp_min(1e-12)
    relative_raw_shift = raw_shift / clean_h_norm
    feature_shift = torch.linalg.vector_norm(changed_u - clean_u, dim=-1)
    cosine_similarity = torch.nn.functional.cosine_similarity(clean_u, changed_u, dim=-1, eps=1e-12)
    cosine_distance = 1.0 - cosine_similarity
    margin = top_logits - runner_logits
    class_difference_norm = torch.linalg.vector_norm(pairs[:, 0] - pairs[:, 1], dim=-1)
    denominator = class_difference_norm * scale
    fp32_scaled_tolerance = 8.0 * torch.finfo(torch.float32).eps * torch.maximum(
        torch.ones_like(margin), torch.maximum(top_logits.abs(), runner_logits.abs())
    )
    denominator_tolerance = 8.0 * torch.finfo(torch.float32).eps * torch.maximum(torch.ones_like(denominator), torch.full_like(denominator, abs(scale)))
    valid_margin = (margin > fp32_scaled_tolerance) & (denominator > denominator_tolerance)
    hyperplane_distance = torch.full_like(margin, float("nan"))
    hyperplane_distance[valid_margin] = margin[valid_margin] / denominator[valid_margin]
    normalized_margin_ratio = torch.full_like(margin, float("nan"))
    normalized_margin_ratio[valid_margin] = feature_shift[valid_margin] / hyperplane_distance[valid_margin]
    ratio_summary = _summary(normalized_margin_ratio, label="feature shift / clean hyperplane distance")
    if int((~valid_margin).sum().item()) > 0:
        ratio_summary.update({
            "status": "UNKNOWN",
            "reason": "one or more rows have a margin or readout denominator at/below the FP32 tolerance",
            "valid_rows": int(valid_margin.sum().item()),
            "invalid_rows": int((~valid_margin).sum().item()),
        })

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPUTED",
        "rows": rows,
        "metrics": {
            "raw_activation_l2": _summary(raw_shift, label="||H_changed-H_clean||_2"),
            "relative_raw_activation_l2": _summary(relative_raw_shift, label="raw activation relative L2"),
            "projected_feature_l2": _summary(feature_shift, label="||u_changed-u_clean||_2"),
            "projected_feature_cosine_distance": _summary(cosine_distance, label="1-cosine(u_changed,u_clean)"),
            "clean_predicted_margin": _summary(margin, label="clean predicted-class logit margin"),
            "clean_hyperplane_distance": _summary(hyperplane_distance, label="clean predicted-class hyperplane distance"),
            "margin_normalized_feature_shift": ratio_summary,
        },
        "margin_geometry": {
            "equation": "(z_top-z_runner)/(exp(s)*||E_top-E_runner||_2)",
            "decoder_loader": P09_LOADER_MODULE,
            "decoder_state_tensor_key_count": P09_TENSOR_KEY_COUNT,
            "decoder_state_has_per_token_gain_or_bias": False,
            "comparison_scope": "clean predicted class versus clean runner-up only",
            "all_class_boundaries_evaluated": False,
            "nearest_boundary_claim": False,
            "certificate": False,
            "logit_scale": scale,
            "valid_margin_rows": int(valid_margin.sum().item()),
            "invalid_margin_rows": int((~valid_margin).sum().item()),
            "near_zero_margin_rows": int((margin <= fp32_scaled_tolerance).sum().item()),
            "near_zero_denominator_rows": int((denominator <= denominator_tolerance).sum().item()),
            "conditioning_status": "COMPUTED" if bool(valid_margin.all().item()) else "UNKNOWN",
            "fixed_ratio_reference": 1.0,
            "fp32_scaled_margin_tolerance": "8*eps_float32*max(1,abs(clean_top_logit),abs(clean_runner_logit))",
            "fp32_scaled_denominator_tolerance": "8*eps_float32*max(1,abs(logit_scale))",
            "threshold_fitted": False,
        },
        "parameter_distance": _parameter_distance(parameter_delta_l2, parameter_base_l2),
        "prediction_change": _prediction_change(clean_top_ids, changed_top_ids, rows=rows),
        "truth_free": True,
        "truth_dependent": False,
    }
    return result


def _parameter_distance(delta: float | None, base: float | None) -> dict[str, Any]:
    if delta is None or base is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "target-prefix parameter distance remains producer-side metadata",
            "deployable": False,
        }
    delta_value = float(delta)
    base_value = float(base)
    if not math.isfinite(delta_value) or not math.isfinite(base_value) or delta_value < 0.0 or base_value <= 0.0:
        raise TransferDiagnosticError("target-prefix parameter distances must be finite, delta>=0, base>0")
    return {
        "status": "COMPUTED",
        "delta_l2": delta_value,
        "base_l2": base_value,
        "relative_l2": delta_value / base_value,
        "deployable": False,
        "source": "target-side producer binding",
    }


def _prediction_change(clean_ids: Any | None, changed_ids: Any | None, *, rows: int) -> dict[str, Any]:
    if clean_ids is None or changed_ids is None:
        return {"status": "UNAVAILABLE", "reason": "top-ID arrays were not supplied"}
    left = _flatten_ids(clean_ids, label="clean top IDs")
    right = _flatten_ids(changed_ids, label="changed top IDs")
    if int(left.numel()) != rows or int(right.numel()) != rows:
        raise TransferDiagnosticError("prediction top-ID rows differ from transfer rows")
    changed = left.ne(right)
    return {
        "status": "COMPUTED",
        "changed_rows": int(changed.sum().item()),
        "unchanged_rows": int((~changed).sum().item()),
        "fraction_changed": float(changed.float().mean().item()),
    }


def validate_after_cut_null(
    clean_activation: Any,
    null_activation: Any,
    clean_prediction: Any | None = None,
    null_prediction: Any | None = None,
) -> dict[str, Any]:
    """Require exact equality for the suffix perturbation null control."""

    clean = _as_float_tensor(clean_activation, label="clean null-control activation")
    null = _as_float_tensor(null_activation, label="after-cut null activation")
    if clean.shape != null.shape:
        raise TransferDiagnosticError("after-cut null activation geometry changed")
    if not torch.equal(clean, null):
        raise TransferDiagnosticError("after-cut null changed the cut activation")
    result = {"status": "PASS", "activation_exact_equal": True}
    if clean_prediction is not None or null_prediction is not None:
        if clean_prediction is None or null_prediction is None:
            raise TransferDiagnosticError("null-control predictions must be supplied together")
        left = _flatten_ids(clean_prediction, label="clean null-control prediction")
        right = _flatten_ids(null_prediction, label="after-cut null prediction")
        if left.shape != right.shape or not torch.equal(left, right):
            raise TransferDiagnosticError("after-cut null changed the frozen decoder prediction")
        result["prediction_exact_equal"] = True
    return result


def evaluator_only_transfer_inventory(
    baseline_prediction: Any,
    changed_prediction: Any,
    truth_tokens: Any,
) -> dict[str, Any]:
    """Count baseline-wrong, broken, and improved rows after the truth gate.

    This is intentionally evaluator-only.  It accepts prediction IDs only
    after the caller has passed the frozen public gate and opened the separate
    truth sidecar.  The returned counts are paired at the same source/position
    unit and are never a deployable transfer signal.
    """

    baseline = _flatten_ids(baseline_prediction, label="baseline prediction")
    changed = _flatten_ids(changed_prediction, label="changed prediction")
    truth = _flatten_ids(truth_tokens, label="truth tokens")
    if baseline.numel() != changed.numel() or baseline.numel() != truth.numel():
        raise TransferDiagnosticError("paired transfer inventory tensors have different sizes")
    baseline_correct = baseline.eq(truth)
    changed_correct = changed.eq(truth)
    baseline_wrong = ~baseline_correct
    changed_wrong = ~changed_correct
    broken = baseline_correct & changed_wrong
    improved = baseline_wrong & changed_correct
    unchanged_correct = baseline_correct & changed_correct
    unchanged_wrong = baseline_wrong & changed_wrong
    total = int(baseline.numel())
    return {
        "status": "COMPUTED_AFTER_TRUTH_GATE",
        "truth_dependent": True,
        "deployable": False,
        "denominator": total,
        "baseline_wrong": int(baseline_wrong.sum().item()),
        "changed_wrong": int(changed_wrong.sum().item()),
        "broken": int(broken.sum().item()),
        "improved": int(improved.sum().item()),
        "unchanged_correct": int(unchanged_correct.sum().item()),
        "unchanged_wrong": int(unchanged_wrong.sum().item()),
        "baseline_changed_rows": int(baseline.ne(changed).sum().item()),
        "baseline_wrong_rate": float(baseline_wrong.float().mean().item()),
        "broken_rate": float(broken.float().mean().item()),
        "improved_rate": float(improved.float().mean().item()),
    }



def evaluator_only_fixed_auc(
    score: Any,
    baseline_prediction: Any,
    changed_prediction: Any,
    truth_tokens: Any,
    *,
    score_name: str,
) -> dict[str, Any]:
    """Compute one predeclared, fixed-direction distortion AUC after truth.

    The risk set is the baseline-correct population.  Within that set the
    positive class is fixed as ``broken`` (changed becomes wrong) and the
    negative class is ``remaining_correct`` (changed remains correct).
    Baseline-wrong rows, including improvements, are excluded from this
    breakage-risk AUC but remain in the paired error inventory.  There is no
    fitted threshold or outcome-based orientation.
    """

    if score_name not in FIXED_AUC_SCORE_NAMES:
        raise TransferDiagnosticError(f"unknown fixed AUC score: {score_name}")
    baseline = _flatten_ids(baseline_prediction, label="baseline prediction")
    changed = _flatten_ids(changed_prediction, label="changed prediction")
    truth = _flatten_ids(truth_tokens, label="truth tokens")
    values = _flatten_vector(score, label=f"fixed AUC score {score_name}")
    if baseline.numel() != changed.numel() or baseline.numel() != truth.numel() or values.numel() != baseline.numel():
        raise TransferDiagnosticError("fixed AUC rows differ from paired predictions")
    baseline_correct = baseline.eq(truth)
    changed_correct = changed.eq(truth)
    broken = baseline_correct & ~changed_correct
    remaining_correct = baseline_correct & changed_correct
    improved = ~baseline_correct & changed_correct
    n_broken = int(broken.sum().item())
    n_remaining_correct = int(remaining_correct.sum().item())
    n_improved_excluded = int(improved.sum().item())
    risk_set = baseline_correct
    if n_broken == 0 or n_remaining_correct == 0:
        return {
            "status": "UNKNOWN",
            "truth_dependent": True,
            "deployable": False,
            "score_name": score_name,
            "risk_set": "baseline_correct",
            "positive_label": "broken",
            "negative_label": "remaining_correct",
            "n_broken": n_broken,
            "n_remaining_correct": n_remaining_correct,
            "n_improved_excluded": n_improved_excluded,
            "reason": "baseline-correct risk set lacks both broken and remaining-correct classes",
        }
    selected_scores = values[risk_set]
    score_range = float(selected_scores.max().item() - selected_scores.min().item())
    near_zero_score_spread_epsilon = 1e-12
    if score_range <= near_zero_score_spread_epsilon:
        return {
            "status": "UNKNOWN",
            "truth_dependent": True,
            "deployable": False,
            "score_name": score_name,
            "risk_set": "baseline_correct",
            "positive_label": "broken",
            "negative_label": "remaining_correct",
            "n_broken": n_broken,
            "n_remaining_correct": n_remaining_correct,
            "n_improved_excluded": n_improved_excluded,
            "score_range": score_range,
            "near_zero_score_spread_epsilon": near_zero_score_spread_epsilon,
            "reason": "baseline-correct risk-set scores have no resolved variation",
        }
    selected_labels = broken[risk_set]
    order = torch.argsort(selected_scores, stable=True)
    sorted_scores = selected_scores[order]
    ranks = torch.empty_like(selected_scores)
    start = 0
    while start < int(sorted_scores.numel()):
        stop = start + 1
        while stop < int(sorted_scores.numel()) and sorted_scores[stop].item() == sorted_scores[start].item():
            stop += 1
        rank = (float(start + 1) + float(stop)) / 2.0
        ranks[order[start:stop]] = rank
        start = stop
    positive_rank_sum = float(ranks[selected_labels].sum().item())
    auc = (positive_rank_sum - (n_broken * (n_broken + 1) / 2.0)) / float(n_broken * n_remaining_correct)
    return {
        "status": "COMPUTED_AFTER_TRUTH_GATE",
        "truth_dependent": True,
        "deployable": False,
        "score_name": score_name,
        "risk_set": "baseline_correct",
        "positive_label": "broken",
        "negative_label": "remaining_correct",
        "higher_score_orientation_fixed": True,
        "auc": float(auc),
        "n_broken": n_broken,
        "n_remaining_correct": n_remaining_correct,
        "n_improved_excluded": n_improved_excluded,
        "risk_set_rows": n_broken + n_remaining_correct,
        "score_range": score_range,
        "near_zero_score_spread_epsilon": near_zero_score_spread_epsilon,
        "threshold_fitted": False,
    }


def evaluator_only_error_inventory(
    candidate_prediction: Any,
    comparator_prediction: Any,
    truth_tokens: Any,
) -> dict[str, Any]:
    """Summarize paired errors after truth opening; never a deployable signal."""

    transfer = evaluator_only_transfer_inventory(candidate_prediction, comparator_prediction, truth_tokens)
    return {
        **transfer,
        "both_correct": transfer["unchanged_correct"],
        "only_candidate_correct": transfer["improved"],
        "only_comparator_correct": transfer["broken"],
        "both_wrong": transfer["unchanged_wrong"],
        "candidate_comparator_discordance": transfer["broken"] + transfer["improved"],
    }




def _transfer_file_record(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    """Validate one create-only/read-only file binding for the transfer bridge."""

    if not isinstance(value, Mapping):
        raise TransferDiagnosticError(f"{description} binding is malformed")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TransferDiagnosticError(f"{description} path is absent")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() and value.get("readonly") is not True:
        raise TransferDiagnosticError(f"{description} symlink must be explicitly readonly")
    path = path.resolve()
    if not path.is_file():
        raise TransferDiagnosticError(f"{description} is unavailable: {path}")
    try:
        relative = path.relative_to(root)
    except ValueError:
        if value.get("readonly") is not True:
            raise TransferDiagnosticError(f"{description} outside the task root must be readonly")
        relative = None
    size = value.get("bytes")
    digest = value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TransferDiagnosticError(f"{description} byte count is malformed")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TransferDiagnosticError(f"{description} SHA-256 is malformed")
    actual_size = path.stat().st_size
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_size != size or actual_digest != digest:
        raise TransferDiagnosticError(f"{description} bytes or SHA-256 changed")
    result = dict(value)
    result.update({"path": str(path), "bytes": actual_size, "sha256": actual_digest})
    if relative is not None:
        result["relative_path"] = relative.as_posix()
    return result


def _load_transfer_json(path: Path) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferDiagnosticError(f"invalid transfer JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TransferDiagnosticError(f"transfer JSON must be an object: {path}")
    return payload


def _validate_transfer_observation_header(
    binding: Mapping[str, Any],
    *,
    records: int,
    root: Path,
    description: str,
) -> dict[str, Any]:
    """Check public H/mask/position headers without loading source text or truth."""

    checked = _transfer_file_record(binding, root=root, description=description)
    try:
        with safe_open(checked["path"], framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise TransferDiagnosticError(f"{description} tensor keys changed")
            activation = handle.get_slice("activations")
            mask = handle.get_slice("attention_mask")
            positions = handle.get_slice("position_ids")
            if tuple(activation.get_shape()) != (records, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
                raise TransferDiagnosticError(f"{description} activation geometry changed")
            if tuple(mask.get_shape()) != (records, STORED_SEQUENCE_TOKENS) or tuple(positions.get_shape()) != (records, STORED_SEQUENCE_TOKENS):
                raise TransferDiagnosticError(f"{description} sidecar geometry changed")
            if activation.get_dtype() != "BF16":
                raise TransferDiagnosticError(f"{description} activation dtype changed")
            if mask.get_dtype() not in {"BOOL", "U8", "I64"} or positions.get_dtype() not in {"I64", "I32"}:
                raise TransferDiagnosticError(f"{description} mask/position dtype changed")
            mask_values = mask[:].to(dtype=torch.bool)
            position_values = positions[:].to(dtype=torch.long)
            expected_positions = torch.arange(STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0).expand(records, -1)
            if not bool(mask_values.all().item()) or not torch.equal(position_values, expected_positions):
                raise TransferDiagnosticError(f"{description} mask or positions changed")
    except TransferDiagnosticError:
        raise
    except Exception as exc:
        raise TransferDiagnosticError(f"{description} safetensors header is unreadable") from exc
    return checked | {
        "records": records,
        "hidden_size": HIDDEN_SIZE,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "dtype": "BF16",
    }


def validate_transfer_observation_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate paired public transfer observations before any decoder call.

    The manifest binds source-order digests, actual safetensor headers, loader
    code, and decoder resources.  It carries no source text, token labels, or
    candidate arrays.  This is an input boundary for the runner bridge; the
    frozen TRR-0010 public-output gate remains authoritative for a complete
    prediction matrix.
    """

    if not isinstance(manifest, Mapping):
        raise TransferDiagnosticError("transfer manifest must be a mapping")
    if manifest.get("schema") != TRANSFER_RUN_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise TransferDiagnosticError("transfer manifest schema or task identity changed")
    if manifest.get("status") != TRANSFER_RUN_STATUS:
        raise TransferDiagnosticError("transfer manifest status is not pre-truth validated")
    for flag in TRUTH_FLAGS:
        value = manifest.get(flag)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise TransferDiagnosticError(f"transfer manifest records forbidden access: {flag}")
    method_id = manifest.get("method_id")
    if not isinstance(method_id, str) or not method_id:
        raise TransferDiagnosticError("transfer manifest method_id is absent")
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise TransferDiagnosticError(f"transfer repository root is unavailable: {root}")

    source_order = manifest.get("source_order_sha256")
    if not isinstance(source_order, Mapping) or not source_order:
        raise TransferDiagnosticError("transfer source-order digests are absent")
    for domain, digest in source_order.items():
        if not isinstance(domain, str) or not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise TransferDiagnosticError(f"source-order digest is malformed: {domain}")

    code_bindings = manifest.get("code_bindings")
    state_bindings = manifest.get("state_bindings")
    decoder_resources = manifest.get("decoder_resources")
    if not isinstance(code_bindings, Mapping) or not code_bindings:
        raise TransferDiagnosticError("transfer code bindings are absent")
    if not isinstance(state_bindings, Mapping) or not state_bindings:
        raise TransferDiagnosticError("transfer state bindings are absent")
    if not isinstance(decoder_resources, Mapping) or not {"embedding", "state", "loader"}.issubset(decoder_resources):
        raise TransferDiagnosticError("transfer decoder resources are incomplete")
    checked_code = {str(name): _transfer_file_record(binding, root=root, description=f"code {name}") for name, binding in code_bindings.items()}
    checked_state = {str(name): _transfer_file_record(binding, root=root, description=f"state {name}") for name, binding in state_bindings.items()}
    checked_resources = {str(name): _transfer_file_record(binding, root=root, description=f"decoder resource {name}") for name, binding in decoder_resources.items()}
    if checked_resources["loader"].get("module") not in (None, P09_LOADER_MODULE):
        raise TransferDiagnosticError("transfer decoder loader module changed")
    if method_id not in {"current_fixed", "expanded_fixed", "continued_fixed_readout"}:
        raise TransferDiagnosticError("transfer bridge only accepts the frozen fixed decoder methods")

    observation_bindings = manifest.get("observation_bindings")
    variant_ids = manifest.get("variant_ids")
    if not isinstance(observation_bindings, Mapping) or not observation_bindings:
        raise TransferDiagnosticError("transfer observation bindings are absent")
    if not isinstance(variant_ids, Sequence) or isinstance(variant_ids, (str, bytes, bytearray)):
        raise TransferDiagnosticError("transfer variant_ids are absent")
    expected_variants = [str(value) for value in variant_ids]
    if len(set(expected_variants)) != len(expected_variants) or set(expected_variants) != set(observation_bindings):
        raise TransferDiagnosticError("transfer variant binding set is incomplete or duplicated")
    checked_observations: dict[str, dict[str, Any]] = {}
    per_domain_digests: dict[str, set[str]] = {}
    for variant_id in expected_variants:
        domain_bindings = observation_bindings[variant_id]
        if not isinstance(domain_bindings, Mapping) or not domain_bindings:
            raise TransferDiagnosticError(f"transfer domains are absent: {variant_id}")
        checked_observations[variant_id] = {}
        for domain, raw_binding in domain_bindings.items():
            if domain not in source_order:
                raise TransferDiagnosticError(f"transfer domain lacks a source-order digest: {domain}")
            if not isinstance(raw_binding, Mapping):
                raise TransferDiagnosticError(f"transfer observation binding is malformed: {variant_id}/{domain}")
            records = raw_binding.get("records")
            if isinstance(records, bool) or not isinstance(records, int) or not 32 <= records <= 64:
                raise TransferDiagnosticError(f"transfer records must be 32 to 64: {variant_id}/{domain}")
            record_digest = raw_binding.get("record_ids_sha256")
            if record_digest != source_order[domain]:
                raise TransferDiagnosticError(f"transfer source order changed: {variant_id}/{domain}")
            checked = _validate_transfer_observation_header(raw_binding, records=records, root=root, description=f"transfer observation {variant_id}/{domain}")
            checked["domain"] = domain
            checked["record_ids_sha256"] = record_digest
            checked_observations[variant_id][str(domain)] = checked
            per_domain_digests.setdefault(str(domain), set()).add(str(record_digest))
    if any(len(digests) != 1 for digests in per_domain_digests.values()):
        raise TransferDiagnosticError("paired transfer variants do not share source order")
    return {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "status": TRANSFER_RUN_STATUS,
        "method_id": method_id,
        "variant_ids": expected_variants,
        "source_order_sha256": {str(key): str(value) for key, value in source_order.items()},
        "code_bindings": checked_code,
        "state_bindings": checked_state,
        "decoder_resources": checked_resources,
        "observation_bindings": checked_observations,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def _capture_variant_id(raw_variant_id: str) -> str:
    try:
        return TRANSFER_VARIANT_ALIASES[str(raw_variant_id)]
    except KeyError as exc:
        raise TransferDiagnosticError(f"capture manifest has an unknown transfer variant: {raw_variant_id}") from exc


def _capture_source_order_digests(panel: Mapping[str, Any]) -> dict[str, str]:
    records = panel.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise TransferDiagnosticError("capture panel does not expose opaque ordered IDs")
    grouped: dict[str, list[tuple[int, str]]] = {domain: [] for domain in TRANSFER_DOMAIN_ORDER}
    for raw in records:
        if not isinstance(raw, Mapping) or raw.get("domain") not in grouped:
            raise TransferDiagnosticError("capture panel opaque record binding is malformed")
        order = raw.get("order")
        opaque_id = raw.get("opaque_id")
        if isinstance(order, bool) or not isinstance(order, int) or not isinstance(opaque_id, str) or not opaque_id:
            raise TransferDiagnosticError("capture panel opaque record order/ID is malformed")
        grouped[str(raw["domain"])].append((int(order), opaque_id))
    result: dict[str, str] = {}
    for domain, rows in grouped.items():
        if len(rows) != TRANSFER_RECORDS_PER_DOMAIN or sorted(order for order, _ in rows) != list(range(TRANSFER_RECORDS_PER_DOMAIN)):
            raise TransferDiagnosticError(f"capture panel record order is incomplete: {domain}")
        digest = hashlib.sha256()
        for order, opaque_id in sorted(rows):
            digest.update(f"{domain}\t{order}\t{opaque_id}\n".encode("utf-8"))
        result[domain] = digest.hexdigest()
    return result


def build_transfer_manifest_from_capture(
    capture_manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    method_id: str,
    code_bindings: Mapping[str, Any],
    state_bindings: Mapping[str, Any],
    decoder_resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the capture-agent H/mask/position manifest to the decoder gate."""

    if not isinstance(capture_manifest, Mapping) or capture_manifest.get("schema") != CAPTURE_MANIFEST_SCHEMA or capture_manifest.get("task_id") != TASK_ID:
        raise TransferDiagnosticError("capture manifest schema or task identity changed")
    if capture_manifest.get("status") != "OBSERVATIONS_TRUTH_FREE":
        raise TransferDiagnosticError("capture manifest is not truth-free")
    for flag in TRUTH_FLAGS:
        if capture_manifest.get(flag) is True or (isinstance(capture_manifest.get(flag), str) and capture_manifest.get(flag).lower() == "true"):
            raise TransferDiagnosticError(f"capture manifest records forbidden access: {flag}")
    panel = capture_manifest.get("panel")
    if not isinstance(panel, Mapping):
        raise TransferDiagnosticError("capture manifest panel is absent")
    source_order = _capture_source_order_digests(panel)
    observations = capture_manifest.get("observations")
    if not isinstance(observations, Mapping):
        raise TransferDiagnosticError("capture manifest observations are absent")
    raw_ids = [str(value) for value in observations]
    canonical_ids = [_capture_variant_id(value) for value in raw_ids]
    if len(set(canonical_ids)) != len(canonical_ids) or set(canonical_ids) != set(TRANSFER_VARIANT_ORDER):
        raise TransferDiagnosticError("capture manifest does not cover all seven mandatory transfer variants")
    root = Path(repository_root).expanduser().resolve()
    checked_observations: dict[str, dict[str, Any]] = {}
    for raw_variant_id, domain_bindings in observations.items():
        variant_id = _capture_variant_id(str(raw_variant_id))
        if not isinstance(domain_bindings, Mapping) or set(domain_bindings) != set(TRANSFER_DOMAIN_ORDER):
            raise TransferDiagnosticError(f"capture observation domains are incomplete: {raw_variant_id}")
        checked_observations[variant_id] = {}
        for domain in TRANSFER_DOMAIN_ORDER:
            raw_binding = domain_bindings[domain]
            if not isinstance(raw_binding, Mapping):
                raise TransferDiagnosticError(f"capture observation binding is malformed: {raw_variant_id}/{domain}")
            if raw_binding.get("variant_id") not in {raw_variant_id, variant_id} or raw_binding.get("domain") != domain:
                raise TransferDiagnosticError(f"capture observation identity changed: {raw_variant_id}/{domain}")
            if raw_binding.get("opaque_record_ids_sha256") != source_order[domain]:
                raise TransferDiagnosticError(f"capture opaque pairing changed: {raw_variant_id}/{domain}")
            binding = dict(raw_binding)
            binding.update({"records": TRANSFER_RECORDS_PER_DOMAIN, "record_ids_sha256": source_order[domain]})
            checked = _validate_transfer_observation_header(
                binding, records=TRANSFER_RECORDS_PER_DOMAIN, root=root,
                description=f"capture observation {raw_variant_id}/{domain}",
            )
            checked_observations[variant_id][domain] = checked | {"record_ids_sha256": source_order[domain]}
    checked_code = {str(name): _transfer_file_record(binding, root=root, description=f"transfer code {name}") for name, binding in code_bindings.items()}
    checked_state = {str(name): _transfer_file_record(binding, root=root, description=f"transfer state {name}") for name, binding in state_bindings.items()}
    checked_resources = {str(name): _transfer_file_record(binding, root=root, description=f"transfer decoder resource {name}") for name, binding in decoder_resources.items()}
    if method_id not in {"expanded_fixed", "current_fixed"}:
        raise TransferDiagnosticError("capture transfer adapter only accepts a frozen fixed decoder")
    variant_parameters: dict[str, Any] = {}
    timing = capture_manifest.get("timing")
    receipts = timing.get("variant_receipts") if isinstance(timing, Mapping) else None
    if isinstance(receipts, Mapping):
        for raw_variant_id, receipt in receipts.items():
            if isinstance(receipt, Mapping) and isinstance(receipt.get("update"), Mapping):
                variant_parameters[_capture_variant_id(str(raw_variant_id))] = dict(receipt["update"])
    return {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "status": TRANSFER_RUN_STATUS,
        "method_id": method_id,
        "variant_ids": list(TRANSFER_VARIANT_ORDER),
        "source_order_sha256": source_order,
        "code_bindings": checked_code,
        "state_bindings": checked_state,
        "decoder_resources": checked_resources,
        "observation_bindings": checked_observations,
        "variant_parameters": variant_parameters,
        "capture_manifest_schema": CAPTURE_MANIFEST_SCHEMA,
        "panel_opaque_reservation_sha256": panel.get("opaque_reservation_sha256"),
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def _validate_historical_runner_sources(
    registration: Mapping[str, Any],
    *,
    repository_root: Path,
    runner_module: Any,
) -> dict[str, Any]:
    """Allow a historical registration only when current sources match hashes."""

    bindings = registration.get("code_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise TransferDiagnosticError("historical registration code bindings are absent")
    root = Path(repository_root).expanduser().resolve()
    try:
        from scripts import trr0010_eval_gate as gate_module
        from scripts import trr0010_p09_fixed_loader as loader_module
        from token_reconstruction import trr0007_positionwise as positionwise_module
    except ImportError as exc:
        raise TransferDiagnosticError("frozen TRR-0010 runtime source modules are unavailable") from exc
    modules = {
        "scripts.trr0010_eval_runner": runner_module,
        "scripts.trr0010_eval_gate": gate_module,
        "scripts.trr0010_p09_fixed_loader": loader_module,
        "token_reconstruction.trr0007_positionwise": positionwise_module,
    }
    evidence: dict[str, Any] = {}
    for module_name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise TransferDiagnosticError(f"frozen runtime source has no file: {module_name}")
        source_path = Path(raw_path).expanduser().resolve()
        if source_path.suffix == ".pyc" and source_path.with_suffix(".py").is_file():
            source_path = source_path.with_suffix(".py")
        if not source_path.is_file():
            raise TransferDiagnosticError(f"frozen runtime source is unavailable: {module_name}")
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        matching = []
        for name, binding in bindings.items():
            if not isinstance(binding, Mapping):
                continue
            declared_path = str(binding.get("path", ""))
            declared_sha = binding.get("sha256")
            if declared_sha == actual_sha and (declared_path.endswith(source_path.as_posix()) or Path(declared_path).name == source_path.name):
                matching.append(str(name))
        if not matching:
            raise TransferDiagnosticError(
                f"current TRR-0011 runtime source does not match frozen registration hash: {module_name}"
            )
        evidence[module_name] = {"path": str(source_path), "sha256": actual_sha, "registration_binding_names": matching}
    return {"mode": "historical_registration_hash_checked", "sources": evidence, "current_head_required": False}


def validate_transfer_inputs_before_inference(
    manifest_path: Path,
    *,
    repository_root: Path,
    registration_path: Path | None = None,
    require_current_head: bool = False,
    runner_module: Any | None = None,
) -> dict[str, Any]:
    """Validate transfer inputs and, when supplied, the real TRR0010 registration."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_transfer_json(manifest_path)
    checked = validate_transfer_observation_manifest(manifest, repository_root=repository_root)
    result: dict[str, Any] = {"transfer_inputs": checked, "manifest_path": str(manifest_path)}
    if registration_path is not None and runner_module is None:
        bridge_binding = checked["code_bindings"].get("transfer_bridge")
        bridge_path = Path(__file__).resolve()
        if not isinstance(bridge_binding, Mapping) or Path(str(bridge_binding.get("path", ""))).resolve() != bridge_path:
            raise TransferDiagnosticError("transfer bridge binding must point to the current TRR-0011 source")
        actual_bridge_sha = hashlib.sha256(bridge_path.read_bytes()).hexdigest()
        if bridge_binding.get("sha256") != actual_bridge_sha:
            raise TransferDiagnosticError("transfer bridge source binding changed")
    if registration_path is not None:
        if runner_module is None:
            try:
                from scripts import trr0010_eval_runner as runner_module
            except ImportError:  # PYTHONPATH=scripts execution
                import trr0010_eval_runner as runner_module
        root = Path(repository_root).expanduser().resolve()
        registration, registration_checked, registration_record = runner_module._load_registration(
            Path(registration_path),
            root=root,
            require_current_head=require_current_head,
            # The registration is historical.  We separately require exact
            # frozen source hashes below instead of bypassing all source checks.
            require_runtime_sources=False,
        )
        result["registration"] = registration_record
        if runner_module is not None and getattr(runner_module, "__file__", None):
            result["historical_source_check"] = _validate_historical_runner_sources(
                registration, repository_root=root, runner_module=runner_module
            )
        result["registered_method_ids"] = [str(row["id"]) for row in registration.get("methods", [])]
        result["registration_checked"] = registration_checked
    return result



def _decoder_geometry_for_record(
    adapter: Any,
    activation: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute projected features and top/runner logits through the loaded decoder.

    This intentionally uses the actual runner adapter's loaded model and
    readout.  Callers cannot supply arbitrary feature/logit arrays and have
    them accepted as if they were produced by the frozen decoder.
    """

    del positions
    model = getattr(adapter, "model", None)
    embedding = getattr(adapter, "embedding", None)
    device = getattr(adapter, "device", None)
    if not isinstance(model, torch.nn.Module) or not isinstance(embedding, torch.Tensor):
        raise TransferDiagnosticError("loaded fixed adapter does not expose model and readout")
    if not isinstance(device, torch.device):
        device = embedding.device
    staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    staged_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    with torch.inference_mode():
        try:
            projected = model.projected_hidden(staged, staged_mask)
            position_slots = torch.arange(1, STORED_SEQUENCE_TOKENS, device=projected.device, dtype=torch.long)
            record_slots = torch.zeros_like(position_slots)
            logits = model.logits_from_rows(projected, record_slots, position_slots, embedding)
        except AttributeError as exc:
            raise TransferDiagnosticError("loaded adapter does not expose frozen decoder geometry") from exc
        if tuple(logits.shape) != (SCORED_POST_BOS_TOKENS, VOCABULARY_SIZE):
            raise TransferDiagnosticError("loaded decoder logits geometry changed")
        if not torch.isfinite(logits).all().item():
            raise TransferDiagnosticError("loaded decoder logits are non-finite")
        # Use the exact argmax rule used by the prediction adapter for the
        # top class.  torch.topk may choose a different tied index, which
        # would make the diagnostic disagree with the emitted prediction.
        top_ids = logits.argmax(dim=-1)
        top_values = logits.gather(1, top_ids[:, None]).squeeze(1)
        runner_logits = logits.clone()
        runner_logits.scatter_(1, top_ids[:, None], float("-inf"))
        runner_values, runner_ids = runner_logits.max(dim=-1)
        top_ids = top_ids.to(dtype=torch.long)
        runner_ids = runner_ids.to(dtype=torch.long)
        top_values = top_values.to(dtype=torch.float32)
        runner_values = runner_values.to(dtype=torch.float32)
        readout_ids = torch.stack((top_ids, runner_ids), dim=1)
        readout = embedding[readout_ids]
        features = projected[0, 1:].detach().cpu().contiguous()
    return {
        "projected_features": features,
        "top_runner_logits": torch.stack((top_values, runner_values), dim=1).detach().cpu().contiguous(),
        "top_runner_ids": readout_ids.detach().cpu().contiguous(),
        "top_runner_readout": readout.detach().cpu().contiguous(),
    }


def _collect_decoder_geometry(
    adapter: Any,
    *,
    cell: Mapping[str, Any],
    records: int,
    hidden_size: int,
    runner_module: Any,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {
        "projected_features": [],
        "top_runner_logits": [],
        "top_runner_ids": [],
        "top_runner_readout": [],
    }
    iterator = getattr(runner_module, "_iter_rows", None)
    if not callable(iterator):
        raise TransferDiagnosticError("TRR0010 runner row iterator is unavailable")
    for _index, activation, mask, positions in iterator(cell, records=records, hidden_size=hidden_size):
        computed = _decoder_geometry_for_record(adapter, activation, mask, positions)
        for name, value in computed.items():
            rows[name].append(value)
    if any(len(values) != records for values in rows.values()):
        raise TransferDiagnosticError("decoder geometry row count changed")
    return {name: torch.stack(values, dim=0).contiguous() for name, values in rows.items()}


def _adapter_logit_scale(adapter: Any) -> float:
    model = getattr(adapter, "model", None)
    scale = getattr(model, "logit_scale", None)
    if isinstance(scale, torch.Tensor):
        if scale.ndim != 0:
            raise TransferDiagnosticError("loaded decoder logit scale is not scalar")
        value = float(scale.detach().cpu().item())
    else:
        value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise TransferDiagnosticError("loaded decoder logit scale is not finite and positive")
    return value


def _write_transfer_geometry(
    output_path: Path,
    geometry: Mapping[str, torch.Tensor],
    *,
    root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    observation_sha256: str | None = None,
    logit_scale: float | None = None,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise TransferDiagnosticError(f"transfer geometry output is not create-only: {output_path}")
    output_path = output_path.expanduser().resolve()
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        output_path.relative_to(task_root)
    except ValueError as exc:
        raise TransferDiagnosticError("transfer geometry output is outside the task root") from exc
    required = {"projected_features", "top_runner_logits", "top_runner_ids", "top_runner_readout"}
    if set(geometry) != required:
        raise TransferDiagnosticError("decoder geometry fields are incomplete")
    expected = {
        "projected_features": (records, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE),
        "top_runner_logits": (records, SCORED_POST_BOS_TOKENS, 2),
        "top_runner_ids": (records, SCORED_POST_BOS_TOKENS, 2),
        "top_runner_readout": (records, SCORED_POST_BOS_TOKENS, 2, HIDDEN_SIZE),
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, shape in expected.items():
        value = torch.as_tensor(geometry[name]).detach().cpu().contiguous()
        if tuple(value.shape) != shape or not torch.isfinite(value.float()).all().item():
            raise TransferDiagnosticError(f"decoder geometry changed: {name}")
        if name == "top_runner_ids" and (value.dtype not in (torch.int32, torch.int64) or value.lt(0).any().item() or value.ge(VOCABULARY_SIZE).any().item()):
            raise TransferDiagnosticError("decoder top/runner IDs changed")
        tensors[name] = value
    metadata = {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "records": str(records),
        "decoder_loader": P09_LOADER_MODULE,
        "decoder_state_tensor_key_count": str(P09_TENSOR_KEY_COUNT),
        "geometry_scope": "frozen decoder projected features and top/runner logits; no truth",
        "geometry_computation": "loaded_decoder.projected_hidden_then_logits_from_rows",
        "projected_feature_normalized": "true",
        "top_class_rule": "same_full_vocabulary_argmax_as_prediction",
        "runner_rule": "full_vocabulary_argmax_after_masking_top_class",
        "readout_equation": "logit=logit_scale*dot(projected_feature,readout_row)",
        "nearest_boundary_claim": "false",
        "logit_scale": repr(float(logit_scale)) if logit_scale is not None else "",
        "truth_opened": "false",
        "source_text_loaded": "false",
        "target_labels_loaded": "false",
        "candidate_arrays_persisted": "false",
    }
    if observation_sha256 is not None:
        metadata["observation_sha256"] = str(observation_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "tensor_digests": {name: tensor_digest(value) for name, value in tensors.items()},
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
    }


def _write_transfer_prediction(
    output_path: Path,
    values: torch.Tensor,
    *,
    root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    observation_sha256: str | None = None,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise TransferDiagnosticError(f"transfer prediction output is not create-only: {output_path}")
    output_path = output_path.expanduser().resolve()
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        output_path.relative_to(task_root)
    except ValueError as exc:
        raise TransferDiagnosticError("transfer prediction output is outside the task root") from exc
    checked = torch.as_tensor(values).detach().cpu().contiguous()
    if checked.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or tuple(checked.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise TransferDiagnosticError("transfer prediction geometry or dtype changed")
    if checked[:, 0].ne(BOS_TOKEN_ID).any().item() or checked.lt(0).any().item() or checked.ge(VOCABULARY_SIZE).any().item():
        raise TransferDiagnosticError("transfer prediction contains an invalid token")
    metadata = {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "records": str(records),
        "geometry_json": json.dumps({"records": records, "stored_sequence_tokens": STORED_SEQUENCE_TOKENS, "vocabulary_size": VOCABULARY_SIZE}, sort_keys=True),
        "truth_opened": "false",
        "source_text_loaded": "false",
        "target_labels_loaded": "false",
        "candidate_arrays_persisted": "false",
    }
    if observation_sha256 is not None:
        metadata["observation_sha256"] = str(observation_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": checked}, str(output_path), metadata=metadata)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "prediction_sha256": tensor_digest(checked),
        "records": records,
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
    }



def _same_transfer_binding(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if left.get(key) != right.get(key):
            raise TransferDiagnosticError(f"{description} binding changed: {key}")


def _validate_transfer_prediction_file(
    binding: Mapping[str, Any],
    *,
    root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    observation_sha256: str,
) -> dict[str, Any]:
    checked = _transfer_file_record(binding, root=root, description=f"transfer prediction {method_id}/{variant_id}/{domain}")
    try:
        with safe_open(checked["path"], framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            if set(handle.keys()) != {"predictions"}:
                raise TransferDiagnosticError("transfer prediction keys changed")
            values = handle.get_tensor("predictions").detach().cpu().contiguous()
    except TransferDiagnosticError:
        raise
    except Exception as exc:
        raise TransferDiagnosticError("transfer prediction artifact is unreadable") from exc
    expected_metadata = {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "records": str(records),
        "truth_opened": "false",
        "source_text_loaded": "false",
        "target_labels_loaded": "false",
        "candidate_arrays_persisted": "false",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise TransferDiagnosticError(f"transfer prediction metadata changed: {method_id}/{variant_id}/{domain}/{key}")
    if metadata.get("observation_sha256") != observation_sha256:
        raise TransferDiagnosticError(f"transfer prediction observation binding changed: {method_id}/{variant_id}/{domain}")
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or tuple(values.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise TransferDiagnosticError(f"transfer prediction geometry or dtype changed: {method_id}/{variant_id}/{domain}")
    if values[:, 0].ne(BOS_TOKEN_ID).any().item() or values.lt(0).any().item() or values.ge(VOCABULARY_SIZE).any().item():
        raise TransferDiagnosticError(f"transfer prediction contains an invalid token: {method_id}/{variant_id}/{domain}")
    return checked | {"metadata": metadata, "prediction_sha256": tensor_digest(values)}


def _validate_transfer_geometry_file(
    binding: Mapping[str, Any],
    *,
    root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    observation_sha256: str,
) -> dict[str, Any]:
    checked = _transfer_file_record(binding, root=root, description=f"transfer geometry {method_id}/{variant_id}/{domain}")
    try:
        with safe_open(checked["path"], framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            expected_keys = {"projected_features", "top_runner_logits", "top_runner_ids", "top_runner_readout"}
            if set(handle.keys()) != expected_keys:
                raise TransferDiagnosticError("transfer decoder geometry keys changed")
            shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in expected_keys}
            dtypes = {name: handle.get_slice(name).get_dtype() for name in expected_keys}
            projected_values = handle.get_tensor("projected_features").detach().cpu().float().contiguous()
            top_runner_values = handle.get_tensor("top_runner_logits").detach().cpu().float().contiguous()
            top_runner_ids = handle.get_tensor("top_runner_ids").detach().cpu().contiguous()
            readout_values = handle.get_tensor("top_runner_readout").detach().cpu().float().contiguous()
    except TransferDiagnosticError:
        raise
    except Exception as exc:
        raise TransferDiagnosticError("transfer decoder geometry artifact is unreadable") from exc
    expected_shapes = {
        "projected_features": (records, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE),
        "top_runner_logits": (records, SCORED_POST_BOS_TOKENS, 2),
        "top_runner_ids": (records, SCORED_POST_BOS_TOKENS, 2),
        "top_runner_readout": (records, SCORED_POST_BOS_TOKENS, 2, HIDDEN_SIZE),
    }
    if shapes != expected_shapes:
        raise TransferDiagnosticError(f"transfer decoder geometry shape changed: {method_id}/{variant_id}/{domain}")
    expected_metadata = {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "records": str(records),
        "decoder_loader": P09_LOADER_MODULE,
        "decoder_state_tensor_key_count": str(P09_TENSOR_KEY_COUNT),
        "truth_opened": "false",
        "source_text_loaded": "false",
        "target_labels_loaded": "false",
        "candidate_arrays_persisted": "false",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise TransferDiagnosticError(f"transfer geometry metadata changed: {method_id}/{variant_id}/{domain}/{key}")
    if metadata.get("observation_sha256") != observation_sha256:
        raise TransferDiagnosticError(f"transfer geometry observation binding changed: {method_id}/{variant_id}/{domain}")
    try:
        scale = float(metadata["logit_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferDiagnosticError(f"transfer geometry logit scale is absent: {method_id}/{variant_id}/{domain}") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise TransferDiagnosticError(f"transfer geometry logit scale is invalid: {method_id}/{variant_id}/{domain}")
    if metadata.get("geometry_computation") != "loaded_decoder.projected_hidden_then_logits_from_rows":
        raise TransferDiagnosticError(f"transfer geometry computation provenance changed: {method_id}/{variant_id}/{domain}")
    if metadata.get("projected_feature_normalized") != "true":
        raise TransferDiagnosticError(f"transfer projected-feature normalization binding changed: {method_id}/{variant_id}/{domain}")
    if metadata.get("top_class_rule") != "same_full_vocabulary_argmax_as_prediction" or metadata.get("runner_rule") != "full_vocabulary_argmax_after_masking_top_class":
        raise TransferDiagnosticError(f"transfer top/runner decision rule changed: {method_id}/{variant_id}/{domain}")
    if metadata.get("nearest_boundary_claim") != "false":
        raise TransferDiagnosticError(f"transfer geometry nearest-boundary claim is not explicitly false: {method_id}/{variant_id}/{domain}")
    if not torch.isfinite(projected_values).all().item() or not torch.isfinite(top_runner_values).all().item() or not torch.isfinite(readout_values).all().item():
        raise TransferDiagnosticError(f"transfer geometry contains non-finite values: {method_id}/{variant_id}/{domain}")
    if top_runner_ids.dtype not in (torch.int32, torch.int64) or top_runner_ids.lt(0).any().item() or top_runner_ids.ge(VOCABULARY_SIZE).any().item():
        raise TransferDiagnosticError(f"transfer top/runner IDs are invalid: {method_id}/{variant_id}/{domain}")
    if top_runner_ids[..., 0].eq(top_runner_ids[..., 1]).any().item():
        raise TransferDiagnosticError(f"transfer top/runner IDs are not distinct: {method_id}/{variant_id}/{domain}")
    if (top_runner_values[..., 0] < top_runner_values[..., 1]).any().item():
        raise TransferDiagnosticError(f"transfer top logits are below runner logits: {method_id}/{variant_id}/{domain}")
    norms = torch.linalg.vector_norm(projected_values, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=2e-3, atol=2e-3):
        raise TransferDiagnosticError(f"transfer projected features are not normalized: {method_id}/{variant_id}/{domain}")
    expected_logits = scale * (projected_values[:, :, None, :] * readout_values).sum(dim=-1)
    if not torch.allclose(top_runner_values, expected_logits, rtol=2e-4, atol=2e-4):
        raise TransferDiagnosticError(f"transfer top/runner logits violate the bound readout equation: {method_id}/{variant_id}/{domain}")
    return checked | {"metadata": metadata, "logit_scale": scale, "shapes": {name: list(shape) for name, shape in shapes.items()}, "dtypes": dtypes}


def _validate_prediction_top_agreement(
    prediction_binding: Mapping[str, Any],
    geometry_binding: Mapping[str, Any],
    *,
    method_id: str,
    variant_id: str,
    domain: str,
    root: Path,
    records: int,
) -> None:
    """Require geometry top IDs to equal the exact emitted decoder IDs."""

    try:
        with safe_open(str(prediction_binding["path"]), framework="pt", device="cpu") as prediction_handle:
            predictions = prediction_handle.get_tensor("predictions").detach().cpu().to(torch.long)
        with safe_open(str(geometry_binding["path"]), framework="pt", device="cpu") as geometry_handle:
            top_ids = geometry_handle.get_tensor("top_runner_ids").detach().cpu().to(torch.long)
    except Exception as exc:
        raise TransferDiagnosticError(
            f"transfer top/prediction agreement could not be checked: {method_id}/{variant_id}/{domain}"
        ) from exc
    if tuple(predictions.shape) != (records, STORED_SEQUENCE_TOKENS) or tuple(top_ids.shape) != (records, SCORED_POST_BOS_TOKENS, 2):
        raise TransferDiagnosticError(f"transfer top/prediction geometry changed: {method_id}/{variant_id}/{domain}")
    if not torch.equal(predictions[:, 1:], top_ids[:, :, 0]):
        raise TransferDiagnosticError(f"transfer top class disagrees with emitted prediction: {method_id}/{variant_id}/{domain}")


def validate_transfer_prediction_matrix(
    matrix: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the complete TRR-0011 transfer matrix before truth.

    This task-local gate is deliberately separate from the TRR-0010 gate,
    whose fixed six-method/128-record matrix cannot represent this diagnostic's
    seven target conditions, two fixed methods, paired domains, and 32-record
    cells.  Every prediction has a matching directly computed decoder-geometry
    artifact and an observation hash.
    """

    if not isinstance(matrix, Mapping) or matrix.get("schema") != TRANSFER_MATRIX_SCHEMA or matrix.get("task_id") != TASK_ID:
        raise TransferDiagnosticError("transfer prediction matrix schema or task identity changed")
    if matrix.get("status") != TRANSFER_MATRIX_STATUS:
        raise TransferDiagnosticError("transfer prediction matrix is not complete before truth")
    for flag in TRUTH_FLAGS:
        value = matrix.get(flag)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise TransferDiagnosticError(f"transfer prediction matrix records forbidden access: {flag}")
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise TransferDiagnosticError(f"transfer repository root is unavailable: {root}")
    input_binding = matrix.get("input_manifest")
    plan_binding = matrix.get("variant_plan")
    panel_binding = matrix.get("panel_descriptor")
    if not isinstance(input_binding, Mapping) or not isinstance(plan_binding, Mapping) or not isinstance(panel_binding, Mapping):
        raise TransferDiagnosticError("transfer prediction matrix is missing input/plan/panel bindings")
    checked_input_file = _transfer_file_record(input_binding, root=root, description="transfer input manifest")
    input_manifest = _load_transfer_json(Path(checked_input_file["path"]))
    checked_input = validate_transfer_observation_manifest(input_manifest, repository_root=root)
    checked_plan_file = _transfer_file_record(plan_binding, root=root, description="transfer variant plan")
    checked_panel_file = _transfer_file_record(panel_binding, root=root, description="transfer panel descriptor")
    checked_plan = validate_variant_plan(_load_transfer_json(Path(checked_plan_file["path"])))
    checked_panel = validate_panel_descriptor(_load_transfer_json(Path(checked_panel_file["path"])))
    if len(checked_plan["variant_ids"]) != TRANSFER_EXPECTED_VARIANT_COUNT:
        raise TransferDiagnosticError("transfer variant plan does not declare the frozen seven target conditions")
    if checked_panel["records_per_domain"] != TRANSFER_RECORDS_PER_DOMAIN:
        raise TransferDiagnosticError("transfer panel record count is not the frozen 32 per domain")
    method_ids = matrix.get("method_ids")
    variant_ids = matrix.get("variant_ids")
    domains = matrix.get("domain_order")
    records = matrix.get("records_per_domain")
    if list(method_ids or ()) != list(TRANSFER_METHOD_ORDER):
        raise TransferDiagnosticError("transfer prediction method order changed")
    if not isinstance(variant_ids, Sequence) or isinstance(variant_ids, (str, bytes, bytearray)) or not variant_ids:
        raise TransferDiagnosticError("transfer prediction variants are absent")
    variant_ids = [str(value) for value in variant_ids]
    if len(set(variant_ids)) != len(variant_ids) or set(variant_ids) != set(checked_input["variant_ids"]):
        raise TransferDiagnosticError("transfer prediction variant set differs from validated inputs")
    if set(variant_ids) != set(checked_plan["variant_ids"]):
        raise TransferDiagnosticError("transfer prediction variant set differs from the bound frozen plan")
    if list(domains or ()) != list(TRANSFER_DOMAIN_ORDER):
        raise TransferDiagnosticError("transfer prediction domain order changed")
    if records != TRANSFER_RECORDS_PER_DOMAIN:
        raise TransferDiagnosticError("transfer prediction record count changed")
    if set(checked_input["source_order_sha256"]) != set(TRANSFER_DOMAIN_ORDER):
        raise TransferDiagnosticError("transfer input domains are not the frozen Finance/Pile pair")
    predictions = matrix.get("predictions")
    geometries = matrix.get("decoder_geometry")
    if not isinstance(predictions, Mapping) or not isinstance(geometries, Mapping):
        raise TransferDiagnosticError("transfer prediction/geometry maps are absent")
    expected_keys = {f"{method}/{variant}/{domain}" for method in method_ids for variant in variant_ids for domain in domains}
    declared_cell_count = matrix.get("cell_count")
    if declared_cell_count is not None and declared_cell_count != len(expected_keys):
        raise TransferDiagnosticError("transfer prediction cell count changed")
    if set(predictions) != expected_keys or set(geometries) != expected_keys:
        raise TransferDiagnosticError("transfer prediction matrix is incomplete, duplicated, or foreign")
    checked_predictions: dict[str, Any] = {}
    checked_geometries: dict[str, Any] = {}
    for method_id in method_ids:
        for variant_id in variant_ids:
            for domain in domains:
                key = f"{method_id}/{variant_id}/{domain}"
                observation = checked_input["observation_bindings"][variant_id][domain]
                checked_predictions[key] = _validate_transfer_prediction_file(
                    predictions[key], root=root, method_id=method_id, variant_id=variant_id, domain=domain,
                    records=records, observation_sha256=observation["sha256"],
                )
                checked_geometries[key] = _validate_transfer_geometry_file(
                    geometries[key], root=root, method_id=method_id, variant_id=variant_id, domain=domain,
                    records=records, observation_sha256=observation["sha256"],
                )
                _validate_prediction_top_agreement(
                    checked_predictions[key], checked_geometries[key],
                    method_id=method_id, variant_id=variant_id, domain=domain,
                    root=root, records=records,
                )
    return {
        "schema": TRANSFER_MATRIX_SCHEMA,
        "task_id": TASK_ID,
        "status": TRANSFER_MATRIX_STATUS,
        "method_ids": list(method_ids),
        "variant_ids": variant_ids,
        "domain_order": list(domains),
        "records_per_domain": records,
        "cell_count": len(expected_keys),
        "prediction_shape": [records, STORED_SEQUENCE_TOKENS],
        "decoder_geometry_shape": [records, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE],
        "input_manifest": checked_input_file,
        "variant_plan": checked_plan_file,
        "panel_descriptor": checked_panel_file,
        "predictions": checked_predictions,
        "decoder_geometry": checked_geometries,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }



def _load_bound_transfer_geometry(
    binding: Mapping[str, Any],
    *,
    root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    observation_sha256: str,
) -> dict[str, torch.Tensor]:
    checked = _validate_transfer_geometry_file(
        binding,
        root=root,
        method_id=method_id,
        variant_id=variant_id,
        domain=domain,
        records=records,
        observation_sha256=observation_sha256,
    )
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(checked["path"], framework="pt", device="cpu") as handle:
        for name in ("projected_features", "top_runner_logits", "top_runner_ids", "top_runner_readout"):
            tensors[name] = handle.get_tensor(name).detach().cpu().contiguous()
    tensors["logit_scale"] = torch.tensor(float(checked["logit_scale"]), dtype=torch.float32)
    return tensors


def _load_bound_transfer_activation(
    binding: Mapping[str, Any],
    *,
    root: Path,
    records: int,
    description: str,
) -> torch.Tensor:
    checked = _validate_transfer_observation_header(binding, records=records, root=root, description=description)
    with safe_open(checked["path"], framework="pt", device="cpu") as handle:
        activation = handle.get_tensor("activations").detach().cpu().contiguous()
    return activation[:, 1:].float().contiguous()


def summarize_bound_transfer_geometry(
    clean_geometry_binding: Mapping[str, Any],
    changed_geometry_binding: Mapping[str, Any],
    *,
    clean_observation_binding: Mapping[str, Any],
    changed_observation_binding: Mapping[str, Any],
    repository_root: Path,
    method_id: str,
    variant_id: str,
    domain: str,
    records: int,
    parameter_delta_l2: float | None = None,
    parameter_base_l2: float | None = None,
) -> dict[str, Any]:
    """Summarize geometry artifacts emitted by the frozen decoder bridge.

    Both geometry files and public H observations are revalidated by hash and
    header before the arithmetic.  This is the production-facing companion
    to the small-array ``summarize_transfer`` unit primitive; callers cannot
    substitute arbitrary projected features or logits while retaining a
    production provenance label.
    """

    root = Path(repository_root).expanduser().resolve()
    clean_observation = _transfer_file_record(clean_observation_binding, root=root, description="clean transfer observation")
    changed_observation = _transfer_file_record(changed_observation_binding, root=root, description="changed transfer observation")
    clean_geometry = _load_bound_transfer_geometry(
        clean_geometry_binding,
        root=root,
        method_id=method_id,
        variant_id="clean_public_base",
        domain=domain,
        records=records,
        observation_sha256=clean_observation["sha256"],
    )
    changed_geometry = _load_bound_transfer_geometry(
        changed_geometry_binding,
        root=root,
        method_id=method_id,
        variant_id=variant_id,
        domain=domain,
        records=records,
        observation_sha256=changed_observation["sha256"],
    )
    clean_h = _load_bound_transfer_activation(clean_observation, root=root, records=records, description="clean transfer observation")
    changed_h = _load_bound_transfer_activation(changed_observation, root=root, records=records, description="changed transfer observation")
    expected_activation_shape = (records, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE)
    if tuple(clean_h.shape) != expected_activation_shape or tuple(changed_h.shape) != expected_activation_shape:
        raise TransferDiagnosticError("bound transfer activations must be [records, 127, 2048] after BOS masking")
    for label, artifact in (("clean", clean_geometry), ("changed", changed_geometry)):
        if tuple(artifact["projected_features"].shape) != expected_activation_shape:
            raise TransferDiagnosticError(f"bound {label} projected feature geometry changed")
        if tuple(artifact["top_runner_logits"].shape) != (records, SCORED_POST_BOS_TOKENS, 2):
            raise TransferDiagnosticError(f"bound {label} top/runner geometry changed")
    result = summarize_transfer(
        clean_h,
        changed_h,
        clean_geometry["projected_features"],
        changed_geometry["projected_features"],
        clean_geometry["top_runner_logits"][..., 0],
        clean_geometry["top_runner_logits"][..., 1],
        clean_geometry["top_runner_readout"],
        logit_scale=float(clean_geometry["logit_scale"].item()),
        clean_top_ids=clean_geometry["top_runner_ids"][..., 0],
        changed_top_ids=changed_geometry["top_runner_ids"][..., 0],
        parameter_delta_l2=parameter_delta_l2,
        parameter_base_l2=parameter_base_l2,
        validate_readout_equation=True,
    )
    result["geometry_provenance"] = {
        "status": "BOUND_FROZEN_DECODER_ARTIFACTS",
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "clean_observation_sha256": clean_observation["sha256"],
        "changed_observation_sha256": changed_observation["sha256"],
        "decoder_loader": P09_LOADER_MODULE,
        "decoder_state_tensor_key_count": P09_TENSOR_KEY_COUNT,
    }
    return result


def run_registered_transfer_predictions(
    manifest_path: Path,
    *,
    registration_path: Path,
    repository_root: Path,
    output_root: Path,
    device_name: str = "cpu",
    require_current_head: bool = False,
    runner_module: Any | None = None,
    write_geometry: bool = True,
) -> dict[str, Any]:
    """Run the existing TRR0010 decoder on transfer observations, truth-free.

    The transfer manifest is validated before registration, embedding, method,
    or observation payloads are loaded.  The default path calls the actual
    ``trr0010_eval_runner`` internals; ``runner_module`` exists only for small
    synthetic tests and cannot bypass the manifest validation.
    """

    root = Path(repository_root).expanduser().resolve()
    checked_inputs = validate_transfer_inputs_before_inference(
        Path(manifest_path),
        repository_root=root,
        registration_path=Path(registration_path),
        require_current_head=require_current_head,
        runner_module=runner_module,
    )
    transfer = checked_inputs["transfer_inputs"]
    if runner_module is None:
        try:
            from scripts import trr0010_eval_runner as runner_module
        except ImportError:
            import trr0010_eval_runner as runner_module
    registration, checked_registration, registration_record = runner_module._load_registration(
        Path(registration_path),
        root=root,
        require_current_head=require_current_head,
        require_runtime_sources=False,
    )
    if runner_module is not None and getattr(runner_module, "__file__", None):
        source_check = _validate_historical_runner_sources(
            registration, repository_root=root, runner_module=runner_module
        )
    else:
        source_check = {"mode": "synthetic_injection"}
    method_id = str(transfer["method_id"])
    rows = {str(row["id"]): row for row in registration.get("methods", [])}
    if method_id not in rows:
        raise TransferDiagnosticError(f"registered transfer method is absent: {method_id}")
    device = torch.device(device_name)
    embedding, embedding_evidence = runner_module._load_embedding(registration, root=root, device=device)
    loaded = runner_module._load_method(
        method_id,
        rows[method_id],
        root=root,
        device=device,
        embedding=embedding,
        method_factory=None,
        code_bindings=checked_registration.get("code_bindings"),
        allow_materialization=False,
    )
    output_root = Path(output_root).expanduser().resolve()
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        output_root.relative_to(task_root)
    except ValueError as exc:
        raise TransferDiagnosticError("transfer output root is outside the task root") from exc
    predictions: dict[str, Any] = {}
    for variant_id in transfer["variant_ids"]:
        for domain, binding in transfer["observation_bindings"][variant_id].items():
            cell = {"observation": binding}
            records = int(binding["records"])
            values, timing = runner_module._run_cell(
                adapter=loaded.adapter,
                cell=cell,
                records=records,
                hidden_size=HIDDEN_SIZE,
                device=device,
                method_id=method_id,
                guard_callback=None,
            )
            key = f"{variant_id}/{domain}"
            prediction_record = _write_transfer_prediction(
                output_root / variant_id / domain / "predictions.safetensors",
                values,
                root=root,
                method_id=method_id,
                variant_id=variant_id,
                domain=domain,
                records=records,
                observation_sha256=str(binding["sha256"]),
            ) | {"timing": timing}
            if write_geometry:
                geometry = _collect_decoder_geometry(
                    loaded.adapter,
                    cell=cell,
                    records=records,
                    hidden_size=HIDDEN_SIZE,
                    runner_module=runner_module,
                )
                prediction_record["decoder_geometry"] = _write_transfer_geometry(
                    output_root / variant_id / domain / "decoder_geometry.safetensors",
                    geometry,
                    root=root,
                    method_id=method_id,
                    variant_id=variant_id,
                    domain=domain,
                    records=records,
                    observation_sha256=str(binding["sha256"]),
                    logit_scale=_adapter_logit_scale(loaded.adapter),
                )
            predictions[key] = prediction_record
    return {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_TRANSFER_PREDICTIONS_COMPLETE_BEFORE_TRUTH",
        "method_id": method_id,
        "registration": registration_record,
        "historical_source_check": source_check,
        "embedding": embedding_evidence,
        "predictions": predictions,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


INFERENCE_PACKAGE_V2_SHA256 = "a74f687f8287cacfeafa0acc81a97f208672dedf797792c60f88ba93b0318b25"

def _write_transfer_json_create(path: Path, value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    task_root = (Path(root).expanduser().resolve() / "experiments" / TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise TransferDiagnosticError("transfer JSON output is outside the task root") from exc
    if path.exists() or path.is_symlink():
        raise TransferDiagnosticError(f"transfer JSON output is not create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _package_v2_binding(package_path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    package_file = _transfer_file_record(
        {"path": str(package_path), "bytes": package_path.stat().st_size, "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(), "readonly": True},
        root=root, description="TRR-0011 inference package v2",
    )
    if package_file["sha256"] != INFERENCE_PACKAGE_V2_SHA256:
        raise TransferDiagnosticError("immutable TRR-0011 inference package v2 hash changed")
    package = _load_transfer_json(Path(package_file["path"]))
    if package.get("schema") != "token-reconstruction.trr0011-inference-package.v1" or package.get("package_revision") != 2 or package.get("status") != "IMMUTABLE_SELECTED_RUNTIME_BINDING_NO_NEW_TRUTH":
        raise TransferDiagnosticError("inference package v2 identity/status changed")
    methods = package.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(TRANSFER_METHOD_ORDER):
        raise TransferDiagnosticError("inference package does not bind both frozen fixed methods")
    source_binding = package.get("package_source_binding")
    if not isinstance(source_binding, Mapping):
        raise TransferDiagnosticError("inference package source binding is absent")
    source_path = Path(str(source_binding.get("path", ""))).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    source_checked = _transfer_file_record(source_binding, root=root, description="TRR-0011 package source")
    if source_checked["path"] != str((root / "scripts" / "trr0011_package.py").resolve()) or source_checked["sha256"] != hashlib.sha256((root / "scripts" / "trr0011_package.py").read_bytes()).hexdigest():
        raise TransferDiagnosticError("TRR-0011 package source binding is not the unchanged local wrapper")
    return package_file, package


def _current_code_binding(path: Path, *, root: Path, readonly: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise TransferDiagnosticError(f"runtime code source is unavailable: {path}")
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if readonly:
        record["readonly"] = True
    return record


def run_registered_transfer_matrix(
    capture_manifest_path: Path,
    *,
    package_path: Path,
    registration_path: Path,
    variant_plan_path: Path,
    panel_descriptor_path: Path,
    repository_root: Path,
    output_root: Path,
    device_name: str = "cuda",
    require_current_head: bool = False,
    runner_module: Any | None = None,
) -> dict[str, Any]:
    """Run both frozen fixed decoders over all seven capture variants.

    The capture manifest is adapted and hash-validated before either decoder
    loads.  Every method is explicit; a missing method, cell, or geometry
    artifact fails closed and no partial matrix is emitted as complete.
    """

    root = Path(repository_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        output_root.relative_to(task_root)
    except ValueError as exc:
        raise TransferDiagnosticError("transfer matrix output root is outside the task root") from exc
    capture_path = Path(capture_manifest_path).expanduser().resolve()
    capture = _load_transfer_json(capture_path)
    package_binding, package = _package_v2_binding(Path(package_path).expanduser().resolve(), root=root)
    variant_plan_binding = _transfer_file_record(
        {"path": str(Path(variant_plan_path).expanduser().resolve()), "bytes": Path(variant_plan_path).expanduser().resolve().stat().st_size, "sha256": hashlib.sha256(Path(variant_plan_path).expanduser().resolve().read_bytes()).hexdigest(), "readonly": True},
        root=root, description="transfer variant plan",
    )
    panel_binding = _transfer_file_record(
        {"path": str(Path(panel_descriptor_path).expanduser().resolve()), "bytes": Path(panel_descriptor_path).expanduser().resolve().stat().st_size, "sha256": hashlib.sha256(Path(panel_descriptor_path).expanduser().resolve().read_bytes()).hexdigest(), "readonly": True},
        root=root, description="transfer panel descriptor",
    )
    plan = _load_transfer_json(Path(variant_plan_binding["path"]))
    panel = _load_transfer_json(Path(panel_binding["path"]))
    validate_variant_plan(plan)
    validate_panel_descriptor(panel, strict_reservation=True)
    capture_binding = _transfer_file_record(
        {"path": str(capture_path), "bytes": capture_path.stat().st_size, "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(), "readonly": True},
        root=root, description="capture manifest",
    )
    code_bindings = {
        "transfer_bridge": _current_code_binding(Path(__file__), root=root),
        "trr0010_runner": _current_code_binding(root / "scripts" / "trr0010_eval_runner.py", root=root),
        "trr0010_loader": _current_code_binding(root / "scripts" / "trr0010_p09_fixed_loader.py", root=root),
        "trr0010_gate": _current_code_binding(root / "scripts" / "trr0010_eval_gate.py", root=root),
        "positionwise_decoder": _current_code_binding(root / "src" / "token_reconstruction" / "trr0007_positionwise.py", root=root),
    }
    predictions: dict[str, Any] = {}
    geometries: dict[str, Any] = {}
    method_receipts: dict[str, Any] = {}
    variant_parameters: dict[str, Any] = {}
    for method_id in TRANSFER_METHOD_ORDER:
        method = package["methods"].get(method_id)
        if not isinstance(method, Mapping):
            raise TransferDiagnosticError(f"frozen package method binding is absent: {method_id}")
        state = method.get("state")
        readout = method.get("public_embedding_table") or method.get("readout")
        if not isinstance(state, Mapping) or not isinstance(readout, Mapping):
            raise TransferDiagnosticError(f"frozen package method resources are incomplete: {method_id}")
        transfer_manifest = build_transfer_manifest_from_capture(
            capture, repository_root=root, method_id=method_id,
            code_bindings=code_bindings,
            state_bindings={"decoder_state": state},
            decoder_resources={"embedding": readout, "state": state, "loader": code_bindings["trr0010_loader"]},
        )
        transfer_manifest["package_v2"] = package_binding
        transfer_manifest["capture_manifest"] = capture_binding
        manifest_path = output_root / f"transfer_input_{method_id}.json"
        transfer_manifest_record = _write_transfer_json_create(manifest_path, transfer_manifest, root=root)
        transfer_result = run_registered_transfer_predictions(
            manifest_path, registration_path=Path(registration_path), repository_root=root,
            output_root=output_root / method_id, device_name=device_name,
            require_current_head=require_current_head, runner_module=runner_module, write_geometry=True,
        )
        if set(transfer_result["predictions"]) != {f"{variant}/{domain}" for variant in TRANSFER_VARIANT_ORDER for domain in TRANSFER_DOMAIN_ORDER}:
            raise TransferDiagnosticError(f"transfer method matrix is incomplete: {method_id}")
        method_receipts[method_id] = {"manifest": transfer_manifest_record, "run": transfer_result}
        for key, record in transfer_result["predictions"].items():
            variant_id, domain = key.split("/", 1)
            matrix_key = f"{method_id}/{variant_id}/{domain}"
            predictions[matrix_key] = {key: record[key] for key in ("path", "bytes", "sha256", "prediction_sha256")}
            geometry = record.get("decoder_geometry")
            if not isinstance(geometry, Mapping):
                raise TransferDiagnosticError(f"decoder geometry is missing: {matrix_key}")
            geometries[matrix_key] = {key: geometry[key] for key in ("path", "bytes", "sha256", "tensor_digests")}
        variant_parameters.update(transfer_manifest.get("variant_parameters", {}))
    matrix = {
        "schema": TRANSFER_MATRIX_SCHEMA,
        "task_id": TASK_ID,
        "status": TRANSFER_MATRIX_STATUS,
        "method_ids": list(TRANSFER_METHOD_ORDER),
        "variant_ids": list(TRANSFER_VARIANT_ORDER),
        "domain_order": list(TRANSFER_DOMAIN_ORDER),
        "records_per_domain": TRANSFER_RECORDS_PER_DOMAIN,
        "cell_count": len(predictions),
        "input_manifest": method_receipts[TRANSFER_METHOD_ORDER[0]]["manifest"],
        "capture_manifest": capture_binding,
        "inference_package_v2": package_binding,
        "variant_plan": variant_plan_binding,
        "panel_descriptor": panel_binding,
        "code_bindings": code_bindings,
        "variant_parameters": variant_parameters,
        "predictions": predictions,
        "decoder_geometry": geometries,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "private_or_truth_payload_read": False,
        "fresh_evaluation_started": False,
    }
    matrix_path = output_root / "prediction_matrix.json"
    matrix_record = _write_transfer_json_create(matrix_path, matrix, root=root)
    checked_matrix = validate_transfer_prediction_matrix(matrix, repository_root=root)
    return {
        "status": "PUBLIC_TRANSFER_PREDICTIONS_COMPLETE_BEFORE_TRUTH",
        "matrix": checked_matrix,
        "matrix_record": matrix_record,
        "method_receipts": method_receipts,
        "inference_package_v2": package_binding,
        "truth_opened": False,
    }


__all__ = [
    "BOS_TOKEN_ID",
    "CUT_DEPTH",
    "HIDDEN_SIZE",
    "SCHEMA",
    "SCORED_POST_BOS_TOKENS",
    "STORED_SEQUENCE_TOKENS",
    "TASK_ID",
    "TransferDiagnosticError",
    "VARIANT_ROLES",
    "TRANSFER_RUN_SCHEMA",
    "TRANSFER_RUN_STATUS",
    "TRANSFER_MATRIX_SCHEMA",
    "TRANSFER_MATRIX_STATUS",
    "validate_transfer_prediction_matrix",
    "evaluator_only_error_inventory",
    "evaluator_only_fixed_auc",
    "evaluator_only_transfer_inventory",
    "run_registered_transfer_predictions",
    "summarize_bound_transfer_geometry",
    "summarize_transfer",
    "tensor_digest",
    "validate_after_cut_null",
    "validate_panel_descriptor",
    "validate_transfer_inputs_before_inference",
    "validate_transfer_observation_manifest",
    "validate_variant_plan",
    "main",
]


def _load_json_object(path: str) -> dict[str, Any]:
    import json
    from pathlib import Path

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferDiagnosticError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TransferDiagnosticError(f"JSON object required: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a truth-free plan or run the registered transfer bridge."""

    import argparse

    parser = argparse.ArgumentParser(description="Validate or run a TRR-0011 transfer diagnostic")
    parser.add_argument("--plan", help="truth-free controlled-target variant plan JSON")
    parser.add_argument("--panel", help="reserved source-paired panel descriptor JSON")
    parser.add_argument("--transfer-manifest", help="hashed transfer observation/resource manifest JSON")
    parser.add_argument("--prediction-matrix", help="complete task-local transfer prediction matrix JSON")
    parser.add_argument("--registration", help="frozen TRR-0010 registration JSON for the runner bridge")
    parser.add_argument("--repository-root", default=".", help="repository root containing experiments/TRR-0011")
    parser.add_argument("--output-root", help="create-only transfer prediction output root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true", help="invoke the existing TRR-0010 runner after validation")
    args = parser.parse_args(argv)
    if not args.plan and not args.transfer_manifest and not args.prediction_matrix:
        parser.error("one of --plan, --transfer-manifest, or --prediction-matrix is required")
    result: dict[str, Any] = {"schema": SCHEMA, "task_id": TASK_ID}
    if args.plan:
        result["variant_plan"] = validate_variant_plan(_load_json_object(args.plan))
    if args.panel:
        result["panel"] = validate_panel_descriptor(_load_json_object(args.panel))
    if args.prediction_matrix:
        result["prediction_matrix"] = validate_transfer_prediction_matrix(
            _load_json_object(args.prediction_matrix), repository_root=Path(args.repository_root)
        )
    if args.transfer_manifest:
        if not args.registration:
            result["transfer"] = validate_transfer_inputs_before_inference(
                Path(args.transfer_manifest), repository_root=Path(args.repository_root)
            )
        elif args.execute:
            if not args.output_root:
                parser.error("--output-root is required with --execute")
            result["transfer"] = run_registered_transfer_predictions(
                Path(args.transfer_manifest),
                registration_path=Path(args.registration),
                repository_root=Path(args.repository_root),
                output_root=Path(args.output_root),
                device_name=str(args.device),
            )
        else:
            result["transfer"] = validate_transfer_inputs_before_inference(
                Path(args.transfer_manifest),
                repository_root=Path(args.repository_root),
                registration_path=Path(args.registration),
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI smoke
    raise SystemExit(main())
