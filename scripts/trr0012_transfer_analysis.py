"""Truth-gated evaluator callback for the TRR-0012 transfer matrix.

The public prediction matrix is validated completely before this module opens
the evaluator truth manifest.  The public gate is the existing TRR-0011
``validate_transfer_prediction_matrix`` contract; this file only adapts its
bound prediction and geometry artifacts to the existing evaluator-only
inventory and fixed-AUC helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

try:  # Repository-root invocation: ``python scripts/...``.
    from scripts import trr0011_transfer as transfer
except ImportError:  # PYTHONPATH=scripts invocation.
    import trr0011_transfer as transfer


TASK_ID = "TRR-0012"
METHOD_ID = "expanded_fixed"
VARIANT_ORDER = tuple(transfer.TRANSFER_VARIANT_ORDER)
DOMAIN_ORDER = tuple(transfer.TRANSFER_DOMAIN_ORDER)
RECORDS_PER_DOMAIN = int(transfer.TRANSFER_RECORDS_PER_DOMAIN)
STORED_SEQUENCE_TOKENS = int(transfer.STORED_SEQUENCE_TOKENS)
SCORED_POST_BOS_TOKENS = int(transfer.SCORED_POST_BOS_TOKENS)
BOS_TOKEN_ID = int(transfer.BOS_TOKEN_ID)
HIDDEN_SIZE = int(transfer.HIDDEN_SIZE)
EXPECTED_CELL_COUNT = len(VARIANT_ORDER) * len(DOMAIN_ORDER)
FIXED_SCORE_NAMES = (
    "raw_activation_l2",
    "relative_raw_activation_l2",
    "margin_normalized_feature_shift",
    "parameter_relative_l2",
)
FREEZE_SCHEMA = "token-reconstruction.trr0012-transfer-public-prediction-freeze.v1"
EXECUTED_FREEZE_SCHEMA = "token-reconstruction.trr0012-transfer-freeze.v1"
TRUTH_SCHEMA = "token-reconstruction.trr0012-transfer-evaluator-truth.v1"
ANALYSIS_SCHEMA = "token-reconstruction.trr0012-transfer-evaluator-analysis.v1"


class TransferAnalysisError(RuntimeError):
    """Raised when the public or evaluator-side contract fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferAnalysisError(f"invalid transfer-analysis JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TransferAnalysisError(f"transfer-analysis JSON must be an object: {path}")
    return value


def _record(path: Path, *, root: Path, description: str, readonly: bool = True) -> dict[str, Any]:
    try:
        checked = transfer._transfer_file_record(  # noqa: SLF001
            {
                "path": str(Path(path).expanduser().resolve()),
                "bytes": Path(path).expanduser().resolve().stat().st_size,
                "sha256": _sha256(Path(path).expanduser().resolve()),
                "readonly": readonly,
            },
            root=root,
            description=description,
        )
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(str(exc)) from exc
    return checked


def _require_flag_false(payload: Mapping[str, Any], *, label: str) -> None:
    for flag in transfer.TRUTH_FLAGS:
        value = payload.get(flag)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise TransferAnalysisError(f"{label} records forbidden access before truth gate: {flag}")


def _validate_freeze_receipt(
    freeze_path: Path,
    *,
    matrix_path: Path,
    matrix_record: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Validate the public freeze/audit receipt after the matrix gate."""

    receipt_record = _record(freeze_path, root=root, description="public prediction freeze receipt")
    receipt = _json(freeze_path)
    if receipt.get("schema") not in {FREEZE_SCHEMA, EXECUTED_FREEZE_SCHEMA} or receipt.get("task_id") != TASK_ID:
        raise TransferAnalysisError("public prediction freeze receipt schema or task identity changed")
    if receipt.get("status") not in {"FROZEN_BEFORE_TRUTH", "COMPLETE_14_CELL_PREDICTIONS_FROZEN_BEFORE_TRUTH"}:
        raise TransferAnalysisError("public prediction freeze receipt is not before-truth frozen")
    _require_flag_false(receipt, label="public prediction freeze receipt")
    matrix_binding = receipt.get("matrix_binding") or receipt.get("matrix")
    if not isinstance(matrix_binding, Mapping):
        raise TransferAnalysisError("public prediction freeze receipt lacks matrix_binding")
    bound_matrix = _record(
        Path(str(matrix_binding.get("path", ""))),
        root=root,
        description="frozen public prediction matrix",
    )
    expected_matrix = Path(matrix_path).expanduser().resolve()
    if Path(bound_matrix["path"]) != expected_matrix:
        raise TransferAnalysisError("public prediction freeze binds a different matrix path")
    for key in ("bytes", "sha256"):
        if bound_matrix[key] != matrix_record[key] or matrix_binding.get(key) != matrix_record[key]:
            raise TransferAnalysisError(f"public prediction freeze matrix binding changed: {key}")
    if list(receipt.get("method_ids", ())) != [METHOD_ID]:
        raise TransferAnalysisError("public prediction freeze method order changed")
    variant_ids = receipt.get("variant_ids")
    if variant_ids is not None and list(variant_ids) != list(VARIANT_ORDER):
        raise TransferAnalysisError("public prediction freeze variant order changed")
    domain_order = receipt.get("domain_order")
    if domain_order is not None and list(domain_order) != list(DOMAIN_ORDER):
        raise TransferAnalysisError("public prediction freeze domain order changed")
    if receipt.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise TransferAnalysisError("public prediction freeze record denominator changed")
    if receipt.get("cell_count", receipt.get("cells")) != EXPECTED_CELL_COUNT:
        raise TransferAnalysisError("public prediction freeze cell count changed")
    if receipt.get("variants", len(VARIANT_ORDER)) != len(VARIANT_ORDER):
        raise TransferAnalysisError("public prediction freeze variant count changed")
    if receipt.get("source_records", len(DOMAIN_ORDER) * RECORDS_PER_DOMAIN) != len(DOMAIN_ORDER) * RECORDS_PER_DOMAIN:
        raise TransferAnalysisError("public prediction freeze source-record count changed")
    if receipt.get("scored_tokens_per_record", SCORED_POST_BOS_TOKENS) != SCORED_POST_BOS_TOKENS:
        raise TransferAnalysisError("public prediction freeze scored-token denominator changed")
    if receipt.get("scored_tokens_per_cell", RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS) != RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS:
        raise TransferAnalysisError("public prediction freeze scored-cell denominator changed")
    prediction_shape = receipt.get("prediction_shape", [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS])
    if list(prediction_shape) != [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS]:
        raise TransferAnalysisError("public prediction freeze prediction shape changed")
    geometry_shape = receipt.get("decoder_geometry_shape", [RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE])
    if list(geometry_shape) != [RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE]:
        raise TransferAnalysisError("public prediction freeze geometry shape changed")
    return {
        "record": receipt_record,
        "schema": receipt["schema"],
        "status": receipt["status"],
        "matrix_binding": bound_matrix,
        "method_ids": [METHOD_ID],
        "variant_ids": list(VARIANT_ORDER),
        "domain_order": list(DOMAIN_ORDER),
        "records_per_domain": RECORDS_PER_DOMAIN,
        "cell_count": EXPECTED_CELL_COUNT,
        "prediction_shape": [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS],
        "decoder_geometry_shape": [RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE],
        "truth_opened": False,
    }


def _validate_public_matrix(matrix_path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the complete public gate before any evaluator truth is opened."""

    matrix_record = _record(matrix_path, root=root, description="public prediction matrix")
    matrix = _json(matrix_path)
    _require_flag_false(matrix, label="public prediction matrix")
    try:
        checked = transfer.validate_transfer_prediction_matrix(
            matrix,
            repository_root=root,
            expected_method_ids=(METHOD_ID,),
        )
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(str(exc)) from exc
    if checked["method_ids"] != [METHOD_ID]:
        raise TransferAnalysisError("public prediction matrix method selection changed")
    if checked["variant_ids"] != list(VARIANT_ORDER):
        raise TransferAnalysisError("public prediction matrix variant order changed")
    if checked["domain_order"] != list(DOMAIN_ORDER):
        raise TransferAnalysisError("public prediction matrix domain order changed")
    if checked["records_per_domain"] != RECORDS_PER_DOMAIN or checked["cell_count"] != EXPECTED_CELL_COUNT:
        raise TransferAnalysisError("public prediction matrix denominators or cell count changed")
    if checked["prediction_shape"] != [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS]:
        raise TransferAnalysisError("public prediction matrix prediction shape changed")
    if checked["decoder_geometry_shape"] != [RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE]:
        raise TransferAnalysisError("public prediction matrix geometry shape changed")

    # The matrix gate validated this manifest already.  Reading it again is
    # public metadata/header work and supplies the bound observation paths for
    # the exact after-cut null and geometry calculations.
    input_path = Path(str(checked["input_manifest"]["path"]))
    input_manifest = _json(input_path)
    try:
        checked_input = transfer.validate_transfer_observation_manifest(input_manifest, repository_root=root)
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(str(exc)) from exc
    if checked_input["variant_ids"] != list(VARIANT_ORDER):
        raise TransferAnalysisError("validated public input variant order changed")

    # Bind the canonical opaque panel identity and ordered IDs separately from
    # the observation headers.  The evaluator truth manifest must repeat these
    # values before any truth tensor is opened.
    panel_path = Path(str(checked["panel_descriptor"]["path"]))
    panel_record = _record(panel_path, root=root, description="public opaque panel descriptor")
    panel = _json(panel_path)
    panel_sha256 = panel.get("panel_sha256")
    domain_record_order_sha256 = panel.get("domain_record_order_sha256")
    if not isinstance(panel_sha256, str) or len(panel_sha256) != 64:
        raise TransferAnalysisError("public opaque panel canonical hash is absent")
    if not isinstance(domain_record_order_sha256, Mapping):
        raise TransferAnalysisError("public opaque panel ordered-ID digests are absent")
    expected_order = {domain: str(checked_input["source_order_sha256"][domain]) for domain in DOMAIN_ORDER}
    actual_order = {str(domain): str(value) for domain, value in domain_record_order_sha256.items()}
    if actual_order != expected_order:
        raise TransferAnalysisError("public opaque panel ordered-ID digests disagree with transfer inputs")
    if panel.get("records_per_domain") != RECORDS_PER_DOMAIN or panel.get("same_record_order_across_variants") is not True:
        raise TransferAnalysisError("public opaque panel record-order contract changed")
    panel_binding = {
        "record": panel_record,
        "panel_sha256": panel_sha256,
        "domain_record_order_sha256": expected_order,
    }
    return matrix_record, matrix, checked, checked_input, input_manifest, panel_binding


def validate_truth_order_declarations(
    manifest: Mapping[str, Any],
    *,
    expected_panel_sha256: str,
    expected_domain_record_order_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Reject a truth binding whose opaque panel/order identity was swapped."""

    if manifest.get("panel_sha256") != expected_panel_sha256:
        raise TransferAnalysisError("evaluator truth canonical panel hash does not match the public panel")
    declared = manifest.get("domain_record_order_sha256")
    if not isinstance(declared, Mapping):
        raise TransferAnalysisError("evaluator truth ordered-ID digests are absent")
    normalized = {str(domain): str(value) for domain, value in declared.items()}
    expected = {str(domain): str(value) for domain, value in expected_domain_record_order_sha256.items()}
    if normalized != expected:
        raise TransferAnalysisError("evaluator truth ordered-ID digests do not match the public panel")
    return {"panel_sha256": expected_panel_sha256, "domain_record_order_sha256": expected}


def _load_prediction(binding: Mapping[str, Any]) -> torch.Tensor:
    try:
        with safe_open(str(binding["path"]), framework="pt", device="cpu") as handle:
            values = handle.get_tensor("predictions").detach().cpu().contiguous()
    except Exception as exc:
        raise TransferAnalysisError("public prediction artifact could not be opened after the gate") from exc
    if tuple(values.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
        raise TransferAnalysisError("public prediction tensor shape changed after the gate")
    return values.to(dtype=torch.long)


def _load_cell(
    checked_matrix: Mapping[str, Any],
    checked_input: Mapping[str, Any],
    *,
    root: Path,
    variant_id: str,
    domain: str,
) -> dict[str, Any]:
    key = f"{METHOD_ID}/{variant_id}/{domain}"
    try:
        prediction_binding = checked_matrix["predictions"][key]
        geometry_binding = checked_matrix["decoder_geometry"][key]
        observation_binding = checked_input["observation_bindings"][variant_id][domain]
    except (KeyError, TypeError) as exc:
        raise TransferAnalysisError(f"public cell binding is absent: {key}") from exc
    try:
        prediction = _load_prediction(prediction_binding)
        geometry = transfer._load_bound_transfer_geometry(  # noqa: SLF001
            geometry_binding,
            root=root,
            method_id=METHOD_ID,
            variant_id=variant_id,
            domain=domain,
            records=RECORDS_PER_DOMAIN,
            observation_sha256=str(observation_binding["sha256"]),
        )
        activation = transfer._load_bound_transfer_activation(  # noqa: SLF001
            observation_binding,
            root=root,
            records=RECORDS_PER_DOMAIN,
            description=f"public transfer observation {variant_id}/{domain}",
        )
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(str(exc)) from exc
    return {
        "prediction": prediction,
        "geometry": geometry,
        "activation": activation,
        "observation": observation_binding,
        "prediction_binding": prediction_binding,
        "geometry_binding": geometry_binding,
    }


def _validate_after_cut_null(
    checked_matrix: Mapping[str, Any],
    checked_input: Mapping[str, Any],
    *,
    root: Path,
    domain: str,
) -> dict[str, Any]:
    clean = _load_cell(checked_matrix, checked_input, root=root, variant_id=VARIANT_ORDER[0], domain=domain)
    null = _load_cell(
        checked_matrix,
        checked_input,
        root=root,
        variant_id="after_cut_suffix_eps1e2_null",
        domain=domain,
    )
    try:
        result = transfer.validate_after_cut_null(
            clean["activation"],
            null["activation"],
            clean["prediction"],
            null["prediction"],
        )
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(f"after-cut null invalid for {domain}: {exc}") from exc
    return result | {
        "domain": domain,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "status": "PASS",
    }


def _parameter_amplitude(plan: Mapping[str, Any], variant_id: str) -> float | None:
    for variant in plan.get("variants", ()):
        if not isinstance(variant, Mapping) or variant.get("variant_id") != variant_id:
            continue
        location = variant.get("update_location")
        if not isinstance(location, Mapping):
            return None
        value = location.get("relative_parameter_l2")
        if value is None:
            return None
        try:
            amplitude = float(value)
        except (TypeError, ValueError) as exc:
            raise TransferAnalysisError(f"parameter amplitude is malformed: {variant_id}") from exc
        if not math.isfinite(amplitude) or amplitude < 0.0:
            raise TransferAnalysisError(f"parameter amplitude is invalid: {variant_id}")
        return amplitude
    raise TransferAnalysisError(f"parameter amplitude is absent from frozen plan: {variant_id}")


def _parameter_binding(
    plan: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    """Bind nominal plan amplitude separately from producer-measured BF16 L2."""

    nominal = _parameter_amplitude(plan, variant_id)
    parameters = input_manifest.get("variant_parameters")
    raw = parameters.get(variant_id) if isinstance(parameters, Mapping) else None
    binding: dict[str, Any] = {
        "nominal_relative_parameter_l2": nominal,
        "actual_relative_parameter_l2": None,
        "actual_delta_l2": None,
        "actual_base_l2": None,
        "source": "unavailable",
    }
    if not isinstance(raw, Mapping) or raw.get("status") != "UPDATED":
        binding["status"] = "UNAVAILABLE"
        binding["reason"] = "producer did not bind an actual quantized update"
        return binding
    try:
        actual = float(raw["effective_relative_parameter_l2"])
        delta = float(raw["effective_delta_l2_after_parameter_dtype"])
        base = float(raw["base_parameter_l2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferAnalysisError(f"actual quantized parameter metadata is incomplete: {variant_id}") from exc
    if not all(math.isfinite(value) for value in (actual, delta, base)) or actual < 0.0 or delta < 0.0 or base <= 0.0:
        raise TransferAnalysisError(f"actual quantized parameter metadata is invalid: {variant_id}")
    if not math.isclose(actual, delta / base, rel_tol=2e-6, abs_tol=2e-9):
        raise TransferAnalysisError(f"actual quantized parameter ratio is inconsistent: {variant_id}")
    binding.update({
        "status": "COMPUTED",
        "actual_relative_parameter_l2": actual,
        "actual_delta_l2": delta,
        "actual_base_l2": base,
        "source": "sanitized producer variant_parameters effective BF16 update",
        "producer_variant_id": raw.get("variant_id"),
        "requested_relative_parameter_l2": raw.get("requested_relative_parameter_l2"),
    })
    return binding


def _truth_accounting(prediction: torch.Tensor, truth: torch.Tensor, *, label: str) -> dict[str, Any]:
    if tuple(prediction.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS) or tuple(truth.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
        raise TransferAnalysisError(f"{label} accounting geometry changed")
    token_equal = prediction[:, 1:].eq(truth[:, 1:])
    record_equal = token_equal.all(dim=1)
    return {
        "label": label,
        "token_scope": "post_bos_scored_tokens",
        "token_correct": int(token_equal.sum().item()),
        "token_total": RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS,
        "exact_record_correct": int(record_equal.sum().item()),
        "exact_record_total": RECORDS_PER_DOMAIN,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
    }


def _row_scores(clean: Mapping[str, Any], changed: Mapping[str, Any], *, parameter_score: float | None) -> dict[str, torch.Tensor]:
    clean_h = clean["activation"].float()
    changed_h = changed["activation"].float()
    clean_u = clean["geometry"]["projected_features"].float()
    changed_u = changed["geometry"]["projected_features"].float()
    raw = torch.linalg.vector_norm(changed_h - clean_h, dim=-1).reshape(-1)
    relative = raw / torch.linalg.vector_norm(clean_h, dim=-1).clamp_min(1e-12).reshape(-1)
    feature = torch.linalg.vector_norm(changed_u - clean_u, dim=-1)
    clean_logits = clean["geometry"]["top_runner_logits"].float()
    readout = clean["geometry"]["top_runner_readout"].float()
    scale = float(clean["geometry"]["logit_scale"].item())
    margin = clean_logits[..., 0] - clean_logits[..., 1]
    class_difference_norm = torch.linalg.vector_norm(readout[..., 0, :] - readout[..., 1, :], dim=-1)
    denominator = class_difference_norm * scale
    fp32_scaled_tolerance = 8.0 * torch.finfo(torch.float32).eps * torch.maximum(
        torch.ones_like(margin), torch.maximum(clean_logits[..., 0].abs(), clean_logits[..., 1].abs())
    )
    denominator_tolerance = 8.0 * torch.finfo(torch.float32).eps * torch.maximum(
        torch.ones_like(denominator), torch.full_like(denominator, abs(scale))
    )
    valid = (margin > fp32_scaled_tolerance) & (denominator > denominator_tolerance)
    margin_score = torch.full_like(margin, float("nan"))
    margin_score[valid] = feature[valid] / (margin[valid] / denominator[valid])
    if parameter_score is None:
        parameter = torch.full_like(raw, float("nan"))
    else:
        parameter = torch.full_like(raw, float(parameter_score))
    return {
        "raw_activation_l2": raw,
        "relative_raw_activation_l2": relative,
        "margin_normalized_feature_shift": margin_score.reshape(-1),
        "parameter_relative_l2": parameter,
    }


def _summarize_loaded_cell_geometry(
    clean: Mapping[str, Any],
    changed: Mapping[str, Any],
    *,
    method_id: str,
    variant_id: str,
    domain: str,
    parameter_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt already-loaded bound geometry to the frozen summary primitive.

    The production geometry files retain record and token axes on their
    readout, ``[records, positions, 2, hidden]``.  The frozen
    ``summarize_transfer`` primitive deliberately accepts only paired rows,
    so flatten that one boundary in memory.  All arrays here have already
    passed the public matrix and bound-file checks in ``_load_cell``; this
    helper does not reopen or replace any artifact.
    """

    expected_activation_shape = (RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, HIDDEN_SIZE)
    expected_logits_shape = (RECORDS_PER_DOMAIN, SCORED_POST_BOS_TOKENS, 2)
    expected_readout_shape = (
        RECORDS_PER_DOMAIN,
        SCORED_POST_BOS_TOKENS,
        2,
        HIDDEN_SIZE,
    )
    for label, cell in (("clean", clean), ("changed", changed)):
        try:
            activation = cell["activation"]
            geometry = cell["geometry"]
            projected = geometry["projected_features"]
            logits = geometry["top_runner_logits"]
            readout = geometry["top_runner_readout"]
        except (KeyError, TypeError) as exc:
            raise TransferAnalysisError(f"loaded bound geometry is incomplete: {label}") from exc
        if tuple(activation.shape) != expected_activation_shape:
            raise TransferAnalysisError(f"bound {label} activations must be [records, 127, 2048] after BOS masking")
        if tuple(projected.shape) != expected_activation_shape:
            raise TransferAnalysisError(f"bound {label} projected feature geometry changed")
        if tuple(logits.shape) != expected_logits_shape:
            raise TransferAnalysisError(f"bound {label} top/runner geometry changed")
        if tuple(readout.shape) != expected_readout_shape:
            raise TransferAnalysisError(f"bound {label} readout geometry changed")

    clean_geometry = clean["geometry"]
    changed_geometry = changed["geometry"]
    clean_readout = clean_geometry["top_runner_readout"]
    readout_pairs = clean_readout.reshape(-1, 2, HIDDEN_SIZE).contiguous()
    try:
        result = transfer.summarize_transfer(
            clean["activation"],
            changed["activation"],
            clean_geometry["projected_features"],
            changed_geometry["projected_features"],
            clean_geometry["top_runner_logits"][..., 0],
            clean_geometry["top_runner_logits"][..., 1],
            readout_pairs,
            logit_scale=float(clean_geometry["logit_scale"].item()),
            clean_top_ids=clean_geometry["top_runner_ids"][..., 0],
            changed_top_ids=changed_geometry["top_runner_ids"][..., 0],
            parameter_delta_l2=parameter_binding.get("actual_delta_l2"),
            parameter_base_l2=parameter_binding.get("actual_base_l2"),
            validate_readout_equation=True,
        )
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(f"public geometry summary failed: {variant_id}/{domain}: {exc}") from exc

    result["geometry_provenance"] = {
        "status": "BOUND_FROZEN_DECODER_ARTIFACTS",
        "method_id": method_id,
        "variant_id": variant_id,
        "domain": domain,
        "clean_observation_sha256": clean["observation"]["sha256"],
        "changed_observation_sha256": changed["observation"]["sha256"],
        "clean_geometry_sha256": clean["geometry_binding"]["sha256"],
        "changed_geometry_sha256": changed["geometry_binding"]["sha256"],
        "decoder_loader": transfer.P09_LOADER_MODULE,
        "decoder_state_tensor_key_count": transfer.P09_TENSOR_KEY_COUNT,
        "readout_shape_before_adapter": list(expected_readout_shape),
        "readout_shape_after_adapter": list(readout_pairs.shape),
        "readout_adapter": "reshape_records_positions_to_rows",
    }
    return result


def _unknown_auc(score_name: str, *, reason: str, rows: int, invalid_rows: int = 0) -> dict[str, Any]:
    return {
        "status": "UNKNOWN",
        "truth_dependent": True,
        "deployable": False,
        "score_name": score_name,
        "risk_set": "baseline_correct",
        "positive_label": "broken",
        "negative_label": "remaining_correct",
        "threshold_fitted": False,
        "reason": reason,
        "rows": rows,
        "invalid_score_rows": invalid_rows,
    }


def _truth_bindings_after_gate(
    truth_path: Path,
    *,
    root: Path,
    expected_panel_sha256: str,
    expected_domain_record_order_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Open truth only after the complete public matrix and freeze gates."""

    truth_record = _record(truth_path, root=root, description="evaluator truth manifest")
    manifest = _json(truth_path)
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise TransferAnalysisError("evaluator truth manifest schema or task identity changed")
    if manifest.get("status") != "TRUTH_BINDINGS_READY_AFTER_PUBLIC_FREEZE":
        raise TransferAnalysisError("evaluator truth manifest is not post-freeze")
    if manifest.get("method_id") != METHOD_ID:
        raise TransferAnalysisError("evaluator truth method changed")
    if list(manifest.get("variant_ids", ())) != list(VARIANT_ORDER):
        raise TransferAnalysisError("evaluator truth variant order changed")
    if list(manifest.get("domain_order", ())) != list(DOMAIN_ORDER):
        raise TransferAnalysisError("evaluator truth domain order changed")
    if manifest.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise TransferAnalysisError("evaluator truth record denominator changed")
    if manifest.get("stored_sequence_tokens") != STORED_SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS_TOKENS:
        raise TransferAnalysisError("evaluator truth token geometry changed")
    order_binding = validate_truth_order_declarations(
        manifest,
        expected_panel_sha256=expected_panel_sha256,
        expected_domain_record_order_sha256=expected_domain_record_order_sha256,
    )
    bindings = manifest.get("truth_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(DOMAIN_ORDER):
        raise TransferAnalysisError("evaluator truth bindings are incomplete")
    # Validate all per-domain order declarations before opening any truth
    # safetensor.  This catches a Finance/Pile swap at the metadata boundary.
    for domain in DOMAIN_ORDER:
        raw = bindings[domain]
        if not isinstance(raw, Mapping):
            raise TransferAnalysisError(f"evaluator truth binding is malformed: {domain}")
        if raw.get("record_ids_sha256") != expected_domain_record_order_sha256[domain]:
            raise TransferAnalysisError(f"evaluator truth record order changed: {domain}")
        if raw.get("records") != RECORDS_PER_DOMAIN or raw.get("stored_sequence_tokens") != STORED_SEQUENCE_TOKENS:
            raise TransferAnalysisError(f"evaluator truth binding geometry changed: {domain}")
    tensors: dict[str, torch.Tensor] = {}
    binding_records: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        raw = bindings[domain]
        record = _record(Path(str(raw.get("path", ""))), root=root, description=f"evaluator truth {domain}")
        if raw.get("bytes") != record["bytes"] or raw.get("sha256") != record["sha256"]:
            raise TransferAnalysisError(f"evaluator truth binding changed: {domain}")
        try:
            with safe_open(str(record["path"]), framework="pt", device="cpu") as handle:
                key = str(raw.get("tensor_key", "truth_tokens"))
                if set(handle.keys()) != {key}:
                    raise TransferAnalysisError(f"evaluator truth tensor keys changed: {domain}")
                truth = handle.get_tensor(key).detach().cpu().contiguous()
        except TransferAnalysisError:
            raise
        except Exception as exc:
            raise TransferAnalysisError(f"evaluator truth tensor is unreadable: {domain}") from exc
        if truth.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or tuple(truth.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
            raise TransferAnalysisError(f"evaluator truth tensor geometry or dtype changed: {domain}")
        if truth[:, 0].ne(BOS_TOKEN_ID).any().item() or truth.lt(0).any().item() or truth.ge(transfer.VOCABULARY_SIZE).any().item():
            raise TransferAnalysisError(f"evaluator truth tensor contains an invalid token: {domain}")
        tensors[domain] = truth.to(dtype=torch.long)
        binding_records[domain] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "records": RECORDS_PER_DOMAIN,
            "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
            "record_ids_sha256": expected_domain_record_order_sha256[domain],
        }
    return {
        "record": truth_record,
        "schema": manifest["schema"],
        "status": manifest["status"],
        "domain_order": list(DOMAIN_ORDER),
        "records_per_domain": RECORDS_PER_DOMAIN,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "panel_sha256": order_binding["panel_sha256"],
        "domain_record_order_sha256": order_binding["domain_record_order_sha256"],
        "truth_bindings": binding_records,
        "truth_opened": True,
    }, tensors


def analyze_transfer_matrix(
    *,
    matrix_path: Path,
    freeze_receipt_path: Path,
    truth_manifest_path: Path,
    repository_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate public predictions, then compute evaluator-only diagnostics."""

    root = Path(repository_root).expanduser().resolve()
    matrix_path = Path(matrix_path).expanduser().resolve()
    freeze_receipt_path = Path(freeze_receipt_path).expanduser().resolve()
    truth_manifest_path = Path(truth_manifest_path).expanduser().resolve()
    matrix_record, _matrix, checked_matrix, checked_input, input_manifest, panel_binding = _validate_public_matrix(matrix_path, root=root)
    freeze = _validate_freeze_receipt(
        freeze_receipt_path,
        matrix_path=matrix_path,
        matrix_record=matrix_record,
        root=root,
    )

    # This is still public data.  The evaluator truth manifest is deliberately
    # not opened until after both gates and the exact null control pass.
    null_controls = {
        domain: _validate_after_cut_null(checked_matrix, checked_input, root=root, domain=domain)
        for domain in DOMAIN_ORDER
    }

    plan_path = Path(str(checked_matrix["variant_plan"]["path"]))
    plan = _json(plan_path)
    try:
        transfer.validate_variant_plan(plan)
    except transfer.TransferDiagnosticError as exc:
        raise TransferAnalysisError(str(exc)) from exc

    # The first evaluator-owned file open occurs here, after the complete
    # prediction matrix, freeze receipt, and after-cut null gates pass.
    truth_metadata, truth_tensors = _truth_bindings_after_gate(
        truth_manifest_path,
        root=root,
        expected_panel_sha256=panel_binding["panel_sha256"],
        expected_domain_record_order_sha256=panel_binding["domain_record_order_sha256"],
    )

    results: dict[str, Any] = {}
    clean_by_domain = {
        domain: _load_cell(checked_matrix, checked_input, root=root, variant_id=VARIANT_ORDER[0], domain=domain)
        for domain in DOMAIN_ORDER
    }
    parameter_bindings = {
        variant_id: _parameter_binding(plan, input_manifest, variant_id)
        for variant_id in VARIANT_ORDER[1:]
    }
    for variant_id in VARIANT_ORDER[1:]:
        parameter_binding = parameter_bindings[variant_id]
        parameter_score = parameter_binding["actual_relative_parameter_l2"]
        for domain in DOMAIN_ORDER:
            clean = clean_by_domain[domain]
            changed = _load_cell(checked_matrix, checked_input, root=root, variant_id=variant_id, domain=domain)
            truth = truth_tensors[domain]
            baseline = clean["prediction"][:, 1:]
            candidate = changed["prediction"][:, 1:]
            truth_scored = truth[:, 1:]
            rows = RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS
            try:
                inventory = transfer.evaluator_only_transfer_inventory(
                    baseline,
                    candidate,
                    truth_scored,
                )
            except transfer.TransferDiagnosticError as exc:
                raise TransferAnalysisError(f"truth inventory failed: {variant_id}/{domain}: {exc}") from exc
            accounting = {
                "clean": _truth_accounting(clean["prediction"], truth, label="clean"),
                "changed": _truth_accounting(changed["prediction"], truth, label=variant_id),
            }
            scores = _row_scores(clean, changed, parameter_score=parameter_score)
            aucs: dict[str, Any] = {}
            for score_name in FIXED_SCORE_NAMES:
                values = scores[score_name]
                invalid_rows = int((~torch.isfinite(values)).sum().item())
                if invalid_rows:
                    aucs[score_name] = _unknown_auc(
                        score_name,
                        reason="score contains invalid rows under the fixed public geometry contract",
                        rows=rows,
                        invalid_rows=invalid_rows,
                    )
                else:
                    try:
                        aucs[score_name] = transfer.evaluator_only_fixed_auc(
                            values,
                            baseline,
                            candidate,
                            truth_scored,
                            score_name=score_name,
                        )
                    except transfer.TransferDiagnosticError as exc:
                        raise TransferAnalysisError(f"fixed AUC failed: {variant_id}/{domain}/{score_name}: {exc}") from exc
                    aucs[score_name]["invalid_score_rows"] = 0
            geometry = _summarize_loaded_cell_geometry(
                clean,
                changed,
                method_id=METHOD_ID,
                variant_id=variant_id,
                domain=domain,
                parameter_binding=parameter_binding,
            )
            results[f"{variant_id}/{domain}"] = {
                "method_id": METHOD_ID,
                "variant_id": variant_id,
                "domain": domain,
                "records_per_domain": RECORDS_PER_DOMAIN,
                "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
                "paired_rows": rows,
                "accounting": accounting,
                "parameter_binding": parameter_binding,
                "inventory": inventory,
                "fixed_auc": aucs,
                "truth_dependent": True,
                "deployable": False,
                "geometry_summary": geometry,
            }

    return {
        "schema": ANALYSIS_SCHEMA,
        "task_id": TASK_ID,
        "status": "COMPUTED_AFTER_PUBLIC_FREEZE",
        "method_id": METHOD_ID,
        "variant_ids": list(VARIANT_ORDER),
        "domain_order": list(DOMAIN_ORDER),
        "records_per_domain": RECORDS_PER_DOMAIN,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "paired_rows_per_cell": RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS,
        "fixed_auc_score_names": list(FIXED_SCORE_NAMES),
        "risk_set": "baseline_correct",
        "positive_label": "broken",
        "negative_label": "remaining_correct",
        "threshold_fitted": False,
        "public_matrix": matrix_record,
        "public_freeze": freeze,
        "after_cut_null": null_controls,
        "public_panel": panel_binding,
        "parameter_bindings": parameter_bindings,
        "truth_manifest": truth_metadata,
        "results": results,
        "truth_opened": True,
        "source_text_loaded": False,
        "target_labels_loaded": True,
        "candidate_arrays_persisted": False,
        "private_or_truth_payload_read": True,
        "fresh_evaluation_started": True,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    task_root = (Path(root).resolve() / "experiments" / TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise TransferAnalysisError("analysis output is outside the task root") from exc
    if path.exists() or path.is_symlink():
        raise TransferAnalysisError(f"analysis output is not create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="complete public 14-cell prediction matrix")
    parser.add_argument("--freeze-receipt", type=Path, required=True, help="public before-truth freeze/audit receipt")
    parser.add_argument("--truth-manifest", type=Path, required=True, help="Agent2 evaluator-side truth binding")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.repository_root).expanduser().resolve()
        result = analyze_transfer_matrix(
            matrix_path=args.matrix,
            freeze_receipt_path=args.freeze_receipt,
            truth_manifest_path=args.truth_manifest,
            repository_root=root,
            output_path=args.output,
        )
        result["analysis_code"] = {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": _sha256(Path(__file__).resolve()),
        }
        result["output"] = _write_create_only(args.output, result, root=root)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (TransferAnalysisError, transfer.TransferDiagnosticError, OSError, RuntimeError) as exc:
        print(f"TRR0012 transfer analysis error: {exc}")
        return 2


__all__ = ["TransferAnalysisError", "analyze_transfer_matrix"]


if __name__ == "__main__":
    raise SystemExit(main())
