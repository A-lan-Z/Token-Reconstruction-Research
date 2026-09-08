"""Evaluator-side controlled-target capture for TRR-0011.

This module is a narrow producer wrapper around the existing TRR-0004 public
prefix path.  It freezes a truth-free target-update recipe, validates a
separately supplied opaque paired panel, runs a full B8x192 forward through
the public prefix, and emits only BF16 cut-4 observations, masks, positions,
and opaque record bindings.  Target weights and tokenized source inputs stay
in the evaluator process.  The module does not select sources, download
assets, train, open truth, or decode predictions.

Production callers supply a fresh public target model through ``model_factory``
and evaluator-owned ``PaddedTokenBatch`` objects through ``batches``.  The
factory is called once per variant, so a suffix-layer null update is applied to
the full target model before ``ContiguousPublicPrefix`` is constructed; the
post-cut update is therefore real while the observed prefix activation must be
bitwise unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Permit direct ``python scripts/trr0011_capture.py`` invocation while keeping
# the actual public-prefix implementation in the source package.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _published_repository_root() -> Path:
    """Resolve the existing published output root, outside this worktree."""

    candidates = [_ROOT]
    if _ROOT.parent.name == ".worktrees":
        candidates.insert(0, _ROOT.parent.parent)
    for candidate in candidates:
        generation = candidate / "outputs" / "TRR-0002" / "public-calibration" / "generation.json"
        update = candidate / "outputs" / "TRR-0002" / "public-calibration" / "updates" / "public_lora_2601.safetensors"
        if generation.is_file() and update.is_file():
            return candidate.resolve()
    raise CaptureError("published TRR-0002 public_lora_2601 assets are unavailable")


from token_reconstruction.public_activation import (  # noqa: E402
    PaddedTokenBatch,
    capture_public_prefix,
    pad_public_token_sequences,
    validate_padded_token_batch,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402


TASK_ID = "TRR-0011"
CAPTURE_PLAN_SCHEMA = "token-reconstruction.trr0011-controlled-target-capture-plan.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr0011-controlled-target-capture.v1"
CAPTURE_MANIFEST_SCHEMA = "token-reconstruction.trr0011-controlled-target-capture-manifest.v1"
CUT_DEPTH = 4
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
CAPTURE_SEQUENCE_TOKENS = 192
RETAINED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
CAPTURE_BATCH_SIZE = 8
BLOCK_ELEMENTS = 4096
RELATIVE_L2_LEVELS = (1e-3, 1e-2)
DOMAINS = ("finance", "pile")
REQUIRED_RECORDS_PER_DOMAIN = 32
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
KNOWN_MODEL_WEIGHT_BYTES = 2_471_645_608
KNOWN_MODEL_WEIGHT_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
SOURCE_BATCH_SCHEMA = "token-reconstruction.trr0011-evaluator-token-batches.v1"
FULL_REFERENCE_SCHEMA = "token-reconstruction.trr0011-full-activation-reference.v1"
TRANSFER_RUN_SCHEMA = "token-reconstruction.trr0011-transfer-public-inference.v1"
TRANSFER_RUN_STATUS = "PUBLIC_TRANSFER_INPUTS_VALIDATED_BEFORE_TRUTH"
TRANSFER_METHOD_IDS = ("expanded_fixed", "current_fixed", "continued_fixed_readout")

TRUTH_FLAGS = (
    "truth_opened",
    "source_text_loaded",
    "target_labels_loaded",
    "private_or_truth_payload_read",
    "candidate_arrays_persisted",
    "fresh_evaluation_started",
)

EXPECTED_VARIANTS = (
    "clean_public",
    "layer1_block_rel1e-3",
    "layer1_block_rel1e-2",
    "layer3_block_rel1e-3",
    "layer3_block_rel1e-2",
    "layer4_null_block_rel1e-2",
    "historical_public_lora_2601",
)


class CaptureError(ValueError):
    """Raised when a capture recipe, panel, or artifact fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureError(f"{label} must be an object")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise CaptureError(f"{label} must be a lowercase SHA-256")
    return value


def _record_file(path: Path, *, root: Path | None = None, allow_symlink: bool = False) -> dict[str, Any]:
    original = Path(path).expanduser()
    path = original.resolve()
    if (original.is_symlink() and not allow_symlink) or not path.is_file():
        raise CaptureError(f"file binding is unavailable: {path}")
    display = str(path)
    if root is not None:
        try:
            display = str(path.relative_to(Path(root).resolve()))
        except ValueError:
            pass
    return {
        "path": display,
        "resolved_path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "readonly": True,
        "symlink": bool(original.is_symlink()),
    }


def _file_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"JSON object required: {path}")
    return value


def default_capture_plan() -> dict[str, Any]:
    """Return the exact prospective recipe before panel/source binding."""

    def update(variant_id: str, layer: int, amplitude: float, seed: int) -> dict[str, Any]:
        return {
            "variant_id": variant_id,
            "role": "artificial_fixed_sign_block",
            "artificial": True,
            "update_location": {
                "layer_index": layer,
                "parameter_path": f"model.layers.{layer}.self_attn.q_proj.weight",
                "block_offset": 0,
                "block_elements": BLOCK_ELEMENTS,
                "relative_parameter_l2": amplitude,
                "recipe_seed": seed,
                "sign_pattern": "alternating_seed_phase",
                "block_order": "flattened_row_major",
            },
        }

    published_root = _published_repository_root()
    generation_path = published_root / "outputs" / "TRR-0002" / "public-calibration" / "generation.json"
    update_path = published_root / "outputs" / "TRR-0002" / "public-calibration" / "updates" / "public_lora_2601.safetensors"
    generation_binding = _record_file(generation_path)
    update_binding = _record_file(update_path)

    plan = {
        "schema": CAPTURE_PLAN_SCHEMA,
        "task_id": TASK_ID,
        "status": "DRAFT_CAPTURE_RECIPE",
        "geometry": {
            "cut_depth": CUT_DEPTH,
            "hidden_size": HIDDEN_SIZE,
            "vocabulary_size": VOCABULARY_SIZE,
            "capture_batch_size": CAPTURE_BATCH_SIZE,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "retained_sequence_tokens": RETAINED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
            "bos_token_id": BOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
            "activation_dtype": "torch.bfloat16",
            "attention_mask_dtype": "torch.uint8",
            "position_ids_dtype": "torch.int64",
        },
        "target_update_recipe": {
            "mode": "fixed_sign_contiguous_parameter_block",
            "parameter_path_template": "model.layers.{layer_index}.self_attn.q_proj.weight",
            "block_elements": BLOCK_ELEMENTS,
            "block_order": "flattened_row_major",
            "sign_pattern": "sign_i = +1 when (i + recipe_seed) is even, otherwise -1",
            "relative_l2_definition": "requested ||delta||_2 / ||base_parameter||_2; actual ratio is recomputed after BF16 parameter assignment",
            "update_dtype": "FP32 construction then target parameter dtype assignment",
            "mutation": "torch.no_grad in evaluator-side full target model before prefix construction",
            "seeds_fixed_before_observations": True,
        },
        "variants": [
            {"variant_id": "clean_public", "role": "clean_public_target", "artificial": False, "update_location": None},
            update("layer1_block_rel1e-3", 1, 1e-3, 1101),
            update("layer1_block_rel1e-2", 1, 1e-2, 1101),
            update("layer3_block_rel1e-3", 3, 1e-3, 1101),
            update("layer3_block_rel1e-2", 3, 1e-2, 1101),
            {
                "variant_id": "layer4_null_block_rel1e-2",
                "role": "after_cut_suffix_null_control",
                "artificial": True,
                "expected_activation_equivalence": True,
                "update_location": {
                    "layer_index": CUT_DEPTH,
                    "parameter_path": f"model.layers.{CUT_DEPTH}.self_attn.q_proj.weight",
                    "block_offset": 0,
                    "block_elements": BLOCK_ELEMENTS,
                    "relative_parameter_l2": 1e-2,
                    "recipe_seed": 1101,
                    "sign_pattern": "alternating_seed_phase",
                    "block_order": "flattened_row_major",
                },
            },
            {
                "variant_id": "historical_public_lora_2601",
                "role": "historical_trained_benchmark_adaptation",
                "artificial": False,
                "target_condition": "public_lora_2601",
                "mandatory_declared_condition": True,
                "available_for_run": True,
                "independent_adaptation_claim": False,
                "training_provenance": {
                    "status": "HISTORICAL_TRAINED_BENCHMARK_ADAPTATION",
                    "generation_receipt": "outputs/TRR-0002/public-calibration/generation.json",
                    "generation_receipt_sha256": "6bed8aca6668bd749bbad03c0c85bbace306085dac3141895dad7685f03d0682",
                    "update_sha256": "eea7bb49f801b61df2e26a8f59af7c3096f6f3a2604404e16e589443bcfba595",
                    "steps": 30,
                    "records": 32,
                    "domain": "Pile",
                    "layers": [0, 1, 2, 3],
                    "modules": ["q_proj", "v_proj"],
                    "rank": 4,
                    "alpha": 4,
                    "asset_bindings": {},
                },
                "update_location": None,
            },
        ],
        "panel_binding": {
            "status": "PENDING_P10_OPAQUE_RESERVATION",
            "records_per_domain": REQUIRED_RECORDS_PER_DOMAIN,
            "domains": list(DOMAINS),
            "same_record_order_across_variants": True,
            "selection_performed": False,
            "opaque_reservation": None,
        },
        "truth_boundary": {
            "target_full_weights_evaluator_only": True,
            "tokenized_sources_evaluator_only": True,
            "decoder_inputs": ["H", "attention_mask", "position_ids", "opaque_record_ids"],
            "truth_sidecar": "separate post-prediction gate; not emitted by capture runner",
            **{flag: False for flag in TRUTH_FLAGS},
        },
        "resource_policy": {
            "launch_authorized": False,
            "download_or_paid_compute": False,
            "one_variant_per_process": True,
            "watchdog_seconds_per_variant": 900,
            "reserved_gpu_ceiling_bytes": 8 * 2**30,
            "minimum_free_gpu_bytes": 11 * 2**30,
            "host_rss_cap_bytes": 12 * 2**30,
            "minimum_host_headroom_bytes": 8 * 2**30,
            "minimum_disk_free_bytes": 20 * 2**30,
        },
        "observations_started": False,
        "source_selection_performed": False,
        "gpu_run_performed": False,
    }
    historical = next(row for row in plan["variants"] if row["variant_id"] == "historical_public_lora_2601")
    historical["training_provenance"]["asset_bindings"] = {
        "generation_receipt": generation_binding,
        "update": update_binding,
    }
    return plan


def freeze_capture_plan(plan: Mapping[str, Any], *, frozen_utc: str | None = None, source_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the recipe before any panel inputs or observations are admitted."""

    value = json.loads(json.dumps(dict(plan)))
    validate_capture_plan(value, require_frozen=False)
    value["status"] = "FROZEN_CAPTURE_RECIPE_BEFORE_OBSERVATIONS"
    value["frozen_utc"] = frozen_utc or _utc_now()
    value["observations_started"] = False
    value["source_selection_performed"] = False
    value["gpu_run_performed"] = False
    if source_binding is not None:
        value["capture_source_binding"] = dict(source_binding)
    validate_capture_plan(value, require_frozen=True)
    return value


def _validate_update_location(variant: Mapping[str, Any], *, expected_id: str) -> dict[str, Any]:
    location = _require_mapping(variant.get("update_location"), label=f"{expected_id}.update_location")
    layer = location.get("layer_index")
    if layer not in (1, 3, 4):
        raise CaptureError(f"{expected_id} has an invalid layer index")
    if location.get("parameter_path") != f"model.layers.{layer}.self_attn.q_proj.weight":
        raise CaptureError(f"{expected_id} parameter path is not the frozen q_proj block")
    if location.get("block_offset") != 0 or location.get("block_elements") != BLOCK_ELEMENTS:
        raise CaptureError(f"{expected_id} block geometry changed")
    amplitude = location.get("relative_parameter_l2")
    if not isinstance(amplitude, (int, float)) or isinstance(amplitude, bool) or not any(math.isclose(float(amplitude), level, rel_tol=0.0, abs_tol=1e-12) for level in RELATIVE_L2_LEVELS):
        raise CaptureError(f"{expected_id} amplitude is not preregistered")
    if not isinstance(location.get("recipe_seed"), int) or isinstance(location.get("recipe_seed"), bool):
        raise CaptureError(f"{expected_id} recipe seed is malformed")
    if location.get("sign_pattern") != "alternating_seed_phase" or location.get("block_order") != "flattened_row_major":
        raise CaptureError(f"{expected_id} sign/block recipe changed")
    return dict(location)


def validate_capture_plan(plan: Mapping[str, Any], *, require_frozen: bool = True) -> dict[str, Any]:
    """Validate the frozen recipe and all truth/source-selection boundaries."""

    if not isinstance(plan, Mapping) or plan.get("schema") != CAPTURE_PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise CaptureError("capture plan schema or task identity changed")
    if require_frozen and plan.get("status") != "FROZEN_CAPTURE_RECIPE_BEFORE_OBSERVATIONS":
        raise CaptureError("capture plan is not frozen before observations")
    if not require_frozen and plan.get("status") not in {"DRAFT_CAPTURE_RECIPE", "FROZEN_CAPTURE_RECIPE_BEFORE_OBSERVATIONS"}:
        raise CaptureError("capture plan status is invalid")
    for flag in TRUTH_FLAGS:
        if _truthy(plan.get(flag)) or _truthy(_require_mapping(plan.get("truth_boundary", {}), label="truth_boundary").get(flag)):
            raise CaptureError(f"capture plan records forbidden access: {flag}")
    if plan.get("observations_started") is True or plan.get("source_selection_performed") is True or plan.get("gpu_run_performed") is True:
        raise CaptureError("capture plan cannot be reused after execution or selection")
    geometry = _require_mapping(plan.get("geometry"), label="geometry")
    expected_geometry = {
        "cut_depth": CUT_DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "vocabulary_size": VOCABULARY_SIZE,
        "capture_batch_size": CAPTURE_BATCH_SIZE,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "retained_sequence_tokens": RETAINED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "bos_token_id": BOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "activation_dtype": "torch.bfloat16",
        "attention_mask_dtype": "torch.uint8",
        "position_ids_dtype": "torch.int64",
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            raise CaptureError(f"geometry changed: {key}")
    recipe = _require_mapping(plan.get("target_update_recipe"), label="target_update_recipe")
    for key, expected in {
        "mode": "fixed_sign_contiguous_parameter_block",
        "parameter_path_template": "model.layers.{layer_index}.self_attn.q_proj.weight",
        "block_elements": BLOCK_ELEMENTS,
        "block_order": "flattened_row_major",
        "sign_pattern": "sign_i = +1 when (i + recipe_seed) is even, otherwise -1",
        "update_dtype": "FP32 construction then target parameter dtype assignment",
        "mutation": "torch.no_grad in evaluator-side full target model before prefix construction",
    }.items():
        if recipe.get(key) != expected:
            raise CaptureError(f"target update recipe changed: {key}")
    if recipe.get("seeds_fixed_before_observations") is not True:
        raise CaptureError("target update seeds are not frozen")
    variants = plan.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes, bytearray)):
        raise CaptureError("capture variants are absent")
    rows = {str(row.get("variant_id")): row for row in variants if isinstance(row, Mapping)}
    if tuple(rows) != EXPECTED_VARIANTS or len(rows) != len(EXPECTED_VARIANTS):
        raise CaptureError("capture variant order or completeness changed")
    clean = rows["clean_public"]
    if clean.get("role") != "clean_public_target" or clean.get("update_location") is not None:
        raise CaptureError("clean variant must have no update")
    expected_prefix = {
        "layer1_block_rel1e-3": (1, 1e-3, 1101),
        "layer1_block_rel1e-2": (1, 1e-2, 1101),
        "layer3_block_rel1e-3": (3, 1e-3, 1101),
        "layer3_block_rel1e-2": (3, 1e-2, 1101),
    }
    for variant_id, (layer, amplitude, seed) in expected_prefix.items():
        row = rows[variant_id]
        if row.get("role") != "artificial_fixed_sign_block" or row.get("artificial") is not True:
            raise CaptureError(f"{variant_id} role changed")
        location = _validate_update_location(row, expected_id=variant_id)
        if location["layer_index"] != layer or not math.isclose(float(location["relative_parameter_l2"]), amplitude, rel_tol=0.0, abs_tol=1e-12) or location["recipe_seed"] != seed:
            raise CaptureError(f"{variant_id} layer, amplitude, or seed changed")
    null = rows["layer4_null_block_rel1e-2"]
    if null.get("role") != "after_cut_suffix_null_control" or null.get("artificial") is not True or null.get("expected_activation_equivalence") is not True:
        raise CaptureError("after-cut null contract changed")
    null_location = _validate_update_location(null, expected_id="layer4_null_block_rel1e-2")
    if null_location["layer_index"] != CUT_DEPTH or not math.isclose(float(null_location["relative_parameter_l2"]), 1e-2, rel_tol=0.0, abs_tol=1e-12) or null_location["recipe_seed"] != 1101:
        raise CaptureError("after-cut null location or recipe changed")
    historical = rows["historical_public_lora_2601"]
    if historical.get("role") != "historical_trained_benchmark_adaptation" or historical.get("target_condition") != "public_lora_2601":
        raise CaptureError("historical public_lora condition is mislabeled")
    if historical.get("independent_adaptation_claim") is not False or historical.get("available_for_run") is not True or historical.get("mandatory_declared_condition") is not True or "optional_heldout_family" in historical:
        raise CaptureError("historical public_lora must be a mandatory available condition, never an optional held-out family")
    training = _require_mapping(historical.get("training_provenance"), label="historical training_provenance")
    if training.get("status") != "HISTORICAL_TRAINED_BENCHMARK_ADAPTATION" or training.get("steps") != 30 or training.get("records") != 32 or training.get("domain") != "Pile":
        raise CaptureError("historical public_lora training provenance changed")
    if tuple(training.get("layers", ())) != (0, 1, 2, 3) or tuple(training.get("modules", ())) != ("q_proj", "v_proj") or training.get("rank") != 4 or training.get("alpha") != 4:
        raise CaptureError("historical public_lora LoRA recipe changed")
    _require_sha(training.get("generation_receipt_sha256"), label="historical generation receipt")
    _require_sha(training.get("update_sha256"), label="historical update")
    assets = _require_mapping(training.get("asset_bindings"), label="historical asset_bindings")
    expected_assets = {
        "generation_receipt": (training.get("generation_receipt_sha256"), "historical generation asset"),
        "update": (training.get("update_sha256"), "historical update asset"),
    }
    for asset_name, (expected_digest, label) in expected_assets.items():
        binding = _require_mapping(assets.get(asset_name), label=f"historical asset {asset_name}")
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise CaptureError(f"{label} must bind an absolute existing path")
        actual = _record_file(Path(raw_path))
        if actual["bytes"] != binding.get("bytes") or actual["sha256"] != binding.get("sha256") or actual["sha256"] != expected_digest:
            raise CaptureError(f"{label} bytes or SHA-256 changed")
    if historical.get("update_location") is not None:
        raise CaptureError("historical public_lora cannot acquire an artificial update recipe")
    panel = _require_mapping(plan.get("panel_binding"), label="panel_binding")
    if panel.get("records_per_domain") != REQUIRED_RECORDS_PER_DOMAIN or tuple(panel.get("domains", ())) != DOMAINS or panel.get("same_record_order_across_variants") is not True or panel.get("selection_performed") is not False:
        raise CaptureError("panel binding changed")
    resources = _require_mapping(plan.get("resource_policy"), label="resource_policy")
    if resources.get("launch_authorized") is not False or resources.get("download_or_paid_compute") is not False or resources.get("one_variant_per_process") is not True:
        raise CaptureError("capture launch/resource policy changed")
    return {
        "schema": CAPTURE_PLAN_SCHEMA,
        "task_id": TASK_ID,
        "status": str(plan.get("status")),
        "variant_ids": list(EXPECTED_VARIANTS),
        "records_per_domain": REQUIRED_RECORDS_PER_DOMAIN,
        "geometry": dict(geometry),
        "truth_free": True,
        "source_selection_performed": False,
    }


def opaque_record_ids_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered domain/position/opaque-ID triples without source content."""

    digest = hashlib.sha256()
    for row in records:
        digest.update(f"{row['domain']}\t{int(row['order'])}\t{row['opaque_id']}\n".encode("utf-8"))
    return digest.hexdigest()


def canonical_panel_sha256(panel: Mapping[str, Any]) -> str:
    """Hash a panel descriptor while excluding its self-referential digest field."""

    if not isinstance(panel, Mapping):
        raise CaptureError("opaque panel must be a mapping")
    value = json.loads(json.dumps(dict(panel)))
    value.pop("panel_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_opaque_panel_descriptor(panel: Mapping[str, Any], *, require_reserved: bool = True) -> dict[str, Any]:
    """Validate the separately supplied 32/domain paired opaque panel."""

    if not isinstance(panel, Mapping) or panel.get("schema") != "token-reconstruction.trr0011-opaque-capture-panel.v1" or panel.get("task_id") != TASK_ID:
        raise CaptureError("opaque panel schema or task identity changed")
    if require_reserved and panel.get("status") != "RESERVED_OPAQUE_PANEL_BEFORE_CAPTURE":
        raise CaptureError("opaque panel is not reserved before capture")
    if panel.get("records_per_domain") != REQUIRED_RECORDS_PER_DOMAIN or tuple(panel.get("domains", ())) != DOMAINS:
        raise CaptureError("opaque panel domain/count contract changed")
    if panel.get("same_record_order_across_variants") is not True or panel.get("selection_performed") is not False:
        raise CaptureError("opaque panel is not paired or records were selected here")
    for flag in TRUTH_FLAGS:
        if _truthy(panel.get(flag)):
            raise CaptureError(f"opaque panel records forbidden access: {flag}")
    reservation = _require_mapping(panel.get("opaque_reservation"), label="opaque_reservation")
    _require_sha(reservation.get("receipt_sha256"), label="opaque_reservation.receipt_sha256")
    rows = panel.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or len(rows) != len(DOMAINS) * REQUIRED_RECORDS_PER_DOMAIN:
        raise CaptureError("opaque panel must contain exactly 64 rows")
    normalized: list[dict[str, Any]] = []
    forbidden = {"source", "source_id", "source_path", "source_text", "text", "tokens", "token_ids", "answers", "labels", "row_index"}
    seen: set[str] = set()
    per_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    for raw in rows:
        row = _require_mapping(raw, label="opaque panel record")
        if set(row) & forbidden:
            raise CaptureError("opaque panel contains source/token/truth fields")
        domain = row.get("domain")
        opaque_id = row.get("opaque_id")
        order = row.get("order")
        if domain not in DOMAINS or not isinstance(opaque_id, str) or not opaque_id or not isinstance(order, int) or isinstance(order, bool):
            raise CaptureError("opaque panel record identity is malformed")
        if opaque_id in seen:
            raise CaptureError("opaque panel contains duplicate opaque IDs")
        seen.add(opaque_id)
        per_domain[str(domain)].append(int(order))
        normalized.append({"domain": str(domain), "opaque_id": opaque_id, "order": int(order)})
    for domain in DOMAINS:
        if sorted(per_domain[domain]) != list(range(REQUIRED_RECORDS_PER_DOMAIN)):
            raise CaptureError(f"opaque panel order is not 0..31 for {domain}")
    digest = opaque_record_ids_digest(normalized)
    declared = panel.get("opaque_record_ids_sha256")
    if declared is not None and declared != digest:
        raise CaptureError("opaque panel ordered-ID digest changed")
    panel_sha = panel.get("panel_sha256")
    if panel_sha is not None:
        _require_sha(panel_sha, label="panel_sha256")
        if panel_sha != canonical_panel_sha256(panel):
            raise CaptureError("opaque panel complete SHA-256 changed")
    return {
        "schema": str(panel["schema"]),
        "task_id": TASK_ID,
        "status": str(panel["status"]),
        "records_per_domain": REQUIRED_RECORDS_PER_DOMAIN,
        "record_count": len(normalized),
        "domains": list(DOMAINS),
        "opaque_reservation_sha256": str(reservation["receipt_sha256"]),
        "opaque_record_ids_sha256": digest,
        "panel_sha256": panel_sha if panel_sha is not None else canonical_panel_sha256(panel),
        "records": normalized,
        "truth_free": True,
    }


def _resolve_parameter(root: Any, path: str) -> torch.nn.Parameter:
    current = root
    for part in path.split("."):
        if not part:
            raise CaptureError("parameter path contains an empty component")
        if part.isdigit() and isinstance(current, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
            index = int(part)
            try:
                current = current[index]
            except (IndexError, TypeError) as exc:
                raise CaptureError(f"parameter path index is unavailable: {path}") from exc
        else:
            try:
                current = getattr(current, part)
            except AttributeError as exc:
                raise CaptureError(f"parameter path is unavailable: {path}") from exc
    if not isinstance(current, torch.nn.Parameter):
        raise CaptureError(f"parameter path does not resolve to a Parameter: {path}")
    return current


def _fixed_signs(length: int, seed: int) -> torch.Tensor:
    indices = torch.arange(length, dtype=torch.int64)
    return torch.where(((indices + int(seed)) % 2) == 0, torch.ones(length), -torch.ones(length)).to(torch.float32)


def apply_fixed_sign_block_update(model: Any, variant: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one frozen evaluator-side update and report BF16-effective distance."""

    role = variant.get("role")
    if role in {"clean_public_target", "historical_trained_benchmark_adaptation"}:
        return {"status": "NO_UPDATE", "variant_id": str(variant.get("variant_id")), "update_applied": False}
    location = _require_mapping(variant.get("update_location"), label="update_location")
    path = str(location.get("parameter_path", ""))
    parameter = _resolve_parameter(model, path)
    if not parameter.dtype.is_floating_point or parameter.numel() <= 0:
        raise CaptureError("target update parameter must be non-empty floating point")
    block_offset = int(location.get("block_offset", -1))
    block_elements = int(location.get("block_elements", -1))
    seed = int(location.get("recipe_seed", -1))
    amplitude = float(location.get("relative_parameter_l2", float("nan")))
    if block_offset < 0 or block_elements <= 0 or block_offset + block_elements > parameter.numel():
        raise CaptureError("target update block is outside the parameter")
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise CaptureError("target update amplitude is invalid")
    # Clone before mutation: on CPU, contiguous float32 views can alias the
    # parameter and would erase the effective-distance comparison.
    before = parameter.detach().float().cpu().contiguous().clone()
    base_norm = float(torch.linalg.vector_norm(before).item())
    if not math.isfinite(base_norm) or base_norm <= 0.0:
        raise CaptureError("target update base parameter norm is not positive")
    requested_delta_norm = amplitude * base_norm
    per_element = requested_delta_norm / math.sqrt(block_elements)
    signs = _fixed_signs(block_elements, seed)
    updated = before.reshape(-1).clone()
    updated[block_offset : block_offset + block_elements] += signs * per_element
    updated = updated.reshape_as(before)
    with torch.no_grad():
        parameter.copy_(updated.to(device=parameter.device, dtype=parameter.dtype))
    after = parameter.detach().float().cpu().contiguous()
    effective_delta = after - before
    effective_delta_norm = float(torch.linalg.vector_norm(effective_delta).item())
    effective_relative = effective_delta_norm / base_norm
    return {
        "status": "UPDATED",
        "variant_id": str(variant.get("variant_id")),
        "update_applied": True,
        "parameter_path": path,
        "layer_index": int(location["layer_index"]),
        "block_offset": block_offset,
        "block_elements": block_elements,
        "recipe_seed": seed,
        "requested_relative_parameter_l2": amplitude,
        "requested_delta_l2": requested_delta_norm,
        "base_parameter_l2": base_norm,
        "effective_delta_l2_after_parameter_dtype": effective_delta_norm,
        "effective_relative_parameter_l2": effective_relative,
        "effective_update_nonzero": bool(torch.count_nonzero(effective_delta).item()),
        "base_parameter_digest": tensor_digest(before),
        "updated_parameter_digest": tensor_digest(after),
        "bf16_rounding_may_zero_small_update": parameter.dtype == torch.bfloat16,
    }


def validate_after_cut_null(clean_activation: torch.Tensor, changed_activation: torch.Tensor) -> dict[str, Any]:
    """Require exact prefix equality after a suffix update is actually applied."""

    clean = torch.as_tensor(clean_activation).detach().cpu().contiguous()
    changed = torch.as_tensor(changed_activation).detach().cpu().contiguous()
    if tuple(clean.shape) != tuple(changed.shape):
        raise CaptureError("after-cut null activation geometry changed")
    if not torch.equal(clean, changed):
        raise CaptureError("after-cut null changed cut activation")
    return {"status": "PASS", "activation_exact_equal": True}


def _validate_capture_batch(batch: PaddedTokenBatch, *, records: int) -> None:
    try:
        validate_padded_token_batch(
            batch,
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            vocab_size=VOCABULARY_SIZE,
            small_post_bos_positions=5000,
        )
    except Exception as exc:
        raise CaptureError("evaluator token batch failed the frozen public geometry") from exc
    if batch.token_ids.shape[0] != records:
        raise CaptureError("evaluator token batch record count differs from opaque panel")


def _retained_observation_tensors(full_activation: torch.Tensor, batch: PaddedTokenBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(full_activation.shape) != (batch.token_ids.shape[0], CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE) or full_activation.dtype != torch.bfloat16:
        raise CaptureError("public full-forward activation geometry or dtype changed")
    retained = full_activation[:, :RETAINED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    mask = batch.attention_mask[:, :RETAINED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    positions = batch.position_ids[:, :RETAINED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    return retained, mask, positions


def save_observation_artifact(
    path: Path,
    *,
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    variant_id: str,
    domain: str,
    opaque_record_ids: Sequence[str],
    panel_sha256: str,
    capture_code_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write only H/mask/positions and opaque metadata, never token IDs."""

    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"observation artifact is create-only: {path}")
    if domain not in DOMAINS or len(opaque_record_ids) != REQUIRED_RECORDS_PER_DOMAIN:
        raise CaptureError("observation domain or opaque ID count changed")
    activations = torch.as_tensor(activations).detach().cpu().contiguous()
    attention_mask = torch.as_tensor(attention_mask).detach().cpu().contiguous()
    position_ids = torch.as_tensor(position_ids).detach().cpu().contiguous()
    if tuple(activations.shape) != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS, HIDDEN_SIZE) or activations.dtype != torch.bfloat16:
        raise CaptureError("retained observation activation geometry changed")
    if tuple(attention_mask.shape) != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS) or attention_mask.dtype != torch.uint8:
        raise CaptureError("retained observation mask geometry changed")
    if tuple(position_ids.shape) != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS) or position_ids.dtype != torch.int64:
        raise CaptureError("retained observation positions geometry changed")
    if not torch.isfinite(activations.float()).all().item():
        raise CaptureError("retained observation contains non-finite activations")
    if not torch.logical_or(attention_mask.eq(0), attention_mask.eq(1)).all().item():
        raise CaptureError("retained observation mask is not binary")
    metadata = {
        "schema": CAPTURE_SCHEMA,
        "task_id": TASK_ID,
        "variant_id": str(variant_id),
        "domain": domain,
        "record_count": str(REQUIRED_RECORDS_PER_DOMAIN),
        "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
        "retained_sequence_tokens": str(RETAINED_SEQUENCE_TOKENS),
        "cut_depth": str(CUT_DEPTH),
        "hidden_size": str(HIDDEN_SIZE),
        "activation_dtype": "torch.bfloat16",
        "attention_mask_dtype": "torch.uint8",
        "position_ids_dtype": "torch.int64",
        "opaque_record_ids_sha256": opaque_record_ids_digest([{"domain": domain, "order": i, "opaque_id": value} for i, value in enumerate(opaque_record_ids)]),
        "panel_sha256": _require_sha(panel_sha256, label="panel_sha256"),
        "capture_code_binding": json.dumps(dict(capture_code_binding), sort_keys=True, separators=(",", ":")),
        "target_full_weights_evaluator_only": "true",
        "tokenized_sources_evaluator_only": "true",
        "token_ids_written": "false",
        "source_text_loaded": "false",
        "target_labels_loaded": "false",
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"activations": activations, "attention_mask": attention_mask, "position_ids": position_ids}, str(path), metadata=metadata)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path), "variant_id": str(variant_id), "domain": domain, "opaque_record_ids_sha256": metadata["opaque_record_ids_sha256"]}


def validate_observation_artifact(path: Path, *, expected_variant_id: str, expected_domain: str, expected_panel_sha256: str, expected_opaque_record_ids_sha256: str) -> dict[str, Any]:
    """Validate emitted observation tensors without reading source or truth."""

    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CaptureError(f"observation artifact unavailable: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
            raise CaptureError("observation contains an unexpected tensor, including possible token IDs")
        # The installed safetensors API exposes tensor shapes through loaded
        # tensors, not a get_shape method.  Loading only these permitted public
        # tensors also keeps validation independent of token/source payloads.
        activation = handle.get_tensor("activations")
        mask = handle.get_tensor("attention_mask")
        positions = handle.get_tensor("position_ids")
        activation_shape = tuple(activation.shape)
        mask_shape = tuple(mask.shape)
        position_shape = tuple(positions.shape)
        metadata = dict(handle.metadata() or {})
        if activation_shape != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS, HIDDEN_SIZE) or mask_shape != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS) or position_shape != (REQUIRED_RECORDS_PER_DOMAIN, RETAINED_SEQUENCE_TOKENS):
            raise CaptureError("observation tensor geometry changed")
    if activation.dtype != torch.bfloat16 or mask.dtype != torch.uint8 or positions.dtype != torch.int64:
        raise CaptureError("observation tensor dtype changed")
    if metadata.get("schema") != CAPTURE_SCHEMA or metadata.get("task_id") != TASK_ID or metadata.get("variant_id") != expected_variant_id or metadata.get("domain") != expected_domain:
        raise CaptureError("observation metadata identity changed")
    if metadata.get("panel_sha256") != expected_panel_sha256 or metadata.get("opaque_record_ids_sha256") != expected_opaque_record_ids_sha256:
        raise CaptureError("observation panel/opaque-ID binding changed")
    for flag in ("truth_opened", "source_text_loaded", "target_labels_loaded", "token_ids_written", "candidate_arrays_persisted"):
        if metadata.get(flag) != "false":
            raise CaptureError(f"observation metadata records forbidden field: {flag}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path), "metadata": metadata, "tensor_keys": ["activations", "attention_mask", "position_ids"]}


def _domain_opaque_ids(panel: Mapping[str, Any]) -> dict[str, list[str]]:
    checked = validate_opaque_panel_descriptor(panel)
    result = {domain: [] for domain in DOMAINS}
    for row in checked["records"]:
        result[row["domain"]].append(row["opaque_id"])
    for domain in DOMAINS:
        result[domain] = [value for _, value in sorted(zip([row["order"] for row in checked["records"] if row["domain"] == domain], result[domain]))]
    return result


def load_evaluator_token_batches(
    path: Path,
    panel: Mapping[str, Any],
) -> tuple[dict[str, PaddedTokenBatch], dict[str, Any]]:
    """Load evaluator-owned token IDs and convert them to fixed B8x192 batches.

    The input is deliberately tokenized-only.  The capture process never
    accepts source text, labels, answers, candidate arrays, or source-selection
    metadata, and emits only an immutable source-file binding.
    """

    path = Path(path).expanduser().resolve()
    payload = _file_json(path)
    if payload.get("schema") != SOURCE_BATCH_SCHEMA or payload.get("task_id") != TASK_ID:
        raise CaptureError("evaluator source-batch schema or task identity changed")
    if payload.get("status") != "EVALUATOR_TOKENIZED_SOURCES_ONLY":
        raise CaptureError("evaluator source-batch status is not frozen")
    for flag in TRUTH_FLAGS:
        if _truthy(payload.get(flag)):
            raise CaptureError(f"evaluator source-batch records forbidden access: {flag}")
    if payload.get("source_text_loaded") is not False or payload.get("target_labels_loaded") is not False:
        raise CaptureError("tokenized source input cannot claim source text or labels")
    domains = payload.get("domains")
    if not isinstance(domains, Mapping) or tuple(domains) != DOMAINS:
        raise CaptureError("evaluator source-batch domains changed")
    checked_panel = validate_opaque_panel_descriptor(panel)
    expected_ids: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for row in checked_panel["records"]:
        expected_ids[row["domain"]].append(row["opaque_id"])
    batches: dict[str, PaddedTokenBatch] = {}
    for domain in DOMAINS:
        rows = domains.get(domain)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or len(rows) != REQUIRED_RECORDS_PER_DOMAIN:
            raise CaptureError(f"evaluator source-batch count changed: {domain}")
        observed_ids: list[str] = []
        sequences: list[list[int]] = []
        for order, raw in enumerate(rows):
            row = _require_mapping(raw, label=f"evaluator source row {domain}/{order}")
            forbidden = {
                "source",
                "source_id",
                "source_path",
                "source_text",
                "text",
                "answers",
                "labels",
                "target",
                "target_tokens",
                "truth",
                "candidate_arrays",
            }
            if set(row) & forbidden:
                raise CaptureError("evaluator source input contains source/truth payload fields")
            if set(row) - {"opaque_id", "order", "token_ids"}:
                raise CaptureError("evaluator source input contains undeclared fields")
            opaque_id = row.get("opaque_id")
            declared_order = row.get("order", order)
            token_ids = row.get("token_ids")
            if not isinstance(opaque_id, str) or opaque_id != expected_ids[domain][order]:
                raise CaptureError(f"evaluator source opaque order changed: {domain}/{order}")
            if isinstance(declared_order, bool) or not isinstance(declared_order, int) or declared_order != order:
                raise CaptureError(f"evaluator source order changed: {domain}/{order}")
            if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes, bytearray)):
                raise CaptureError(f"evaluator token IDs are missing: {domain}/{order}")
            sequence = [int(value) if isinstance(value, int) and not isinstance(value, bool) else -1 for value in token_ids]
            if len(sequence) != CAPTURE_SEQUENCE_TOKENS:
                raise CaptureError(f"evaluator source sequence must be exactly {CAPTURE_SEQUENCE_TOKENS} tokens")
            if sequence[0] != BOS_TOKEN_ID or any(value < 0 or value >= VOCABULARY_SIZE for value in sequence):
                raise CaptureError(f"evaluator source token geometry changed: {domain}/{order}")
            observed_ids.append(opaque_id)
            sequences.append(sequence)
        if observed_ids != expected_ids[domain]:
            raise CaptureError(f"evaluator source rows are not paired with the opaque panel: {domain}")
        batches[domain] = pad_public_token_sequences(
            sequences,
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            vocab_size=VOCABULARY_SIZE,
            small_post_bos_positions=5000,
        )
        _validate_capture_batch(batches[domain], records=REQUIRED_RECORDS_PER_DOMAIN)
    return batches, {
        "schema": SOURCE_BATCH_SCHEMA,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "readonly": True,
        "tokenized_sources_evaluator_only": True,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "truth_opened": False,
    }


def _host_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError):
        return 0


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system().lower() == "linux" else value


def capture_resource_snapshot(device: torch.device, *, output_root: Path | None = None) -> dict[str, Any]:
    """Return the producer-side resource state without allocating model tensors."""

    snapshot: dict[str, Any] = {
        "utc": _utc_now(),
        "pid": os.getpid(),
        "device": str(device),
        "host_available_bytes": _host_available_bytes(),
        "host_peak_rss_bytes": _max_rss_bytes(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if output_root is not None:
        try:
            snapshot["disk_free_bytes"] = int(shutil.disk_usage(Path(output_root).expanduser().resolve()).free)
        except OSError:
            snapshot["disk_free_bytes"] = None
    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "gpu_free_bytes": int(free),
                "gpu_total_bytes": int(total),
                "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "gpu_name": torch.cuda.get_device_name(device),
            }
        )
    else:
        snapshot.update(
            {
                "gpu_free_bytes": None,
                "gpu_total_bytes": None,
                "gpu_allocated_bytes": None,
                "gpu_reserved_bytes": None,
                "gpu_peak_allocated_bytes": None,
                "gpu_peak_reserved_bytes": None,
                "gpu_name": None,
            }
        )
    return snapshot


def preflight_resources(
    device: torch.device,
    *,
    output_root: Path,
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed before model load when the declared resource margin is absent."""

    snapshot = capture_resource_snapshot(device, output_root=output_root)
    min_headroom = int(resource_policy.get("minimum_host_headroom_bytes", 0))
    min_disk = int(resource_policy.get("minimum_disk_free_bytes", 0))
    if snapshot["host_available_bytes"] < min_headroom:
        raise CaptureError("host available memory is below the declared safety margin")
    if snapshot.get("disk_free_bytes") is not None and snapshot["disk_free_bytes"] < min_disk:
        raise CaptureError("disk free space is below the declared safety margin")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CaptureError("CUDA capture requested but CUDA is unavailable")
        if int(snapshot.get("gpu_free_bytes") or 0) < int(resource_policy.get("minimum_free_gpu_bytes", 0)):
            raise CaptureError("free GPU memory is below the declared safety margin")
        if int(snapshot.get("gpu_reserved_bytes") or 0) > int(resource_policy.get("reserved_gpu_ceiling_bytes", 0)):
            raise CaptureError("existing GPU reservation exceeds the declared ceiling")
    return {
        "schema": "token-reconstruction.trr0011-resource-preflight.v1",
        "status": "PREFLIGHT_PASS",
        "policy": dict(resource_policy),
        "snapshot": snapshot,
        "one_variant_per_process": resource_policy.get("one_variant_per_process") is True,
    }


def enforce_resource_policy(
    device: torch.device,
    *,
    output_root: Path,
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Check declared RSS/GPU/disk ceilings after a capture stage."""

    snapshot = capture_resource_snapshot(device, output_root=output_root)
    rss_cap = int(resource_policy.get("host_rss_cap_bytes", 0))
    disk_floor = int(resource_policy.get("minimum_disk_free_bytes", 0))
    if rss_cap and snapshot["host_peak_rss_bytes"] > rss_cap:
        raise CaptureError("host peak RSS exceeded the declared ceiling")
    if disk_floor and snapshot.get("disk_free_bytes") is not None and snapshot["disk_free_bytes"] < disk_floor:
        raise CaptureError("disk free space crossed the declared floor")
    if device.type == "cuda":
        reserved = int(snapshot.get("gpu_reserved_bytes") or 0)
        free = int(snapshot.get("gpu_free_bytes") or 0)
        if reserved > int(resource_policy.get("reserved_gpu_ceiling_bytes", 0)):
            raise CaptureError("GPU reservation exceeded the declared ceiling")
        if free < int(resource_policy.get("minimum_free_gpu_bytes", 0)):
            raise CaptureError("free GPU memory crossed the declared floor")
    return snapshot


def _model_file_binding(model_snapshot: Path, name: str) -> dict[str, Any]:
    return _record_file(model_snapshot / name, allow_symlink=True)


def public_model_resource_bindings(
    model_snapshot: Path,
    *,
    historical_lora_path: Path | None = None,
    generation_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the exact public model and historical update assets used by a producer."""

    model_snapshot = Path(model_snapshot).expanduser().resolve()
    if not model_snapshot.is_dir():
        raise CaptureError(f"public model snapshot directory is unavailable: {model_snapshot}")
    weights = _model_file_binding(model_snapshot, "model.safetensors")
    if weights["bytes"] != KNOWN_MODEL_WEIGHT_BYTES or weights["sha256"] != KNOWN_MODEL_WEIGHT_SHA256:
        raise CaptureError("pinned public model weight bytes or SHA-256 changed")
    result: dict[str, Any] = {
        "model_snapshot": {
            "path": str(model_snapshot),
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "config": _model_file_binding(model_snapshot, "config.json"),
            "generation_config": _model_file_binding(model_snapshot, "generation_config.json"),
            "weights": weights,
        }
    }
    if generation_path is not None:
        result["historical_generation"] = _record_file(Path(generation_path), allow_symlink=False)
    if historical_lora_path is not None:
        result["historical_lora"] = _record_file(Path(historical_lora_path), allow_symlink=False)
    return result


def build_capture_code_bindings(root: Path | None = None) -> dict[str, Any]:
    """Bind producer and public-prefix source files used by actual capture."""

    root = Path(root or _ROOT).expanduser().resolve()
    files = {
        "capture_runner": root / "scripts" / "trr0011_capture.py",
        "public_activation": root / "src" / "token_reconstruction" / "public_activation.py",
        "public_prefix": root / "src" / "token_reconstruction" / "public_prefix.py",
    }
    return {name: _record_file(path, root=root, allow_symlink=False) for name, path in files.items()}


def load_public_target_model(
    model_snapshot: Path,
    *,
    variant: Mapping[str, Any],
    device: torch.device,
    historical_lora_path: Path | None = None,
) -> torch.nn.Module:
    """Load a fresh pinned target model and apply only the declared variant."""

    model_snapshot = Path(model_snapshot).expanduser().resolve()
    public_model_resource_bindings(model_snapshot, historical_lora_path=historical_lora_path)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise CaptureError("transformers is required for evaluator model capture") from exc
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_snapshot),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
    except Exception as exc:
        raise CaptureError("pinned public target model could not be loaded locally") from exc
    if int(getattr(model.config, "hidden_size", -1)) != HIDDEN_SIZE or int(getattr(model.config, "vocab_size", -1)) != VOCABULARY_SIZE:
        raise CaptureError("pinned public target model geometry changed")
    if int(getattr(model.config, "num_hidden_layers", 0)) <= CUT_DEPTH:
        raise CaptureError("pinned public target model has no downstream cut layer")
    model.requires_grad_(False)
    if variant.get("role") == "historical_trained_benchmark_adaptation":
        if historical_lora_path is None:
            raise CaptureError("historical public_lora_2601 requires its published update asset")
        try:
            from token_reconstruction.target_update import TargetLoRAConfig, install_target_lora, load_target_lora
            installed = install_target_lora(
                model,
                TargetLoRAConfig(layers=(0, 1, 2, 3), modules=("q_proj", "v_proj"), rank=4, alpha=4.0, seed=2601),
            )
            load_target_lora(installed, Path(historical_lora_path).expanduser().resolve())
            for module in installed.values():
                module.requires_grad_(False)
        except Exception as exc:
            raise CaptureError("historical public_lora_2601 could not be loaded") from exc
        model.requires_grad_(False)
    return model


def load_full_reference_activations(path: Path, *, panel: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Load a private evaluator-side full-H reference for the after-cut null check."""

    path = Path(path).expanduser().resolve()
    checked_panel = validate_opaque_panel_descriptor(panel)
    result: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(DOMAINS):
            raise CaptureError("full-H null reference must contain exactly Finance/Pile tensors")
        metadata = dict(handle.metadata() or {})
        if metadata.get("schema") != FULL_REFERENCE_SCHEMA or metadata.get("panel_sha256") != checked_panel["panel_sha256"]:
            raise CaptureError("full-H null reference binding changed")
        for domain in DOMAINS:
            value = handle.get_tensor(domain)
            if tuple(value.shape) != (REQUIRED_RECORDS_PER_DOMAIN, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE) or value.dtype != torch.bfloat16:
                raise CaptureError("full-H null reference geometry changed")
            if not torch.isfinite(value.float()).all().item():
                raise CaptureError("full-H null reference contains non-finite values")
            result[domain] = value.contiguous()
    return result


def save_full_reference_artifact(
    path: Path,
    *,
    full_references: Mapping[str, torch.Tensor],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an evaluator-only full-H clean reference for a future null process."""

    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"full-H reference is create-only: {path}")
    checked_panel = validate_opaque_panel_descriptor(panel)
    if set(full_references) != set(DOMAINS):
        raise CaptureError("full-H reference domains are incomplete")
    tensors: dict[str, torch.Tensor] = {}
    for domain in DOMAINS:
        value = torch.as_tensor(full_references[domain]).detach().cpu().contiguous()
        if tuple(value.shape) != (REQUIRED_RECORDS_PER_DOMAIN, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE) or value.dtype != torch.bfloat16:
            raise CaptureError("full-H reference geometry changed")
        if not torch.isfinite(value.float()).all().item():
            raise CaptureError("full-H reference contains non-finite values")
        tensors[domain] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(path),
        metadata={
            "schema": FULL_REFERENCE_SCHEMA,
            "task_id": TASK_ID,
            "panel_sha256": checked_panel["panel_sha256"],
            "truth_opened": "false",
            "source_text_loaded": "false",
            "target_labels_loaded": "false",
            "token_ids_written": "false",
        },
    )
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path), "readonly": True}


def capture_full_forward(model: torch.nn.Module, batch: PaddedTokenBatch, *, device: torch.device, batch_size: int = CAPTURE_BATCH_SIZE, resource_check: Callable[[], None] | None = None) -> torch.Tensor:
    """Run the actual full width-192 public forward and retain all H for checks."""

    _validate_capture_batch(batch, records=int(batch.token_ids.shape[0]))
    prefix = ContiguousPublicPrefix(model, cut_depth=CUT_DEPTH).to(device).eval()
    return capture_public_prefix(prefix, batch, device=device, batch_size=batch_size, hidden_size=HIDDEN_SIZE, resource_check=resource_check)



def run_capture_variants(
    plan: Mapping[str, Any],
    panel: Mapping[str, Any],
    batches: Mapping[str, PaddedTokenBatch],
    model_factory: Callable[[Mapping[str, Any]], torch.nn.Module],
    *,
    output_root: Path,
    device: torch.device,
    command: Sequence[str],
    code_bindings: Mapping[str, Any],
    target_resource_bindings: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    resource_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    resource_check: Callable[[], None] | None = None,
    variant_ids: Sequence[str] | None = None,
    clean_full_reference: Mapping[str, torch.Tensor] | None = None,
    source_binding: Mapping[str, Any] | None = None,
    access_scope: Mapping[str, Any] | None = None,
    include_historical_public_lora: bool = True,
) -> dict[str, Any]:
    """Capture declared variants with a fresh evaluator model per process.

    Production callers pass one variant_ids entry per invocation and launch
    this function under an external watchdog. The default all-variant mode
    remains useful for synthetic integration, but its returned manifest is
    still truth-free and requires every mandatory condition.
    """

    validate_capture_plan(plan, require_frozen=True)
    if include_historical_public_lora is not True:
        raise CaptureError("historical public_lora_2601 is a mandatory declared capture condition")
    checked_panel = validate_opaque_panel_descriptor(panel)
    ids_by_domain = _domain_opaque_ids(panel)
    if set(batches) != set(DOMAINS):
        raise CaptureError("evaluator batches must cover Finance and Pile")
    for domain in DOMAINS:
        _validate_capture_batch(batches[domain], records=REQUIRED_RECORDS_PER_DOMAIN)
    panel_sha_value = panel.get("panel_sha256")
    if not isinstance(panel_sha_value, str):
        raise CaptureError("production capture requires a SHA-256 binding for the complete opaque panel descriptor")
    panel_sha = _require_sha(panel_sha_value, label="panel_sha256")
    if panel_sha != checked_panel["panel_sha256"]:
        raise CaptureError("opaque panel complete SHA-256 binding changed")
    variants = {str(row["variant_id"]): row for row in plan["variants"]}
    runnable_ids = list(variant_ids) if variant_ids is not None else list(EXPECTED_VARIANTS)
    if not runnable_ids or len(set(runnable_ids)) != len(runnable_ids) or any(value not in EXPECTED_VARIANTS for value in runnable_ids):
        raise CaptureError("capture variant selection is malformed")
    if "layer4_null_block_rel1e-2" in runnable_ids and "clean_public" not in runnable_ids and clean_full_reference is None:
        raise CaptureError("after-cut null requires a private full-H clean reference or the same-process clean capture")
    clean_full: dict[str, torch.Tensor] = {}
    if clean_full_reference is not None:
        if set(clean_full_reference) != set(DOMAINS):
            raise CaptureError("clean full-H reference domains are incomplete")
        for domain in DOMAINS:
            value = torch.as_tensor(clean_full_reference[domain]).detach().cpu().contiguous()
            if tuple(value.shape) != (REQUIRED_RECORDS_PER_DOMAIN, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE) or value.dtype != torch.bfloat16:
                raise CaptureError("clean full-H reference geometry changed")
            clean_full[domain] = value
    started = time.perf_counter()
    started_utc = _utc_now()
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    full_references: dict[str, torch.Tensor] = {}
    variant_receipts: dict[str, Any] = {}
    for variant_id in runnable_ids:
        variant = variants[variant_id]
        if resource_check is not None:
            resource_check()
        variant_started = time.perf_counter()
        resource_before = dict(resource_snapshot()) if resource_snapshot is not None else None
        model = model_factory(variant)
        if not isinstance(model, torch.nn.Module):
            raise CaptureError(f"model_factory did not return a torch module for {variant_id}")
        update_receipt = apply_fixed_sign_block_update(model, variant)
        by_domain: dict[str, dict[str, Any]] = {}
        null_checks: dict[str, Any] = {}
        try:
            # Construct the prefix after the mutation, including layer-4 null.
            for domain in DOMAINS:
                full = capture_full_forward(
                    model,
                    batches[domain],
                    device=device,
                    batch_size=CAPTURE_BATCH_SIZE,
                    resource_check=resource_check,
                )
                if variant_id == "clean_public":
                    clean_full[domain] = full.detach().cpu().contiguous()
                    full_references[domain] = clean_full[domain]
                if variant.get("role") == "after_cut_suffix_null_control":
                    if domain not in clean_full:
                        raise CaptureError("after-cut null was attempted before clean reference capture")
                    null_checks[domain] = validate_after_cut_null(clean_full[domain], full)
                retained, mask, positions = _retained_observation_tensors(full, batches[domain])
                artifact_path = Path(output_root) / variant_id / f"{domain}.safetensors"
                descriptor = save_observation_artifact(
                    artifact_path,
                    activations=retained,
                    attention_mask=mask,
                    position_ids=positions,
                    variant_id=variant_id,
                    domain=domain,
                    opaque_record_ids=ids_by_domain[domain],
                    panel_sha256=panel_sha,
                    capture_code_binding=code_bindings,
                )
                by_domain[domain] = descriptor
                if resource_check is not None:
                    resource_check()
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        resource_after = dict(resource_snapshot()) if resource_snapshot is not None else None
        variant_receipts[variant_id] = {
            "status": "CAPTURED_TRUTH_FREE",
            "variant_id": variant_id,
            "role": variant.get("role"),
            "update": update_receipt,
            "null_checks": null_checks,
            "observations": by_domain,
            "wall_seconds": time.perf_counter() - variant_started,
            "resource_snapshots": [item for item in (resource_before, resource_after) if item is not None],
            "truth_opened": False,
            "source_text_loaded": bool((access_scope or {}).get("producer", {}).get("source_text_loaded", False)),
            "tokenized_sources_loaded": True,
            "target_labels_loaded": False,
            "target_full_weights_evaluator_only": True,
        }
        observations[variant_id] = by_domain
    manifest = None
    if set(observations) == set(EXPECTED_VARIANTS):
        manifest = build_capture_manifest(
            plan,
            panel,
            observations,
            command=command,
            code_bindings=code_bindings,
            target_resource_bindings=target_resource_bindings,
            resource_receipt=resource_receipt,
            timing={
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "variant_receipts": variant_receipts,
                "scope": "full B8x192 public forward, retained first128; one evaluator model process per variant",
            },
            source_binding=source_binding,
            access_scope=access_scope,
        )
    return {
        "manifest": manifest,
        "variant_receipts": variant_receipts,
        "observations": observations,
        "full_references": full_references,
        "truth_opened": False,
    }


def build_capture_manifest(
    plan: Mapping[str, Any],
    panel: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    command: Sequence[str],
    code_bindings: Mapping[str, Any],
    target_resource_bindings: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    timing: Mapping[str, Any],
    truth_sidecar: Mapping[str, Any] | None = None,
    source_binding: Mapping[str, Any] | None = None,
    access_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a compact truth-free manifest after a producer run."""

    validate_capture_plan(plan, require_frozen=True)
    checked_panel = validate_opaque_panel_descriptor(panel)
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)) or not command:
        raise CaptureError("capture command must be a non-empty argv sequence")
    if not isinstance(code_bindings, Mapping) or not code_bindings:
        raise CaptureError("capture code bindings are absent")
    if not isinstance(target_resource_bindings, Mapping):
        raise CaptureError("target resource bindings are malformed")
    if not isinstance(resource_receipt, Mapping) or not isinstance(timing, Mapping):
        raise CaptureError("resource/timing receipts are malformed")
    normalized_observations: dict[str, dict[str, Any]] = {}
    for variant_id, by_domain in observations.items():
        if variant_id not in EXPECTED_VARIANTS:
            raise CaptureError(f"unknown observation variant: {variant_id}")
        if not isinstance(by_domain, Mapping) or set(by_domain) != set(DOMAINS):
            raise CaptureError(f"observation cells incomplete for {variant_id}")
        normalized_observations[variant_id] = {domain: dict(value) for domain, value in by_domain.items()}
    expected_variants = set(EXPECTED_VARIANTS)
    if set(normalized_observations) != expected_variants:
        raise CaptureError("capture observations must cover every mandatory declared variant")
    sidecar = dict(truth_sidecar or {"status": "PENDING_POSTPREDICTION_GATE", "truth_opened": False, "path": None})
    if sidecar.get("truth_opened") is not False:
        raise CaptureError("capture manifest cannot bind an opened truth sidecar")
    return {
        "schema": CAPTURE_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "status": "OBSERVATIONS_TRUTH_FREE",
        "plan": {"schema": plan["schema"], "status": plan["status"], "frozen_utc": plan.get("frozen_utc")},
        "panel": checked_panel,
        "geometry": dict(plan["geometry"]),
        "command": [str(item) for item in command],
        "code_bindings": json.loads(json.dumps(dict(code_bindings))),
        "target_resource_bindings": json.loads(json.dumps(dict(target_resource_bindings))),
        "resource_receipt": json.loads(json.dumps(dict(resource_receipt))),
        "timing": json.loads(json.dumps(dict(timing))),
        "observations": normalized_observations,
        "source_binding": json.loads(json.dumps(dict(source_binding or {}))),
        "access_scope": json.loads(json.dumps(dict(access_scope or {
            "producer": {
                "source_text_loaded": False,
                "tokenized_sources_loaded": True,
                "target_full_weights_loaded": True,
                "truth_opened": False,
            },
            "decoder": {
                "source_text_loaded": False,
                "tokenized_sources_loaded": False,
                "target_full_weights_loaded": False,
                "truth_opened": False,
                "permitted_inputs": ["activations", "attention_mask", "position_ids", "opaque_record_ids"],
            },
            "scorer": {"truth_opened": False},
        }))),
        "truth_sidecar": sidecar,
        "truth_opened": False,
        "source_text_loaded": bool((access_scope or {}).get("producer", {}).get("source_text_loaded", False)),
        "tokenized_sources_evaluator_only": True,
        "target_full_weights_evaluator_only": True,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "source_selection_performed": False,
        "target_weights_written_to_observations": False,
        "decoder_input_contract": ["activations", "attention_mask", "position_ids", "opaque_record_ids"],
    }


def build_transfer_observation_manifest(
    capture_manifest: Mapping[str, Any],
    *,
    method_id: str,
    transfer_code_bindings: Mapping[str, Any],
    state_bindings: Mapping[str, Any],
    decoder_resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the seven-condition capture into the transfer-agent input schema."""

    if capture_manifest.get("schema") != CAPTURE_MANIFEST_SCHEMA or capture_manifest.get("status") != "OBSERVATIONS_TRUTH_FREE":
        raise CaptureError("capture manifest is not a complete truth-free observation package")
    if method_id not in TRANSFER_METHOD_IDS:
        raise CaptureError("transfer method is outside the frozen fixed-decoder set")
    for flag in TRUTH_FLAGS:
        if _truthy(capture_manifest.get(flag)):
            raise CaptureError(f"capture manifest records forbidden access: {flag}")
    observations = capture_manifest.get("observations")
    if not isinstance(observations, Mapping) or set(observations) != set(EXPECTED_VARIANTS):
        raise CaptureError("transfer bridge requires all seven capture variants")
    checked_panel = validate_opaque_panel_descriptor(capture_manifest.get("panel"))
    source_order: dict[str, str] = {}
    for domain in DOMAINS:
        rows = [row for row in checked_panel["records"] if row["domain"] == domain]
        ordered = sorted(rows, key=lambda row: int(row["order"]))
        source_order[domain] = opaque_record_ids_digest(ordered)
    observation_bindings: dict[str, dict[str, Any]] = {}
    for variant_id in EXPECTED_VARIANTS:
        by_domain = observations[variant_id]
        if not isinstance(by_domain, Mapping) or set(by_domain) != set(DOMAINS):
            raise CaptureError(f"capture transfer cell is incomplete: {variant_id}")
        observation_bindings[variant_id] = {}
        for domain in DOMAINS:
            descriptor = _require_mapping(by_domain[domain], label=f"capture observation {variant_id}/{domain}")
            path = Path(str(descriptor.get("path", ""))).expanduser().resolve()
            checked = _record_file(path)
            if checked["bytes"] != descriptor.get("bytes") or checked["sha256"] != descriptor.get("sha256"):
                raise CaptureError(f"capture observation binding changed: {variant_id}/{domain}")
            record_digest = descriptor.get("opaque_record_ids_sha256")
            if record_digest != source_order[domain]:
                raise CaptureError(f"capture source order digest changed: {variant_id}/{domain}")
            observation_bindings[variant_id][domain] = {
                "path": str(path),
                "bytes": checked["bytes"],
                "sha256": checked["sha256"],
                "readonly": True,
                "records": REQUIRED_RECORDS_PER_DOMAIN,
                "record_ids_sha256": record_digest,
            }
    return {
        "schema": TRANSFER_RUN_SCHEMA,
        "task_id": TASK_ID,
        "status": TRANSFER_RUN_STATUS,
        "method_id": method_id,
        "variant_ids": list(EXPECTED_VARIANTS),
        "source_order_sha256": source_order,
        "code_bindings": json.loads(json.dumps(dict(transfer_code_bindings))),
        "state_bindings": json.loads(json.dumps(dict(state_bindings))),
        "decoder_resources": json.loads(json.dumps(dict(decoder_resources))),
        "observation_bindings": observation_bindings,
        "capture_manifest_binding": {
            "schema": capture_manifest["schema"],
            "panel_sha256": checked_panel["panel_sha256"],
            "truth_opened": False,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "private_or_truth_payload_read": False,
        "fresh_evaluation_started": False,
    }


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"JSON receipt is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _root_release_binding(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = _file_json(path)
    if value.get("task_id") != TASK_ID or value.get("status") != "ROOT_RELEASED_TRR0011_CAPTURE":
        raise CaptureError("root release receipt is absent or not for TRR-0011 capture")
    for flag in TRUTH_FLAGS:
        if _truthy(value.get(flag)):
            raise CaptureError(f"root release receipt records forbidden access: {flag}")
    return _record_file(path, allow_symlink=False)


def _historical_asset_paths(plan: Mapping[str, Any]) -> tuple[Path, Path]:
    row = next(item for item in plan["variants"] if item["variant_id"] == "historical_public_lora_2601")
    training = _require_mapping(row["training_provenance"], label="historical training provenance")
    assets = _require_mapping(training["asset_bindings"], label="historical asset bindings")
    generation = _require_mapping(assets["generation_receipt"], label="historical generation binding")
    update = _require_mapping(assets["update"], label="historical update binding")
    return Path(str(generation["path"])).expanduser().resolve(), Path(str(update["path"])).expanduser().resolve()


def _capture_variant_cli(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve()
    panel_path = Path(args.panel).expanduser().resolve()
    source_path = Path(args.source_input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    plan = _file_json(plan_path)
    validate_capture_plan(plan, require_frozen=True)
    panel = _file_json(panel_path)
    checked_panel = validate_opaque_panel_descriptor(panel)
    if panel.get("panel_sha256") != checked_panel["panel_sha256"]:
        raise CaptureError("capture CLI requires the panel complete SHA-256 binding")
    batches, source_binding = load_evaluator_token_batches(source_path, panel)
    device = torch.device(str(args.device))
    release_binding = _root_release_binding(Path(args.root_release_receipt).expanduser().resolve() if args.root_release_receipt else None)
    if release_binding is None and device.type == "cuda":
        raise CaptureError("CUDA capture is blocked until root supplies a release receipt")
    if release_binding is None and plan.get("resource_policy", {}).get("launch_authorized") is not False:
        raise CaptureError("capture launch authorization is not bound to a root release")
    policy = _require_mapping(plan.get("resource_policy"), label="resource_policy")
    preflight = preflight_resources(device, output_root=output_root, resource_policy=policy)
    generation_path, historical_path = _historical_asset_paths(plan)
    if args.historical_lora:
        historical_path = Path(args.historical_lora).expanduser().resolve()
    target_resources = public_model_resource_bindings(
        Path(args.model_snapshot),
        historical_lora_path=historical_path,
        generation_path=generation_path,
    )
    code_bindings = build_capture_code_bindings(_ROOT)
    if args.tokenizer:
        tokenizer_binding = _record_file(Path(args.tokenizer), allow_symlink=True)
        target_resources["tokenizer"] = tokenizer_binding
    clean_reference = None
    if args.variant == "layer4_null_block_rel1e-2":
        if not args.clean_reference:
            raise CaptureError("layer4 null capture requires --clean-reference from the clean producer")
        clean_reference = load_full_reference_activations(Path(args.clean_reference), panel=panel)
    snapshots: list[dict[str, Any]] = []
    resource_snapshot = lambda: capture_resource_snapshot(device, output_root=output_root)
    resource_check = lambda: snapshots.append(enforce_resource_policy(device, output_root=output_root, resource_policy=policy))
    run = run_capture_variants(
        plan,
        panel,
        batches,
        lambda variant: load_public_target_model(
            Path(args.model_snapshot),
            variant=variant,
            device=device,
            historical_lora_path=historical_path,
        ),
        output_root=output_root,
        device=device,
        command=[str(value) for value in sys.argv],
        code_bindings=code_bindings,
        target_resource_bindings=target_resources,
        resource_receipt={
            "preflight": preflight,
            "root_release": release_binding,
            "external_watchdog_seconds": int(policy.get("watchdog_seconds_per_variant", 900)),
            "resource_checks": snapshots,
        },
        resource_snapshot=resource_snapshot,
        resource_check=resource_check,
        variant_ids=[args.variant],
        clean_full_reference=clean_reference,
        source_binding=source_binding,
        access_scope={
            "producer": {
                "source_text_loaded": False,
                "tokenized_sources_loaded": True,
                "target_full_weights_loaded": True,
                "truth_opened": False,
            },
            "decoder": {
                "source_text_loaded": False,
                "tokenized_sources_loaded": False,
                "target_full_weights_loaded": False,
                "truth_opened": False,
                "permitted_inputs": ["activations", "attention_mask", "position_ids", "opaque_record_ids"],
            },
            "scorer": {"truth_opened": False},
        },
    )
    variant_receipt = run["variant_receipts"][args.variant]
    receipt = {
        "schema": "token-reconstruction.trr0011-capture-variant-receipt.v1",
        "task_id": TASK_ID,
        "status": "CAPTURED_TRUTH_FREE",
        "variant_id": args.variant,
        "plan_binding": _record_file(plan_path, allow_symlink=False),
        "panel_binding": _record_file(panel_path, allow_symlink=False) | {"panel_sha256": checked_panel["panel_sha256"]},
        "source_binding": source_binding,
        "code_bindings": code_bindings,
        "target_resource_bindings": target_resources,
        "resource_receipt": {
            "preflight": preflight,
            "root_release": release_binding,
            "external_watchdog_seconds": int(policy.get("watchdog_seconds_per_variant", 900)),
            "resource_checks": snapshots,
            "peak": capture_resource_snapshot(device, output_root=output_root),
        },
        "command": [str(value) for value in sys.argv],
        "access_scope": {
            "producer": {"source_text_loaded": False, "tokenized_sources_loaded": True, "target_full_weights_loaded": True, "truth_opened": False},
            "decoder": {"source_text_loaded": False, "tokenized_sources_loaded": False, "target_full_weights_loaded": False, "truth_opened": False, "permitted_inputs": ["activations", "attention_mask", "position_ids", "opaque_record_ids"]},
            "scorer": {"truth_opened": False},
        },
        "variant_receipt": variant_receipt,
        "observations": run["observations"][args.variant],
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "source_selection_performed": False,
    }
    receipt_path = Path(args.receipt).expanduser().resolve() if args.receipt else output_root / args.variant / "capture_variant_receipt.json"
    receipt_file = write_create_only_json(receipt_path, receipt)
    if args.full_reference_output:
        if args.variant != "clean_public":
            raise CaptureError("--full-reference-output is valid only for clean_public")
        reference_file = save_full_reference_artifact(
            Path(args.full_reference_output),
            full_references=run["full_references"],
            panel=panel,
        )
    else:
        reference_file = None
    return {"receipt": receipt_file, "full_reference": reference_file, "variant": args.variant, "observations": receipt["observations"]}


def _assemble_manifest_cli(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve()
    panel_path = Path(args.panel).expanduser().resolve()
    capture_root = Path(args.capture_root).expanduser().resolve()
    plan = _file_json(plan_path)
    validate_capture_plan(plan, require_frozen=True)
    panel = _file_json(panel_path)
    checked_panel = validate_opaque_panel_descriptor(panel)
    receipts: list[dict[str, Any]] = []
    for variant_id in EXPECTED_VARIANTS:
        path = capture_root / variant_id / "capture_variant_receipt.json"
        if not path.is_file():
            raise CaptureError(f"variant receipt is missing: {variant_id}")
        value = _file_json(path)
        if value.get("schema") != "token-reconstruction.trr0011-capture-variant-receipt.v1" or value.get("variant_id") != variant_id or value.get("truth_opened") is not False:
            raise CaptureError(f"variant receipt is not a frozen truth-free capture: {variant_id}")
        receipts.append(value)
    first = receipts[0]
    observations = {value["variant_id"]: value["observations"] for value in receipts}
    if any(value.get("panel_binding", {}).get("panel_sha256") != checked_panel["panel_sha256"] for value in receipts):
        raise CaptureError("variant receipts do not share the panel binding")
    if any(value.get("source_binding", {}).get("sha256") != first.get("source_binding", {}).get("sha256") for value in receipts):
        raise CaptureError("variant receipts do not share the evaluator source binding")
    timing = {
        "scope": "seven one-variant-per-process captures; full B8x192 public forward, retained first128",
        "variant_receipts": {value["variant_id"]: value["variant_receipt"] for value in receipts},
        "assembled_utc": _utc_now(),
    }
    resource_receipt = dict(first["resource_receipt"])
    resource_receipt["variant_receipts"] = {value["variant_id"]: value["resource_receipt"] for value in receipts}
    manifest = build_capture_manifest(
        plan,
        panel,
        observations,
        command=first["command"],
        code_bindings=first["code_bindings"],
        target_resource_bindings=first["target_resource_bindings"],
        resource_receipt=resource_receipt,
        timing=timing,
        source_binding=first["source_binding"],
        access_scope=first["access_scope"],
    )
    receipt = write_create_only_json(Path(args.output), manifest)
    return {"manifest": receipt}


def _build_transfer_manifest_cli(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    capture_manifest = _file_json(Path(args.capture_manifest).expanduser().resolve())
    transfer_bridge = _record_file(Path(args.transfer_bridge), root=root, allow_symlink=True)
    state = _record_file(Path(args.decoder_state), root=root, allow_symlink=True)
    embedding = _record_file(Path(args.decoder_embedding), root=root, allow_symlink=True)
    loader = _record_file(Path(args.decoder_loader), root=root, allow_symlink=True) | {"module": "scripts.trr0010_p09_fixed_loader"}
    value = build_transfer_observation_manifest(
        capture_manifest,
        method_id=str(args.method_id),
        transfer_code_bindings={"transfer_bridge": transfer_bridge},
        state_bindings={"decoder_state": state},
        decoder_resources={"embedding": embedding, "state": state, "loader": loader},
    )
    receipt = write_create_only_json(Path(args.output), value)
    return {"transfer_manifest": receipt}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("write-default-plan")
    plan.add_argument("--output", type=Path, required=True)
    validate_plan = sub.add_parser("validate-plan")
    validate_plan.add_argument("--plan", type=Path, required=True)
    validate_panel = sub.add_parser("validate-panel")
    validate_panel.add_argument("--panel", type=Path, required=True)
    capture = sub.add_parser("capture-variant")
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--panel", type=Path, required=True)
    capture.add_argument("--source-input", type=Path, required=True, help="evaluator-only tokenized source batch JSON")
    capture.add_argument("--model-snapshot", type=Path, required=True)
    capture.add_argument("--tokenizer", type=Path)
    capture.add_argument("--historical-lora", type=Path)
    capture.add_argument("--variant", choices=EXPECTED_VARIANTS, required=True)
    capture.add_argument("--output-root", type=Path, required=True)
    capture.add_argument("--receipt", type=Path)
    capture.add_argument("--full-reference-output", type=Path)
    capture.add_argument("--clean-reference", type=Path)
    capture.add_argument("--root-release-receipt", type=Path)
    capture.add_argument("--device", default="cuda")
    assemble = sub.add_parser("assemble-manifest")
    assemble.add_argument("--plan", type=Path, required=True)
    assemble.add_argument("--panel", type=Path, required=True)
    assemble.add_argument("--capture-root", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    transfer = sub.add_parser("build-transfer-manifest")
    transfer.add_argument("--capture-manifest", type=Path, required=True)
    transfer.add_argument("--transfer-bridge", type=Path, required=True)
    transfer.add_argument("--decoder-embedding", type=Path, required=True)
    transfer.add_argument("--decoder-state", type=Path, required=True)
    transfer.add_argument("--decoder-loader", type=Path, required=True)
    transfer.add_argument("--method-id", choices=TRANSFER_METHOD_IDS, default="current_fixed")
    transfer.add_argument("--repository-root", type=Path, default=_ROOT)
    transfer.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-default-plan":
            plan = freeze_capture_plan(default_capture_plan(), source_binding=_record_file(Path(__file__), root=_ROOT))
            receipt = write_create_only_json(args.output, plan)
            print(json.dumps({"status": "FROZEN_CAPTURE_RECIPE_BEFORE_OBSERVATIONS", **receipt}, sort_keys=True))
        elif args.command == "validate-plan":
            plan = _file_json(args.plan)
            print(json.dumps(validate_capture_plan(plan, require_frozen=True), sort_keys=True))
        elif args.command == "validate-panel":
            panel = _file_json(args.panel)
            print(json.dumps(validate_opaque_panel_descriptor(panel), sort_keys=True))
        elif args.command == "capture-variant":
            print(json.dumps(_capture_variant_cli(args), indent=2, sort_keys=True))
        elif args.command == "assemble-manifest":
            print(json.dumps(_assemble_manifest_cli(args), indent=2, sort_keys=True))
        else:
            print(json.dumps(_build_transfer_manifest_cli(args), indent=2, sort_keys=True))
    except CaptureError as exc:
        print(f"trr0011_capture: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
