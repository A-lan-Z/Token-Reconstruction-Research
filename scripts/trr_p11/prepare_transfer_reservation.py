"""Prepare a hash-only TRR-P11 transfer reservation and deferred capture handoff.

This command validates only immutable JSON/byte bindings.  It deliberately
never loads public rows, tokenizes text, creates source identities, or opens
truth.  Selection remains a later root-released operation using the existing
TRR-0011 natural 128-token selector and its opaque panel gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRANSFER_ROOT = ROOT.parent / "TRR-0012"
TASK_ID = "TRR-P11"
TRANSFER_TASK_ID = "TRR-0011"
SELECTION_SEED = 5011
SOURCE_RANGES = {"finance": [28000, 30000], "pile": [9000, 10000]}
DOMAIN_ORDER = ["finance", "pile"]
TRANSFER_RECORDS_BY_DOMAIN = {"finance": 32, "pile": 32}
TRANSFER_VARIANTS = [
    "clean_public_base",
    "early_prefix_eps1e3",
    "early_prefix_eps1e2",
    "near_cut_prefix_eps1e3",
    "near_cut_prefix_eps1e2",
    "after_cut_suffix_eps1e2_null",
]
RUNNER_VARIANTS = [
    "clean_public",
    "layer1_block_rel1e-3",
    "layer1_block_rel1e-2",
    "layer3_block_rel1e-3",
    "layer3_block_rel1e-2",
    "layer4_null_block_rel1e-2",
    "historical_public_lora_2601",
]
FORBIDDEN_TRUE_KEYS = (
    "truth_opened",
    "source_text_loaded",
    "source_text_written",
    "target_labels_loaded",
    "token_ids_written",
    "candidate_arrays_persisted",
    "private_or_truth_payload_read",
    "fresh_evaluation_started",
    "selection_performed",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, label: str, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is unavailable or symlinked: {path}")
    rendered = str(path)
    if relative_to is not None:
        try:
            rendered = str(path.relative_to(relative_to.resolve()))
        except ValueError:
            rendered = str(path)
    return {"path": rendered, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "readonly": True}


def _load(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return dict(value), record


def _require_false(value: Mapping[str, Any], *, label: str) -> None:
    for key in FORBIDDEN_TRUE_KEYS:
        item = value.get(key)
        if item is True or (isinstance(item, str) and item.lower() == "true"):
            raise RuntimeError(f"{label} records forbidden state: {key}")


def _load_existing_selector_validator(template: Mapping[str, Any]) -> dict[str, Any]:
    module_path = TRANSFER_ROOT / "scripts" / "trr0011_select_diagnostic.py"
    spec = importlib.util.spec_from_file_location("trr0011_select_diagnostic_immutable", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable TRR-0011 selector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        result = module.validate_panel_spec(template)
    except Exception as exc:
        raise RuntimeError("immutable TRR-0011 panel validator rejected the template") from exc
    if result.get("selection_performed") is not False or result.get("reservation_status") != "PENDING":
        raise RuntimeError("immutable panel validator did not preserve the pending boundary")
    return {
        "module": "../TRR-0012/scripts/trr0011_select_diagnostic.py",
        "function": "validate_panel_spec",
        "result": result,
    }


def _validate_plan(plan: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != "token-reconstruction.trr0011-controlled-target-transfer.v1" or plan.get("task_id") != TRANSFER_TASK_ID:
        raise RuntimeError("transfer design plan schema/task changed")
    if plan.get("status") != "DESIGN_ONLY_NO_SELECTION_NO_TRUTH":
        raise RuntimeError("transfer design plan is not design-only")
    geometry = plan.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("stored_sequence_tokens") != 128 or geometry.get("scored_post_bos_tokens") != 127 or geometry.get("bos_token_id") != 128000:
        raise RuntimeError("transfer geometry changed")
    panel = plan.get("panel")
    if not isinstance(panel, Mapping) or panel.get("records_per_domain") != 32 or panel.get("same_record_order_across_targets") is not True or panel.get("selection_performed") is not False:
        raise RuntimeError("transfer panel contract changed")
    proposed = panel.get("proposed_source_ranges")
    if not isinstance(proposed, Mapping) or proposed.get("finance") != "[28000,30000)" or proposed.get("pile") != "[9000,10000)" or proposed.get("selection_seed") != SELECTION_SEED or proposed.get("records_per_domain_max") != 32:
        raise RuntimeError("transfer ranges/seed/count changed")
    if plan.get("truth_opened") is not False or plan.get("source_text_loaded") is not False:
        raise RuntimeError("transfer plan truth boundary changed")
    _require_false(plan, label="transfer design plan")
    if template.get("target_variants") != TRANSFER_VARIANTS:
        raise RuntimeError("transfer panel target variant order changed")
    return {
        "schema": str(plan["schema"]),
        "status": str(plan["status"]),
        "geometry": dict(geometry),
        "target_variants": list(template["target_variants"]),
        "selection_performed": False,
    }


def _path_from_root(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(Path(__import__("os").path.relpath(path, ROOT)))


def _runner_commands(*, plan: Path, output_root: str, release: Path, historical_lora: Path) -> list[dict[str, Any]]:
    common = [
        "python3",
        "../TRR-0012/scripts/trr0011_capture.py",
        "capture-variant",
        "--plan",
        _path_from_root(plan),
        "--panel",
        "${AGENT2_OPAQUE_PANEL_JSON:?set evaluator-owned opaque panel JSON}",
        "--source-input",
        "${AGENT2_EVALUATOR_TOKEN_BATCH_JSON:?set evaluator-owned token batch JSON}",
        "--model-snapshot",
        "${MODEL_SNAPSHOT:?set public model snapshot}",
        "--tokenizer",
        "${MODEL_SNAPSHOT:?set public model snapshot}/tokenizer.json",
    ]
    result: list[dict[str, Any]] = []
    for variant in RUNNER_VARIANTS:
        command = list(common) + ["--variant", variant, "--output-root", output_root, "--receipt", f"{output_root}/{variant}/capture_variant_receipt.json", "--root-release-receipt", _path_from_root(release), "--device", "cuda"]
        extra: dict[str, Any] = {}
        if variant == "clean_public":
            command += ["--full-reference-output", f"{output_root}/clean_public/full_reference.safetensors"]
        if variant == "layer4_null_block_rel1e-2":
            command += ["--clean-reference", f"{output_root}/clean_public/full_reference.safetensors"]
        if variant == "historical_public_lora_2601":
            command += ["--historical-lora", str(historical_lora)]
            extra["independent_target"] = False
        result.append({"step": f"capture_{variant}", "command": command, "run": False, "watchdog": "timeout --kill-after=30s 900s", **extra})
    result.append({
        "step": "assemble_manifest",
        "command": [
            "python3", "../TRR-0012/scripts/trr0011_capture.py", "assemble-manifest",
            "--plan", _path_from_root(plan),
            "--panel", "${AGENT2_OPAQUE_PANEL_JSON:?set evaluator-owned opaque panel JSON}",
            "--capture-root", output_root,
            "--output", f"{output_root}/transfer_capture_manifest.json",
        ],
        "run": False,
        "requires": "all seven truth-free variant receipts",
    })
    return result


def prepare(*, reservation_output: Path, capture_output: Path) -> dict[str, Any]:
    plan_path = TRANSFER_ROOT / "experiments" / "TRR-0011" / "transfer" / "variant_plan_v2.json"
    template_path = TRANSFER_ROOT / "experiments" / "TRR-0011" / "transfer" / "panel_descriptor_template_v2.json"
    selector_path = TRANSFER_ROOT / "scripts" / "trr0011_select_diagnostic.py"
    capture_runner = TRANSFER_ROOT / "scripts" / "trr0011_capture.py"
    capture_plan_path = TRANSFER_ROOT / "experiments" / "outputs" / "TRR-0012" / "transfer" / "capture_plan_exec_v2.json"
    capture_commands_path = TRANSFER_ROOT / "experiments" / "outputs" / "TRR-0012" / "transfer" / "capture_commands_v3.json"
    capture_config_path = TRANSFER_ROOT / "experiments" / "outputs" / "TRR-0012" / "transfer" / "capture_config_v3.json"
    release_path = TRANSFER_ROOT / "experiments" / "outputs" / "TRR-0012" / "transfer" / "root_release_transfer_v2.json"
    proof_path = TRANSFER_ROOT / "experiments" / "outputs" / "TRR-0012" / "transfer" / "seed_compatibility_test_receipt_v2.json"
    historical_lora = ROOT.parent.parent / "outputs" / "TRR-0002" / "public-calibration" / "updates" / "public_lora_2601.safetensors"
    source_inputs_path = ROOT / "experiments" / TASK_ID / "selector" / "public_source_inputs_r1.json"
    p11_selector_path = ROOT / "scripts" / "trr_p11" / "source_selector.py"
    p11_wrapper_path = ROOT / "scripts" / "trr_p11" / "run_source_selection.py"
    plan, plan_record = _load(plan_path, label="TRR-0011 transfer variant plan")
    template, template_record = _load(template_path, label="TRR-0011 transfer panel template")
    template_validation = _load_existing_selector_validator(template)
    plan_summary = _validate_plan(plan, template)
    reservation_output = reservation_output.expanduser().resolve()
    capture_output = capture_output.expanduser().resolve()
    for output in (reservation_output, capture_output):
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"output is create-only: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    bindings = {
        "agent1_selector": _record(selector_path, label="immutable TRR-0011 selector", relative_to=ROOT),
        "variant_plan_v2": plan_record,
        "panel_template_v2": template_record,
        "capture_runner": _record(capture_runner, label="immutable transfer capture runner", relative_to=ROOT),
        "capture_plan_exec_v2": _record(capture_plan_path, label="corrected executable capture plan", relative_to=ROOT),
        "capture_commands_v3": _record(capture_commands_path, label="final transfer capture commands", relative_to=ROOT),
        "capture_config_v3": _record(capture_config_path, label="final transfer capture config", relative_to=ROOT),
        "root_release_transfer_v2": _record(release_path, label="final transfer root release receipt", relative_to=ROOT),
        "seed_compatibility_proof_v2": _record(proof_path, label="transfer seed compatibility proof", relative_to=ROOT),
        "historical_public_lora_2601": _record(historical_lora, label="historical public LoRA", relative_to=ROOT.parent.parent),
        "p11_source_inputs": _record(source_inputs_path, label="P11 public source descriptor", relative_to=ROOT),
        "p11_selector": _record(p11_selector_path, label="P11 selector", relative_to=ROOT),
        "p11_selector_wrapper": _record(p11_wrapper_path, label="P11 selector wrapper", relative_to=ROOT),
    }
    output_root_rel = "outputs/TRR-P11/private-evaluation/transfer-capture"
    release_slot = "experiments/TRR-P11/exclusions/recovery_identity_audit_release.json"
    reservation = {
        "schema": "token-reconstruction.trr-p11-transfer-opaque-reservation.v1",
        "task_id": TASK_ID,
        "transfer_task_id": TRANSFER_TASK_ID,
        "status": "READY_PENDING_ROOT_EXCLUSION_RELEASE_NO_SELECTION",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reservation_receipt_sha256": "PENDING_ROOT_RELEASE_AND_SELECTION",
        "source_ranges_half_open": dict(SOURCE_RANGES),
        "domain_order": list(DOMAIN_ORDER),
        "records_by_domain": dict(TRANSFER_RECORDS_BY_DOMAIN),
        "record_count": 64,
        "selection_seed": SELECTION_SEED,
        "geometry": {
            "stored_sequence_tokens_including_bos": 128,
            "scored_post_bos_tokens": 127,
            "bos_token_id": 128000,
            "hidden_size": 2048,
            "vocabulary_size": 128256,
            "cut_depth": 4,
        },
        "target_variants": list(TRANSFER_VARIANTS),
        "selection_reuse": {
            "selector": "../TRR-0012/scripts/trr0011_select_diagnostic.py",
            "gate_function": "validate_panel_spec",
            "sampler": "existing natural 128-token trusted selector injected after release",
            "new_sampling_rule": False,
            "metadata_adaptation_only": True,
            "same_record_order_across_targets": True,
            "source_scan_performed": False,
            "selection_performed": False,
        },
        "immutable_inputs": bindings,
        "exclusion_gate": {
            "identity_union_required": True,
            "identity_union_scope": "full canonical union across fitting, inherited fitting, checkpoint selection, calibration, opened development/evaluation, duplicates, cross-study, and TRR-0009 ledgers",
            "release_audit_slot": release_slot,
            "release_requirements": ["coverage_complete=true", "selection_release=true", "root_reviewed_final_audit=true"],
            "current_audit_not_released": True,
            "current_audit_status": "PARTIAL_CANONICAL_SEQUENCE_EXCLUSION_AUDIT",
            "h40_conventions": {
                "trr0001": "raw little-endian signed-int32 H40 sequence digest",
                "trr0002": "canonical int64 tensor-header plus little-endian signed-int64 H40 token digest",
                "trr0003": "raw little-endian signed-int32 H40 sequence digest",
            },
        },
        "separation": {
            "p11_confirmation_records": 512,
            "transfer_records": 64,
            "transfer_is_separate_from_confirmation": True,
            "used_record_disjointness_proof": "deferred until released full-union audit and identity-only selection; this reservation is not proof of disjointness",
        },
        "private_artifacts": {
            "root": output_root_rel,
            "selection_panel_path": f"{output_root_rel}/opaque_panel.json",
            "evaluator_token_batch_path": f"{output_root_rel}/evaluator_token_batch.json",
            "capture_root": output_root_rel,
            "handoff_to_agent1": "PENDING_NO_HANDOFF",
            "agent1_receives_source_ids": False,
            "agent1_receives_source_text": False,
            "agent1_receives_token_ids": False,
            "agent1_receives_observations_before_root_release": False,
            "agent1_receives_predictions": False,
            "agent1_receives_truth": False,
            "agent1_receives_scores": False,
            "decoder_permitted_inputs_after_capture": ["activations", "attention_mask", "position_ids", "opaque_record_slots"],
        },
        "access_boundary": {
            "source_rows_read": False,
            "source_text_read": False,
            "token_ids_read": False,
            "source_id_hashes_emitted": False,
            "observations_captured": False,
            "predictions_started": False,
            "truth_opened": False,
            "p03_holdout_accessed": False,
            "gpu_used": False,
        },
        "template_validation": template_validation,
        "plan_summary": plan_summary,
    }
    capture_plan = {
        "schema": "token-reconstruction.trr-p11-transfer-capture-preflight.v1",
        "task_id": TASK_ID,
        "transfer_task_id": TRANSFER_TASK_ID,
        "status": "READY_DEFERRED_AUDIT_RELEASE_PANEL_AND_LIVE_PREFLIGHT",
        "reservation": {"path": str(reservation_output.relative_to(ROOT)), "status": reservation["status"], "record_count": 64},
        "source_selection_required_before_capture": True,
        "capture_runner_custody": "Agent2 evaluator process; raw token batch and target weights remain evaluator-only",
        "immutable_bindings": {key: bindings[key] for key in ("capture_runner", "capture_plan_exec_v2", "capture_commands_v3", "capture_config_v3", "root_release_transfer_v2", "seed_compatibility_proof_v2", "historical_public_lora_2601")},
        "geometry": {"full_forward_sequence_tokens": 192, "retained_sequence_tokens": 128, "scored_post_bos_tokens": 127, "hidden_size": 2048, "capture_batch_size": 8, "activation_dtype": "torch.bfloat16", "attention_mask_dtype": "torch.uint8", "position_ids_dtype": "torch.int64"},
        "variants": list(RUNNER_VARIANTS),
        "historical_variant_independent_target": False,
        "output_root": output_root_rel,
        "root_release_receipt": bindings["root_release_transfer_v2"],
        "resource_policy": {"min_free_gpu_bytes": 11811160064, "min_host_headroom_bytes": 8589934592, "host_rss_cap_bytes": 12884901888, "min_disk_free_bytes": 21474836480, "one_variant_per_process": True, "watchdog_seconds_per_variant": 900, "launch_authorized": False},
        "required_before_run": ["root release of complete P11 exclusion audit", "opaque 64-record panel produced without source payload handoff", "evaluator-owned token batch", "live GPU/host/disk preflight", "Agent1 narrow seed clarification for inherited transfer recipe"],
        "commands": _runner_commands(plan=capture_plan_path, output_root=output_root_rel, release=release_path, historical_lora=historical_lora),
        "access_boundary": {"source_text_loaded_by_decoder": False, "token_ids_loaded_by_decoder": False, "target_weights_loaded_by_decoder": False, "truth_opened": False, "selection_performed": False, "observations_started": False, "handoff_to_agent1": False},
        "no_data_expansion": True,
        "run_commands": False,
    }
    reservation_output.write_text(json.dumps(reservation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    capture_plan["reservation"]["sha256"] = _sha256_file(reservation_output)
    capture_output.write_text(json.dumps(capture_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"reservation": {"path": str(reservation_output), "bytes": reservation_output.stat().st_size, "sha256": _sha256_file(reservation_output)}, "capture_preflight": {"path": str(capture_output), "bytes": capture_output.stat().st_size, "sha256": _sha256_file(capture_output)}}


def main() -> int:
    result = prepare(
        reservation_output=ROOT / "experiments" / TASK_ID / "transfer" / "opaque64_reservation_r2.json",
        capture_output=ROOT / "experiments" / TASK_ID / "transfer" / "capture_preflight_r2.json",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
