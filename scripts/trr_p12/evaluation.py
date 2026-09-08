"""Freeze-gated P12 evaluation and source-paired scoring.

The evaluator has two deliberately separate phases.  ``freeze_evaluation``
validates a curator supplied descriptor and hashes every B1 stage artifact
(and the optional A1+A2 comparator) without loading evaluator truth.
``score_evaluation`` revalidates that receipt and only then opens the
already-materialized truth tensors named by a second descriptor.  No source
loader, tokenizer, target model, or target prefix is present in this module.

Descriptor shape (JSON) is intentionally explicit::

    {
      "schema": "token-reconstruction.trr-p12-evaluation-freeze.v1",
      "status": "FREEZE_COMPLETE_BEFORE_TRUTH",
      "truth_opened": false, "source_text_loaded": false,
      "target_weights_loaded": false,
      "source_orders": {"finance": {"hashes": [...], "sha256": "..."}},
      "methods": {
        "B1": {"domains": {"finance": {"stages": {
          "forecast": {"path": ..., "sha256": ...},
          "stages": {
            "0": {"artifacts": {
              "observation": {"path": ..., "sha256": ...},
              "prediction": {"path": ..., "sha256": ..., "key": "expanded_fixed"},
              "projected": {"path": ..., "sha256": ...}},
              "costs": {"elapsed_seconds": ...},
              "code": {"pipeline": {"path": ..., "sha256": ...}}},
            "64": {...}, "128": {...}, "256": {...}}}}
      }
    }

A domain's ``source_order`` may be supplied on the domain instead of in
``source_orders``.  A truth descriptor contains one ``cells`` entry per
method/domain/stage and a truth safetensors binding.  Prediction paths always
come from the frozen receipt; truth cannot replace them after the gate.  B1
uses one shared early forecast artifact per domain; A1+A2 uses a candidate
receipt binding and has no projected/forecast requirement. Forecast rows are
read from the frozen artifact only after the gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from scripts.trr_p12 import analysis


TASK_ID = "TRR-P12"
FREEZE_SCHEMA = "token-reconstruction.trr-p12-evaluation-freeze.v1"
FREEZE_RECEIPT_SCHEMA = "token-reconstruction.trr-p12-evaluation-freeze-receipt.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p12-evaluation-truth.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p12-evaluation-score.v1"
BOOTSTRAP_SEED = 9012
BOOTSTRAP_DRAWS = 10_000
ONE_SIDED_ALPHA = 0.025
B1_STAGES = (0, 64, 128, 256)
A1_STAGES = (0, 256)
B1_STAGE_ARTIFACTS = ("observation", "prediction", "projected")
A1_STAGE_ARTIFACTS = ("observation", "prediction", "candidate_receipt")
A1_ALIASES = frozenset({"A1+A2", "A1_A2", "A1A2", "A1-A2"})


class EvaluationError(ValueError):
    """Raised when a freeze descriptor, truth descriptor, or score is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_path(value: str | Path, *, base: Path | None = None, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() and base is not None:
        raw = base / raw
    if raw.is_symlink():
        raise EvaluationError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise EvaluationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _create_only(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() or raw.is_symlink():
        raise EvaluationError(f"{label} is create-only: {raw}")
    path = raw.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_path(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationError("non-finite number cannot enter a receipt")
    return value


def _file_binding(
    value: Any,
    *,
    base: Path,
    label: str,
    require_declared_hash: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "path" not in value:
        raise EvaluationError(f"{label} must be an explicit path/sha256 binding")
    if require_declared_hash and not isinstance(value.get("sha256"), str):
        raise EvaluationError(f"{label} must declare sha256")
    declared = value.get("sha256")
    if declared is not None:
        declared = str(declared).lower()
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise EvaluationError(f"{label} sha256 is malformed")
    path = _regular_path(str(value["path"]), base=base, label=label)
    actual = sha256_file(path)
    if declared is not None and actual != declared:
        raise EvaluationError(f"{label} sha256 changed: expected {declared}, got {actual}")
    actual_bytes = int(path.stat().st_size)
    if "bytes" in value and int(value["bytes"]) != actual_bytes:
        raise EvaluationError(f"{label} byte count changed")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual,
    }
    for key in ("key", "format", "role"):
        if key in value:
            result[key] = str(value[key])
    return result


def _source_order(value: Any, *, label: str, expected_count: int | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} must contain ordered source hashes")
    hashes_value = value.get("hashes", value.get("source_hashes"))
    if not isinstance(hashes_value, list) or not hashes_value:
        raise EvaluationError(f"{label} hashes are missing")
    hashes = [str(item).lower() for item in hashes_value]
    if len(set(hashes)) != len(hashes):
        raise EvaluationError(f"{label} hashes are not unique")
    for item in hashes:
        if len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise EvaluationError(f"{label} contains a malformed source hash")
    computed = analysis.source_order_digest(hashes)
    declared = value.get("sha256", value.get("order_sha256"))
    if not isinstance(declared, str) or declared.lower() != computed:
        raise EvaluationError(f"{label} digest does not match ordered hashes")
    if expected_count is not None and len(hashes) != int(expected_count):
        raise EvaluationError(f"{label} has {len(hashes)} records; expected {expected_count}")
    return {"hashes": hashes, "sha256": computed, "record_count": len(hashes)}


def _normalise_costs(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationError(f"{label} must be a non-empty cost/timing mapping")
    result = _json_safe(dict(value))
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise EvaluationError(f"{label} is malformed")
    return result


def _normalise_code(value: Any, *, base: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationError(f"{label} must bind at least one code file")
    files_value = value.get("files", value)
    if not isinstance(files_value, Mapping) or not files_value:
        raise EvaluationError(f"{label}.files must bind at least one code file")
    files = {
        str(name): _file_binding(binding, base=base, label=f"{label}.{name}")
        for name, binding in files_value.items()
    }
    result: dict[str, Any] = {"files": files}
    for key in ("commit", "revision", "description"):
        if key in value:
            result[key] = str(value[key])
    return result



def _hash_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise EvaluationError(f"{label} must be a sha256 string")
    value = value.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvaluationError(f"{label} sha256 is malformed")
    return value


def _validate_public_tensor_artifact(
    binding: Mapping[str, Any],
    *,
    method: str,
    stage: int,
    artifact: str,
    record_count: int,
) -> dict[str, Any]:
    """Check public prediction/feature metadata while truth is still closed."""

    if artifact not in {"prediction", "projected"}:
        return dict(binding)
    path = _regular_path(binding["path"], label=f"{method}/stage{stage}/{artifact}")
    default_key = "expanded_fixed" if artifact == "prediction" else "projected_hidden"
    key = str(binding.get("key", default_key))
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise EvaluationError(f"{method}/stage{stage}/{artifact} key {key!r} is absent")
            tensor_slice = handle.get_slice(key)
            shape = tuple(int(item) for item in tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"{method}/stage{stage}/{artifact} is not readable safetensors") from exc
    if not shape or shape[0] != int(record_count):
        raise EvaluationError(
            f"{method}/stage{stage}/{artifact} record count {shape[0] if shape else 0} differs from source order {record_count}"
        )
    if artifact == "prediction":
        if len(shape) != 2 or shape[1] <= 1 or dtype not in {"I8", "I16", "I32", "I64", "U8"}:
            raise EvaluationError(f"{method}/stage{stage}/prediction must be integer [records,tokens>1]")
    elif len(shape) != 3 or shape[1] <= 0 or shape[2] <= 0 or dtype not in {"BF16", "F16", "F32", "F64"}:
        raise EvaluationError(f"{method}/stage{stage}/projected must be floating [records,positions,hidden]")
    result = dict(binding)
    result["key"] = key
    result["shape"] = list(shape)
    result["dtype"] = dtype
    return result


def _validate_observation_artifact(binding: Mapping[str, Any], *, method: str, stage: int, record_count: int) -> dict[str, Any]:
    """Validate count metadata for a sanitized observation artifact if available."""

    path = _regular_path(binding["path"], label=f"{method}/stage{stage}/observation")
    result = dict(binding)
    suffix = path.suffix.lower()
    if suffix in {".safetensors", ".safetensor"}:
        key = str(binding.get("key", "observations"))
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if key not in set(handle.keys()):
                    if len(handle.keys()) != 1:
                        raise EvaluationError(f"{method}/stage{stage}/observation key is absent")
                    key = next(iter(handle.keys()))
                shape = tuple(int(item) for item in handle.get_slice(key).get_shape())
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"{method}/stage{stage}/observation is not readable safetensors") from exc
        if not shape or shape[0] != int(record_count):
            raise EvaluationError(f"{method}/stage{stage}/observation record count differs from source order")
        result["key"] = key
        result["shape"] = list(shape)
    elif suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"{method}/stage{stage}/observation is not valid JSON") from exc
        if isinstance(payload, Mapping) and "record_count" in payload and int(payload["record_count"]) != int(record_count):
            raise EvaluationError(f"{method}/stage{stage}/observation record count differs from source order")
    return result


def _collect_decoder_bindings(value: Any, *, base: Path, label: str, found: list[dict[str, Any]], hash_keys: list[str]) -> None:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            found.append(_file_binding(value, base=base, label=label))
            return
        if "sha256" in value:
            hash_keys.append(label)
            _hash_string(value["sha256"], label=f"{label}.sha256")
        for key, child in value.items():
            if key in {"path", "bytes", "sha256", "package_root"}:
                continue
            _collect_decoder_bindings(child, base=base, label=f"{label}/{key}", found=found, hash_keys=hash_keys)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_decoder_bindings(child, base=base, label=f"{label}/{index}", found=found, hash_keys=hash_keys)
    elif isinstance(value, str) and (label.endswith("sha256") or label.endswith("_sha256")):
        hash_keys.append(label)
        _hash_string(value, label=label)


def _normalise_registrations(descriptor: Mapping[str, Any], *, base: Path, strict: bool) -> dict[str, Any]:
    if not strict:
        return {}
    geometry = descriptor.get("registered_geometry")
    if geometry is None:
        geometry = descriptor.get("geometry_registration", descriptor.get("actual_boundary_analysis"))
    if not isinstance(geometry, Mapping):
        raise EvaluationError("strict freeze requires registered geometry diagnostics")
    pairs = geometry.get("stage_pairs")
    if not isinstance(pairs, list) or {str(item) for item in pairs} != {"0_to_64", "0_to_128", "0_to_256"}:
        raise EvaluationError("registered geometry must bind 0_to_64, 0_to_128, and 0_to_256")
    if geometry.get("full_vocabulary_required") is not True:
        raise EvaluationError("registered geometry must require full-vocabulary competitors")
    if geometry.get("retrospective_only") is not True:
        raise EvaluationError("registered actual-boundary geometry must be retrospective-only")
    decoder = descriptor.get("frozen_decoder_binding", descriptor.get("decoder_bindings"))
    if not isinstance(decoder, Mapping):
        raise EvaluationError("strict freeze requires frozen decoder/package bindings")
    files: list[dict[str, Any]] = []
    hash_keys: list[str] = []
    _collect_decoder_bindings(decoder, base=base, label="frozen_decoder_binding", found=files, hash_keys=hash_keys)
    if len(files) < 3:
        raise EvaluationError("frozen decoder/package binding must include at least three path/sha256 files")
    labels = " ".join(item["label"].lower() for item in files)
    if "decoder" not in labels or not any(term in labels for term in ("package", "readout", "b1_state")):
        raise EvaluationError("frozen decoder/package binding lacks decoder and package/readout evidence")
    return {
        "registered_geometry": {
            "stage_pairs": ["0_to_64", "0_to_128", "0_to_256"],
            "full_vocabulary_required": True,
            "retrospective_only": True,
            "no_nearest_euclidean_claim_from_runner_up": bool(geometry.get("no_nearest_euclidean_claim_from_runner_up", True)),
        },
        "decoder_files": files,
        "decoder_hash_bindings": sorted(hash_keys),
    }


def _verify_normalized_registrations(value: Any, *, base: Path, strict: bool) -> dict[str, Any]:
    if not strict:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationError("freeze receipt registration bindings are missing")
    geometry = value.get("registered_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("stage_pairs") != ["0_to_64", "0_to_128", "0_to_256"]:
        raise EvaluationError("freeze receipt registered geometry differs")
    if geometry.get("full_vocabulary_required") is not True or geometry.get("retrospective_only") is not True:
        raise EvaluationError("freeze receipt geometry flags differ")
    files = value.get("decoder_files")
    if not isinstance(files, list) or len(files) < 3:
        raise EvaluationError("freeze receipt decoder/package files are incomplete")
    checked = [_file_binding(item, base=base, label=f"freeze receipt decoder file {index}") for index, item in enumerate(files)]
    labels = " ".join(item["label"].lower() for item in checked)
    if "decoder" not in labels or not any(term in labels for term in ("package", "readout", "b1_state")):
        raise EvaluationError("freeze receipt decoder/package evidence is incomplete")
    return {"registered_geometry": dict(geometry), "decoder_files": checked, "decoder_hash_bindings": list(value.get("decoder_hash_bindings", []))}

def _canonical_method(name: str) -> str:
    name = str(name)
    return "A1+A2" if name in A1_ALIASES else name


def _method_stages(method: str) -> tuple[int, ...]:
    if method == "B1":
        return B1_STAGES
    if method == "A1+A2":
        return A1_STAGES
    raise EvaluationError(f"unsupported method in P12 descriptor: {method}")


def _normalise_methods(
    raw_methods: Any,
    *,
    source_orders: Mapping[str, Any],
    base: Path,
    strict_record_counts: bool,
) -> dict[str, Any]:
    if not isinstance(raw_methods, Mapping) or "B1" not in raw_methods:
        raise EvaluationError("freeze descriptor must include method B1")
    result: dict[str, Any] = {}
    for raw_name, raw_method in raw_methods.items():
        method = _canonical_method(str(raw_name))
        if method in result:
            raise EvaluationError(f"duplicate method after canonicalization: {method}")
        if method not in ("B1", "A1+A2"):
            raise EvaluationError(f"unsupported method in P12 descriptor: {raw_name}")
        if not isinstance(raw_method, Mapping):
            raise EvaluationError(f"method {method} must be an object")
        raw_domains = raw_method.get("domains")
        if not isinstance(raw_domains, Mapping) or not raw_domains:
            raise EvaluationError(f"method {method} must contain domains")
        expected_count = 128 if method == "B1" else 32
        domains: dict[str, Any] = {}
        for raw_domain, raw_domain_value in raw_domains.items():
            domain = str(raw_domain)
            if not isinstance(raw_domain_value, Mapping):
                raise EvaluationError(f"method {method} domain {domain} is malformed")
            source_value = raw_domain_value.get("source_order")
            if source_value is None:
                source_value = source_orders.get(domain)
            if source_value is None:
                raise EvaluationError(f"source order is missing for {method}/{domain}")
            order = _source_order(
                source_value,
                label=f"{method}/{domain} source order",
                expected_count=expected_count if strict_record_counts else None,
            )
            raw_stages = raw_domain_value.get("stages")
            if not isinstance(raw_stages, Mapping):
                raise EvaluationError(f"{method}/{domain} stages are missing")
            stage_keys = {int(str(key)) for key in raw_stages}
            required_stages = set(_method_stages(method))
            if stage_keys != required_stages:
                raise EvaluationError(
                    f"{method}/{domain} stages must be {sorted(required_stages)}, got {sorted(stage_keys)}"
                )
            stages: dict[str, Any] = {}
            method_code = raw_method.get("code")
            raw_domain_forecast = raw_domain_value.get("forecast")
            if method == "B1" and raw_domain_forecast is None:
                for early_stage in (0, 64):
                    early_raw = raw_stages.get(str(early_stage), raw_stages.get(early_stage))
                    if isinstance(early_raw, Mapping) and isinstance(early_raw.get("artifacts"), Mapping) and "forecast" in early_raw["artifacts"]:
                        raw_domain_forecast = early_raw["artifacts"]["forecast"]
                        break
            if method == "B1" and raw_domain_forecast is None:
                raise EvaluationError(f"B1/{domain} lacks its shared early forecast binding")
            normalized_forecast = None if method != "B1" else _file_binding(
                raw_domain_forecast, base=base, label=f"B1/{domain}/forecast"
            )
            for stage in _method_stages(method):
                raw_stage = raw_stages.get(str(stage), raw_stages.get(stage))
                if not isinstance(raw_stage, Mapping):
                    raise EvaluationError(f"{method}/{domain}/stage{stage} is malformed")
                raw_artifacts = raw_stage.get("artifacts")
                if not isinstance(raw_artifacts, Mapping):
                    raise EvaluationError(f"{method}/{domain}/stage{stage} artifacts are missing")
                count = raw_stage.get("record_count", order["record_count"])
                if int(count) != order["record_count"]:
                    raise EvaluationError(f"{method}/{domain}/stage{stage} record count differs from source order")
                if method == "B1":
                    required_artifacts = B1_STAGE_ARTIFACTS
                    artifacts: dict[str, Any] = {}
                    for artifact in required_artifacts:
                        if artifact not in raw_artifacts:
                            raise EvaluationError(f"{method}/{domain}/stage{stage} lacks {artifact} binding")
                        artifacts[artifact] = _file_binding(
                            raw_artifacts[artifact], base=base, label=f"{method}/{domain}/stage{stage}/{artifact}"
                        )
                    artifacts["observation"] = _validate_observation_artifact(
                        artifacts["observation"], method=method, stage=stage, record_count=int(count)
                    )
                    artifacts["prediction"] = _validate_public_tensor_artifact(
                        artifacts["prediction"], method=method, stage=stage, artifact="prediction", record_count=int(count)
                    )
                    artifacts["projected"] = _validate_public_tensor_artifact(
                        artifacts["projected"], method=method, stage=stage, artifact="projected", record_count=int(count)
                    )
                else:
                    candidate_value = raw_artifacts.get("candidate_receipt", raw_artifacts.get("candidate", raw_artifacts.get("receipt")))
                    if candidate_value is None:
                        candidate_value = raw_domain_value.get("candidate_receipt", raw_method.get("candidate_receipt"))
                    if candidate_value is None:
                        raise EvaluationError(f"{method}/{domain}/stage{stage} lacks candidate receipt binding")
                    artifacts = {}
                    for artifact, artifact_value in (("observation", raw_artifacts.get("observation")), ("prediction", raw_artifacts.get("prediction")), ("candidate_receipt", candidate_value)):
                        if artifact_value is None:
                            raise EvaluationError(f"{method}/{domain}/stage{stage} lacks {artifact} binding")
                        artifacts[artifact] = _file_binding(
                            artifact_value, base=base, label=f"{method}/{domain}/stage{stage}/{artifact}"
                        )
                    artifacts["observation"] = _validate_observation_artifact(
                        artifacts["observation"], method=method, stage=stage, record_count=int(count)
                    )
                    artifacts["prediction"] = _validate_public_tensor_artifact(
                        artifacts["prediction"], method=method, stage=stage, artifact="prediction", record_count=int(count)
                    )
                stage_order_digest = raw_stage.get("source_order_sha256")
                if stage_order_digest is not None and str(stage_order_digest).lower() != order["sha256"]:
                    raise EvaluationError(f"{method}/{domain}/stage{stage} source order differs")
                raw_costs = raw_stage.get("costs", raw_stage.get("cost"))
                costs = _normalise_costs(raw_costs, label=f"{method}/{domain}/stage{stage}/costs")
                raw_code = raw_stage.get("code", method_code)
                code = _normalise_code(raw_code, base=base, label=f"{method}/{domain}/stage{stage}/code")
                stages[str(stage)] = {
                    "stage": stage,
                    "record_count": int(count),
                    "source_order_sha256": order["sha256"],
                    "artifacts": artifacts,
                    "costs": costs,
                    "code": code,
                }
            domain_output: dict[str, Any] = {"source_order": order, "stages": stages}
            if normalized_forecast is not None:
                domain_output["forecast"] = normalized_forecast
            domains[domain] = domain_output
        result[method] = {
            "method": method,
            "record_scope": "all128_per_domain" if method == "B1" else "first32_per_domain",
            "domains": domains,
        }
    if "A1+A2" in result:
        b1_domains = result["B1"]["domains"]
        for domain, a1_value in result["A1+A2"]["domains"].items():
            if domain not in b1_domains:
                raise EvaluationError(f"A1+A2 domain {domain} has no B1 common source order")
            b1_hashes = b1_domains[domain]["source_order"]["hashes"]
            a1_hashes = a1_value["source_order"]["hashes"]
            if a1_hashes != b1_hashes[: len(a1_hashes)]:
                raise EvaluationError(f"A1+A2/{domain} is not the B1 common-prefix source order")
    return result


def validate_freeze_descriptor(
    descriptor: Mapping[str, Any],
    *,
    base: Path,
    strict_record_counts: bool = True,
) -> dict[str, Any]:
    """Validate and normalize a pretruth descriptor without opening truth."""

    if descriptor.get("schema") != FREEZE_SCHEMA:
        raise EvaluationError("freeze descriptor schema differs")
    if descriptor.get("status") != "FREEZE_COMPLETE_BEFORE_TRUTH":
        raise EvaluationError("freeze descriptor status is not pretruth-complete")
    for flag in ("truth_opened", "source_text_loaded", "target_weights_loaded"):
        if descriptor.get(flag) is not False:
            raise EvaluationError(f"freeze descriptor violates {flag}=false")
    raw_source_orders = descriptor.get("source_orders", {})
    if not isinstance(raw_source_orders, Mapping):
        raise EvaluationError("source_orders must be an object")
    if strict_record_counts:
        raw_method_names = {_canonical_method(str(name)) for name in descriptor.get("methods", {}).keys()} if isinstance(descriptor.get("methods"), Mapping) else set()
        if raw_method_names != {"B1", "A1+A2"}:
            raise EvaluationError("strict freeze requires both B1 and A1+A2")
        for raw_name, raw_method in descriptor["methods"].items():
            method_name = _canonical_method(str(raw_name))
            if isinstance(raw_method, Mapping) and isinstance(raw_method.get("domains"), Mapping) and set(str(item) for item in raw_method["domains"]) != {"finance", "pile"}:
                raise EvaluationError(f"strict freeze requires exactly finance and pile domains for {method_name}")
    methods = _normalise_methods(
        descriptor.get("methods"),
        source_orders=raw_source_orders,
        base=base,
        strict_record_counts=strict_record_counts,
    )
    if strict_record_counts:
        if set(methods) != {"B1", "A1+A2"}:
            raise EvaluationError("strict freeze requires both B1 and A1+A2")
        if set(methods["B1"]["domains"]) != {"finance", "pile"} or set(methods["A1+A2"]["domains"]) != {"finance", "pile"}:
            raise EvaluationError("strict freeze requires exactly finance and pile domains for both methods")
    return {
        "schema": FREEZE_SCHEMA,
        "status": "FREEZE_COMPLETE_BEFORE_TRUTH",
        "source_orders": {
            str(domain): _source_order(value, label=f"source_orders/{domain}")
            for domain, value in raw_source_orders.items()
        },
        "methods": methods,
        "registrations": _normalise_registrations(descriptor, base=base, strict=strict_record_counts),
        "truth_opened": False,
        "source_text_loaded": False,
        "target_weights_loaded": False,
    }


def freeze_evaluation(
    descriptor_path: str | Path,
    receipt_output: str | Path,
    *,
    strict_record_counts: bool = True,
) -> dict[str, Any]:
    """Hash and freeze all required B1/A1 artifacts before truth is opened."""

    receipt_path = _create_only(receipt_output, label="freeze receipt")
    descriptor_file = _regular_path(descriptor_path, label="freeze descriptor")
    descriptor = _json_object(descriptor_file, label="freeze descriptor")
    normalized = validate_freeze_descriptor(
        descriptor,
        base=descriptor_file.parent,
        strict_record_counts=strict_record_counts,
    )
    receipt = {
        "schema": FREEZE_RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "created_utc": utc_now(),
        "freeze_descriptor": _file_binding(
            {"path": str(descriptor_file), "sha256": sha256_file(descriptor_file)},
            base=descriptor_file.parent,
            label="freeze descriptor",
        ),
        "record_count_policy": "B1=128,A1+A2=32" if strict_record_counts else "compact_fixture_counts_allowed",
        "source_orders": normalized["source_orders"],
        "methods": normalized["methods"],
        "registrations": normalized["registrations"],
        "truth_opened": False,
        "source_text_loaded": False,
        "target_weights_loaded": False,
        "code": {"evaluation": _file_binding(
            {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            base=Path(__file__).resolve().parent,
            label="evaluation code",
        )},
    }
    receipt_path.write_text(json.dumps(_json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return receipt


def _load_freeze_receipt(
    freeze_receipt_path: str | Path,
    *,
    strict_record_counts: bool,
) -> tuple[Path, dict[str, Any]]:
    path = _regular_path(freeze_receipt_path, label="freeze receipt")
    receipt = _json_object(path, label="freeze receipt")
    if receipt.get("schema") != FREEZE_RECEIPT_SCHEMA or receipt.get("status") != "PREDICTIONS_FROZEN_BEFORE_TRUTH":
        raise EvaluationError("freeze receipt is not a P12 pretruth receipt")
    for flag in ("truth_opened", "source_text_loaded", "target_weights_loaded"):
        if receipt.get(flag) is not False:
            raise EvaluationError(f"freeze receipt violates {flag}=false")
    _file_binding(receipt.get("freeze_descriptor"), base=path.parent, label="freeze descriptor binding")
    raw_code = receipt.get("code")
    if not isinstance(raw_code, Mapping) or not raw_code:
        raise EvaluationError("freeze receipt code bindings are missing")
    for code_name, code_binding in raw_code.items():
        _file_binding(code_binding, base=path.parent, label=f"freeze receipt code/{code_name}")
    normalized = _normalise_methods(
        receipt.get("methods"),
        source_orders=receipt.get("source_orders", {}),
        base=path.parent,
        strict_record_counts=strict_record_counts,
    )
    if strict_record_counts:
        if set(normalized) != {"B1", "A1+A2"}:
            raise EvaluationError("strict freeze receipt lacks B1 or A1+A2")
        if set(normalized["B1"]["domains"]) != {"finance", "pile"} or set(normalized["A1+A2"]["domains"]) != {"finance", "pile"}:
            raise EvaluationError("strict freeze receipt domains differ")
    receipt["registrations"] = _verify_normalized_registrations(receipt.get("registrations"), base=path.parent, strict=strict_record_counts)
    receipt["methods"] = normalized
    return path, receipt


def _cell_key(method: str, domain: str, stage: int) -> tuple[str, str, int]:
    return _canonical_method(method), str(domain), int(stage)


def _normalise_truth_descriptor(
    descriptor: Mapping[str, Any],
    *,
    descriptor_path: Path,
    freeze_path: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    if descriptor.get("schema") != TRUTH_SCHEMA:
        raise EvaluationError("truth descriptor schema differs")
    if descriptor.get("status") not in {"TRUTH_READY_AFTER_FREEZE", "TRUTH_OPENED_AFTER_FREEZE"}:
        raise EvaluationError("truth descriptor status differs")
    if descriptor.get("truth_opened") is not True:
        raise EvaluationError("truth descriptor must declare truth_opened=true")
    expected_freeze_hash = descriptor.get("freeze_receipt_sha256")
    if expected_freeze_hash is not None and str(expected_freeze_hash).lower() != sha256_file(freeze_path):
        raise EvaluationError("truth descriptor is bound to a different freeze receipt")
    raw_cells = descriptor.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise EvaluationError("truth descriptor cells are missing")
    freeze_methods = freeze.get("methods")
    if not isinstance(freeze_methods, Mapping):
        raise EvaluationError("freeze receipt methods are missing")
    expected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for method_name, method_value in freeze_methods.items():
        for domain, domain_value in method_value["domains"].items():
            for stage_text, stage_value in domain_value["stages"].items():
                expected[_cell_key(str(method_name), domain, int(stage_text))] = stage_value
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            raise EvaluationError(f"truth cell {index} is malformed")
        key = _cell_key(str(raw_cell.get("method")), str(raw_cell.get("domain")), int(raw_cell.get("stage")))
        if key in cells:
            raise EvaluationError(f"duplicate truth cell {key}")
        if key not in expected:
            raise EvaluationError(f"truth cell {key} is not present in freeze")
        truth_value = raw_cell.get("truth", raw_cell.get("evaluator_truth"))
        truth_binding = _file_binding(
            truth_value,
            base=descriptor_path.parent,
            label=f"truth/{key}",
        )
        mask_value = raw_cell.get("valid_mask")
        mask_binding = None if mask_value is None else _file_binding(
            mask_value,
            base=descriptor_path.parent,
            label=f"valid_mask/{key}",
        )
        source_digest = raw_cell.get("source_order_sha256")
        expected_digest = freeze["methods"][key[0]]["domains"][key[1]]["source_order"]["sha256"]
        if source_digest is not None and str(source_digest).lower() != expected_digest:
            raise EvaluationError(f"truth cell {key} source order differs from freeze")
        cells[key] = {
            "method": key[0],
            "domain": key[1],
            "stage": key[2],
            "truth": truth_binding,
            "valid_mask": mask_binding,
            "source_order_sha256": expected_digest,
        }
    if set(cells) != set(expected):
        missing = sorted(set(expected) - set(cells))
        raise EvaluationError(f"truth descriptor lacks frozen cells: {missing[:4]}")
    return {
        "schema": TRUTH_SCHEMA,
        "status": "TRUTH_OPENED_AFTER_FREEZE",
        "truth_opened": True,
        "cells": cells,
    }


def _load_safetensor(binding: Mapping[str, Any], *, label: str) -> torch.Tensor:
    path = _regular_path(binding["path"], label=label)
    key = str(binding.get("key", "tensor"))
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise EvaluationError(f"{label} key {key!r} is absent")
            value = handle.get_tensor(key).detach().cpu().contiguous()
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"{label} cannot be loaded as safetensors") from exc
    return value


def _load_cell_tensors(
    freeze_cell: Mapping[str, Any],
    truth_cell: Mapping[str, Any],
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    prediction = _load_safetensor(freeze_cell["artifacts"]["prediction"], label=f"{label} prediction")
    truth = _load_safetensor(truth_cell["truth"], label=f"{label} evaluator truth")
    mask_binding = truth_cell.get("valid_mask")
    mask = None if mask_binding is None else _load_safetensor(mask_binding, label=f"{label} valid mask")
    if prediction.ndim != 2 or truth.shape != prediction.shape:
        raise EvaluationError(f"{label} prediction/truth geometry differs")
    if mask is not None and tuple(mask.shape) != tuple(prediction.shape):
        raise EvaluationError(f"{label} valid mask geometry differs")
    return prediction.to(dtype=torch.long), truth.to(dtype=torch.long), mask


def _result_for_stage(
    baseline: torch.Tensor,
    stage_prediction: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    method: str,
    domain: str,
    stage: int,
    source_hashes: Sequence[str],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    if stage == 0:
        result = analysis.absolute_stage_summary(
            baseline, truth, valid_mask=mask, domain=domain, stage=stage,
        )
    else:
        result = analysis.paired_stage_analysis(
            baseline,
            stage_prediction,
            truth,
            valid_mask=mask,
            source_ids=source_hashes,
            domain=domain,
            stage=stage,
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
            one_sided_alpha=ONE_SIDED_ALPHA,
        )
    result = analysis.json_ready(result)
    absolute = result.get("score", result.get("stage_score"))
    if isinstance(absolute, Mapping):
        result["absolute"] = dict(absolute)
        result["exact"] = {
            "exact_records": int(absolute["exact_records"]),
            "records": int(absolute["records"]),
            "exact_rate": float(absolute["exact_rate"]),
        }
    if "paired" not in result:
        result["paired"] = {
            "status": "BASELINE_REFERENCE",
            "broken_tokens": 0,
            "improved_tokens": 0,
            "broken_exact_records": 0,
            "improved_exact_records": 0,
        }
    result["method"] = method
    result["domain"] = domain
    result["stage"] = str(stage)
    result["source_order_sha256"] = analysis.source_order_digest(source_hashes)
    return result


def _forecast_rows(binding: Mapping[str, Any], *, label: str, records: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = _regular_path(binding["path"], label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} is not valid forecast JSON") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise EvaluationError(f"{label} has no compact forecast rows")
    rows = payload["rows"]
    if len(rows) != records:
        raise EvaluationError(f"{label} has {len(rows)} rows; expected {records}")
    strict_rows: dict[int, list[bool]] = {}
    valid_rows: dict[int, list[bool]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or int(row.get("record_index", -1)) != index:
            raise EvaluationError(f"{label} record order differs at {index}")
        valid = row.get("valid")
        if not isinstance(valid, list) or not valid:
            raise EvaluationError(f"{label} row {index} valid mask is missing")
        for stage in (128, 256):
            strict = row.get("predicted_strict_change_by_stage", {}).get(str(stage))
            if not isinstance(strict, list) or len(strict) != len(valid):
                raise EvaluationError(f"{label} row {index} stage {stage} forecast geometry differs")
            strict_rows.setdefault(stage, []).append([bool(item) for item in strict])
            valid_rows.setdefault(stage, []).append([bool(item) for item in valid])
    return (
        torch.tensor(strict_rows[128], dtype=torch.bool),
        torch.tensor(strict_rows[256], dtype=torch.bool),
        torch.tensor(valid_rows[128], dtype=torch.bool),
        payload,
    )


def _forecast_table(
    baseline: torch.Tensor,
    later: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor | None,
    predicted: torch.Tensor,
    forecast_valid: torch.Tensor,
    *,
    record_mask: torch.Tensor,
) -> dict[str, Any]:
    if baseline.ndim != 2 or later.shape != baseline.shape or truth.shape != baseline.shape:
        raise EvaluationError("forecast prediction/truth geometry differs")
    if predicted.shape != (baseline.shape[0], baseline.shape[1] - 1) or forecast_valid.shape != predicted.shape:
        raise EvaluationError("forecast rows must align with post-BOS tokens")
    valid = torch.ones_like(baseline, dtype=torch.bool) if valid_mask is None else valid_mask.to(dtype=torch.bool)
    valid = valid.clone()
    valid[:, 0] = False
    eligible = valid[:, 1:] & forecast_valid & record_mask[:, None]
    base_correct = baseline[:, 1:].eq(truth[:, 1:])
    broken = base_correct & ~later[:, 1:].eq(truth[:, 1:])
    def table(mask: torch.Tensor) -> dict[str, int]:
        actual = broken & mask
        guess = predicted & mask
        return {
            "eligible_tokens": int(mask.sum().item()),
            "baseline_correct_tokens": int((base_correct & mask).sum().item()),
            "broken_tokens": int(actual.sum().item()),
            "predicted_strict_change_tokens": int(guess.sum().item()),
            "true_positive": int((actual & guess).sum().item()),
            "false_positive": int((~actual & guess & mask).sum().item()),
            "true_negative": int((~actual & ~guess & mask).sum().item()),
            "false_negative": int((actual & ~guess).sum().item()),
        }
    baseline_correct_eligible = eligible & base_correct
    return {
        "record_count": int(record_mask.sum().item()),
        "eligible_records": int(record_mask.sum().item()),
        "primary_baseline_correct_only": table(baseline_correct_eligible),
        "descriptive_all_valid": table(eligible),
        "decision_rule": "strict predicted change iff estimated matching-stage baseline-winner margin < 0; ties are excluded from strict positives",
    }


def _score_forecast_for_domain(
    freeze_method: Mapping[str, Any],
    truth_cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    domain: str,
    source_hashes: Sequence[str],
    baseline_prediction: torch.Tensor,
    baseline_truth: torch.Tensor,
    baseline_mask: torch.Tensor | None,
    stage_predictions: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    records = len(source_hashes)
    if records < 2 or records % 2:
        raise EvaluationError("forecast second64 split requires an even record count")
    second_half = torch.zeros(records, dtype=torch.bool)
    second_half[records // 2 :] = True
    results: dict[str, Any] = {
        "method": "B1",
        "domain": domain,
        "record_scope_primary": "second64_per_domain",
        "record_scope_descriptive": "full128_per_domain",
        "stages": {},
    }
    for later_stage in (128, 256):
        frozen_domain = freeze_method["domains"][domain]
        frozen_stage = frozen_domain["stages"][str(later_stage)]
        strict128, strict256, _, forecast_payload = _forecast_rows(
            frozen_domain["forecast"],
            label=f"B1/{domain}/stage{later_stage}/forecast",
            records=records,
        )
        predicted = strict128 if later_stage == 128 else strict256
        forecast_valid = torch.tensor(
            [[bool(item) for item in row.get("valid", [])] for row in forecast_payload["rows"]],
            dtype=torch.bool,
        )
        table = _forecast_table(
            baseline_prediction,
            stage_predictions[later_stage],
            baseline_truth,
            baseline_mask,
            predicted,
            forecast_valid,
            record_mask=second_half,
        )
        full = _forecast_table(
            baseline_prediction,
            stage_predictions[later_stage],
            baseline_truth,
            baseline_mask,
            predicted,
            forecast_valid,
            record_mask=torch.ones(records, dtype=torch.bool),
        )
        results["stages"][str(later_stage)] = {
            "matching_stage": later_stage,
            "primary_second64": table,
            "descriptive_full128": full,
            "forecast_artifact_sha256": frozen_domain["forecast"]["sha256"],
        }
    return results


def score_evaluation(
    freeze_receipt_path: str | Path,
    truth_descriptor_path: str | Path,
    score_output: str | Path,
    *,
    strict_record_counts: bool = True,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Revalidate a pretruth receipt, then score materialized truth exactly once."""

    output_path = _create_only(score_output, label="score output")
    started = time.perf_counter()
    freeze_path, freeze = _load_freeze_receipt(
        freeze_receipt_path,
        strict_record_counts=strict_record_counts,
    )
    truth_path = _regular_path(truth_descriptor_path, label="truth descriptor")
    truth_descriptor = _json_object(truth_path, label="truth descriptor")
    normalized_truth = _normalise_truth_descriptor(
        truth_descriptor,
        descriptor_path=truth_path,
        freeze_path=freeze_path,
        freeze=freeze,
    )
    truth_cells = normalized_truth["cells"]
    methods_output: dict[str, Any] = {}
    all_tensor_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
    for method, method_value in freeze["methods"].items():
        method_out: dict[str, Any] = {"method": method, "domains": {}}
        for domain, domain_value in method_value["domains"].items():
            source_hashes = domain_value["source_order"]["hashes"]
            cells = {}
            for stage in _method_stages(method):
                key = _cell_key(method, domain, stage)
                freeze_cell = domain_value["stages"][str(stage)]
                truth_cell = truth_cells[key]
                tensors = _load_cell_tensors(freeze_cell, truth_cell, label=f"{method}/{domain}/stage{stage}")
                all_tensor_cache[key] = tensors
                cells[stage] = tensors
            baseline_prediction, baseline_truth, baseline_mask = cells[0]
            domain_out: dict[str, Any] = {
                "source_order_sha256": analysis.source_order_digest(source_hashes),
                "record_count": len(source_hashes),
                "stages": {},
            }
            for stage in _method_stages(method):
                prediction, truth, mask = cells[stage]
                if not torch.equal(truth, baseline_truth):
                    raise EvaluationError(f"{method}/{domain} evaluator truth changed across stages")
                if mask is not None and baseline_mask is not None and not torch.equal(mask, baseline_mask):
                    raise EvaluationError(f"{method}/{domain} valid mask changed across stages")
                domain_out["stages"][str(stage)] = _result_for_stage(
                    baseline_prediction,
                    prediction,
                    truth,
                    mask if mask is not None else baseline_mask,
                    method=method,
                    domain=domain,
                    stage=stage,
                    source_hashes=source_hashes,
                    bootstrap_seed=int(bootstrap_seed),
                    bootstrap_draws=int(bootstrap_draws),
                )
            if method == "B1":
                domain_out["forecast"] = _score_forecast_for_domain(
                    method_value,
                    truth_cells,
                    domain=domain,
                    source_hashes=source_hashes,
                    baseline_prediction=baseline_prediction,
                    baseline_truth=baseline_truth,
                    baseline_mask=baseline_mask,
                    stage_predictions={stage: cells[stage][0] for stage in (128, 256)},
                )
            method_out["domains"][domain] = domain_out
        methods_output[method] = method_out

    comparator: dict[str, Any] | None = None
    if "A1+A2" in freeze["methods"]:
        comparator = {"method": "A1+A2_vs_B1", "domains": {}}
        for domain in freeze["methods"]["A1+A2"]["domains"]:
            b1_order = freeze["methods"]["B1"]["domains"][domain]["source_order"]["hashes"]
            a1_order = freeze["methods"]["A1+A2"]["domains"][domain]["source_order"]["hashes"]
            common_count = len(a1_order)
            if a1_order != b1_order[:common_count]:
                raise EvaluationError(f"common32 source order differs for {domain}")
            domain_compare: dict[str, Any] = {"common_record_count": common_count, "stages": {}}
            for stage in A1_STAGES:
                b1_pred, b1_truth, b1_mask = all_tensor_cache[("B1", domain, stage)]
                a1_pred, a1_truth, a1_mask = all_tensor_cache[("A1+A2", domain, stage)]
                b1_pred = b1_pred[:common_count]
                b1_truth = b1_truth[:common_count]
                a1_truth = a1_truth[:common_count]
                if not torch.equal(b1_truth, a1_truth):
                    raise EvaluationError(f"common32 truth differs for {domain}/stage{stage}")
                mask = None
                if b1_mask is not None:
                    mask = b1_mask[:common_count]
                if a1_mask is not None:
                    a1_mask = a1_mask[:common_count]
                    if mask is not None and not torch.equal(mask, a1_mask):
                        raise EvaluationError(f"common32 valid mask differs for {domain}/stage{stage}")
                    mask = a1_mask
                contrast = analysis.paired_stage_analysis(
                    b1_pred,
                    a1_pred,
                    a1_truth,
                    valid_mask=mask,
                    source_ids=a1_order,
                    domain=domain,
                    stage=f"A1+A2_vs_B1_{stage}",
                    bootstrap_seed=int(bootstrap_seed),
                    bootstrap_draws=int(bootstrap_draws),
                    one_sided_alpha=ONE_SIDED_ALPHA,
                )
                contrast = analysis.json_ready(contrast)
                contrast.update({"candidate": "A1+A2", "reference": "B1", "stage": str(stage)})
                domain_compare["stages"][str(stage)] = contrast
            comparator["domains"][domain] = domain_compare

    result = {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "SCORED_AFTER_FREEZE",
        "created_utc": utc_now(),
        "freeze_receipt": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "truth_descriptor": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "methods": methods_output,
        "common32_comparator": comparator,
        "uncertainty": {
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_draws": int(bootstrap_draws),
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "unit": "source_record",
        },
        "truth_opened": True,
        "source_text_loaded": False,
        "target_weights_loaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["score_output"] = str(output_path)
    result["score_output_sha256"] = sha256_file(output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="validate and freeze predictions before truth")
    freeze.add_argument("--descriptor", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--allow-compact-record-counts", action="store_true")
    score = subparsers.add_parser("score", help="score materialized truth after freeze")
    score.add_argument("--freeze-receipt", type=Path, required=True)
    score.add_argument("--truth-descriptor", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--allow-compact-record-counts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            receipt = freeze_evaluation(
                args.descriptor,
                args.output,
                strict_record_counts=not args.allow_compact_record_counts,
            )
        else:
            receipt = score_evaluation(
                args.freeze_receipt,
                args.truth_descriptor,
                args.output,
                strict_record_counts=not args.allow_compact_record_counts,
            )
    except (EvaluationError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "output": receipt.get("receipt_path", receipt.get("score_output"))}, sort_keys=True))
    return 0


__all__ = [
    "A1_STAGES",
    "B1_STAGES",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "EvaluationError",
    "FREEZE_RECEIPT_SCHEMA",
    "FREEZE_SCHEMA",
    "SCORE_SCHEMA",
    "TRUTH_SCHEMA",
    "freeze_evaluation",
    "main",
    "score_evaluation",
    "validate_freeze_descriptor",
]


if __name__ == "__main__":
    raise SystemExit(main())
