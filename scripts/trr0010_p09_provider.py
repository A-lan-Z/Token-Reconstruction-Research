"""Thin production provider for the shared TRR-P09 runner.

The common runner, bank loader, schedule semantics, and validation metrics are
owned by Agent 2.  This module only binds those interfaces to the frozen
TRR-0010 artifacts.  It deliberately refuses incomplete contracts instead of
inventing paths, validation rows, or schedule settings.

The provider is imported by ``trr0010_p09_qualifier`` as
``trr0010_p09_provider:build_inputs``.  It performs public artifact loading
only; it does not select records, sample positions, open evaluation truth, or
run an update itself.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from token_reconstruction.trr0007_positionwise import (
    RESIDUAL_MLP_METHOD_ID,
    load_positionwise_model_state,
    save_positionwise_state,
)
from trr0010_model import export_effective_embedding, save_directional_state


TASK_ID = "TRR-0010"
REQUIRED_DOMAINS = ("Finance", "Pile")
REQUIRED_SCHEDULE_KEYS = (
    "batch_record_indices",
    "draw_position_slots",
    "draw_record_slots",
    "used_replacement",
)


class ProviderError(RuntimeError):
    """Raised when a final public provider binding is incomplete or changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_descriptor(value: Mapping[str, Any], *, label: str) -> Path:
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ProviderError(f"{label} path is missing")
    try:
        expected_bytes = int(value["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"{label} byte binding is malformed") from exc
    expected_sha = value.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ProviderError(f"{label} SHA-256 binding is malformed")
    path = Path(raw_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ProviderError(f"{label} is not a regular file: {path}")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha:
        raise ProviderError(f"{label} changed after its binding")
    return path


def _artifact(receipt: Mapping[str, Any], role: str) -> tuple[Path, Mapping[str, Any]]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(role), Mapping):
        raise ProviderError(f"bound artifact is missing: {role}")
    descriptor = artifacts[role]
    return _verify_descriptor(descriptor, label=role), descriptor


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderError(f"{label} must contain an object")
    return value


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ProviderError(f"cannot import bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise ProviderError(f"bound module failed to import: {path}") from exc
    return module


def _load_a2_modules(receipt: Mapping[str, Any]) -> tuple[ModuleType, ModuleType, ModuleType]:
    sources = receipt.get("a2_sources")
    if not isinstance(sources, Mapping):
        raise ProviderError("A2 source bindings are missing")
    names = (
        "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
        "scripts/trr_p09/fixed_control_runner.py",
        "scripts/trr_p09/prepare_streamed_bank.py",
    )
    paths: dict[str, Path] = {}
    for relative in names:
        value = sources.get(relative)
        if not isinstance(value, Mapping):
            raise ProviderError(f"A2 source binding is missing: {relative}")
        paths[relative] = _verify_descriptor(value, label=f"A2 source {relative}")
    # fixed_control_runner imports the adapter using this canonical package
    # name.  Injecting the hash-bound module first avoids importing an
    # arbitrary copy from sys.path.
    adapter = _load_module(
        paths[names[0]], "token_reconstruction.trr_p09_fixed_control_adapter"
    )
    runner = _load_module(paths[names[1]], "trr0010_bound_p09_runner")
    loader = _load_module(paths[names[2]], "trr0010_bound_p09_loader")
    return adapter, runner, loader


def _metadata_int(metadata: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"metadata field {key!r} is not an integer") from exc
    return None


def _load_embedding(path: Path, *, device: torch.device, expected_shape: tuple[int, int]) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if "embeddings" not in handle.keys():
            raise ProviderError("public embedding must expose the 'embeddings' tensor")
        value = handle.get_tensor("embeddings")
    if tuple(value.shape) != expected_shape or value.ndim != 2:
        raise ProviderError(f"public embedding shape differs: {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise ProviderError("public embedding contains non-finite values")
    return value.to(device=device, dtype=torch.float32)


def _load_support(path: Path, *, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ProviderError(f"support artifact lacks tensor {key!r}")
        value = handle.get_tensor(key)
    if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point():
        raise ProviderError(f"support tensor {key!r} must be a rank-1 integer vector")
    return value.to(dtype=torch.long).contiguous()


def deserialize_schedule(
    path: Path,
    *,
    runner: ModuleType,
    expected: Mapping[str, Any],
    sequence_tokens: int,
    batch_records: int,
    position_budget: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Deserialize the frozen schedule into A2 ``ScheduleStep`` objects.

    This is intentionally a format adapter only.  It never generates a draw
    or changes the serialized order.
    """

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = set(REQUIRED_SCHEDULE_KEYS) - keys
        if missing:
            raise ProviderError(f"schedule is missing tensors: {sorted(missing)}")
        batch_rows = handle.get_tensor("batch_record_indices").to(dtype=torch.long)
        draw_positions = handle.get_tensor("draw_position_slots").to(dtype=torch.long)
        draw_records = handle.get_tensor("draw_record_slots").to(dtype=torch.long)
        replacement = handle.get_tensor("used_replacement").to(dtype=torch.bool)
        metadata = dict(handle.metadata() or {})
    if batch_rows.ndim != 2 or draw_positions.ndim != 2 or draw_records.ndim != 2:
        raise ProviderError("schedule tensors must be rank two")
    steps = int(batch_rows.shape[0])
    if tuple(draw_positions.shape) != (steps, int(position_budget)):
        raise ProviderError("schedule position budget differs from its binding")
    if tuple(draw_records.shape) != (steps, int(position_budget)):
        raise ProviderError("schedule draw geometry differs from its binding")
    if tuple(batch_rows.shape[1:]) != (int(batch_records),) or replacement.numel() != steps:
        raise ProviderError("schedule batch geometry differs from its binding")
    try:
        seed = int(expected["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError("schedule seed binding is malformed") from exc
    expected_steps = int(expected.get("steps", steps))
    if expected_steps != steps:
        raise ProviderError("serialized schedule step count differs from its binding")
    schedule_step_type = getattr(runner, "ScheduleStep", None)
    schedule_plan_type = getattr(runner, "SchedulePlan", None)
    if schedule_step_type is None or schedule_plan_type is None:
        raise ProviderError("A2 runner does not expose ScheduleStep/SchedulePlan")
    schedule_steps = tuple(
        schedule_step_type(
            step=index,
            batch_global_rows=tuple(int(value) for value in batch_rows[index].tolist()),
            draw_record_slots=tuple(int(value) for value in draw_records[index].tolist()),
            draw_position_slots=tuple(int(value) for value in draw_positions[index].tolist()),
            used_replacement=bool(replacement[index].item()),
        )
        for index in range(steps)
    )
    plan = schedule_plan_type.from_steps(seed=seed, steps=schedule_steps)
    plan.validate(
        record_batch_size=int(batch_records),
        position_budget=int(position_budget),
        sequence_tokens=int(sequence_tokens),
    )
    expected_digest = str(expected.get("semantic_sha256", ""))
    if plan.semantic_sha256 != expected_digest:
        raise ProviderError("schedule semantic digest differs from its binding")
    return schedule_steps, {
        "seed": seed,
        "steps": steps,
        "semantic_sha256": plan.semantic_sha256,
        "metadata": metadata,
        "exposure": plan.exposure_summary(),
    }


@dataclass(frozen=True)
class _ValidationBatch:
    activations: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    global_rows: tuple[int, ...]


class _ValidationView:
    """Lazy direct safetensors view for one declared validation domain."""

    def __init__(self, descriptor: Mapping[str, Any], *, label: str, expected_tokens: int, expected_batch: int) -> None:
        raw_file = descriptor.get("file", descriptor)
        if not isinstance(raw_file, Mapping):
            raise ProviderError(f"validation {label} file binding is malformed")
        self.path = _verify_descriptor(raw_file, label=f"validation {label}")
        keys = descriptor.get("keys", {})
        if not isinstance(keys, Mapping):
            raise ProviderError(f"validation {label} tensor-key binding is malformed")
        self.activation_key = str(keys.get("activations", "activations"))
        self.token_key = str(keys.get("token_ids", "token_ids"))
        self.mask_key = str(keys.get("attention_mask", "attention_mask"))
        self.position_key = keys.get("position_ids")
        self.sequence_tokens = int(descriptor.get("sequence_tokens", expected_tokens))
        self.batch_records = int(descriptor.get("batch_records", expected_batch))
        if self.sequence_tokens != int(expected_tokens) or self.batch_records != int(expected_batch):
            raise ProviderError(f"validation {label} geometry differs from the frozen binding")
        records = descriptor.get("record_indices")
        if records is None:
            try:
                record_count = int(descriptor["record_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(f"validation {label} record indices are missing") from exc
            records = list(range(record_count))
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ProviderError(f"validation {label} record indices are malformed")
        self.record_indices = tuple(int(value) for value in records)
        if not self.record_indices or len(self.record_indices) % self.batch_records:
            raise ProviderError(f"validation {label} records are not batch aligned")
        self.expected_post_bos_rows = int(
            descriptor.get("expected_post_bos_rows", len(self.record_indices) * (self.sequence_tokens - 1))
        )

    def batches(self) -> Iterable[_ValidationBatch]:
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            required = (self.activation_key, self.token_key, self.mask_key)
            if any(key not in handle.keys() for key in required):
                raise ProviderError(f"validation file lacks required tensor keys: {required}")
            for start in range(0, len(self.record_indices), self.batch_records):
                stop = start + self.batch_records
                activations = handle.get_slice(self.activation_key)[start:stop].contiguous()
                token_ids = handle.get_slice(self.token_key)[start:stop].contiguous()
                attention_mask = handle.get_slice(self.mask_key)[start:stop].contiguous().to(dtype=torch.bool)
                if self.position_key is None:
                    position_ids = torch.arange(self.sequence_tokens, dtype=torch.long).expand(self.batch_records, -1).clone()
                else:
                    key = str(self.position_key)
                    if key not in handle.keys():
                        raise ProviderError("validation position_ids key is absent")
                    position_ids = handle.get_slice(key)[start:stop].contiguous().to(dtype=torch.long)
                yield _ValidationBatch(
                    activations=activations,
                    token_ids=token_ids.to(dtype=torch.long),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    global_rows=tuple(self.record_indices[start:stop]),
                )


def _manifest_validation_views(
    manifest_descriptor: Mapping[str, Any],
    *,
    expected_tokens: int,
    expected_batch: int,
) -> dict[str, _ValidationView]:
    """Adapt a hash-bound public validation manifest to direct views.

    The manifest remains the source of record ordering and resource keys.  A
    caller must provide an explicit Finance/Pile grouping; style names are
    never guessed from row counts.
    """

    manifest_path = _verify_descriptor(manifest_descriptor, label="validation manifest")
    manifest = _read_json(manifest_path, label="validation manifest")
    resources = manifest.get("resources")
    grouping = manifest.get("validation_grouping")
    if not isinstance(resources, Mapping) or not isinstance(grouping, Mapping):
        raise ProviderError("validation manifest lacks resources or grouping")
    groups = grouping.get("groups_in_record_order")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ProviderError("validation manifest grouping is malformed")
    if len(groups) != int(grouping.get("record_count", len(groups))):
        raise ProviderError("validation manifest record count differs from grouping")
    if set(str(value) for value in groups) != set(REQUIRED_DOMAINS):
        raise ProviderError("validation manifest must explicitly name Finance and Pile")
    try:
        obs = resources["validation_observations"]
        truth = resources["validation_truth"]
        mask = resources["validation_valid_mask"]
    except KeyError as exc:
        raise ProviderError("validation manifest lacks public validation resources") from exc
    if not all(isinstance(value, Mapping) for value in (obs, truth, mask)):
        raise ProviderError("validation resource descriptors are malformed")
    resource_descriptors: dict[str, dict[str, Any]] = {}
    for key, value in (("observations", obs), ("truth", truth), ("mask", mask)):
        raw_path = value.get("path")
        if not isinstance(raw_path, str):
            raise ProviderError(f"validation {key} path is missing")
        resolved = (manifest_path.parent / raw_path).resolve()
        descriptor = {**value, "path": str(resolved)}
        _verify_descriptor(descriptor, label=f"validation {key}")
        resource_descriptors[key] = descriptor
    shapes = obs.get("shape")
    if not isinstance(shapes, Sequence) or len(shapes) != 3:
        raise ProviderError("validation observation shape is missing")
    if int(shapes[1]) != int(expected_tokens):
        raise ProviderError("validation manifest sequence width differs from binding")
    _verify_descriptor(resource_descriptors["truth"], label="validation truth")
    _verify_descriptor(resource_descriptors["mask"], label="validation mask")
    views: dict[str, _ValidationView] = {}
    declared_positions = grouping.get("post_bos_positions_by_style")
    for domain in REQUIRED_DOMAINS:
        indices = tuple(index for index, value in enumerate(groups) if str(value) == domain)
        if not indices or len(indices) % int(expected_batch):
            raise ProviderError(f"validation {domain} records are not batch aligned")
        descriptor = {
            "file": resource_descriptors["observations"],
            "keys": {
                "activations": resource_descriptors["observations"].get("tensor_key", "activations"),
                "token_ids": resource_descriptors["truth"].get("tensor_key", "token_ids"),
                "attention_mask": resource_descriptors["mask"].get("tensor_key", "attention_mask"),
            },
            "sequence_tokens": int(expected_tokens),
            "batch_records": int(expected_batch),
            "record_indices": indices,
            "expected_post_bos_rows": (
                int(declared_positions[domain])
                if isinstance(declared_positions, Mapping) and domain in declared_positions
                else len(indices) * (int(expected_tokens) - 1)
            ),
        }
        views[domain] = _ValidationView(
            descriptor,
            label=domain,
            expected_tokens=expected_tokens,
            expected_batch=expected_batch,
        )
    return views


def _validation_views(contract: Mapping[str, Any], *, expected_tokens: int, expected_batch: int) -> dict[str, _ValidationView]:
    section = contract.get("validation_views")
    if section is None and isinstance(contract.get("validation"), Mapping):
        section = contract["validation"].get("views")
    if section is None and isinstance(contract.get("validation"), Mapping):
        manifest = contract["validation"].get("manifest")
        if isinstance(manifest, Mapping):
            return _manifest_validation_views(
                manifest,
                expected_tokens=expected_tokens,
                expected_batch=expected_batch,
            )
    if not isinstance(section, Mapping) or set(section) != set(REQUIRED_DOMAINS):
        raise ProviderError("final contract must bind exactly Finance and Pile validation views")
    return {
        domain: _ValidationView(section[domain], label=domain, expected_tokens=expected_tokens, expected_batch=expected_batch)
        for domain in REQUIRED_DOMAINS
    }


def _contract_settings(contract: Mapping[str, Any], receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    training = contract.get("training")
    if not isinstance(training, Mapping):
        training = contract.get("qualification")
    if not isinstance(training, Mapping):
        raise ProviderError("final contract training settings are missing")
    return training


def _checkpoint_export_factory(
    *,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    embedding: torch.Tensor,
    fixture_source: Any,
    fixture_rows: Sequence[int],
) -> Callable[[Any, Path], Mapping[str, Any]]:
    """Build the complete deployment export/reload probe.

    The callback is invoked after discarded updates while Adam is still live.
    It writes a base-only decoder and a full effective readout, then reloads
    both and compares nonzero-Delta logits and argmax IDs on the first two
    scheduled public fitting records at positions 1 and 2.  The probe is
    intentionally bounded and produces no selected contender.
    """

    _, base_binding = _artifact(receipt, "base_state")
    _, bank_binding = _artifact(receipt, "bank_manifest")
    settings = receipt["settings"]
    contract_digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def checkpoint_export(runtime: Any, output_root: Path) -> Mapping[str, Any]:
        root = Path(output_root).expanduser().resolve() / "provider_probe"
        root.mkdir(parents=True, exist_ok=True)
        step = int(settings["probe_steps"])
        base_path = root / f"base_decoder_step_{step:06d}.safetensors"
        effective_path = root / f"effective_readout_step_{step:06d}.safetensors"

        # Fetch a fixed public fitting fixture through the already bound random
        # access source.  This is H/mask only for the equivalence diagnostic;
        # no private labels or evaluation records are opened.
        batch = fixture_source.batch_for_global_rows(tuple(int(row) for row in fixture_rows))
        if len(fixture_rows) < 2:
            raise ProviderError("deployment fixture must contain two records")
        device = next(runtime.decoder.parameters()).device
        activation = batch.activations[:2].to(device=device)
        valid_mask = batch.attention_mask[:2].to(device=device, dtype=torch.bool)
        record_slots = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        position_slots = torch.tensor([1, 2, 1, 2], device=device, dtype=torch.long)
        if not bool(valid_mask[record_slots, position_slots].all().item()):
            raise ProviderError("deployment fixture contains an invalid post-BOS position")
        with torch.inference_mode():
            projected = runtime.decoder.projected_hidden(activation, valid_mask)
            query_rows = projected[record_slots, position_slots]
            reference_logits = runtime.hook.score_rows(
                query_rows,
                runtime.hook.logit_scale,
                embedding,
            ).detach().cpu().contiguous()

        base_export = save_positionwise_state(
            base_path,
            runtime.decoder,
            method_id=RESIDUAL_MLP_METHOD_ID,
            selected_step=step,
            initialization="discarded directional probe base parameters",
            distribution="TRR-0010 qualification-only base deployment export",
            bottleneck_size=runtime.decoder.bottleneck_size,
            metadata={
                "serialization_only": True,
                "directional_readout_excluded": True,
                "provider_contract_sha256": contract_digest,
                "base_binding_sha256": base_binding["sha256"],
            },
        )
        effective_export = export_effective_embedding(
            effective_path,
            runtime.hook,
            embedding,
            metadata={
                "serialization_only": True,
                "selected_step": step,
                "provider_contract_sha256": contract_digest,
                "fit_manifest_sha256": bank_binding["sha256"],
            },
        )
        # export_effective_embedding materializes its own full W and releases
        # it before returning.  Empty the CUDA allocator cache before loading
        # the serialized W, so the probe does not retain two full dictionaries.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        reloaded_base = load_positionwise_model_state(
            Path(base_export["path"]),
            method_id=RESIDUAL_MLP_METHOD_ID,
            hidden_size=runtime.decoder.hidden_size,
            vocabulary_size=runtime.decoder.vocabulary_size,
            context_width=runtime.decoder.context_width,
            bottleneck_size=runtime.decoder.bottleneck_size,
        ).to(device)
        effective_state = load_file(str(effective_export["path"]), device="cpu")
        if set(effective_state) != {"embeddings"}:
            raise ProviderError("effective readout export tensor keys differ")
        reloaded_embedding = effective_state["embeddings"].to(device=device, dtype=torch.float32)
        del effective_state
        with torch.inference_mode():
            reloaded_projected = reloaded_base.projected_hidden(activation, valid_mask)
            reloaded_logits = reloaded_base.logits_from_rows(
                reloaded_projected,
                record_slots,
                position_slots,
                reloaded_embedding,
            ).detach().cpu().contiguous()
        exact_logits = bool(torch.equal(reference_logits, reloaded_logits))
        exact_argmax = bool(torch.equal(reference_logits.argmax(-1), reloaded_logits.argmax(-1)))
        max_abs = float((reference_logits - reloaded_logits).abs().max().item())
        if not exact_logits or not exact_argmax:
            raise ProviderError(
                f"deployment export/reload changed fixture outputs: exact={exact_logits}, max_abs={max_abs}"
            )
        del reloaded_embedding, reloaded_base, activation, valid_mask, projected, query_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "PROBE_EXPORTED_RELOADED_EXACT",
            "base_decoder": base_export,
            "effective_readout": effective_export,
            "fixture": {
                "kind": "public_fitting_bank_schedule_step_zero_batch_first_two_records",
                "global_rows": [int(row) for row in fixture_rows],
                "positions": [1, 2],
                "rows_scored": 4,
            },
            "reload_check": {
                "exact_logits": exact_logits,
                "exact_argmax": exact_argmax,
                "max_abs": max_abs,
                "materialized_readout_released_before_reload": True,
            },
        }

    return checkpoint_export


def build_inputs(
    binding_receipt: Mapping[str, Any],
    lease_caps: Mapping[str, Any],
    device: torch.device,
    preparation_guard: Callable[[str], None],
) -> Mapping[str, Any]:
    """Build the exact provider mapping required by the shared qualifier."""

    del lease_caps
    settings = binding_receipt.get("settings")
    if not isinstance(settings, Mapping):
        raise ProviderError("qualifier settings are missing")
    contract_path, _ = _artifact(binding_receipt, "contract")
    bank_path, _ = _artifact(binding_receipt, "bank_manifest")
    schedule_path, _ = _artifact(binding_receipt, "schedule")
    base_path, _ = _artifact(binding_receipt, "base_state")
    embedding_path, _ = _artifact(binding_receipt, "public_embedding")
    support_ids_path, _ = _artifact(binding_receipt, "support_ids")
    support_counts_path, _ = _artifact(binding_receipt, "support_counts")
    contract = _read_json(contract_path, label="final contract")
    a2_adapter, runner, loader_module = _load_a2_modules(binding_receipt)
    preparation_guard("after_a2_import")

    hidden_size = int(settings["hidden_size"])
    vocabulary_size = int(settings["vocabulary_size"])
    sequence_tokens = int(settings["sequence_tokens"])
    batch_records = int(settings["batch_records"])
    position_budget = int(settings["position_budget"])

    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        base_metadata = dict(handle.metadata() or {})
    context_width = _metadata_int(base_metadata, "context_width", "sequence_tokens")
    bottleneck_size = _metadata_int(base_metadata, "bottleneck_size")
    if context_width is None or bottleneck_size is None:
        raise ProviderError("selected base state lacks context/bottleneck metadata")
    preparation_guard("before_base_load")
    base_decoder = load_positionwise_model_state(
        base_path,
        method_id=RESIDUAL_MLP_METHOD_ID,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        bottleneck_size=bottleneck_size,
    ).to(device)
    preparation_guard("after_base_load")

    preparation_guard("before_embedding_load")
    public_embedding = _load_embedding(
        embedding_path,
        device=device,
        expected_shape=(vocabulary_size, hidden_size),
    )
    preparation_guard("after_embedding_load")
    support_ids = _load_support(support_ids_path, key="support_ids")
    support_counts = _load_support(support_counts_path, key="support_counts")
    if support_ids.numel() != support_counts.numel() or support_ids.numel() <= 0:
        raise ProviderError("support ID/count vectors differ or are empty")
    if not torch.equal(support_ids, torch.sort(support_ids).values):
        raise ProviderError("support IDs are not sorted")
    expected_support = contract.get("directional_readout", {}).get("support_count") if isinstance(contract.get("directional_readout"), Mapping) else None
    if expected_support is not None and int(expected_support) != int(support_ids.numel()):
        raise ProviderError("support count differs from the final contract")
    preparation_guard("after_support_load")

    loader_type = getattr(loader_module, "StreamedBankLoader", None)
    if loader_type is None:
        raise ProviderError("A2 loader does not expose StreamedBankLoader")
    loader = loader_type(bank_path, device="cpu")
    source_type = getattr(runner, "RandomAccessLoaderSource", None)
    if source_type is None:
        raise ProviderError("A2 runner does not expose RandomAccessLoaderSource")
    source = source_type(loader)
    preparation_guard("after_bank_loader")

    schedule_steps, schedule_receipt = deserialize_schedule(
        schedule_path,
        runner=runner,
        expected=binding_receipt["schedule"],
        sequence_tokens=sequence_tokens,
        batch_records=batch_records,
        position_budget=position_budget,
    )
    preparation_guard("after_schedule_load")

    training = _contract_settings(contract, binding_receipt)
    try:
        validation_every = int(training["validation_every"])
        gradient_clip_norm = float(training["gradient_clip_norm"])
        selection_metric = str(training["selection_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError("final contract lacks validation/gradient settings") from exc
    if validation_every <= 0 or gradient_clip_norm <= 0.0 or not selection_metric:
        raise ProviderError("final contract validation/gradient settings are invalid")
    config = runner.RunnerConfig(
        steps=int(settings["probe_steps"]),
        record_batch_size=batch_records,
        position_budget=position_budget,
        validation_every=validation_every,
        selection_metric=selection_metric,
        seed=int(binding_receipt["schedule"]["seed"]),
        learning_rate=float(settings["base_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        gradient_clip_norm=gradient_clip_norm,
        train_sequence_tokens=sequence_tokens,
        hidden_size=hidden_size,
        expected_activation_dtype="torch.bfloat16",
    )
    config.validate()

    validation_geometry = binding_receipt["validation_geometry"]
    validation_tokens = int(validation_geometry["sequence_tokens"])
    validation_batch_records = int(validation_geometry["batch_records"])
    views = _validation_views(
        contract,
        expected_tokens=validation_tokens,
        expected_batch=validation_batch_records,
    )
    if any(view.expected_post_bos_rows <= 0 for view in views.values()):
        raise ProviderError("validation views have no declared scored rows")

    def validation_callback(step: int, evaluate_view: Callable[[Iterable[Any]], Mapping[str, Any]]) -> Mapping[str, Any]:
        del step
        per_domain = {domain: evaluate_view(views[domain].batches()) for domain in REQUIRED_DOMAINS}
        return runner.aggregate_domain_validation(per_domain, required_domains=REQUIRED_DOMAINS)

    preparation_guard("after_validation_binding")
    provider_receipt = {
        "schema": "token-reconstruction.trr0010-p09-provider.v1",
        "task_id": TASK_ID,
        "a2_modules": {
            "adapter": str(getattr(a2_adapter, "__file__", "")),
            "runner": str(getattr(runner, "__file__", "")),
            "loader": str(getattr(loader_module, "__file__", "")),
        },
        "support_count": int(support_ids.numel()),
        "schedule": schedule_receipt,
        "validation_domains": {
            domain: {"record_count": len(view.record_indices), "expected_post_bos_rows": view.expected_post_bos_rows}
            for domain, view in views.items()
        },
        "merged_primary": True,
        "compute_base_logits": False,
    }
    return {
        "runner": runner,
        "adapter_module": a2_adapter,
        "runner_module": runner,
        "loader_module": loader_module,
        "base_decoder": base_decoder,
        "public_embedding": public_embedding,
        "support_ids": support_ids,
        "support_counts": support_counts,
        "source": source,
        "schedule_steps": schedule_steps,
        "config": config,
        "validation_callback": validation_callback,
        "validation_sequence_tokens": validation_tokens,
        "validation_batch_records": validation_batch_records,
        "validation_activation_dtype": torch.bfloat16,
        "training_activation_dtype": torch.bfloat16,
        "checkpoint_export": _checkpoint_export_factory(
            receipt=binding_receipt,
            contract=contract,
            embedding=public_embedding,
            fixture_source=source,
            fixture_rows=schedule_steps[0].batch_global_rows,
        ),
        "provider_receipt": provider_receipt,
    }


__all__ = ["ProviderError", "build_inputs", "deserialize_schedule"]
